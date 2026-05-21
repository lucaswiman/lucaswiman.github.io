// Self-play between two MCTS+NN players.
//
// Used by the training loop to generate AlphaZero-style replay data for
// the AI by pitting it against a player-imitator network. Both sides
// run MCTS; each side gets its own `Policy` (i.e. its own NN).
//
// Compute envelope: each MCTS run is O(simulations) NN evaluations, and
// a game can run ~80–120 plies, so even short batches are heavy on a
// phone. Callers control the cost with `search.simulations`, `count`,
// and `maxPlies`, and can interrupt with an `AbortSignal`. The module
// itself only awaits the policy and a microtask between plies so the
// event loop stays responsive; the heavy lifting is in `search`.

import { encodeState } from '../encoding/index.js';
import {
  applyMove,
  initialState,
  isTerminal,
  outcome,
  sideToMove,
  toFen,
  type Color,
  type GameState,
  type Move,
  type Outcome,
} from '../rules/index.js';
import {
  search,
  type Policy,
  type SearchConfig,
} from '../mcts/index.js';
import { buildTrainingExamples, type GameMoveRecord } from '../agent/index.js';
import type { TrainingExample } from '../nn/index.js';

/** Outcome of one self-play game from one side's perspective. */
export type SideResult = 'win' | 'loss' | 'draw';

/** Map a game outcome onto whether the given color won/lost/drew. */
function resultForColor(out: Outcome, color: Color): SideResult {
  if (out === 'draw') return 'draw';
  if (out === 'ongoing') return 'draw';
  return (out === 'white-wins' && color === 'w') ||
         (out === 'black-wins' && color === 'b')
    ? 'win'
    : 'loss';
}

export interface PlayGameConfig {
  white: Policy;
  black: Policy;
  search: SearchConfig;
  /**
   * Hard ply cap. A game that hits the cap before a natural terminal
   * state is treated as a draw — both sides failed to finish. Defaults
   * to 240 plies (= 120 full moves), which is generous for short MCTS
   * games and well below chess's actual 50-move/75-move limits.
   */
  maxPlies?: number;
  /** Optional starting state; defaults to the standard initial position. */
  startingState?: GameState;
  /** Abort the game between plies. The half-finished game returns 'draw'. */
  signal?: AbortSignal;
  /**
   * AlphaZero-style sampling temperature. For the first
   * `temperaturePlies` plies, the move is sampled from the visit-count
   * distribution raised to 1/`temperature` instead of taking the argmax.
   * After that, plies use the argmax (greedy) move. Defaults to 0
   * plies of sampling — i.e. argmax everywhere — which preserves the
   * "play your best move" behaviour used outside training (human play,
   * Stockfish sparring). Self-play callers opt in to widen the
   * exploration of opening / mid-game lines so the replay buffer isn't
   * filled with the same shuffle every game.
   */
  temperaturePlies?: number;
  /**
   * Temperature applied during the sampled plies above. Lower → sharper
   * toward argmax; higher → flatter toward uniform. Ignored when
   * `temperaturePlies <= 0`. Defaults to 1.0 (sample proportional to
   * raw visit counts, AlphaZero's choice).
   */
  temperature?: number;
  /**
   * RNG used for visit-count sampling. Defaults to Math.random. Tests
   * pass a seeded PRNG to make sampled games reproducible.
   */
  rng?: () => number;
  /**
   * Optional per-ply observer. Fires once with `plyIndex = 0` and
   * `lastMove = null` for the starting position, then once after each
   * move with the post-move FEN. Used by the UI to render a live board
   * during self-play; the core loop itself does not depend on it.
   */
  onPly?: (info: { fen: string; plyIndex: number; lastMove: Move | null }) => void;
}

/**
 * Sample a move from a visit-count distribution raised to 1/temperature.
 *
 * `n_i^(1/τ)` is the AlphaZero recipe: τ=1 samples proportional to raw
 * visit counts, τ→0 collapses to argmax, τ→∞ flattens to uniform. We
 * never get an empty map in practice (`search` throws on terminal
 * positions) but guard the totals against zero so a flaky search call
 * can't sample NaN.
 */
function sampleMoveByVisits(
  visitCounts: Map<Move, number>,
  temperature: number,
  rng: () => number,
): Move {
  const entries = Array.from(visitCounts.entries());
  if (entries.length === 0) {
    throw new Error('sampleMoveByVisits: empty visit-count distribution');
  }
  const invT = 1 / temperature;
  let total = 0;
  const weights: number[] = entries.map(([, n]) => {
    const w = n > 0 ? Math.pow(n, invT) : 0;
    total += w;
    return w;
  });
  if (total === 0) return entries[0][0];
  let r = rng() * total;
  for (let i = 0; i < entries.length; i++) {
    r -= weights[i];
    if (r <= 0) return entries[i][0];
  }
  return entries[entries.length - 1][0];
}

export interface PlayGameResult {
  outcome: Outcome;          // 'white-wins' | 'black-wins' | 'draw' (never 'ongoing')
  plies: number;             // total plies actually played
  whiteMoves: GameMoveRecord[];
  blackMoves: GameMoveRecord[];
}

/**
 * Play a single game between two policies. Each ply runs MCTS for the
 * side to move and applies either the argmax visit count or, for the
 * first `temperaturePlies` plies, a visit-count sample.
 *
 * Sampling early plies is the standard AlphaZero recipe for training-
 * data generation: it spreads the replay buffer over many different
 * opening / mid-game lines instead of grinding the same forced line
 * each game. After the temperature window, the loop reverts to argmax
 * so the late game still plays each side's strongest reply.
 *
 * Outside training (human play, Stockfish sparring) callers leave
 * `temperaturePlies` at its default of 0 and the loop is greedy
 * throughout.
 */
export async function playOneGame(
  config: PlayGameConfig,
): Promise<PlayGameResult> {
  let state = config.startingState ?? initialState();
  const whiteMoves: GameMoveRecord[] = [];
  const blackMoves: GameMoveRecord[] = [];
  const maxPlies = config.maxPlies ?? 240;
  const temperaturePlies = config.temperaturePlies ?? 0;
  const temperature = config.temperature ?? 1.0;
  const rng = config.rng ?? Math.random.bind(Math);

  // Announce the starting position before any moves have been played
  // so the observer can render the board immediately.
  config.onPly?.({ fen: toFen(state), plyIndex: 0, lastMove: null });

  let plies = 0;
  while (plies < maxPlies && !isTerminal(state)) {
    if (config.signal?.aborted) break;

    const side = sideToMove(state);
    const policy = side === 'w' ? config.white : config.black;
    const features = encodeState(state);

    const result = await search(state, policy, config.search);
    const useSampling = plies < temperaturePlies && temperature > 0;
    const move = useSampling
      ? sampleMoveByVisits(result.visitCounts, temperature, rng)
      : result.bestMove;

    const record: GameMoveRecord = { features, move, sideToMove: side };
    if (side === 'w') whiteMoves.push(record);
    else blackMoves.push(record);

    state = applyMove(state, move);
    plies++;
    config.onPly?.({ fen: toFen(state), plyIndex: plies, lastMove: move });

    // Yield to the event loop between plies so the UI thread is not
    // starved for several seconds on phones.
    await new Promise<void>(resolve => setTimeout(resolve, 0));
  }

  const finalOutcome: Outcome = isTerminal(state) ? outcome(state) : 'draw';
  return { outcome: finalOutcome, plies, whiteMoves, blackMoves };
}

export interface SelfPlayBatchConfig {
  /** Policy for the side the AI plays. */
  aiPolicy: Policy;
  /** Policy for the side the player-imitator plays. */
  playerPolicy: Policy;
  /** MCTS configuration applied to both sides. */
  search: SearchConfig;
  /** Number of games to play. */
  count: number;
  /** Per-game ply cap. See `playOneGame`. */
  maxPlies?: number;
  /**
   * If true (default), alternate which color the AI plays so the agent
   * sees both sides of the same imitator. If false, the AI always plays
   * white.
   */
  alternateColors?: boolean;
  /** See `PlayGameConfig.temperaturePlies`. Applied to every game. */
  temperaturePlies?: number;
  /** See `PlayGameConfig.temperature`. Applied to every game. */
  temperature?: number;
  /** RNG used for visit-count sampling. Defaults to Math.random. */
  rng?: () => number;
  /**
   * Optional curriculum hook: returns the starting state for game
   * `gameIndex` (0-based). When unset, every game starts from the
   * standard chess initial position. Used by the endgame-curriculum
   * self-play mode: callers wrap a pool of preset FENs and sample one
   * per game so the value head sees decisive ±1 outcomes inside the
   * ply cap.
   */
  startingStateForGame?: (gameIndex: number) => GameState;
  /** Per-game progress callback. */
  onGame?: (idx: number, info: {
    aiColor: Color;
    aiResult: SideResult;
    plies: number;
  }) => void;
  /**
   * Optional per-ply observer, scoped to the current game. Used by the
   * UI to render a live board; pure training callers can leave it unset.
   */
  onPly?: (gameIndex: number, info: {
    aiColor: Color;
    fen: string;
    plyIndex: number;
    lastMove: Move | null;
  }) => void;
  /** Abort the batch between games. The current game finishes. */
  signal?: AbortSignal;
}

export interface SelfPlayBatchResult {
  /** Examples to feed into the AI's replay buffer. */
  aiTrainingExamples: TrainingExample[];
  games: number;
  wins: number;
  losses: number;
  draws: number;
}

/**
 * Run a batch of self-play games and return AlphaZero-style training
 * examples for the AI's side.
 *
 * Training-example construction reuses the existing winner rule
 * (`buildTrainingExamples`): only the moves the AI played on the
 * winning side carry a policy target. Losing-side AI positions still
 * contribute a value target. The imitator's moves are not turned into
 * AI training examples — those are not the AI's decisions.
 */
export async function runSelfPlayBatch(
  config: SelfPlayBatchConfig,
): Promise<SelfPlayBatchResult> {
  const aiTrainingExamples: TrainingExample[] = [];
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let played = 0;

  for (let i = 0; i < config.count; i++) {
    if (config.signal?.aborted) break;

    const aiColor: Color =
      config.alternateColors === false ? 'w' : (Math.random() < 0.5 ? 'w' : 'b');

    const game = await playOneGame({
      white: aiColor === 'w' ? config.aiPolicy : config.playerPolicy,
      black: aiColor === 'b' ? config.aiPolicy : config.playerPolicy,
      search: config.search,
      maxPlies: config.maxPlies,
      startingState: config.startingStateForGame?.(i),
      temperaturePlies: config.temperaturePlies,
      temperature: config.temperature,
      rng: config.rng,
      signal: config.signal,
      onPly: config.onPly
        ? (info) => config.onPly!(i, { aiColor, ...info })
        : undefined,
    });

    const aiMoves = aiColor === 'w' ? game.whiteMoves : game.blackMoves;
    const aiExamples = buildTrainingExamples(aiMoves, game.outcome);
    aiTrainingExamples.push(...aiExamples);

    const aiResult = resultForColor(game.outcome, aiColor);
    if (aiResult === 'win') wins++;
    else if (aiResult === 'loss') losses++;
    else draws++;
    played++;

    config.onGame?.(i, { aiColor, aiResult, plies: game.plies });
  }

  return {
    aiTrainingExamples,
    games: played,
    wins,
    losses,
    draws,
  };
}
