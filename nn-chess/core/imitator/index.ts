// PlayerImitator — a second ChessNet whose only job is to predict the
// user's next move from a position the user is facing. The opponent's
// training loop plays self-play games against this imitator under MCTS
// so we can generate replay data faster than waiting for the user to
// finish a game.
//
// Storage layout: weights + replay buffer + meta, namespaced under
// `imitator/…` so the imitator's persistence does not collide with the
// opponent's `agent/…` keys (the legacy `agent/` prefix is kept on disk
// for backward compatibility with users' trained networks; see
// OPPONENT_STORAGE_KEYS in core/opponent). Both pieces of state are
// independently versioned via NN_VERSION.

import type { GameMoveRecord } from '../opponent/index.js';
import { buildImitationExamples, policyFromNet } from '../opponent/index.js';
import {
  ChessNet,
  deserialize,
  NN_VERSION,
  type TrainResult,
} from '../nn/index.js';
import type { GameState, Move, Outcome } from '../rules/index.js';
import { isTerminal, legalMoves } from '../rules/index.js';
import { search, type SearchConfig } from '../mcts/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
  type ReplayBuffer,
} from '../training/index.js';
import {
  type BlobStorage,
  getJson,
  putJson,
} from '../storage/index.js';

export const IMITATOR_STORAGE_KEYS = {
  WEIGHTS: 'imitator/weights',
  REPLAY: 'imitator/replay',
  META: 'imitator/meta',
} as const;

// ── selectImitatorMove ───────────────────────────────────────────────────────

export interface ImitatorSelectOptions {
  /**
   * When provided, drive move selection with MCTS using the imitator's
   * own net for priors and value. When absent, pick straight from the
   * policy head (the raw "what would the user play here?" view, useful
   * for tests and for interpretability inspection).
   *
   * MCTS is the same `search` function the opponent and self-play use,
   * so the mate-in-1 override that lives inside it applies here too.
   * That's consistent with the project's single approved heuristic
   * ("checkmate is terminal") — see `core/mcts` for details — and with
   * the existing `core/selfplay` flow which already drives the imitator
   * with MCTS.
   */
  search?: SearchConfig;
  /**
   * Sampling temperature.
   *
   * - With `search` set, this is the visit-count sampling temperature
   *   (T=0 picks argmax of visits; T=1 samples proportional to visits).
   * - Without `search`, this is the raw-policy sampling temperature
   *   (T=0 greedy over priors; T>0 samples ∝ P^(1/T)).
   *
   * Defaults to 0.
   */
  temperature?: number;
  rng?: () => number;
}

export interface ImitatorSelectResult {
  move: Move;
  /** Legal-move priors out of the policy head. */
  priors: Map<Move, number>;
  /** Set only when `options.search` was provided. */
  search?: {
    rootValue: number;
    visitCounts: Map<Move, number>;
    edgeValues: Map<Move, number>;
    mateInOneMoves: Move[];
  };
}

/**
 * Pick a move the imitator would play from `state`.
 *
 * Two modes:
 *
 * - **MCTS mode** (`options.search` provided, default in the UI): runs
 *   the same `search()` MCTS the opponent and self-play use, with the
 *   imitator's net supplying priors and value. The mate-in-1 override
 *   baked into `search` applies; this is the project's one approved
 *   heuristic ("checkmate is terminal") and is also what self-play uses
 *   when the imitator moves there, so MCTS-vs-imitator play is the
 *   same algorithm as in-loop self-play.
 *
 * - **Policy-head mode** (`options.search` absent): asks the net "what
 *   would the user play here?" directly, no value-based lookahead.
 *   Greedy by default; `temperature > 0` samples ∝ P^(1/T). Useful for
 *   tests and for an interpretability view that shows the raw policy
 *   without MCTS smoothing it out.
 *
 * Reuses `policyFromNet` from core/opponent so legal-move masking +
 * renormalization stays a single source of truth.
 *
 * Throws if `state` is terminal — callers should never reach here.
 */
export async function selectImitatorMove(
  net: ChessNet,
  state: GameState,
  options: ImitatorSelectOptions = {},
): Promise<ImitatorSelectResult> {
  if (isTerminal(state)) {
    throw new Error('selectImitatorMove: called on a terminal GameState');
  }
  const moves = legalMoves(state);
  if (moves.length === 0) {
    throw new Error('selectImitatorMove: no legal moves at non-terminal state');
  }

  const policy = policyFromNet(net);
  const temperature = options.temperature ?? 0;

  if (options.search !== undefined) {
    const result = await search(state, policy, {
      ...options.search,
      rng: options.search.rng ?? options.rng,
    });
    const { rootValue, visitCounts, priorPolicy, edgeValues, mateInOneMoves } = result;
    const move =
      mateInOneMoves.length > 0
        ? mateInOneMoves[0]
        : pickFromVisits(visitCounts, priorPolicy, temperature, options.rng);
    return {
      move,
      priors: priorPolicy,
      search: { rootValue, visitCounts, edgeValues, mateInOneMoves },
    };
  }

  // Raw policy-head path.
  const { priors } = await policy.evaluate(state);

  if (temperature === 0 || moves.length === 1) {
    let bestMove = moves[0];
    let bestP = priors.get(bestMove) ?? 0;
    for (const m of moves) {
      const p = priors.get(m) ?? 0;
      if (p > bestP || (p === bestP && m < bestMove)) {
        bestP = p;
        bestMove = m;
      }
    }
    return { move: bestMove, priors };
  }

  const invT = 1 / temperature;
  const weights = moves.map(m => Math.pow(priors.get(m) ?? 0, invT));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) {
    const idx = Math.floor((options.rng ?? Math.random)() * moves.length);
    return { move: moves[Math.min(idx, moves.length - 1)], priors };
  }
  const draw = (options.rng ?? Math.random)() * total;
  let cumulative = 0;
  for (let i = 0; i < moves.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return { move: moves[i], priors };
  }
  return { move: moves[moves.length - 1], priors };
}

function pickFromVisits(
  visitCounts: Map<Move, number>,
  priors: Map<Move, number>,
  temperature: number,
  rng: (() => number) | undefined,
): Move {
  const entries = Array.from(visitCounts.entries());
  if (entries.length === 0) {
    throw new Error('pickFromVisits: empty visit-count map (root had no legal moves?)');
  }
  if (temperature === 0 || entries.length === 1) {
    // Argmax of visits; ties broken by prior, then lex by UCI for determinism.
    let bestMove = entries[0][0];
    let bestN = entries[0][1];
    let bestP = priors.get(bestMove) ?? 0;
    for (let i = 1; i < entries.length; i++) {
      const [m, n] = entries[i];
      const p = priors.get(m) ?? 0;
      if (n > bestN || (n === bestN && p > bestP) || (n === bestN && p === bestP && m < bestMove)) {
        bestMove = m;
        bestN = n;
        bestP = p;
      }
    }
    return bestMove;
  }
  // Sample ∝ N^(1/T).
  const invT = 1 / temperature;
  const weights = entries.map(([, n]) => Math.pow(n, invT));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) {
    const idx = Math.floor((rng ?? Math.random)() * entries.length);
    return entries[Math.min(idx, entries.length - 1)][0];
  }
  const draw = (rng ?? Math.random)() * total;
  let cumulative = 0;
  for (let i = 0; i < entries.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return entries[i][0];
  }
  return entries[entries.length - 1][0];
}

export interface PlayerImitator {
  /**
   * Add the user's moves from a finished game to the imitator's
   * replay buffer. Every move becomes a positive example for the
   * policy head; the value head learns the eventual outcome from the
   * user's side.
   */
  recordUserMoves(userMoves: GameMoveRecord[], finalOutcome: Outcome): void;

  /**
   * Pick a move from this imitator. Thin wrapper around
   * `selectImitatorMove(net, state, …)` for callers that already hold a
   * `PlayerImitator`. See `selectImitatorMove` for MCTS vs raw-policy
   * mode semantics.
   */
  selectMove(
    state: GameState,
    options?: ImitatorSelectOptions,
  ): Promise<ImitatorSelectResult>;

  /**
   * One gradient step on a minibatch sampled from the replay buffer.
   * Returns null if the buffer holds fewer examples than `batchSize`.
   */
  trainStep(batchSize: number, rng?: () => number): Promise<TrainResult | null>;

  /** Persist weights + replay buffer + meta to storage. */
  saveTo(storage: BlobStorage): Promise<void>;

  /** Read-only handles for the UI / interpretability viewer / self-play. */
  readonly net: ChessNet;
  readonly replayBuffer: ReplayBuffer;
}

export interface CreatePlayerImitatorOptions {
  net: ChessNet;
  replayBuffer: ReplayBuffer;
}

export function createPlayerImitator(
  opts: CreatePlayerImitatorOptions,
): PlayerImitator {
  const { net, replayBuffer } = opts;

  const recordUserMoves = (
    userMoves: GameMoveRecord[],
    finalOutcome: Outcome,
  ): void => {
    if (userMoves.length === 0) return;
    const examples = buildImitationExamples(userMoves, finalOutcome);
    replayBuffer.addAll(examples);
  };

  const selectMove = (
    state: GameState,
    options: ImitatorSelectOptions = {},
  ): Promise<ImitatorSelectResult> =>
    selectImitatorMove(net, state, options);

  const trainStep = async (
    batchSize: number,
    rng?: () => number,
  ): Promise<TrainResult | null> => {
    if (replayBuffer.size() < batchSize) return null;
    const batch = replayBuffer.sample(batchSize, rng);
    return net.trainBatch(batch);
  };

  const saveTo = async (storage: BlobStorage): Promise<void> => {
    const meta = { NN_VERSION };
    await Promise.all([
      net.serialize().then(bytes => storage.put(IMITATOR_STORAGE_KEYS.WEIGHTS, bytes)),
      storage.put(IMITATOR_STORAGE_KEYS.REPLAY, replayBuffer.serialize()),
      putJson(storage, IMITATOR_STORAGE_KEYS.META, meta),
    ]);
  };

  return {
    recordUserMoves,
    selectMove,
    trainStep,
    saveTo,
    get net() { return net; },
    get replayBuffer() { return replayBuffer; },
  };
}

/**
 * Restore an imitator from storage if compatible weights are present,
 * otherwise return a fresh one built from `defaults`. Mirrors
 * `loadOpponent` but reads from the imitator's keyspace.
 */
export async function loadPlayerImitator(
  storage: BlobStorage,
  defaults: { net: ChessNet; replayCapacity: number },
): Promise<PlayerImitator> {
  const fresh = () =>
    createPlayerImitator({
      net: defaults.net,
      replayBuffer: createReplayBuffer(defaults.replayCapacity),
    });

  const meta = await getJson<{ NN_VERSION: number }>(
    storage,
    IMITATOR_STORAGE_KEYS.META,
  );

  if (meta === null || meta.NN_VERSION !== NN_VERSION) {
    return fresh();
  }

  const [weightBytes, replayBytes] = await Promise.all([
    storage.get(IMITATOR_STORAGE_KEYS.WEIGHTS),
    storage.get(IMITATOR_STORAGE_KEYS.REPLAY),
  ]);

  if (weightBytes === null) return fresh();

  let net: ChessNet;
  try {
    net = await deserialize(weightBytes);
  } catch {
    return fresh();
  }

  const replayBuffer =
    replayBytes !== null
      ? deserializeReplayBuffer(replayBytes, defaults.replayCapacity)
      : createReplayBuffer(defaults.replayCapacity);

  return createPlayerImitator({ net, replayBuffer });
}
