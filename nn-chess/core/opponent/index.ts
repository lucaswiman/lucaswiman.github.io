// Opponent — glues ChessNet + MCTS + ReplayBuffer into the single object
// the UI talks to. This is the network the user plays against (the AI
// side of the matchup); the companion `imitator` network in
// `core/imitator/` stands in for the user during self-play.
//
// Responsibilities:
//   - Adapt ChessNet to MCTS's Policy interface (mask illegal moves,
//     renormalize, pass value through).
//   - Run MCTS with OpponentConfig, then pick a move from the visit-count
//     distribution (temperature=0 → greedy; temperature>0 → sample).
//   - Convert finished games into TrainingExamples via the
//     winner-reinforcement rule and push them into the ReplayBuffer.
//   - Sample a minibatch from the buffer and drive one gradient step.
//   - Serialize / restore the whole opponent (weights + buffer + meta)
//     through the BlobStorage port.
//
// What this module does NOT do:
//   - Touch the DOM, window, localStorage, React, or Astro.
//   - Implement any chess heuristic beyond the rules.

import { encodeState } from '../encoding/index.js';
import type { GameState, Move, Outcome } from '../rules/index.js';
import { legalMoves, isTerminal, applyMove, terminalValueForMover } from '../rules/index.js';
import {
  search,
  type Policy,
  type SearchConfig,
} from '../mcts/index.js';
import {
  ChessNet,
  deserialize,
  policyIndexFromUci,
  POLICY_SIZE,
  NN_VERSION,
  type TrainingExample,
} from '../nn/index.js';
import {
  type BlobStorage,
  putJson,
  getJson,
} from '../storage/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
  type ReplayBuffer,
} from '../training/index.js';

// ── Public types ─────────────────────────────────────────────────────────────

/**
 * One position from one game, captured before the move was played.
 * Accumulate these during a game, then pass them all to
 * `recordGameResult` once the outcome is known.
 */
export interface GameMoveRecord {
  /** Encoded state BEFORE the move was played. */
  features: Float32Array;
  /** UCI move that was actually played from that state. */
  move: Move;
  /** Whose move it was. */
  sideToMove: 'w' | 'b';
  /**
   * Sorted, de-duplicated set of policy indices for the legal moves at
   * this position. Captured at record time because the position is not
   * recoverable from `features` cheaply. Feeds the illegal-move loss
   * term (`ChessNet.trainBatch`): the network is penalized for putting
   * softmax mass on any index NOT in this list.
   *
   * Promotion underpromotions collapse to a single policy index (the
   * TODO in `core/nn/index.ts`), so this list is a set, not a multiset
   * — `e7e8q` / `e7e8r` / `e7e8b` / `e7e8n` all contribute the same
   * index and the slot is "legal" if any one of them is.
   */
  legalMoveIndices: number[];
  /**
   * Sorted, de-duplicated set of policy indices for the legal moves at
   * this position that deliver immediate checkmate. Empty when no
   * mate-in-1 was available.
   *
   * When this is non-empty, the build-examples functions ignore the
   * actually-played move and emit a corrective policy target that puts
   * positive weight on these mate indices and NEGATIVE weight on every
   * other legal index. Two things follow:
   *   - Every available checkmate becomes a positive training example
   *     for the mate move, regardless of who was on move or which move
   *     they actually chose.
   *   - Non-mate moves played from a position where mate was on the
   *     board get a *negative* learning signal, so the policy is
   *     actively pushed off them instead of being trained to imitate
   *     the miss.
   */
  mateMoveIndices: number[];
}

/**
 * Find every legal move at `state` that delivers immediate checkmate.
 * Returns UCI strings, lexicographically sorted for determinism.
 *
 * "Delivers checkmate" = applying the move yields a terminal position
 * where the side that just moved (us) won. This is the same rules-only
 * check MCTS uses internally to short-circuit search.
 */
export function mateInOneMovesFor(state: GameState): Move[] {
  if (isTerminal(state)) return [];
  const out: Move[] = [];
  for (const m of legalMoves(state)) {
    const child = applyMove(state, m);
    if (isTerminal(child) && terminalValueForMover(child) === 1) out.push(m);
  }
  out.sort();
  return out;
}

/**
 * Policy indices for every mate-in-1 move at `state`. Companion to
 * `legalMoveIndicesFor`; lives here for the same reason.
 */
export function mateInOneIndicesFor(state: GameState): number[] {
  const set = new Set<number>();
  for (const m of mateInOneMovesFor(state)) set.add(policyIndexFromUci(m));
  return Array.from(set).sort((a, b) => a - b);
}

/**
 * Compute the sorted, de-duplicated set of policy indices for the
 * legal moves at `state`. Used by every site that builds a
 * `GameMoveRecord`. Lives here (next to `policyIndexFromUci`'s
 * importer) so the implementation can stay a one-liner.
 */
export function legalMoveIndicesFor(state: GameState): number[] {
  const set = new Set<number>();
  for (const m of legalMoves(state)) set.add(policyIndexFromUci(m));
  return Array.from(set).sort((a, b) => a - b);
}

export interface OpponentConfig {
  /** MCTS simulations per selectMove call. */
  simulations: number;
  /** PUCT exploration constant. */
  cPuct: number;
  dirichletAlpha?: number;
  dirichletEpsilon?: number;
  /**
   * Optional deterministic rng for tests. Both MCTS and temperature
   * sampling use this rng so tests are fully reproducible.
   */
  rng?: () => number;
  /**
   * Temperature for move selection from visit counts.
   *   > 0  sample proportional to N^(1/T)
   *   = 0  greedy argmax (ties broken by prior, then by UCI string)
   * Defaults to 0 (greedy).
   */
  temperature?: number;
}

export interface Opponent {
  /** Pick a move for the opponent to play from `state`. */
  selectMove(
    state: GameState,
  ): Promise<{
    move: Move;
    rootValue: number;
    visitCounts: Map<Move, number>;
    priorPolicy: Map<Move, number>;
    edgeValues: Map<Move, number>;
  }>;

  /**
   * Record a complete game. Converts move records into TrainingExamples
   * and pushes them into the replay buffer per the winner-reinforcement
   * rule (see README). Does NO NN work.
   */
  recordGameResult(moves: GameMoveRecord[], finalOutcome: Outcome): void;

  /** Persist current NN weights + replay buffer to storage. */
  saveTo(storage: BlobStorage): Promise<void>;

  /**
   * Sample one minibatch from the buffer and run one gradient step.
   * Returns the losses, or null if the buffer has fewer examples than
   * `batchSize`.
   */
  trainStep(
    batchSize: number,
  ): Promise<{ valueLoss: number; policyLoss: number; totalLoss: number } | null>;

  /** Update the MCTS simulation budget for future selectMove calls. */
  setSimulations(n: number): void;
  /** Update the PUCT exploration constant for future selectMove calls. */
  setCPuct(c: number): void;

  /** Read-only handles for the UI / interpretability viewer. */
  readonly net: ChessNet;
  readonly replayBuffer: ReplayBuffer;
}

export interface CreateOpponentOptions {
  net: ChessNet;
  replayBuffer: ReplayBuffer;
  config: OpponentConfig;
}

/**
 * Storage keys used by loadOpponent / saveTo.
 *
 * NOTE: the string values ('agent/weights', 'agent/replay', 'agent/meta')
 * intentionally retain the legacy "agent/" prefix even though the type
 * was renamed to Opponent. These strings identify entries inside each
 * user's IndexedDB; changing them would orphan every trained network in
 * the wild. The TypeScript identifiers were renamed for consistency, but
 * the on-disk keyspace must stay backward-compatible.
 */
export const OPPONENT_STORAGE_KEYS = {
  WEIGHTS: 'agent/weights',
  REPLAY: 'agent/replay',
  META: 'agent/meta',
} as const;

// ── policyFromNet ────────────────────────────────────────────────────────────

/**
 * Adapt a ChessNet to MCTS's Policy interface.
 *
 * The NN returns raw logits over all 4096 from→to pairs, most of which
 * are illegal in any given position. We:
 *   1. Run softmax over the raw logits.
 *   2. Zero out all illegal moves.
 *   3. Renormalize so the surviving entries sum to 1.
 *   4. Fall back to a uniform prior if all legal moves had zero mass.
 *
 * For terminal positions the caller should not call this, but if it
 * does we return empty priors and value 0 (safe default).
 *
 * Exported for independent testing and for the future Stockfish-as-
 * policy harness.
 */
export function policyFromNet(net: ChessNet): Policy {
  return {
    async evaluate(
      state: GameState,
    ): Promise<{ priors: Map<Move, number>; value: number }> {
      const moves = legalMoves(state);

      // Safe default for terminal states (caller shouldn't reach here).
      if (moves.length === 0 || isTerminal(state)) {
        return { priors: new Map(), value: 0 };
      }

      const features = encodeState(state);
      const { value, policy: logits } = net.predict(features);

      // Softmax over the full logit vector.
      const probs = softmax(logits);

      // Mask illegal moves.
      const priorMap = new Map<Move, number>();
      let sum = 0;
      for (const move of moves) {
        const idx = policyIndexFromUci(move);
        const p = probs[idx];
        if (p > 0) {
          priorMap.set(move, p);
          sum += p;
        }
      }

      // Renormalize or fall back to uniform.
      let finalPriors: Map<Move, number>;
      if (sum === 0) {
        const u = 1 / moves.length;
        finalPriors = new Map(moves.map(m => [m, u]));
      } else {
        finalPriors = new Map(
          Array.from(priorMap.entries()).map(([m, p]) => [m, p / sum]),
        );
        // Legal moves not in priorMap get 0 (MCTS handles 0-prior edges).
        for (const m of moves) {
          if (!finalPriors.has(m)) finalPriors.set(m, 0);
        }
      }

      return { priors: finalPriors, value };
    },
  };
}

// ── buildTrainingExamples ────────────────────────────────────────────────────

/**
 * Convert a finished game's move records into TrainingExamples per the
 * winner-reinforcement rule documented in nn-chess/README.md.
 *
 * Value target (from the side-to-move's perspective at each position):
 *   white wins → white-to-move positions get +1, black-to-move get -1
 *   black wins → black-to-move positions get +1, white-to-move get -1
 *   draw       → all positions get 0
 *
 * Policy target (one-hot over POLICY_SIZE for the move played):
 *   Only produced for positions where the player who moved went on to
 *   WIN the game. Losing-side positions and all draw positions get
 *   policyTarget = null (policy loss skipped).
 *
 * Throws if `finalOutcome` is 'ongoing' (caller error — can't build
 * training examples from an unfinished game).
 *
 * Exported for unit testing in isolation.
 */
export function buildTrainingExamples(
  moves: GameMoveRecord[],
  finalOutcome: Outcome,
): TrainingExample[] {
  if (finalOutcome === 'ongoing') {
    throw new Error(
      'buildTrainingExamples: finalOutcome is "ongoing" — game is not finished',
    );
  }

  return moves.map(record => {
    // Value target from side-to-move's perspective.
    let valueTarget: number;
    if (finalOutcome === 'draw') {
      valueTarget = 0;
    } else if (finalOutcome === 'white-wins') {
      valueTarget = record.sideToMove === 'w' ? 1 : -1;
    } else {
      // black-wins
      valueTarget = record.sideToMove === 'b' ? 1 : -1;
    }

    // Policy target:
    //   - If a mate-in-1 was on the board at this position, emit the
    //     corrective target (push mass onto mate indices, push it off
    //     every other legal move). This happens regardless of who won
    //     and regardless of which move was actually played — every
    //     available checkmate must produce a training example for the
    //     mate move.
    //   - Otherwise: one-hot on the played move, but only for the
    //     winning side's positions. Draws and losers' moves → null.
    let policyTarget: Float32Array | null = null;
    if (record.mateMoveIndices && record.mateMoveIndices.length > 0) {
      policyTarget = correctivePolicyTarget(
        record.legalMoveIndices,
        record.mateMoveIndices,
      );
    } else {
      const isWinner =
        finalOutcome !== 'draw' &&
        ((finalOutcome === 'white-wins' && record.sideToMove === 'w') ||
          (finalOutcome === 'black-wins' && record.sideToMove === 'b'));
      if (isWinner) {
        policyTarget = new Float32Array(POLICY_SIZE);
        const idx = policyIndexFromUci(record.move);
        policyTarget[idx] = 1;
      }
    }

    return {
      features: record.features,
      valueTarget,
      policyTarget,
      legalMoveIndices: record.legalMoveIndices,
    };
  });
}

/**
 * Build a corrective policy target for a position where mate-in-1 was
 * available. Positive mass is spread uniformly over the mate indices
 * (sums to +1); negative mass is spread uniformly over every other
 * legal index (sums to -1). Cross-entropy then pulls the softmax up on
 * mate moves and pushes it down on every other legal alternative.
 *
 * Illegal indices stay at zero — the existing illegal-move penalty in
 * `ChessNet.trainBatch` is what keeps mass off them.
 */
function correctivePolicyTarget(
  legalIndices: number[],
  mateIndices: number[],
): Float32Array {
  const target = new Float32Array(POLICY_SIZE);
  const mateSet = new Set(mateIndices);
  const nonMate = legalIndices.filter(i => !mateSet.has(i));
  if (mateIndices.length > 0) {
    const w = 1 / mateIndices.length;
    for (const i of mateIndices) target[i] = w;
  }
  if (nonMate.length > 0) {
    const w = -1 / nonMate.length;
    for (const i of nonMate) target[i] = w;
  }
  return target;
}

// ── buildImitationExamples ────────────────────────────────────────────────────

/**
 * Build training examples for the *player-imitator* network.
 *
 * The imitator's job is behaviour-cloning of the user: given a position
 * the user faced, predict the move the user actually played. That goal
 * diverges from `buildTrainingExamples` in two ways:
 *
 *   - Every move gets a policy target — not just the winner's. We are
 *     trying to model what the user does, whether or not it works out.
 *   - Only the user's own move records should be passed in; the
 *     opponent's moves are not part of the imitator's training signal.
 *
 * Value target follows the same convention as `buildTrainingExamples`
 * (outcome from side-to-move's perspective at each position) so the
 * imitator's value head is still a sensible bootstrap prior for MCTS.
 *
 * The caller is responsible for filtering `userMoves` to the user's
 * side of the game; this function will not double-check.
 */
export function buildImitationExamples(
  userMoves: GameMoveRecord[],
  finalOutcome: Outcome,
): TrainingExample[] {
  if (finalOutcome === 'ongoing') {
    throw new Error(
      'buildImitationExamples: finalOutcome is "ongoing" — game is not finished',
    );
  }

  return userMoves.map(record => {
    let valueTarget: number;
    if (finalOutcome === 'draw') {
      valueTarget = 0;
    } else if (finalOutcome === 'white-wins') {
      valueTarget = record.sideToMove === 'w' ? 1 : -1;
    } else {
      valueTarget = record.sideToMove === 'b' ? 1 : -1;
    }

    // When mate-in-1 was on the board, override the user's actual move
    // with the corrective target so the imitator doesn't learn to miss
    // free mates. Otherwise behaviour-clone as usual.
    const policyTarget =
      record.mateMoveIndices && record.mateMoveIndices.length > 0
        ? correctivePolicyTarget(record.legalMoveIndices, record.mateMoveIndices)
        : (() => {
            const t = new Float32Array(POLICY_SIZE);
            t[policyIndexFromUci(record.move)] = 1;
            return t;
          })();

    return {
      features: record.features,
      valueTarget,
      policyTarget,
      legalMoveIndices: record.legalMoveIndices,
    };
  });
}

// ── createOpponent ───────────────────────────────────────────────────────────

export function createOpponent(opts: CreateOpponentOptions): Opponent {
  const { net, replayBuffer, config } = opts;

  const selectMove = async (
    state: GameState,
  ): Promise<{
    move: Move;
    rootValue: number;
    visitCounts: Map<Move, number>;
    priorPolicy: Map<Move, number>;
    edgeValues: Map<Move, number>;
  }> => {
    const policy = policyFromNet(net);

    const searchConfig: SearchConfig = {
      simulations: config.simulations,
      cPuct: config.cPuct,
      dirichletAlpha: config.dirichletAlpha,
      dirichletEpsilon: config.dirichletEpsilon,
      rng: config.rng,
    };

    const result = await search(state, policy, searchConfig);
    const { rootValue, visitCounts, priorPolicy, edgeValues, mateInOneMoves } = result;

    // Mate-in-1 always wins: bypass the temperature sampler entirely
    // so a sampled run can't randomly leave a free checkmate on the
    // board. `search()` already lists every mate-in-1 root move; pick
    // the lexicographically-first one to match `bestMove`.
    const temperature = config.temperature ?? 0;
    const move =
      mateInOneMoves.length > 0
        ? mateInOneMoves[0]
        : pickMove(visitCounts, priorPolicy, temperature, config.rng);

    return { move, rootValue, visitCounts, priorPolicy, edgeValues };
  };

  const recordGameResult = (
    gameMoves: GameMoveRecord[],
    finalOutcome: Outcome,
  ): void => {
    // buildTrainingExamples throws on 'ongoing' — let it propagate.
    const examples = buildTrainingExamples(gameMoves, finalOutcome);
    replayBuffer.addAll(examples);
  };

  const saveTo = async (storage: BlobStorage): Promise<void> => {
    const meta = { NN_VERSION };
    await Promise.all([
      net.serialize().then(bytes => storage.put(OPPONENT_STORAGE_KEYS.WEIGHTS, bytes)),
      storage.put(OPPONENT_STORAGE_KEYS.REPLAY, replayBuffer.serialize()),
      putJson(storage, OPPONENT_STORAGE_KEYS.META, meta),
    ]);
  };

  const trainStep = async (
    batchSize: number,
  ): Promise<{ valueLoss: number; policyLoss: number; totalLoss: number } | null> => {
    if (replayBuffer.size() < batchSize) return null;
    const batch = replayBuffer.sample(batchSize, config.rng);
    return net.trainBatch(batch);
  };

  const setSimulations = (n: number): void => {
    config.simulations = n;
  };

  const setCPuct = (c: number): void => {
    config.cPuct = c;
  };

  return {
    selectMove,
    recordGameResult,
    saveTo,
    trainStep,
    setSimulations,
    setCPuct,
    get net() { return net; },
    get replayBuffer() { return replayBuffer; },
  };
}

// ── loadOpponent ─────────────────────────────────────────────────────────────

/**
 * Restore weights + replay buffer from storage if present and compatible;
 * otherwise return a fresh opponent built from `defaults`.
 *
 * Version mismatch (stored NN_VERSION differs from current NN_VERSION) is
 * treated as a cache miss — the stored state is ignored and a fresh
 * opponent is returned. This ensures that an architecture or encoding
 * change doesn't silently corrupt training.
 */
export async function loadOpponent(
  storage: BlobStorage,
  defaults: {
    net: ChessNet;
    config: OpponentConfig;
    replayCapacity: number;
  },
): Promise<Opponent> {
  const fresh = () =>
    createOpponent({
      net: defaults.net,
      replayBuffer: createReplayBuffer(defaults.replayCapacity),
      config: defaults.config,
    });

  // Read meta first — check version before loading weights.
  const meta = await getJson<{ NN_VERSION: number }>(
    storage,
    OPPONENT_STORAGE_KEYS.META,
  );

  if (meta === null || meta.NN_VERSION !== NN_VERSION) {
    // Nothing stored yet, or stored weights are from a different architecture.
    return fresh();
  }

  // Try to load weights and replay buffer.
  const [weightBytes, replayBytes] = await Promise.all([
    storage.get(OPPONENT_STORAGE_KEYS.WEIGHTS),
    storage.get(OPPONENT_STORAGE_KEYS.REPLAY),
  ]);

  if (weightBytes === null) {
    return fresh();
  }

  let net: ChessNet;
  try {
    net = await deserialize(weightBytes);
  } catch {
    // Corrupt weights — start fresh.
    return fresh();
  }

  const replayBuffer =
    replayBytes !== null
      ? deserializeReplayBuffer(replayBytes, defaults.replayCapacity)
      : createReplayBuffer(defaults.replayCapacity);

  return createOpponent({ net, replayBuffer, config: defaults.config });
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Numerically-stable softmax over a Float32Array.
 * Returns a new Float32Array of the same length with values summing to 1.
 */
export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  if (sum > 0) {
    for (let i = 0; i < out.length; i++) out[i] /= sum;
  }
  return out;
}

/**
 * Pick a move from the MCTS visit counts.
 *
 * temperature = 0: greedy argmax; ties broken by prior, then by UCI string
 *                  (stable, deterministic, matches MCTS's own bestMove
 *                   selection — we intentionally re-implement here so the
 *                   temperature=0 path uses no rng at all).
 * temperature > 0: sample proportional to N^(1/T) using the provided rng
 *                  (or Math.random if none is given).
 */
function pickMove(
  visitCounts: Map<Move, number>,
  priorPolicy: Map<Move, number>,
  temperature: number,
  rng: (() => number) | undefined,
): Move {
  const moves = Array.from(visitCounts.keys());

  if (temperature === 0 || moves.length === 1) {
    // Greedy with tie-breaking: most visits > highest prior > lexicographic.
    let bestMove = moves[0];
    let bestN = visitCounts.get(bestMove) ?? 0;
    let bestP = priorPolicy.get(bestMove) ?? 0;
    for (const m of moves) {
      const n = visitCounts.get(m) ?? 0;
      const p = priorPolicy.get(m) ?? 0;
      if (
        n > bestN ||
        (n === bestN && p > bestP) ||
        (n === bestN && p === bestP && m < bestMove)
      ) {
        bestN = n;
        bestP = p;
        bestMove = m;
      }
    }
    return bestMove;
  }

  // Temperature sampling: weight ∝ N^(1/T).
  const invT = 1 / temperature;
  const weights = moves.map(m => Math.pow(visitCounts.get(m) ?? 0, invT));
  const totalWeight = weights.reduce((a, b) => a + b, 0);

  if (totalWeight === 0) {
    // All visit counts are 0 (shouldn't happen after MCTS, but be safe).
    return moves[0];
  }

  const draw = (rng ?? Math.random)() * totalWeight;
  let cumulative = 0;
  for (let i = 0; i < moves.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return moves[i];
  }
  // Floating-point rounding fallback.
  return moves[moves.length - 1];
}
