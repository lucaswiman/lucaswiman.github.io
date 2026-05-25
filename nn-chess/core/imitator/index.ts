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

/**
 * Pick a move directly from a net's policy head — no MCTS, no value-
 * based reward search. The point is to ask the net "what would the user
 * play here?" the same way the imitator was trained to answer.
 *
 * This is the imitator's "play yourself" move-selection function, used
 * by the play-against-imitator UI. Reuses `policyFromNet` from
 * core/opponent so legal-move masking + renormalization stays a single
 * source of truth.
 *
 * `temperature = 0` (default): greedy argmax over the masked priors.
 * Ties broken lexicographically by UCI string for determinism.
 *
 * `temperature > 0`: sample proportional to `P^(1/T)` over the legal
 * moves, using `rng` (defaults to `Math.random`).
 *
 * Throws if `state` is terminal — callers should never reach here.
 *
 * Notes on heuristics: this function deliberately does NOT add any
 * mate-in-1 override or rules-based search the way `core/opponent`'s
 * `selectMove` does. The imitator's job is behaviour-cloning of the
 * user; any "always seize the mate" behaviour must come from the
 * imitator's policy head learning the corrective targets produced by
 * `buildImitationExamples`, not from a heuristic baked into selection.
 */
export async function selectImitatorMove(
  net: ChessNet,
  state: GameState,
  options: { temperature?: number; rng?: () => number } = {},
): Promise<{ move: Move; priors: Map<Move, number> }> {
  if (isTerminal(state)) {
    throw new Error('selectImitatorMove: called on a terminal GameState');
  }
  const moves = legalMoves(state);
  if (moves.length === 0) {
    throw new Error('selectImitatorMove: no legal moves at non-terminal state');
  }

  const { priors } = await policyFromNet(net).evaluate(state);
  const temperature = options.temperature ?? 0;

  if (temperature === 0 || moves.length === 1) {
    // Greedy: argmax of the masked prior, ties broken lex by UCI string.
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

  // Temperature sampling: weight ∝ P^(1/T) over legal moves.
  const invT = 1 / temperature;
  const weights = moves.map(m => Math.pow(priors.get(m) ?? 0, invT));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) {
    // Every legal move had zero prior (e.g. a freshly-initialized net
    // whose softmax collapsed onto illegal indices). Fall back to a
    // uniform draw so the game keeps progressing.
    const idx = Math.floor((options.rng ?? Math.random)() * moves.length);
    return { move: moves[Math.min(idx, moves.length - 1)], priors };
  }
  const draw = (options.rng ?? Math.random)() * total;
  let cumulative = 0;
  for (let i = 0; i < moves.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return { move: moves[i], priors };
  }
  // Floating-point rounding fallback.
  return { move: moves[moves.length - 1], priors };
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
   * Pick a move from this imitator's policy head. Thin wrapper around
   * `selectImitatorMove(net, state, …)` for callers that already hold a
   * `PlayerImitator`. See `selectImitatorMove` for the temperature /
   * tie-breaking semantics.
   */
  selectMove(
    state: GameState,
    options?: { temperature?: number; rng?: () => number },
  ): Promise<{ move: Move; priors: Map<Move, number> }>;

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
    options: { temperature?: number; rng?: () => number } = {},
  ): Promise<{ move: Move; priors: Map<Move, number> }> =>
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
