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
import { buildImitationExamples } from '../opponent/index.js';
import {
  ChessNet,
  deserialize,
  NN_VERSION,
  type TrainResult,
} from '../nn/index.js';
import type { GameState, Outcome } from '../rules/index.js';
import {
  selectMoveFromNet,
  type MoveSelectionOptions,
  type MoveSelectionResult,
} from '../move-selection/index.js';
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
   * `selectMoveFromNet(net, state, …)` for callers that already hold a
   * `PlayerImitator`. See `core/move-selection` for MCTS vs raw-policy
   * mode semantics.
   */
  selectMove(
    state: GameState,
    options?: MoveSelectionOptions,
  ): Promise<MoveSelectionResult>;

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
    options: MoveSelectionOptions = {},
  ): Promise<MoveSelectionResult> =>
    selectMoveFromNet(net, state, options);

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
