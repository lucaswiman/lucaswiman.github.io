// Cumulative training-stats per net. Persisted in the same BlobStorage as
// weights and replay buffers but in a separate, tiny JSON blob so the
// counters survive the user re-loading the page and aren't bundled into
// the weight bytes.
//
// The numbers come from the workers via the existing `finalStep` /
// `imitatorFinalStep` fields in `done` messages; the main thread
// multiplies by the batch size to get "examples seen" and persists. We
// don't trust the worker to know about lifetime stats — it operates on
// fresh deserialized bytes every call.
//
// On a model reset (danger zone) the matching stats blob is deleted
// alongside the weights so the counter starts from zero.

import { getJson, putJson, type BlobStorage } from '../core/storage/index.js';

export interface NetStats {
  /** Total gradient steps taken across the lifetime of this network. */
  gradientSteps: number;
  /** Total examples consumed = sum of (steps * batchSize) per training pass. */
  examplesSeen: number;
  /** Unix-ms timestamp of the most recent training pass; null if never trained. */
  lastTrainedAt: number | null;
}

/**
 * Storage keys for the per-net stats blobs. Live under the same prefix
 * as the rest of each network's persisted state so the reset handlers
 * can clear them with the existing keyspace convention.
 */
export const OPPONENT_TRAIN_STATS_KEY = 'agent/train-stats';
export const IMITATOR_TRAIN_STATS_KEY = 'imitator/train-stats';

export function emptyNetStats(): NetStats {
  return { gradientSteps: 0, examplesSeen: 0, lastTrainedAt: null };
}

export async function loadNetStats(storage: BlobStorage, key: string): Promise<NetStats> {
  const stored = await getJson<Partial<NetStats>>(storage, key);
  if (stored === null) return emptyNetStats();
  return {
    gradientSteps: stored.gradientSteps ?? 0,
    examplesSeen: stored.examplesSeen ?? 0,
    lastTrainedAt: stored.lastTrainedAt ?? null,
  };
}

export async function saveNetStats(
  storage: BlobStorage,
  key: string,
  stats: NetStats,
): Promise<void> {
  await putJson(storage, key, stats);
}

/**
 * Apply one training pass to existing stats. Returns a new object so
 * React state updates trigger re-renders.
 */
export function addTrainingPass(
  prev: NetStats,
  delta: { gradientSteps: number; batchSize: number },
): NetStats {
  if (delta.gradientSteps <= 0) return prev;
  return {
    gradientSteps: prev.gradientSteps + delta.gradientSteps,
    examplesSeen: prev.examplesSeen + delta.gradientSteps * delta.batchSize,
    lastTrainedAt: Date.now(),
  };
}

/**
 * "Effective epochs over the current buffer" — total examples ever
 * consumed divided by the number of examples sitting in the buffer
 * right now. Crude but useful: if it climbs into the tens while the
 * buffer is small, the net has revisited the same examples many times
 * and overfitting is plausible.
 */
export function effectiveEpochs(stats: NetStats, currentBufferSize: number): number {
  if (currentBufferSize <= 0) return 0;
  return stats.examplesSeen / currentBufferSize;
}
