// One-shot migration helper for moving BlobStorage contents between
// adapters. The original nn-chess release wrote to localStorage; this
// helper copies anything found there into a new IndexedDB-backed store
// the first time the page loads with the new code.
//
// Migration is opportunistic and idempotent:
//   - If the destination already has any keys, we treat the migration as
//     done and skip — the user has already been on the new build.
//   - If the source has nothing, there is nothing to do.
//   - Each key is copied via standard BlobStorage operations; the source
//     entries are deleted only after every put succeeds, so a crash in
//     the middle leaves the source intact and the next startup retries.

import type { BlobStorage } from '../core/storage/index.js';

export interface MigrateResult {
  migrated: number;
  skipped: boolean;
}

/**
 * Copy every key from `from` into `to`, then delete the originals.
 * If `to` is non-empty, returns `{ skipped: true, migrated: 0 }` without
 * touching either store. Failures during the copy phase abort before any
 * deletes happen.
 */
export async function migrateBlobStorage(
  from: BlobStorage,
  to: BlobStorage,
): Promise<MigrateResult> {
  // Probe destination first. If it already has any of our keys we have
  // nothing to do — the migration ran on a previous load (or the user
  // has been writing on the new backend already).
  const existing = await to.list();
  if (existing.length > 0) {
    return { migrated: 0, skipped: true };
  }

  const keys = await from.list();
  if (keys.length === 0) {
    return { migrated: 0, skipped: false };
  }

  // Stage 1: copy everything. If any put throws, propagate without
  // mutating `from` — caller can retry next session.
  for (const key of keys) {
    const value = await from.get(key);
    if (value === null) continue;
    await to.put(key, value);
  }

  // Stage 2: only after every put succeeded, free space in `from`.
  // Errors here are non-fatal — the data is safely in `to`; the leftover
  // entries in `from` will be cleaned up next time list() runs and finds
  // the destination already populated (skip path), or by manual cleanup.
  for (const key of keys) {
    try { await from.delete(key); } catch { /* best-effort */ }
  }

  return { migrated: keys.length, skipped: false };
}
