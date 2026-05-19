// In-memory BlobStorage implementation backed by a Map.
// Used by tests and by any in-process context that does not need
// persistence across JS environments (e.g. MCTS simulations that
// want a throw-away store).

import type { BlobStorage } from '../core/storage/index.js';

/**
 * Returns a fresh, empty BlobStorage whose data lives only in the
 * current JS heap. All reads and writes are defensive copies, so
 * callers cannot corrupt stored state by mutating a buffer they passed
 * in or received back.
 */
export function createMemoryStorage(): BlobStorage {
  const store = new Map<string, Uint8Array>();

  return {
    get(key: string): Promise<Uint8Array | null> {
      const stored = store.get(key);
      if (stored === undefined) return Promise.resolve(null);
      // Return a defensive copy so callers cannot mutate stored state.
      return Promise.resolve(stored.slice());
    },

    put(key: string, value: Uint8Array): Promise<void> {
      // Store a defensive copy so later mutations of `value` by the
      // caller do not affect what is persisted.
      store.set(key, value.slice());
      return Promise.resolve();
    },

    delete(key: string): Promise<void> {
      store.delete(key);
      return Promise.resolve();
    },

    list(prefix?: string): Promise<string[]> {
      const keys: string[] = [];
      for (const key of store.keys()) {
        if (prefix === undefined || key.startsWith(prefix)) {
          keys.push(key);
        }
      }
      return Promise.resolve(keys);
    },
  };
}
