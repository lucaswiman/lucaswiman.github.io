// Abstract persistence layer for trained NN weights, replay buffers,
// and any other artifacts the agent wants to keep around between
// sessions.
//
// `core/` only ever touches this interface. Concrete adapters
// (localStorage, in-memory, IndexedDB, download/upload, remote, …)
// live under `nn-chess/adapters/` and are wired in by the UI or by a
// test harness. This is the seam that lets us move the project to a
// separate repo, or to a Node-based training harness, or to a worker,
// without touching core code.

/**
 * A simple async key→bytes blob store. Keys are arbitrary strings;
 * values are opaque byte buffers (so we can persist tfjs weight
 * tensors, JSON payloads, PGNs, etc. through the same interface).
 *
 * Implementations must be safe to call concurrently from the same
 * caller — e.g. interleaved `put` + `get` should resolve in a
 * consistent order. They are NOT required to be safe across multiple
 * processes; the in-browser callers all live in the same tab.
 */
export interface BlobStorage {
  /** Returns the stored bytes for `key`, or `null` if absent. */
  get(key: string): Promise<Uint8Array | null>;

  /** Stores `value` under `key`, overwriting any existing entry. */
  put(key: string, value: Uint8Array): Promise<void>;

  /** Removes `key`. No-op if the key is not present. */
  delete(key: string): Promise<void>;

  /**
   * Returns all keys that start with `prefix` (or all keys if no
   * prefix is given). Order is unspecified.
   */
  list(prefix?: string): Promise<string[]>;
}

// --- JSON helpers -------------------------------------------------------
//
// Most things we want to persist (training metadata, replay-buffer
// indices, the agent's config) are JSON-shaped. These helpers
// centralize the encode/decode so callers don't sprinkle TextEncoder
// usage everywhere.

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export async function putJson(
  storage: BlobStorage,
  key: string,
  value: unknown,
): Promise<void> {
  await storage.put(key, encoder.encode(JSON.stringify(value)));
}

export async function getJson<T = unknown>(
  storage: BlobStorage,
  key: string,
): Promise<T | null> {
  const bytes = await storage.get(key);
  if (bytes === null) return null;
  return JSON.parse(decoder.decode(bytes)) as T;
}

/**
 * Convenience namespace wrapper. Returns a `BlobStorage` whose keys
 * are all transparently prefixed with `prefix + '/'`. Useful for
 * partitioning a single underlying store into per-agent or per-run
 * scopes.
 */
export function namespaced(
  storage: BlobStorage,
  prefix: string,
): BlobStorage {
  const wrap = (k: string) => `${prefix}/${k}`;
  const unwrap = (k: string) => k.slice(prefix.length + 1);
  return {
    get: (k) => storage.get(wrap(k)),
    put: (k, v) => storage.put(wrap(k), v),
    delete: (k) => storage.delete(wrap(k)),
    list: async (innerPrefix) => {
      const fullPrefix = innerPrefix ? wrap(innerPrefix) : `${prefix}/`;
      const keys = await storage.list(fullPrefix);
      return keys.map(unwrap);
    },
  };
}
