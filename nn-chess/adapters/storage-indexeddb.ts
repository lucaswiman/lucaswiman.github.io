// BlobStorage adapter backed by IndexedDB.
//
// IndexedDB is the right backend for the agent's replay buffer: per-origin
// quotas are typically hundreds of MB to multi-GB, values are stored as
// native binary (no base64), and writes are async so they don't block the
// main thread the way large localStorage strings do.
//
// The adapter opens a single object store keyed by string with Uint8Array
// values. Keys are namespaced by a configurable prefix so the agent does
// not collide with anything else stored in the same database. The DB is
// opened lazily on first use so importing this module in Node (for the
// training harness, or for SSR) does not blow up.

import type { BlobStorage } from '../core/storage/index.js';

export interface IndexedDBAdapterOptions {
  /** IndexedDB database name. Defaults to `"nn-chess"`. */
  dbName?: string;

  /** Object store name inside the database. Defaults to `"blobs"`. */
  storeName?: string;

  /**
   * Prefix prepended to every logical key before it touches IDB.
   * Defaults to `"nn-chess/"`. Matches the localStorage adapter's default
   * so a migrate-from-localStorage step preserves keys 1:1.
   */
  prefix?: string;

  /**
   * The IDBFactory to use. Defaults to `globalThis.indexedDB`. Tests pass
   * a `fake-indexeddb` factory here so they don't need to patch globals.
   */
  factory?: IDBFactory;
}

function awaitRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Returns a BlobStorage backed by IndexedDB. The database connection is
 * opened lazily on the first operation so the adapter is safe to
 * construct in environments where IndexedDB is unavailable until later
 * (e.g. before the browser has finished initializing).
 */
export function createIndexedDBAdapter(
  options?: IndexedDBAdapterOptions,
): BlobStorage {
  const dbName = options?.dbName ?? 'nn-chess';
  const storeName = options?.storeName ?? 'blobs';
  const nsPrefix = options?.prefix ?? 'nn-chess/';
  const explicitFactory = options?.factory;

  // Resolved lazily so we can be constructed before IndexedDB exists
  // (SSR, Node import) and only fail when first used.
  const getFactory = (): IDBFactory => {
    if (explicitFactory !== undefined) return explicitFactory;
    const global = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (global === undefined) {
      throw new Error('IndexedDB is not available in this environment');
    }
    return global;
  };

  const idbKey = (key: string): string => `${nsPrefix}${key}`;
  const stripPrefix = (k: string): string => k.slice(nsPrefix.length);

  let dbPromise: Promise<IDBDatabase> | null = null;
  const getDb = (): Promise<IDBDatabase> => {
    if (dbPromise !== null) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      let factory: IDBFactory;
      try { factory = getFactory(); } catch (err) { reject(err); return; }
      const req = factory.open(dbName, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(storeName)) {
          db.createObjectStore(storeName);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error('IndexedDB open blocked'));
    });
    // If the open fails, clear the cached promise so subsequent calls
    // get a fresh attempt rather than a permanently-broken handle.
    dbPromise.catch(() => { dbPromise = null; });
    return dbPromise;
  };

  const withStore = async <T>(
    mode: IDBTransactionMode,
    fn: (store: IDBObjectStore) => Promise<T>,
  ): Promise<T> => {
    const db = await getDb();
    const tx = db.transaction(storeName, mode);
    const store = tx.objectStore(storeName);
    const result = await fn(store);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error ?? new Error('IDB transaction aborted'));
    });
    return result;
  };

  return {
    async get(key: string): Promise<Uint8Array | null> {
      return withStore('readonly', async (store) => {
        const value = await awaitRequest(store.get(idbKey(key)));
        if (value === undefined) return null;
        // IDB returns the same instance we wrote (structured-clone copy),
        // but to match the memory adapter's defensive-copy contract we
        // hand out a fresh Uint8Array.
        const u8 = value as Uint8Array;
        return u8.slice();
      });
    },

    async put(key: string, value: Uint8Array): Promise<void> {
      // Defensive copy so later mutations of `value` don't affect the
      // structured-cloned payload IDB now owns.
      const copy = value.slice();
      await withStore('readwrite', async (store) => {
        await awaitRequest(store.put(copy, idbKey(key)));
      });
    },

    async delete(key: string): Promise<void> {
      await withStore('readwrite', async (store) => {
        await awaitRequest(store.delete(idbKey(key)));
      });
    },

    async list(prefix?: string): Promise<string[]> {
      const scanPrefix = prefix !== undefined ? idbKey(prefix) : nsPrefix;
      return withStore('readonly', async (store) => {
        const keys = (await awaitRequest(store.getAllKeys())) as IDBValidKey[];
        const out: string[] = [];
        for (const k of keys) {
          if (typeof k !== 'string') continue;
          if (k.startsWith(scanPrefix)) out.push(stripPrefix(k));
        }
        return out;
      });
    },
  };
}
