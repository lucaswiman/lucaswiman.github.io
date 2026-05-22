// BlobStorage adapter backed by IndexedDB.
//
// IndexedDB is the right backend for the opponent's replay buffer:
// per-origin quotas are typically hundreds of MB to multi-GB, values
// are stored as native binary (no base64), and writes are async so they
// don't block the main thread the way large localStorage strings do.
//
// The adapter opens a single object store keyed by string with
// Uint8Array values. Keys are namespaced by a configurable prefix so
// the opponent does not collide with anything else stored in the same
// database. The DB is opened lazily on first use so importing this
// module in Node (for the training harness, or for SSR) does not blow
// up.

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

// Classify the error thrown when we try to use a connection the browser
// has already closed. The spec-mandated synchronous throw from
// `db.transaction()` in that case is a DOMException named
// `InvalidStateError`; the message check is a fallback for browsers
// that surface the same condition under a different name.
function isTransientConnectionError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if (err.name === 'InvalidStateError') return true;
  return /database connection is clos(ing|ed)/i.test(err.message);
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

    // Deferred-resolution pattern so the success/error/blocked handlers
    // below can refer to `promise` for the "is this still the live
    // handle?" guard inside `invalidate`. Without that guard, a late
    // close event from an old connection would clobber a brand-new
    // cached promise.
    let resolveDb!: (db: IDBDatabase) => void;
    let rejectDb!: (err: unknown) => void;
    const promise = new Promise<IDBDatabase>((resolve, reject) => {
      resolveDb = resolve;
      rejectDb = reject;
    });
    dbPromise = promise;
    const invalidate = (): void => {
      if (dbPromise === promise) dbPromise = null;
    };

    let factory: IDBFactory;
    try {
      factory = getFactory();
    } catch (err) {
      invalidate();
      rejectDb(err);
      return promise;
    }

    const req = factory.open(dbName, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName);
      }
    };
    req.onsuccess = () => {
      const db = req.result;
      // Proactive invalidation: when the browser tears down the
      // connection (mobile tab suspension, storage eviction, app
      // backgrounding) `onclose` fires and we drop the stale handle so
      // the next operation transparently re-opens. `onversionchange`
      // fires when another tab/connection wants to upgrade the schema;
      // we close ours so the upgrade can proceed instead of blocking.
      db.onclose = () => invalidate();
      db.onversionchange = () => {
        db.close();
        invalidate();
      };
      resolveDb(db);
    };
    req.onerror = () => { invalidate(); rejectDb(req.error); };
    req.onblocked = () => { invalidate(); rejectDb(new Error('IndexedDB open blocked')); };

    return promise;
  };

  const runTx = async <T>(
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

  const withStore = async <T>(
    mode: IDBTransactionMode,
    fn: (store: IDBObjectStore) => Promise<T>,
  ): Promise<T> => {
    try {
      return await runTx(mode, fn);
    } catch (err) {
      // Reactive recovery: if `db.transaction()` (or the transaction
      // itself) failed because the connection was already in the
      // closing/closed state — typically because the browser closed it
      // silently between operations — drop the stale cached handle and
      // retry exactly once with a fresh connection. Anything else
      // (quota, real read/write errors) bubbles up unchanged.
      if (!isTransientConnectionError(err)) throw err;
      dbPromise = null;
      return await runTx(mode, fn);
    }
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
