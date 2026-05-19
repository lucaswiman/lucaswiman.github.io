// Tests for the localStorage → IndexedDB migration helper.

import { describe, expect, it } from 'vitest';
import { IDBFactory } from 'fake-indexeddb';
import { createMemoryStorage } from './storage-memory.js';
import { createIndexedDBAdapter } from './storage-indexeddb.js';
import { migrateBlobStorage } from './migrate.js';

describe('migrateBlobStorage', () => {
  it('copies all keys from source to destination', async () => {
    const from = createMemoryStorage();
    const to = createMemoryStorage();
    await from.put('a', new Uint8Array([1, 2, 3]));
    await from.put('b', new Uint8Array([4, 5]));

    const result = await migrateBlobStorage(from, to);

    expect(result).toEqual({ migrated: 2, skipped: false });
    expect(await to.get('a')).toEqual(new Uint8Array([1, 2, 3]));
    expect(await to.get('b')).toEqual(new Uint8Array([4, 5]));
  });

  it('clears source after a successful copy', async () => {
    const from = createMemoryStorage();
    const to = createMemoryStorage();
    await from.put('weights', new Uint8Array([7, 7, 7]));

    await migrateBlobStorage(from, to);

    expect(await from.list()).toEqual([]);
    expect(await from.get('weights')).toBeNull();
  });

  it('skips when the destination is already populated', async () => {
    const from = createMemoryStorage();
    const to = createMemoryStorage();
    await from.put('a', new Uint8Array([1]));
    await to.put('existing', new Uint8Array([9]));

    const result = await migrateBlobStorage(from, to);

    expect(result).toEqual({ migrated: 0, skipped: true });
    // Source is left intact so a future migration can still run if the
    // destination is later wiped.
    expect(await from.get('a')).toEqual(new Uint8Array([1]));
    expect(await to.get('existing')).toEqual(new Uint8Array([9]));
  });

  it('is a no-op when both stores are empty', async () => {
    const from = createMemoryStorage();
    const to = createMemoryStorage();
    const result = await migrateBlobStorage(from, to);
    expect(result).toEqual({ migrated: 0, skipped: false });
  });

  it('migrates into an IndexedDB destination', async () => {
    const from = createMemoryStorage();
    const to = createIndexedDBAdapter({ factory: new IDBFactory() });
    await from.put('agent/weights', new Uint8Array([0xde, 0xad, 0xbe, 0xef]));
    await from.put('agent/meta', new Uint8Array([0x01]));

    const result = await migrateBlobStorage(from, to);

    expect(result).toEqual({ migrated: 2, skipped: false });
    expect(await to.get('agent/weights')).toEqual(
      new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
    );
    expect(await to.get('agent/meta')).toEqual(new Uint8Array([0x01]));
  });

  it('leaves source intact when destination put fails', async () => {
    const from = createMemoryStorage();
    await from.put('a', new Uint8Array([1]));
    await from.put('b', new Uint8Array([2]));

    // Wrap a memory store with a failing put on the second call.
    const inner = createMemoryStorage();
    let puts = 0;
    const to = {
      get: inner.get.bind(inner),
      put: (k: string, v: Uint8Array) => {
        puts++;
        if (puts === 2) return Promise.reject(new Error('boom'));
        return inner.put(k, v);
      },
      delete: inner.delete.bind(inner),
      list: inner.list.bind(inner),
    };

    await expect(migrateBlobStorage(from, to)).rejects.toThrow('boom');

    // Both source entries should still be there — the helper only
    // deletes after every put succeeds.
    const keys = (await from.list()).sort();
    expect(keys).toEqual(['a', 'b']);
  });
});
