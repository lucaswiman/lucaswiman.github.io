// Behavioural tests for the BlobStorage adapters and the core storage
// helpers (putJson, getJson, namespaced). The same suite is run against
// three configurations via describe.each so regressions are caught
// regardless of which backend is in use.

import { describe, expect, it } from 'vitest';
import type { BlobStorage } from '../core/storage/index.js';
import {
  getJson,
  namespaced,
  putJson,
} from '../core/storage/index.js';
import { createMemoryStorage } from './storage-memory.js';
import { createLocalStorageAdapter } from './storage-localstorage.js';

// --- Mock Storage -----------------------------------------------------------
//
// A minimal DOM Storage implementation backed by a Map so localStorage
// tests can run in a Node environment without a browser.

class MockStorage implements Storage {
  private readonly map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  key(index: number): string | null {
    // Storage.key() is index-stable within a single synchronous run.
    const keys = Array.from(this.map.keys());
    return keys[index] ?? null;
  }

  clear(): void {
    this.map.clear();
  }
}

// --- Shared suite -----------------------------------------------------------

type StorageFactory = () => BlobStorage;

/**
 * Runs the behavioural contract tests against the given factory.
 * Every `it` block calls the factory to get a fresh, empty store so
 * tests are independent.
 */
function sharedSuite(label: string, factory: StorageFactory): void {
  describe(label, () => {
    it('get on a missing key returns null', async () => {
      const s = factory();
      expect(await s.get('nope')).toBeNull();
    });

    it('put then get returns the same bytes', async () => {
      const s = factory();
      const bytes = new Uint8Array([1, 2, 3, 0x00, 0xff]);
      await s.put('k', bytes);
      expect(await s.get('k')).toEqual(bytes);
    });

    it('put then put overwrites', async () => {
      const s = factory();
      await s.put('k', new Uint8Array([1, 2, 3]));
      await s.put('k', new Uint8Array([9, 8, 7]));
      expect(await s.get('k')).toEqual(new Uint8Array([9, 8, 7]));
    });

    it('put then delete then get returns null', async () => {
      const s = factory();
      await s.put('k', new Uint8Array([42]));
      await s.delete('k');
      expect(await s.get('k')).toBeNull();
    });

    it('delete on a missing key does not throw', async () => {
      const s = factory();
      await expect(s.delete('missing')).resolves.toBeUndefined();
    });

    it('list without prefix returns all keys', async () => {
      const s = factory();
      await s.put('a', new Uint8Array([1]));
      await s.put('b', new Uint8Array([2]));
      await s.put('c', new Uint8Array([3]));
      const keys = await s.list();
      expect(keys.sort()).toEqual(['a', 'b', 'c']);
    });

    it('list with prefix filters keys', async () => {
      const s = factory();
      await s.put('foo/1', new Uint8Array([1]));
      await s.put('foo/2', new Uint8Array([2]));
      await s.put('bar/1', new Uint8Array([3]));
      const keys = await s.list('foo/');
      expect(keys.sort()).toEqual(['foo/1', 'foo/2']);
    });

    it('round-trips bytes including zeros and high bytes', async () => {
      const s = factory();
      const bytes = new Uint8Array([0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff]);
      await s.put('raw', bytes);
      expect(await s.get('raw')).toEqual(bytes);
    });

    it('mutating the input buffer after put does not affect stored value', async () => {
      const s = factory();
      const input = new Uint8Array([10, 20, 30]);
      await s.put('k', input);
      input[0] = 99; // mutate after storing
      const stored = await s.get('k');
      expect(stored?.[0]).toBe(10); // original value, not 99
    });

    it('putJson + getJson round-trip an object with nested arrays/strings', async () => {
      const s = factory();
      const obj = { version: 1, tags: ['alpha', 'beta'], nested: { x: 0.5 } };
      await putJson(s, 'config', obj);
      const result = await getJson<typeof obj>(s, 'config');
      expect(result).toEqual(obj);
    });

    it('getJson on a missing key returns null', async () => {
      const s = factory();
      expect(await getJson(s, 'absent')).toBeNull();
    });
  });
}

// --- Memory-adapter-specific tests ------------------------------------------

describe('memory adapter: get returns a defensive copy', () => {
  // The localStorage adapter decodes a fresh Uint8Array from the string
  // each time, so this property is inherent and not worth a separate test.
  // For the memory adapter we verify it explicitly.
  it('mutating the returned buffer does not corrupt stored value', async () => {
    const s = createMemoryStorage();
    await s.put('k', new Uint8Array([1, 2, 3]));
    const copy = await s.get('k');
    copy![0] = 99; // mutate the returned copy
    const again = await s.get('k');
    expect(again?.[0]).toBe(1); // stored value unchanged
  });
});

// --- namespaced helper tests ------------------------------------------------

describe('namespaced: keys are transparently prefixed', () => {
  it('writes under the prefixed key in the underlying store', async () => {
    const base = createMemoryStorage();
    const ns = namespaced(base, 'scope');
    await ns.put('weights', new Uint8Array([7, 8, 9]));
    // The underlying store must see the prefixed key.
    expect(await base.get('scope/weights')).toEqual(new Uint8Array([7, 8, 9]));
    // The namespaced view should see it under the unprefixed key.
    expect(await ns.get('weights')).toEqual(new Uint8Array([7, 8, 9]));
  });

  it('list returns unprefixed keys', async () => {
    const base = createMemoryStorage();
    const ns = namespaced(base, 'scope');
    await ns.put('a', new Uint8Array([1]));
    await ns.put('b', new Uint8Array([2]));
    // Unrelated key in the base store must not appear in ns.list().
    await base.put('other/c', new Uint8Array([3]));
    const keys = await ns.list();
    expect(keys.sort()).toEqual(['a', 'b']);
  });

  it('list with prefix filters and returns unprefixed keys', async () => {
    const base = createMemoryStorage();
    const ns = namespaced(base, 'ns');
    await ns.put('x/1', new Uint8Array([1]));
    await ns.put('x/2', new Uint8Array([2]));
    await ns.put('y/3', new Uint8Array([3]));
    const keys = await ns.list('x/');
    expect(keys.sort()).toEqual(['x/1', 'x/2']);
  });
});

// --- Run the shared suite against all three configurations ------------------

sharedSuite('memory adapter', () => createMemoryStorage());

sharedSuite('localStorage adapter', () =>
  createLocalStorageAdapter({ storage: new MockStorage() }),
);

sharedSuite(
  'namespaced(memory)',
  () => namespaced(createMemoryStorage(), 'scope'),
);
