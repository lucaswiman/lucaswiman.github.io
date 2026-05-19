// BlobStorage adapter backed by the DOM localStorage API (or any
// Storage-shaped object). Because localStorage only holds strings,
// bytes are base64-encoded on write and decoded on read.
//
// The adapter prepends a configurable namespace prefix to every key
// it writes so it does not collide with other apps sharing the same
// origin's localStorage.

import type { BlobStorage } from '../core/storage/index.js';

// --- base64 helpers ---------------------------------------------------------
//
// We need helpers that work in both Node 22 (for tests) and the browser.
// Node 22 exposes `Buffer`; browsers expose `btoa`/`atob`. We probe for
// the Node Buffer constructor at runtime via a string key access so the
// TypeScript compiler (targeting browsers) does not see a direct reference
// to the Node-only `Buffer` global and therefore does not complain.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NodeBuffer = (globalThis as Record<string, unknown>)['Buffer'] as
  | { from(data: Uint8Array): { toString(encoding: string): string };
      from(data: string, encoding: string): { buffer: ArrayBuffer; byteOffset: number; byteLength: number } }
  | undefined;

/**
 * Encodes a Uint8Array to a base64 string.
 * Works in Node 22 (via Buffer) and in browsers (via btoa).
 */
export function bytesToBase64(u8: Uint8Array): string {
  if (NodeBuffer !== undefined) {
    return NodeBuffer.from(u8).toString('base64');
  }
  let binary = '';
  for (let i = 0; i < u8.length; i++) {
    binary += String.fromCharCode(u8[i]!);
  }
  return btoa(binary);
}

/**
 * Decodes a base64 string back to a Uint8Array.
 * Works in Node 22 (via Buffer) and in browsers (via atob).
 */
export function base64ToBytes(s: string): Uint8Array {
  if (NodeBuffer !== undefined) {
    const buf = NodeBuffer.from(s, 'base64');
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const binary = atob(s);
  const u8 = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    u8[i] = binary.charCodeAt(i);
  }
  return u8;
}

// --- adapter ----------------------------------------------------------------

export interface LocalStorageAdapterOptions {
  /**
   * Prefix prepended to every key stored in localStorage.
   * Defaults to `"nn-chess/"`.
   */
  prefix?: string;

  /**
   * The Storage-shaped object to use.
   * Defaults to `window.localStorage`. Pass a mock for tests.
   */
  storage?: Storage;
}

/**
 * Returns a BlobStorage backed by `localStorage` (or any compatible
 * `Storage` object). Keys visible to callers are the logical keys;
 * the adapter transparently prepends `prefix` before touching
 * `localStorage` and strips it on `list`.
 */
export function createLocalStorageAdapter(
  options?: LocalStorageAdapterOptions,
): BlobStorage {
  const nsPrefix = options?.prefix ?? 'nn-chess/';
  // Defer access to `window.localStorage` until first use so this
  // module can be imported in Node without blowing up at module-eval
  // time. Tests always supply an explicit `storage`.
  const getStorage = (): Storage =>
    options?.storage ?? (typeof window !== 'undefined'
      ? window.localStorage
      : (() => { throw new Error('localStorage is not available in this environment'); })());

  const lsKey = (key: string): string => `${nsPrefix}${key}`;
  const stripPrefix = (lsKey: string): string => lsKey.slice(nsPrefix.length);

  return {
    get(key: string): Promise<Uint8Array | null> {
      const raw = getStorage().getItem(lsKey(key));
      if (raw === null) return Promise.resolve(null);
      return Promise.resolve(base64ToBytes(raw));
    },

    put(key: string, value: Uint8Array): Promise<void> {
      getStorage().setItem(lsKey(key), bytesToBase64(value));
      return Promise.resolve();
    },

    delete(key: string): Promise<void> {
      getStorage().removeItem(lsKey(key));
      return Promise.resolve();
    },

    list(prefix?: string): Promise<string[]> {
      const storage = getStorage();
      // The full prefix we are scanning for in localStorage key-space.
      const scanPrefix = prefix !== undefined
        ? lsKey(prefix)
        : nsPrefix;
      const keys: string[] = [];
      for (let i = 0; i < storage.length; i++) {
        const rawKey = storage.key(i);
        if (rawKey !== null && rawKey.startsWith(scanPrefix)) {
          keys.push(stripPrefix(rawKey));
        }
      }
      return Promise.resolve(keys);
    },
  };
}
