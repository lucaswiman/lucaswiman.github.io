// Maia-2 weight loader.
//
// The conversion script (nn-chess/scripts/convert-maia2.py) writes a
// single flat binary blob (`weights.bin`) plus a JSON manifest mapping
// each tensor name to its byte-offset, shape, and dtype. This module:
//   1. Parses the manifest.
//   2. Slices each tensor out of the blob.
//   3. Promotes fp16 → fp32 on the JS side (TF.js has no native fp16
//      tensors, so we widen at load time).
//   4. Wraps every tensor as a `tf.Tensor`, named the same as the
//      manifest entry. Callers (model.ts) look these up via
//      `maiaWeights.get('chess_cnn.conv1.weight')`.
//
// The module is pure TS with no DOM / fetch / IndexedDB calls — the
// caller passes in a `Uint8Array` (e.g. fetched in a worker and saved
// to IndexedDB) and a manifest object. Tests use the same entry
// point as production.

import * as tf from '@tensorflow/tfjs';

// ── Manifest schema ──────────────────────────────────────────────────────────

export interface MaiaManifestTensor {
  name: string;
  shape: number[];
  offset: number;
  size_bytes: number;
  dtype: 'fp16' | 'fp32';
}

export interface MaiaManifest {
  version: number;
  dtype: 'fp16' | 'fp32';
  model: string;
  model_type: string;
  total_bytes: number;
  sha256: string;
  num_elo_buckets: number;
  policy_size: number;
  tensors: MaiaManifestTensor[];
}

/** Manifest version this loader understands. Bump if the layout changes. */
export const MAIA_MANIFEST_VERSION = 1;

// ── fp16 decode ──────────────────────────────────────────────────────────────

/**
 * Convert a single IEEE 754 binary16 (uint16) to a binary32 (number).
 *
 * We need this because (a) tf.js has no native fp16 tensors, and (b)
 * neither Node nor browsers expose a built-in fp16-decode primitive
 * that round-trips through ArrayBuffer typed-array views. Hand-rolling
 * it keeps the loader dependency-free.
 */
function fp16ToFp32(half: number): number {
  const s = (half & 0x8000) >> 15;
  const e = (half & 0x7c00) >> 10;
  const f = half & 0x03ff;
  if (e === 0) {
    // Subnormal or zero.
    if (f === 0) return s ? -0 : 0;
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 0x1f) {
    // Inf or NaN.
    if (f === 0) return s ? -Infinity : Infinity;
    return NaN;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Decode a packed fp16 blob to a Float32Array. */
function decodeFp16(buf: Uint8Array, offset: number, count: number): Float32Array {
  const view = new DataView(buf.buffer, buf.byteOffset + offset, count * 2);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = fp16ToFp32(view.getUint16(i * 2, true /* little-endian */));
  }
  return out;
}

/** Decode a packed fp32 blob to a Float32Array. */
function decodeFp32(buf: Uint8Array, offset: number, count: number): Float32Array {
  // Need a view aligned to 4 bytes. `Float32Array` requires alignment;
  // copy first if alignment isn't satisfied.
  const byteStart = buf.byteOffset + offset;
  if (byteStart % 4 === 0) {
    return new Float32Array(buf.buffer, byteStart, count).slice();
  }
  const sliced = buf.slice(offset, offset + count * 4);
  return new Float32Array(sliced.buffer, 0, count);
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface MaiaWeights {
  /** Look up a tensor by manifest name. Throws if the name is unknown. */
  get(name: string): tf.Tensor;
  /** Dispose every tensor in the collection. Idempotent. */
  dispose(): void;
  /** True if every expected tensor is present. */
  isComplete(expected: ReadonlyArray<string>): boolean;
  /** All loaded tensor names. */
  names(): string[];
}

/**
 * Decode a Maia-2 weight blob into a dictionary of `tf.Tensor`s.
 *
 * `blob` is the raw bytes of `weights.bin`. `manifest` is the parsed
 * `manifest.json`. The tensors created here are NOT disposed
 * automatically; call `.dispose()` on the returned object when the
 * model is torn down.
 */
export function loadMaiaWeights(
  blob: Uint8Array,
  manifest: MaiaManifest,
): MaiaWeights {
  if (manifest.version !== MAIA_MANIFEST_VERSION) {
    throw new Error(
      `Maia-2 manifest version mismatch: got ${manifest.version}, ` +
      `loader supports ${MAIA_MANIFEST_VERSION}`,
    );
  }
  if (blob.byteLength !== manifest.total_bytes) {
    throw new Error(
      `Maia-2 blob size mismatch: got ${blob.byteLength} bytes, ` +
      `manifest says ${manifest.total_bytes}`,
    );
  }

  const tensors = new Map<string, tf.Tensor>();
  for (const entry of manifest.tensors) {
    const numElements = entry.shape.reduce((a, b) => a * b, 1);
    let data: Float32Array;
    if (entry.dtype === 'fp16') {
      data = decodeFp16(blob, entry.offset, numElements);
    } else if (entry.dtype === 'fp32') {
      data = decodeFp32(blob, entry.offset, numElements);
    } else {
      throw new Error(`unsupported dtype: ${entry.dtype}`);
    }
    const t = tf.tensor(data, entry.shape, 'float32');
    tensors.set(entry.name, t);
  }

  let disposed = false;
  return {
    get(name: string): tf.Tensor {
      const t = tensors.get(name);
      if (t === undefined) {
        throw new Error(`Maia weights: unknown tensor "${name}"`);
      }
      return t;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      for (const t of tensors.values()) t.dispose();
      tensors.clear();
    },
    isComplete(expected: ReadonlyArray<string>): boolean {
      for (const name of expected) {
        if (!tensors.has(name)) return false;
      }
      return true;
    },
    names(): string[] {
      return Array.from(tensors.keys());
    },
  };
}
