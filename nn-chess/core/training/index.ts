// Replay buffer for the nn-chess training pipeline.
//
// The buffer is a bounded ring (circular queue): when it is full, the
// oldest entry is evicted to make room for the newest. This is the
// standard experience-replay approach from DQN / AlphaZero.
//
// Serialization wire format (Uint8Array):
//   [0..3]   little-endian uint32: byte length of the JSON header
//   [4..4+H) JSON header: { count, capacity, featureLen, hasPolicyTarget[] }
//             - count:          number of stored examples
//             - capacity:       ring capacity
//             - featureLen:     length of each features Float32Array
//             - hasPolicyTarget: bool array, one entry per stored example
//   [4+H..)  raw Float32 payload, tightly packed:
//             for each example i in insertion order:
//               features[i]     (featureLen float32s)
//               valueTarget[i]  (1 float32)
//               policyTarget[i] (POLICY_SIZE float32s) if hasPolicyTarget[i],
//                               else nothing
//
// `POLICY_SIZE` (4096) is NOT embedded in the header; it is derived at
// deserialization time from the total byte count and the header metadata.
// We do store `featureLen` in the header so the format is self-describing
// even if FEATURE_COUNT changes between versions (a version bump in the
// encoding/nn module would already invalidate any saved state via META).

import type { TrainingExample } from '../nn/index.js';
import { POLICY_SIZE } from '../nn/index.js';

// ── Public interface ──────────────────────────────────────────────────────────

/**
 * Bounded ring buffer of TrainingExamples. Oldest entries are evicted when the
 * buffer is full. Supports deterministic sampling for tests via an injected rng.
 */
export interface ReplayBuffer {
  add(example: TrainingExample): void;
  addAll(examples: TrainingExample[]): void;
  /** Number of examples currently stored. */
  size(): number;
  /** Maximum number of examples the buffer can hold. */
  capacity(): number;
  /**
   * Returns `count` examples sampled with replacement using the provided rng
   * (defaults to Math.random).
   */
  sample(count: number, rng?: () => number): TrainingExample[];
  /** Returns all examples currently in the buffer in insertion order. */
  snapshot(): TrainingExample[];
  /** Serialize to a self-contained Uint8Array (see format in file header). */
  serialize(): Uint8Array;
}

// ── createReplayBuffer ────────────────────────────────────────────────────────

export function createReplayBuffer(capacity: number): ReplayBuffer {
  if (capacity <= 0) throw new Error('ReplayBuffer capacity must be > 0');

  // Ring buffer internals.
  // `ring` holds up to `capacity` examples.
  // `head` is the index of the next write slot (oldest slot to overwrite).
  // `count` is the current number of stored examples.
  const ring: TrainingExample[] = new Array(capacity);
  let head = 0;
  let count = 0;

  function add(example: TrainingExample): void {
    ring[head] = example;
    head = (head + 1) % capacity;
    if (count < capacity) count++;
  }

  function addAll(examples: TrainingExample[]): void {
    for (const ex of examples) add(ex);
  }

  function size(): number {
    return count;
  }

  function capacityFn(): number {
    return capacity;
  }

  function snapshot(): TrainingExample[] {
    if (count === 0) return [];
    if (count < capacity) {
      // Buffer not yet wrapped — elements are [0..count).
      return ring.slice(0, count);
    }
    // Buffer has wrapped — oldest element is at `head`.
    const result: TrainingExample[] = new Array(capacity);
    for (let i = 0; i < capacity; i++) {
      result[i] = ring[(head + i) % capacity];
    }
    return result;
  }

  function sample(count_: number, rng: () => number = Math.random): TrainingExample[] {
    if (count === 0) return [];
    const snap = snapshot();
    const result: TrainingExample[] = new Array(count_);
    for (let i = 0; i < count_; i++) {
      const idx = Math.floor(rng() * snap.length);
      result[i] = snap[idx];
    }
    return result;
  }

  function serialize(): Uint8Array {
    return serializeBuffer(snapshot(), capacity);
  }

  return { add, addAll, size, capacity: capacityFn, sample, snapshot, serialize };
}

// ── deserializeReplayBuffer ───────────────────────────────────────────────────

export function deserializeReplayBuffer(bytes: Uint8Array, capacity: number): ReplayBuffer {
  const examples = deserializeExamples(bytes);
  const buf = createReplayBuffer(capacity);
  buf.addAll(examples);
  return buf;
}

// ── Serialization helpers ─────────────────────────────────────────────────────

interface SerialHeader {
  count: number;
  capacity: number;
  featureLen: number;
  hasPolicyTarget: boolean[];
}

function serializeBuffer(examples: TrainingExample[], capacity: number): Uint8Array {
  const count = examples.length;
  const featureLen = count > 0 ? examples[0].features.length : 0;

  const hasPolicyTarget = examples.map(ex => ex.policyTarget !== null);

  const header: SerialHeader = { count, capacity, featureLen, hasPolicyTarget };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));

  // Calculate total float32 payload size.
  // Each example: features (featureLen) + valueTarget (1) + optional policyTarget (POLICY_SIZE)
  let floatCount = 0;
  for (let i = 0; i < count; i++) {
    floatCount += featureLen + 1 + (hasPolicyTarget[i] ? POLICY_SIZE : 0);
  }

  // Build the float payload in its own buffer (avoids alignment issues when
  // placing a Float32Array at an arbitrary offset inside the outer Uint8Array).
  const floatPayload = new Float32Array(floatCount);
  let offset = 0;
  for (let i = 0; i < count; i++) {
    const ex = examples[i];
    floatPayload.set(ex.features, offset);
    offset += featureLen;
    floatPayload[offset] = ex.valueTarget;
    offset += 1;
    if (hasPolicyTarget[i] && ex.policyTarget !== null) {
      floatPayload.set(ex.policyTarget, offset);
      offset += POLICY_SIZE;
    }
  }

  const floatBytes = new Uint8Array(floatPayload.buffer);
  const out = new Uint8Array(4 + headerBytes.length + floatBytes.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, headerBytes.length, /* littleEndian= */ true);
  out.set(headerBytes, 4);
  out.set(floatBytes, 4 + headerBytes.length);

  return out;
}

function deserializeExamples(bytes: Uint8Array): TrainingExample[] {
  if (bytes.length < 4) return [];

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(0, /* littleEndian= */ true);

  const headerBytes = bytes.slice(4, 4 + headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as SerialHeader;

  const { count, featureLen, hasPolicyTarget } = header;
  if (count === 0) return [];

  const payloadStart = 4 + headerLen;
  // Must be aligned to 4 bytes for Float32Array. We read via DataView if needed,
  // but slicing gives us a fresh buffer so alignment is guaranteed.
  const payloadBytes = bytes.slice(payloadStart);
  const floatView = new Float32Array(payloadBytes.buffer, payloadBytes.byteOffset, payloadBytes.byteLength / 4);

  const examples: TrainingExample[] = [];
  let offset = 0;

  for (let i = 0; i < count; i++) {
    const features = floatView.slice(offset, offset + featureLen) as Float32Array;
    offset += featureLen;
    const valueTarget = floatView[offset];
    offset += 1;

    let policyTarget: Float32Array | null = null;
    if (hasPolicyTarget[i]) {
      policyTarget = floatView.slice(offset, offset + POLICY_SIZE) as Float32Array;
      offset += POLICY_SIZE;
    }

    examples.push({ features, valueTarget, policyTarget });
  }

  return examples;
}
