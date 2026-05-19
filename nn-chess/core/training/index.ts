// Replay buffer for the nn-chess training pipeline.
//
// The buffer is a bounded ring (circular queue): when it is full, the
// oldest entry is evicted to make room for the newest. This is the
// standard experience-replay approach from DQN / AlphaZero.
//
// Serialization wire format (Uint8Array):
//   [0..3]   little-endian uint32: byte length of the JSON header
//   [4..4+H) JSON header:
//     v2 (current):
//       { version: 2, count, capacity, featureLen, hasPolicyTarget[] }
//       Per-example float payload, tightly packed:
//         features[i]     (featureLen float32s)
//         valueTarget[i]  (1 float32)
//         policyIndex[i]  (1 float32) if hasPolicyTarget[i], else nothing
//       The policy target is reconstituted as a one-hot Float32Array of
//       length POLICY_SIZE at deserialize time. Storing the index instead
//       of the dense vector shrinks each winning-side example from ~21 KB
//       to ~4.6 KB — a ~4.5× reduction overall and the difference between
//       the buffer fitting in storage or not.
//
//     v1 (legacy, still readable):
//       { count, capacity, featureLen, hasPolicyTarget[] }
//       Per-example payload differed only in the policyTarget block,
//       which was POLICY_SIZE float32s (the full dense one-hot).
//
//   [4+H..)  raw Float32 payload (see per-version layouts above).
//
// `POLICY_SIZE` is not embedded in the header; the version determines
// how to interpret the per-example payload. `featureLen` is stored so
// the format remains self-describing even if FEATURE_COUNT changes
// between releases (a version bump in the encoding/nn module already
// invalidates any saved state via META).

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

const SERIAL_VERSION = 2;

interface SerialHeader {
  /** Wire format version. Omitted in v1 blobs; v2+ always set. */
  version?: number;
  count: number;
  capacity: number;
  featureLen: number;
  hasPolicyTarget: boolean[];
}

/**
 * Return the single 1-hot index in a policy target Float32Array.
 * `buildTrainingExamples` is the only producer and always constructs a
 * one-hot, so the first nonzero entry is the answer. We fall back to
 * argmax if the assumption ever breaks (e.g. a future change introduces
 * soft policy targets) — the cost is one extra pass per winning example.
 */
function policyTargetIndex(pt: Float32Array): number {
  for (let i = 0; i < pt.length; i++) if (pt[i] === 1) return i;
  let bestI = 0;
  let bestV = pt[0];
  for (let i = 1; i < pt.length; i++) {
    if (pt[i] > bestV) { bestV = pt[i]; bestI = i; }
  }
  return bestI;
}

function serializeBuffer(examples: TrainingExample[], capacity: number): Uint8Array {
  const count = examples.length;
  const featureLen = count > 0 ? examples[0].features.length : 0;

  const hasPolicyTarget = examples.map(ex => ex.policyTarget !== null);

  const header: SerialHeader = {
    version: SERIAL_VERSION,
    count,
    capacity,
    featureLen,
    hasPolicyTarget,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));

  // v2: each example is features + valueTarget + optional policy index (1 float).
  let floatCount = 0;
  for (let i = 0; i < count; i++) {
    floatCount += featureLen + 1 + (hasPolicyTarget[i] ? 1 : 0);
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
      floatPayload[offset] = policyTargetIndex(ex.policyTarget);
      offset += 1;
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

function policyOneHot(index: number): Float32Array {
  const pt = new Float32Array(POLICY_SIZE);
  if (index >= 0 && index < POLICY_SIZE) pt[index] = 1;
  return pt;
}

function deserializeExamples(bytes: Uint8Array): TrainingExample[] {
  if (bytes.length < 4) return [];

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(0, /* littleEndian= */ true);

  const headerBytes = bytes.slice(4, 4 + headerLen);
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as SerialHeader;

  const { count, featureLen, hasPolicyTarget } = header;
  const version = header.version ?? 1;
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
      if (version >= 2) {
        const idx = floatView[offset];
        offset += 1;
        policyTarget = policyOneHot(Math.round(idx));
      } else {
        // v1: dense one-hot embedded directly in the payload.
        policyTarget = floatView.slice(offset, offset + POLICY_SIZE) as Float32Array;
        offset += POLICY_SIZE;
      }
    }

    examples.push({ features, valueTarget, policyTarget });
  }

  return examples;
}
