// Unit tests for core/training/index.ts (ReplayBuffer).

import { describe, expect, it } from 'vitest';
import { createReplayBuffer, deserializeReplayBuffer } from './index.js';
import type { TrainingExample } from '../nn/index.js';
import { POLICY_SIZE } from '../nn/index.js';
import { FEATURE_COUNT } from '../encoding/index.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Simple seeded PRNG (mulberry32). Matches the one used in mcts.test.ts. */
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Build a minimal TrainingExample with recognizable values for testing. */
function makeExample(
  id: number,
  withPolicy = true,
): TrainingExample {
  const features = new Float32Array(FEATURE_COUNT).fill(id);
  const valueTarget = id % 3 === 0 ? 1 : id % 3 === 1 ? -1 : 0;
  const policyTarget = withPolicy
    ? (() => {
        const pt = new Float32Array(POLICY_SIZE);
        pt[id % POLICY_SIZE] = 1;
        return pt;
      })()
    : null;
  return { features, valueTarget, policyTarget };
}

/** Compare two Float32Arrays byte-for-byte. */
function float32Equal(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Compare two TrainingExamples exactly. */
function exampleEqual(a: TrainingExample, b: TrainingExample): boolean {
  if (!float32Equal(a.features, b.features)) return false;
  if (a.valueTarget !== b.valueTarget) return false;
  if ((a.policyTarget === null) !== (b.policyTarget === null)) return false;
  if (a.policyTarget !== null && b.policyTarget !== null) {
    if (!float32Equal(a.policyTarget, b.policyTarget)) return false;
  }
  return true;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('createReplayBuffer', () => {
  it('starts empty', () => {
    const buf = createReplayBuffer(10);
    expect(buf.size()).toBe(0);
    expect(buf.capacity()).toBe(10);
    expect(buf.snapshot()).toEqual([]);
  });

  it('add then size/sample', () => {
    const buf = createReplayBuffer(5);
    buf.add(makeExample(0));
    buf.add(makeExample(1));
    expect(buf.size()).toBe(2);
    const samples = buf.sample(2, makePrng(1));
    expect(samples).toHaveLength(2);
    // Each sample must be one of the inserted examples.
    for (const s of samples) {
      const snap = buf.snapshot();
      expect(snap.some(e => exampleEqual(e, s))).toBe(true);
    }
  });

  it('eviction: filling past capacity drops the oldest entry', () => {
    const buf = createReplayBuffer(3);
    buf.add(makeExample(0)); // oldest
    buf.add(makeExample(1));
    buf.add(makeExample(2));
    // Buffer is now full. Adding another should evict makeExample(0).
    buf.add(makeExample(3));

    expect(buf.size()).toBe(3);
    const snap = buf.snapshot();
    // example(0) should be gone.
    const hasZero = snap.some(e => float32Equal(e.features, new Float32Array(FEATURE_COUNT).fill(0)));
    expect(hasZero).toBe(false);
    // examples 1, 2, 3 should be present.
    for (const id of [1, 2, 3]) {
      expect(snap.some(e => float32Equal(e.features, new Float32Array(FEATURE_COUNT).fill(id)))).toBe(true);
    }
  });

  it('eviction: adding many entries preserves exactly the last `capacity` ones', () => {
    const capacity = 4;
    const buf = createReplayBuffer(capacity);
    for (let i = 0; i < 10; i++) buf.add(makeExample(i));
    expect(buf.size()).toBe(capacity);
    const snap = buf.snapshot();
    // Should be examples 6, 7, 8, 9 (last 4).
    for (let i = 0; i < capacity; i++) {
      const expected = makeExample(10 - capacity + i);
      expect(exampleEqual(snap[i], expected)).toBe(true);
    }
  });

  it('sample is deterministic given a seeded rng', () => {
    const buf = createReplayBuffer(10);
    for (let i = 0; i < 7; i++) buf.add(makeExample(i));

    const sample1 = buf.sample(5, makePrng(42));
    const sample2 = buf.sample(5, makePrng(42));

    expect(sample1).toHaveLength(5);
    for (let i = 0; i < sample1.length; i++) {
      expect(exampleEqual(sample1[i], sample2[i])).toBe(true);
    }
  });

  it('sample with different seeds produces different results (probabilistically)', () => {
    const buf = createReplayBuffer(100);
    for (let i = 0; i < 100; i++) buf.add(makeExample(i));

    const sample1 = buf.sample(20, makePrng(1));
    const sample2 = buf.sample(20, makePrng(9999));
    // It's astronomically unlikely these two are identical.
    const identical = sample1.every((e, i) => exampleEqual(e, sample2[i]));
    expect(identical).toBe(false);
  });

  it('addAll is equivalent to a loop of add', () => {
    const buf1 = createReplayBuffer(10);
    const buf2 = createReplayBuffer(10);
    const examples = [makeExample(0), makeExample(1, false), makeExample(2)];

    for (const ex of examples) buf1.add(ex);
    buf2.addAll(examples);

    const snap1 = buf1.snapshot();
    const snap2 = buf2.snapshot();
    expect(snap1).toHaveLength(snap2.length);
    for (let i = 0; i < snap1.length; i++) {
      expect(exampleEqual(snap1[i], snap2[i])).toBe(true);
    }
  });

  it('addAll respects eviction just like individual adds', () => {
    const buf = createReplayBuffer(3);
    buf.addAll([makeExample(0), makeExample(1), makeExample(2), makeExample(3)]);
    expect(buf.size()).toBe(3);
    const snap = buf.snapshot();
    // example(0) should be evicted.
    expect(snap.some(e => float32Equal(e.features, new Float32Array(FEATURE_COUNT).fill(0)))).toBe(false);
  });
});

describe('ReplayBuffer serialize/deserialize', () => {
  it('round-trips an empty buffer', () => {
    const buf = createReplayBuffer(5);
    const bytes = buf.serialize();
    const restored = deserializeReplayBuffer(bytes, 5);
    expect(restored.size()).toBe(0);
    expect(restored.snapshot()).toEqual([]);
  });

  it('round-trips examples with and without policy targets', () => {
    const buf = createReplayBuffer(20);
    const examples = [
      makeExample(0, true),
      makeExample(1, false), // null policy
      makeExample(2, true),
      makeExample(3, false),
    ];
    buf.addAll(examples);

    const bytes = buf.serialize();
    const restored = deserializeReplayBuffer(bytes, 20);

    expect(restored.size()).toBe(4);
    const snap = restored.snapshot();
    for (let i = 0; i < examples.length; i++) {
      expect(exampleEqual(snap[i], examples[i])).toBe(true);
    }
  });

  it('preserves snapshot() exactly — features, valueTargets, and policy targets byte-for-byte', () => {
    const buf = createReplayBuffer(10);
    for (let i = 0; i < 6; i++) buf.add(makeExample(i, i % 2 === 0));

    const orig = buf.snapshot();
    const bytes = buf.serialize();
    const restored = deserializeReplayBuffer(bytes, 10);
    const snap = restored.snapshot();

    expect(snap).toHaveLength(orig.length);
    for (let i = 0; i < orig.length; i++) {
      // features
      expect(float32Equal(snap[i].features, orig[i].features)).toBe(true);
      // valueTarget
      expect(snap[i].valueTarget).toBe(orig[i].valueTarget);
      // policyTarget nullness
      expect(snap[i].policyTarget === null).toBe(orig[i].policyTarget === null);
      // policyTarget content
      if (orig[i].policyTarget !== null && snap[i].policyTarget !== null) {
        expect(float32Equal(snap[i].policyTarget, orig[i].policyTarget)).toBe(true);
      }
    }
  });

  it('round-trip with a smaller restored capacity truncates correctly', () => {
    const buf = createReplayBuffer(10);
    for (let i = 0; i < 8; i++) buf.add(makeExample(i));

    const bytes = buf.serialize();
    // Restore with capacity=5: the deserializer uses addAll which evicts
    // if necessary.
    const restored = deserializeReplayBuffer(bytes, 5);
    expect(restored.size()).toBe(5);
    // The last 5 of the 8 stored examples should survive.
    const snap = restored.snapshot();
    const origSnap = buf.snapshot();
    for (let i = 0; i < 5; i++) {
      expect(exampleEqual(snap[i], origSnap[8 - 5 + i])).toBe(true);
    }
  });

  it('round-trips a full (wrapped) buffer correctly', () => {
    const capacity = 5;
    const buf = createReplayBuffer(capacity);
    // Add 8 examples so the ring wraps.
    for (let i = 0; i < 8; i++) buf.add(makeExample(i));

    const orig = buf.snapshot();
    const bytes = buf.serialize();
    const restored = deserializeReplayBuffer(bytes, capacity);
    const snap = restored.snapshot();

    expect(snap).toHaveLength(orig.length);
    for (let i = 0; i < orig.length; i++) {
      expect(exampleEqual(snap[i], orig[i])).toBe(true);
    }
  });
});
