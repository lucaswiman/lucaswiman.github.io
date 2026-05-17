// Unit tests for nn-chess/core/nn/index.ts
//
// These tests run in Node.js via vitest. tfjs defaults to the CPU backend
// in Node, which is what we want for determinism. We explicitly set the
// backend to 'cpu' in the test setup to be sure.

import { beforeAll, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
  ChessNet,
  deserialize,
  policyIndexFromUci,
  uciFromPolicyIndex,
  POLICY_SIZE,
  NN_VERSION,
  type TrainingExample,
} from './index.js';
import { FEATURE_COUNT } from '../encoding/index.js';

// ── Test setup ───────────────────────────────────────────────────────────────

beforeAll(async () => {
  // Force CPU backend for determinism across environments.
  await tf.setBackend('cpu');
  await tf.ready();
});

// Helpers
function makeFeatures(fill = 0): Float32Array {
  return new Float32Array(FEATURE_COUNT).fill(fill);
}

function makeRandomFeatures(seed: number): Float32Array {
  // Simple deterministic pseudo-random fill for test inputs.
  const arr = new Float32Array(FEATURE_COUNT);
  let x = seed;
  for (let i = 0; i < arr.length; i++) {
    // LCG
    x = (x * 1664525 + 1013904223) & 0xffffffff;
    arr[i] = (x >>> 16) / 65535;
  }
  return arr;
}

// ── Policy index helpers ──────────────────────────────────────────────────────

describe('policyIndexFromUci / uciFromPolicyIndex', () => {
  it('round-trips e2e4', () => {
    const idx = policyIndexFromUci('e2e4');
    expect(uciFromPolicyIndex(idx)).toBe('e2e4');
  });

  it('round-trips a1h8', () => {
    const idx = policyIndexFromUci('a1h8');
    expect(uciFromPolicyIndex(idx)).toBe('a1h8');
  });

  it('round-trips h1a8', () => {
    const idx = policyIndexFromUci('h1a8');
    expect(uciFromPolicyIndex(idx)).toBe('h1a8');
  });

  it('policyIndexFromUci returns a value in [0, POLICY_SIZE)', () => {
    const idx = policyIndexFromUci('e2e4');
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(POLICY_SIZE);
  });

  it('POLICY_SIZE equals 4096', () => {
    expect(POLICY_SIZE).toBe(4096);
  });

  it('NN_VERSION is defined and at least 1', () => {
    expect(NN_VERSION).toBeGreaterThanOrEqual(1);
  });
});

// ── Shape checks ──────────────────────────────────────────────────────────────

describe('ChessNet.predict — output shapes', () => {
  it('returns value in [-1, 1] and policy of length POLICY_SIZE', () => {
    const net = ChessNet.create({ seed: 1 });
    try {
      const features = makeFeatures(0.5);
      const { value, policy } = net.predict(features);

      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);

      expect(policy).toBeInstanceOf(Float32Array);
      expect(policy.length).toBe(POLICY_SIZE);
    } finally {
      net.dispose();
    }
  });

  it('predictBatch returns one entry per input', () => {
    const net = ChessNet.create({ seed: 2 });
    try {
      const batch = [makeFeatures(0), makeFeatures(0.5), makeFeatures(1)];
      const results = net.predictBatch(batch);

      expect(results).toHaveLength(3);
      for (const { value, policy } of results) {
        expect(typeof value).toBe('number');
        expect(value).toBeGreaterThanOrEqual(-1);
        expect(value).toBeLessThanOrEqual(1);
        expect(policy).toBeInstanceOf(Float32Array);
        expect(policy.length).toBe(POLICY_SIZE);
      }
    } finally {
      net.dispose();
    }
  });

  it('predictBatch on empty array returns empty array', () => {
    const net = ChessNet.create({ seed: 3 });
    try {
      expect(net.predictBatch([])).toEqual([]);
    } finally {
      net.dispose();
    }
  });
});

// ── Batch consistency ─────────────────────────────────────────────────────────

describe('predictBatch matches individual predict calls', () => {
  it('batch [a, b, c] gives same values as three predict calls (within 1e-5)', () => {
    const net = ChessNet.create({ seed: 42 });
    try {
      const a = makeRandomFeatures(100);
      const b = makeRandomFeatures(200);
      const c = makeRandomFeatures(300);

      const [ra, rb, rc] = net.predictBatch([a, b, c]);
      const sa = net.predict(a);
      const sb = net.predict(b);
      const sc = net.predict(c);

      expect(ra.value).toBeCloseTo(sa.value, 5);
      expect(rb.value).toBeCloseTo(sb.value, 5);
      expect(rc.value).toBeCloseTo(sc.value, 5);

      for (let i = 0; i < POLICY_SIZE; i++) {
        expect(ra.policy[i]).toBeCloseTo(sa.policy[i], 5);
        expect(rb.policy[i]).toBeCloseTo(sb.policy[i], 5);
        expect(rc.policy[i]).toBeCloseTo(sc.policy[i], 5);
      }
    } finally {
      net.dispose();
    }
  });
});

// ── Determinism ───────────────────────────────────────────────────────────────

describe('determinism with fixed seed', () => {
  it('two ChessNet({ seed: 42 }) instances produce identical predictions', () => {
    const net1 = ChessNet.create({ seed: 42 });
    const net2 = ChessNet.create({ seed: 42 });
    try {
      const features = makeRandomFeatures(999);
      const r1 = net1.predict(features);
      const r2 = net2.predict(features);

      expect(r1.value).toBe(r2.value);
      for (let i = 0; i < POLICY_SIZE; i++) {
        expect(r1.policy[i]).toBe(r2.policy[i]);
      }
    } finally {
      net1.dispose();
      net2.dispose();
    }
  });
});

// ── Serialize / deserialize round-trip ────────────────────────────────────────

describe('serialize / deserialize', () => {
  it('round-trip preserves predictions exactly (within 1e-6)', async () => {
    const net = ChessNet.create({ seed: 7 });
    try {
      const features = makeRandomFeatures(77);
      const original = net.predict(features);

      const bytes = await net.serialize();
      expect(bytes).toBeInstanceOf(Uint8Array);
      expect(bytes.length).toBeGreaterThan(0);

      const restored = await deserialize(bytes);
      try {
        const fromRestored = restored.predict(features);

        expect(fromRestored.value).toBeCloseTo(original.value, 6);
        for (let i = 0; i < POLICY_SIZE; i++) {
          expect(fromRestored.policy[i]).toBeCloseTo(original.policy[i], 6);
        }
      } finally {
        restored.dispose();
      }
    } finally {
      net.dispose();
    }
  });

  it('serialize produces a Uint8Array that can be stored in BlobStorage', async () => {
    const net = ChessNet.create({ seed: 8 });
    try {
      const bytes = await net.serialize();
      // BlobStorage.put/get contract: value is a Uint8Array.
      expect(bytes).toBeInstanceOf(Uint8Array);
      // Sanity: at minimum a 4-byte header + some JSON + some weights.
      expect(bytes.length).toBeGreaterThan(100);
    } finally {
      net.dispose();
    }
  });
});

// ── Training convergence ──────────────────────────────────────────────────────

describe('trainBatch', () => {
  it('after 50 steps on valueTarget=1, value output moves toward +1', async () => {
    const net = ChessNet.create({ seed: 13 });
    try {
      const features = makeRandomFeatures(42);

      const before = net.predict(features).value;

      const example: TrainingExample = {
        features,
        valueTarget: 1,
        policyTarget: null, // value-only training
      };

      for (let i = 0; i < 50; i++) {
        await net.trainBatch([example]);
      }

      const after = net.predict(features).value;

      // The value should have moved toward +1.
      expect(after).toBeGreaterThan(before);
      // And should be meaningfully closer to +1.
      expect(after).toBeGreaterThan(0);
    } finally {
      net.dispose();
    }
  }, 60_000 /* 60 s timeout for 50 gradient steps on CPU */);

  it('trainBatch returns non-negative losses', async () => {
    const net = ChessNet.create({ seed: 14 });
    try {
      const features = makeRandomFeatures(55);
      const policyTarget = new Float32Array(POLICY_SIZE);
      policyTarget[policyIndexFromUci('e2e4')] = 1;

      const result = await net.trainBatch([
        { features, valueTarget: 1, policyTarget },
      ]);

      expect(result.valueLoss).toBeGreaterThanOrEqual(0);
      expect(result.policyLoss).toBeGreaterThanOrEqual(0);
      expect(result.totalLoss).toBeGreaterThanOrEqual(0);
    } finally {
      net.dispose();
    }
  });

  it('trainBatch on empty batch returns zero losses', async () => {
    const net = ChessNet.create({ seed: 15 });
    try {
      const result = await net.trainBatch([]);
      expect(result.valueLoss).toBe(0);
      expect(result.policyLoss).toBe(0);
      expect(result.totalLoss).toBe(0);
    } finally {
      net.dispose();
    }
  });

  it('policyTarget=null contributes 0 to policy loss (masked example)', async () => {
    // Compare a batch of one example with policyTarget=null vs one with a
    // real policyTarget. The masked example should have policyLoss == 0.
    const net = ChessNet.create({ seed: 16 });
    try {
      const features = makeRandomFeatures(66);

      const maskedResult = await net.trainBatch([
        { features, valueTarget: 0, policyTarget: null },
      ]);

      expect(maskedResult.policyLoss).toBe(0);

      // With a real policyTarget, the policy loss should be > 0.
      const policyTarget = new Float32Array(POLICY_SIZE);
      policyTarget[policyIndexFromUci('d2d4')] = 1;

      const unmaskedResult = await net.trainBatch([
        { features, valueTarget: 0, policyTarget },
      ]);

      expect(unmaskedResult.policyLoss).toBeGreaterThan(0);
    } finally {
      net.dispose();
    }
  });
});

// ── Activation inspection (interpretability hook) ─────────────────────────────

describe('getActivations / layerNames', () => {
  it('layerNames returns a non-empty array of strings', () => {
    const net = ChessNet.create({ seed: 99 });
    try {
      const names = net.layerNames();
      expect(names.length).toBeGreaterThan(0);
      for (const n of names) expect(typeof n).toBe('string');
    } finally {
      net.dispose();
    }
  });

  it('getActivations returns Float32Arrays for each requested layer', () => {
    const net = ChessNet.create({ seed: 100 });
    try {
      const features = makeRandomFeatures(11);
      const names = net.layerNames();
      const acts = net.getActivations(features, names);

      for (const name of names) {
        expect(acts[name]).toBeInstanceOf(Float32Array);
        expect(acts[name].length).toBeGreaterThan(0);
      }
    } finally {
      net.dispose();
    }
  });

  it('getActivations with unknown layer name silently omits it', () => {
    const net = ChessNet.create({ seed: 101 });
    try {
      const features = makeFeatures(0);
      const acts = net.getActivations(features, ['does_not_exist']);
      expect(acts['does_not_exist']).toBeUndefined();
    } finally {
      net.dispose();
    }
  });
});
