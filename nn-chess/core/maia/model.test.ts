// End-to-end parity test for the Maia-2 forward pass.
//
// This test compares the TS forward pass output against activations
// dumped by `nn-chess/scripts/convert-maia2.py --ref-fen ...`. It
// requires the weights.bin produced by that same script to be present
// at one of:
//   - $MAIA2_WEIGHTS  (env var pointing directly at the .bin file)
//   - /tmp/maia2_out/weights.bin  (the conversion script's default
//     output, used by developers running locally)
//
// CI does NOT have the weights and is expected to skip the parity
// check; the test logs a clear message and exits as `skipped` rather
// than failing. Run the conversion locally and re-run vitest to enable
// the real assertion.

import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import * as tf from '@tensorflow/tfjs';

import {
  encodeForMaia,
  eloToBucket,
} from './encoding.js';
import { loadMaiaWeights, type MaiaManifest } from './weights.js';
import { maiaForward, type MaiaActivationName } from './model.js';
import { maiaPolicyIndex, MAIA_MOVES } from './action-space.js';
import { fromFen, legalMoves } from '../rules/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const FIXTURES_DIR = join(__dirname, '__fixtures__');
const MANIFEST_PATH = join(FIXTURES_DIR, 'manifest.json');
const REFERENCE_PATH = join(FIXTURES_DIR, 'reference.json');

interface ReferenceFixture {
  fen: string;
  elo_self: number;
  elo_oppo: number;
  mirrored: boolean;
  legal_indices: number[];
  value: number;
  policy_top5_legal: Array<{ index: number; uci: string; logit: number }>;
  pooled_summary: ActivationSummary;
  post_patch_summary: ActivationSummary;
  post_attn_summary: ActivationSummary;
  backbone_summary: ActivationSummary;
}

interface ActivationSummary {
  sum: number;
  abs_sum: number;
  max: number;
  min: number;
  first8: number[];
  last8: number[];
  n: number;
}

function summarize(arr: Float32Array): ActivationSummary {
  let sum = 0, abs_sum = 0, max = -Infinity, min = Infinity;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    abs_sum += Math.abs(arr[i]);
    if (arr[i] > max) max = arr[i];
    if (arr[i] < min) min = arr[i];
  }
  return {
    sum,
    abs_sum,
    max,
    min,
    first8: Array.from(arr.slice(0, 8)),
    last8: Array.from(arr.slice(arr.length - 8)),
    n: arr.length,
  };
}

function findWeightsPath(): string | null {
  const fromEnv = process.env.MAIA2_WEIGHTS;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const fallback = '/tmp/maia2_out/weights.bin';
  if (existsSync(fallback)) return fallback;
  return null;
}

const WEIGHTS_PATH = findWeightsPath();

describe('Maia-2 forward pass parity', () => {
  const haveWeights = WEIGHTS_PATH !== null;
  // We can't conditionally skip the test, but we can short-circuit
  // every assertion with a clear status message instead.
  const itc = haveWeights ? it : it.skip;

  if (!haveWeights) {
    it.skip(
      `weights.bin not found (set MAIA2_WEIGHTS or run convert-maia2.py to /tmp/maia2_out)`,
      () => {},
    );
    return;
  }

  const manifest: MaiaManifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf-8'));
  const reference: ReferenceFixture = JSON.parse(readFileSync(REFERENCE_PATH, 'utf-8'));
  const blob = readFileSync(WEIGHTS_PATH!);
  const blobBytes = new Uint8Array(blob.buffer, blob.byteOffset, blob.byteLength);

  // Each `it` builds a fresh forward-pass result. We could share state
  // but the cost (~1-2s on CPU) is tolerable and isolation makes
  // failures easier to read.
  itc('matches the reference value head to within 1e-2', async () => {
    await tf.setBackend('cpu');
    const weights = loadMaiaWeights(blobBytes, manifest);
    try {
      const state = fromFen(reference.fen);
      const { features } = encodeForMaia(state);
      const pred = maiaForward(weights, features, {
        eloSelf: eloToBucket(reference.elo_self),
        eloOppo: eloToBucket(reference.elo_oppo),
      });
      expect(pred.value).toBeCloseTo(reference.value, 2);
    } finally {
      weights.dispose();
    }
  }, 30_000);

  itc('matches the reference top-5 legal policy moves', async () => {
    await tf.setBackend('cpu');
    const weights = loadMaiaWeights(blobBytes, manifest);
    try {
      const state = fromFen(reference.fen);
      const { features } = encodeForMaia(state);
      const pred = maiaForward(weights, features, {
        eloSelf: eloToBucket(reference.elo_self),
        eloOppo: eloToBucket(reference.elo_oppo),
      });
      // Rank legal moves by our logits.
      const legalIndices = reference.legal_indices;
      const ranked = legalIndices
        .map(i => ({ index: i, uci: MAIA_MOVES[i], logit: pred.policyLogits[i] }))
        .sort((a, b) => b.logit - a.logit)
        .slice(0, 5);
      // Compare top-1; the next four should be the same set (order may
      // shuffle for moves with near-identical logits).
      expect(ranked[0].index).toBe(reference.policy_top5_legal[0].index);
      const ourTop5Set = new Set(ranked.map(r => r.index));
      const refTop5Set = new Set(reference.policy_top5_legal.map(r => r.index));
      // At least 4 of 5 should overlap; fp16 rounding can rearrange the tail.
      let overlap = 0;
      for (const idx of ourTop5Set) if (refTop5Set.has(idx)) overlap++;
      expect(overlap).toBeGreaterThanOrEqual(4);
    } finally {
      weights.dispose();
    }
  }, 30_000);

  itc('captures the post-backbone activation tensor at the right scale', async () => {
    await tf.setBackend('cpu');
    const weights = loadMaiaWeights(blobBytes, manifest);
    try {
      const state = fromFen(reference.fen);
      const { features } = encodeForMaia(state);
      const out = new Map<MaiaActivationName, Float32Array>();
      maiaForward(weights, features, {
        eloSelf: eloToBucket(reference.elo_self),
        eloOppo: eloToBucket(reference.elo_oppo),
        collect: new Set<MaiaActivationName>(['backbone', 'pooled']),
        out,
      });
      const bb = out.get('backbone')!;
      const pooled = out.get('pooled')!;
      const bbSummary = summarize(bb);
      const pooledSummary = summarize(pooled);
      // The fp16 storage + reduced precision in tf.js's WASM/CPU
      // kernels can shift sums by a few % over 1000s of accumulated
      // elements. We compare relative magnitudes rather than per-
      // element identity; the top-5 + value tests above pin down the
      // correctness of the head outputs.
      expect(bbSummary.n).toBe(reference.backbone_summary.n);
      expect(pooledSummary.n).toBe(reference.pooled_summary.n);

      const relErrSum =
        Math.abs(bbSummary.abs_sum - reference.backbone_summary.abs_sum) /
        Math.max(reference.backbone_summary.abs_sum, 1e-6);
      expect(relErrSum).toBeLessThan(0.05);

      const pooledRel =
        Math.abs(pooledSummary.abs_sum - reference.pooled_summary.abs_sum) /
        Math.max(reference.pooled_summary.abs_sum, 1e-6);
      expect(pooledRel).toBeLessThan(0.05);
    } finally {
      weights.dispose();
    }
  }, 30_000);

  itc('selects a legal move via MaiaEngine.selectMove', async () => {
    await tf.setBackend('cpu');
    const { MaiaEngine } = await import('./engine.js');
    const weights = loadMaiaWeights(blobBytes, manifest);
    const engine = new MaiaEngine(weights);
    try {
      const state = fromFen(reference.fen);
      const result = engine.selectMove(state, { elo: reference.elo_self });
      const legal = legalMoves(state);
      expect(legal).toContain(result.move);
      // moveProbs should sum to ≈ 1 over legal moves.
      let s = 0;
      for (const p of result.moveProbs.values()) s += p;
      expect(s).toBeCloseTo(1, 4);
    } finally {
      engine.dispose();
    }
  }, 30_000);
});

// Sanity-check that maiaPolicyIndex agrees with MAIA_MOVES for every
// move in the reference's legal-indices set. Cheap and doesn't need
// the weight blob.
describe('Maia-2 policy index <→ move name', () => {
  const reference: ReferenceFixture = JSON.parse(readFileSync(REFERENCE_PATH, 'utf-8'));
  it('maps every reference-legal index to a UCI string that round-trips', () => {
    for (const idx of reference.legal_indices) {
      const move = MAIA_MOVES[idx];
      expect(move).toBeDefined();
      expect(maiaPolicyIndex(move)).toBe(idx);
    }
  });
});
