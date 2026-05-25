// Tests for the `maiaPolicy(...)` MCTS adapter and an MCTS smoke run.
//
// Same skip-if-no-weights pattern as model.test.ts: if `weights.bin`
// isn't on disk we report `it.skip` with a clear message instead of
// failing. Set MAIA2_WEIGHTS=<path> or run convert-maia2.py with its
// default output (/tmp/maia2_out/weights.bin) to enable.

import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import * as tf from '@tensorflow/tfjs';

import { loadMaiaWeights, type MaiaManifest } from './weights.js';
import { MaiaEngine } from './engine.js';
import { maiaPolicy } from './engine.js';
import { fromFen, legalMoves } from '../rules/index.js';
import { search } from '../mcts/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const FIXTURES_DIR = join(__dirname, '__fixtures__');
const MANIFEST_PATH = join(FIXTURES_DIR, 'manifest.json');
const REFERENCE_PATH = join(FIXTURES_DIR, 'reference.json');

interface ReferenceFixture {
  fen: string;
  elo_self: number;
  elo_oppo: number;
}

function findWeightsPath(): string | null {
  const fromEnv = process.env.MAIA2_WEIGHTS;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  const fallback = '/tmp/maia2_out/weights.bin';
  if (existsSync(fallback)) return fallback;
  return null;
}

const WEIGHTS_PATH = findWeightsPath();

describe('maiaPolicy + MCTS smoke', () => {
  const haveWeights = WEIGHTS_PATH !== null;
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

  itc('returns priors over legal moves summing to ≈ 1 and a finite value', async () => {
    await tf.setBackend('cpu');
    const weights = loadMaiaWeights(blobBytes, manifest);
    const engine = new MaiaEngine(weights);
    try {
      const state = fromFen(reference.fen);
      const policy = maiaPolicy(engine, reference.elo_self, reference.elo_oppo);
      const { priors, value } = await policy.evaluate(state);

      const legal = new Set(legalMoves(state));
      // Every key in priors must be a legal move.
      for (const m of priors.keys()) {
        expect(legal.has(m)).toBe(true);
      }
      // Conversely, every legal move should be a key (with possibly-zero mass).
      for (const m of legal) {
        expect(priors.has(m)).toBe(true);
      }
      let sum = 0;
      for (const p of priors.values()) sum += p;
      expect(sum).toBeCloseTo(1, 4);
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(-1.5);
      expect(value).toBeLessThanOrEqual(1.5);
    } finally {
      engine.dispose();
    }
  }, 60_000);

  itc('drives MCTS to a legal move with simulations=2', async () => {
    await tf.setBackend('cpu');
    const weights = loadMaiaWeights(blobBytes, manifest);
    const engine = new MaiaEngine(weights);
    try {
      const state = fromFen(reference.fen);
      const policy = maiaPolicy(engine, reference.elo_self, reference.elo_oppo);
      // Tiny sims budget so this is bounded under WASM/CPU — we only
      // care that search() exits with a legal best move and well-formed
      // visit counts.
      const result = await search(state, policy, { simulations: 2, cPuct: 1.5 });
      const legal = legalMoves(state);
      expect(legal).toContain(result.bestMove);
      const total = Array.from(result.visitCounts.values()).reduce((a, b) => a + b, 0);
      expect(total).toBeGreaterThan(0);
    } finally {
      engine.dispose();
    }
  }, 120_000);
});
