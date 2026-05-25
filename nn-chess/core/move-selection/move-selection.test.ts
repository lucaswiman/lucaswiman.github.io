// Unit tests for the generic move-selection function. Exercises a real
// ChessNet (so tfjs is involved) but otherwise framework-free.

import { beforeAll, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { selectMoveFromNet } from './index.js';
import { ChessNet } from '../nn/index.js';
import { initialState, fromFen, legalMoves } from '../rules/index.js';

// Mulberry32: tiny deterministic PRNG so temperature-sampling tests are
// reproducible without dragging in a dependency.
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

describe('selectMoveFromNet — policy-head mode', () => {
  beforeAll(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
  });

  it('returns one of the legal moves for the starting position (greedy)', async () => {
    const net = ChessNet.create({ seed: 11 });
    try {
      const state = initialState();
      const { move, priors } = await selectMoveFromNet(net, state);
      const legal = new Set(legalMoves(state));
      expect(legal.has(move)).toBe(true);
      // Priors are over the legal moves only — same shape as policyFromNet.
      for (const m of priors.keys()) expect(legal.has(m)).toBe(true);
    } finally {
      net.dispose();
    }
  });

  it('greedy selection is deterministic across calls', async () => {
    const net = ChessNet.create({ seed: 17 });
    try {
      const state = initialState();
      const a = await selectMoveFromNet(net, state);
      const b = await selectMoveFromNet(net, state);
      expect(a.move).toBe(b.move);
    } finally {
      net.dispose();
    }
  });

  it('returns the unique legal move when only one is available', async () => {
    // White king h1 in check from a black queen on g2; the only legal
    // escape is Kxg2.
    const net = ChessNet.create({ seed: 3 });
    try {
      const state = fromFen('4k3/8/8/8/8/8/6q1/7K w - - 0 1');
      const legal = legalMoves(state);
      expect(legal.length).toBe(1);
      expect(legal[0]).toBe('h1g2');
      const { move } = await selectMoveFromNet(net, state);
      expect(move).toBe(legal[0]);
    } finally {
      net.dispose();
    }
  });

  it('throws on a terminal position', async () => {
    const net = ChessNet.create({ seed: 5 });
    try {
      // Fool's mate: black just played Qh4#, white is checkmated.
      const terminal = fromFen(
        'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3',
      );
      await expect(selectMoveFromNet(net, terminal)).rejects.toThrow();
    } finally {
      net.dispose();
    }
  });

  it('temperature > 0 with a seeded rng samples a legal move reproducibly', async () => {
    const net = ChessNet.create({ seed: 23 });
    try {
      const state = initialState();
      const legal = new Set(legalMoves(state));
      const first = await selectMoveFromNet(net, state, {
        temperature: 1,
        rng: makePrng(99),
      });
      const second = await selectMoveFromNet(net, state, {
        temperature: 1,
        rng: makePrng(99),
      });
      expect(legal.has(first.move)).toBe(true);
      expect(first.move).toBe(second.move);
    } finally {
      net.dispose();
    }
  });
});

describe('selectMoveFromNet — MCTS mode', () => {
  beforeAll(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
  });

  it('returns a legal move and exposes search stats', async () => {
    const net = ChessNet.create({ seed: 41 });
    try {
      const state = initialState();
      const legal = new Set(legalMoves(state));
      const result = await selectMoveFromNet(net, state, {
        search: { simulations: 8, cPuct: 1.5 },
      });
      expect(legal.has(result.move)).toBe(true);
      expect(result.search).toBeDefined();
      expect(result.search!.visitCounts.size).toBeGreaterThan(0);
      const totalVisits = Array.from(result.search!.visitCounts.values()).reduce(
        (a, b) => a + b,
        0,
      );
      expect(totalVisits).toBe(8);
      expect(result.search!.rootValue).toBeGreaterThanOrEqual(-1);
      expect(result.search!.rootValue).toBeLessThanOrEqual(1);
    } finally {
      net.dispose();
    }
  });

  it('seizes a mate-in-1 via the search-internal override', async () => {
    // Black king cornered on a8, white king on c7 supports a back-rank
    // mate — White to move with Qb1-b8# (and Qb1-b7# also delivers).
    const net = ChessNet.create({ seed: 53 });
    try {
      const state = fromFen('k7/2K5/8/8/8/8/8/1Q6 w - - 0 1');
      const result = await selectMoveFromNet(net, state, {
        search: { simulations: 4, cPuct: 1.5 },
      });
      expect(result.search!.mateInOneMoves.length).toBeGreaterThan(0);
      expect(result.search!.mateInOneMoves).toContain(result.move);
    } finally {
      net.dispose();
    }
  });
});
