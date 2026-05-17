// Tests for the PUCT MCTS implementation.
//
// All tests use a hand-rolled Policy (uniform priors over legal moves,
// value = 0) unless otherwise noted. That's sufficient because:
//   - The terminal backup propagates +1/-1 from checkmate nodes even
//     when the policy value head returns 0.
//   - MCTS with enough simulations will find forced mates via the terminal
//     signal alone; the uniform prior is the weakest possible prior and
//     therefore the hardest test for the search.

import { describe, expect, it } from 'vitest';
import {
  fromFen,
  legalMoves,
  applyMove,
  isTerminal,
  outcome,
} from '../rules/index.js';
import { search } from './index.js';
import type { Policy, SearchConfig } from './index.js';
import type { GameState, Move } from '../rules/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Uniform-prior, value=0 policy. */
const uniformPolicy: Policy = {
  async evaluate(state: GameState) {
    const moves = legalMoves(state);
    const p = 1 / moves.length;
    const priors = new Map<Move, number>(moves.map(m => [m, p]));
    return { priors, value: 0 };
  },
};

/** A simple seeded PRNG (mulberry32). */
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

function baseConfig(overrides: Partial<SearchConfig> = {}): SearchConfig {
  return {
    simulations: 200,
    cPuct: 1.5,
    rng: makePrng(42),
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Mate-in-1, white to move
// ---------------------------------------------------------------------------
// Position: 6k1/8/5K2/8/8/8/8/6Q1 w - - 0 1
//   White queen on g1, white king on f6, black king on g8.
//   g1g7 is the only legal mate (g1g8 stalemates).
//   This is deliberately the strictest possible test: uniform priors,
//   only one winning move, one stalemate trap.

describe('mcts: mate-in-1, white to move', () => {
  const FEN = '6k1/8/5K2/8/8/8/8/6Q1 w - - 0 1';

  it('selects the mating move with 200 simulations', async () => {
    const state = fromFen(FEN);
    const result = await search(state, uniformPolicy, baseConfig({ simulations: 200 }));
    expect(result.bestMove).toBe('g1g7');
  });

  it('verifies g1g7 is indeed checkmate', () => {
    const state = fromFen(FEN);
    const next = applyMove(state, 'g1g7');
    expect(isTerminal(next)).toBe(true);
    expect(outcome(next)).toBe('white-wins');
  });

  it('verifies g1g8 is stalemate', () => {
    const state = fromFen(FEN);
    const next = applyMove(state, 'g1g8');
    expect(isTerminal(next)).toBe(true);
    expect(outcome(next)).toBe('draw');
  });
});

// ---------------------------------------------------------------------------
// Mate-in-1, black to move
// ---------------------------------------------------------------------------
// Position: 7q/8/8/8/8/1k6/8/K7 b - - 0 1
//   Black queen on h8, black king on b3, white king on a1.
//   h8h1 and h8b2 both deliver checkmate.

describe('mcts: mate-in-1, black to move', () => {
  const FEN = '7q/8/8/8/8/1k6/8/K7 b - - 0 1';

  it('selects a mating move with 200 simulations', async () => {
    const state = fromFen(FEN);
    const result = await search(state, uniformPolicy, baseConfig({ simulations: 200 }));
    const matingMoves = new Set(['h8h1', 'h8b2']);
    expect(matingMoves.has(result.bestMove)).toBe(true);
  });

  it('verifies h8h1 is checkmate', () => {
    const state = fromFen(FEN);
    const next = applyMove(state, 'h8h1');
    expect(isTerminal(next)).toBe(true);
    expect(outcome(next)).toBe('black-wins');
  });

  it('verifies h8b2 is checkmate', () => {
    const state = fromFen(FEN);
    const next = applyMove(state, 'h8b2');
    expect(isTerminal(next)).toBe(true);
    expect(outcome(next)).toBe('black-wins');
  });
});

// ---------------------------------------------------------------------------
// Stalemate avoidance
// ---------------------------------------------------------------------------
// Same position as the white mate-in-1 above (reused): g1g7 mates, g1g8
// stalemates. With a uniform prior and value=0, the stalemate path returns
// 0 while the mate path returns +1 from the backup. MCTS must prefer g1g7.

describe('mcts: stalemate avoidance', () => {
  const FEN = '6k1/8/5K2/8/8/8/8/6Q1 w - - 0 1';

  it('prefers mate over stalemate', async () => {
    const state = fromFen(FEN);
    const result = await search(state, uniformPolicy, baseConfig({ simulations: 200 }));
    expect(result.bestMove).toBe('g1g7');

    // Stalemate should have fewer (or equal) visits than the mating move.
    const mateVisits = result.visitCounts.get('g1g7') ?? 0;
    const stalemateVisits = result.visitCounts.get('g1g8') ?? 0;
    expect(mateVisits).toBeGreaterThan(stalemateVisits);
  });
});

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

describe('mcts: determinism with seeded rng', () => {
  it('produces identical visitCounts for the same seed', async () => {
    // Use a position with some non-trivial branching to make the test meaningful.
    const state = fromFen('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');

    const cfg1 = baseConfig({
      simulations: 100,
      dirichletAlpha: 0.3,
      dirichletEpsilon: 0.25,
      rng: makePrng(1234),
    });
    const cfg2 = baseConfig({
      simulations: 100,
      dirichletAlpha: 0.3,
      dirichletEpsilon: 0.25,
      rng: makePrng(1234),
    });

    const r1 = await search(state, uniformPolicy, cfg1);
    const r2 = await search(state, uniformPolicy, cfg2);

    expect(r1.bestMove).toBe(r2.bestMove);
    expect(Array.from(r1.visitCounts.entries()).sort()).toEqual(
      Array.from(r2.visitCounts.entries()).sort(),
    );
  });
});

// ---------------------------------------------------------------------------
// Root noise changes priors
// ---------------------------------------------------------------------------

describe('mcts: dirichlet noise', () => {
  it('changes priorPolicy when epsilon differs', async () => {
    const state = fromFen('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');

    const r1 = await search(state, uniformPolicy, baseConfig({
      simulations: 50,
      dirichletAlpha: 0.3,
      dirichletEpsilon: 0.0,
      rng: makePrng(99),
    }));
    const r2 = await search(state, uniformPolicy, baseConfig({
      simulations: 50,
      dirichletAlpha: 0.3,
      dirichletEpsilon: 0.5,
      rng: makePrng(99),
    }));

    // Priors must differ when epsilon is non-zero.
    const priors1 = Array.from(r1.priorPolicy.values());
    const priors2 = Array.from(r2.priorPolicy.values());
    expect(priors1).not.toEqual(priors2);

    // Both must sum to ~1.
    const sum1 = priors1.reduce((a, b) => a + b, 0);
    const sum2 = priors2.reduce((a, b) => a + b, 0);
    expect(sum1).toBeCloseTo(1, 5);
    expect(sum2).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// Priors masked to legal moves
// ---------------------------------------------------------------------------
// Supply a policy that returns priors with illegal moves included.
// The search must drop illegal moves and renormalize.

describe('mcts: illegal move masking', () => {
  it('drops illegal moves from policy priors and renormalizes', async () => {
    const state = fromFen('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
    const legal = new Set(legalMoves(state));

    // Policy returns uniform priors over legal + some obviously illegal moves.
    const badPolicy: Policy = {
      async evaluate(s: GameState) {
        const moves = legalMoves(s);
        const priors = new Map<Move, number>();
        // Put half the mass on legal moves...
        for (const m of moves) priors.set(m, 1 / moves.length);
        // ...and some mass on illegal moves (should be discarded).
        priors.set('a1a9', 0.5);  // off-board
        priors.set('z9z9', 0.5);  // nonsense
        return { priors, value: 0 };
      },
    };

    const result = await search(state, badPolicy, baseConfig({ simulations: 50 }));

    // All keys in priorPolicy must be legal.
    for (const move of result.priorPolicy.keys()) {
      expect(legal.has(move)).toBe(true);
    }

    // Priors must sum to ~1.
    const sum = Array.from(result.priorPolicy.values()).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
});

// ---------------------------------------------------------------------------
// Visit counts sum to simulations
// ---------------------------------------------------------------------------

describe('mcts: visit count invariant', () => {
  it('sum of visitCounts equals config.simulations for non-terminal root', async () => {
    const state = fromFen('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');
    const N = 150;
    const result = await search(state, uniformPolicy, baseConfig({ simulations: N }));
    const total = Array.from(result.visitCounts.values()).reduce((a, b) => a + b, 0);
    expect(total).toBe(N);
  });
});

// ---------------------------------------------------------------------------
// Throws on terminal root
// ---------------------------------------------------------------------------

describe('mcts: terminal root', () => {
  it('throws when called on a checkmate position', async () => {
    // Scholar's mate — white just mated black (or use a canned terminal FEN).
    // Position: black king mated on h8, queenmate.
    // Actually construct a terminal position via the rules module.
    let s = fromFen('6k1/8/5K2/8/8/8/8/6Q1 w - - 0 1');
    s = applyMove(s, 'g1g7'); // white delivers checkmate
    expect(isTerminal(s)).toBe(true);

    await expect(search(s, uniformPolicy, baseConfig())).rejects.toThrow();
  });
});
