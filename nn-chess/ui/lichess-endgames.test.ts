import { describe, it, expect } from 'vitest';
import { fromFen, outcome, legalMoves } from '../core/rules/index.js';
import { LICHESS_DATA, LICHESS_CATEGORIES } from './lichess-endgames.js';

describe('lichess endgame puzzles', () => {
  it('metadata is well-formed', () => {
    expect(LICHESS_DATA.license).toBe('CC0-1.0');
    expect(LICHESS_DATA.source).toContain('lichess.org');
    expect(LICHESS_DATA.categories.length).toBeGreaterThan(0);
  });

  it('every puzzle FEN is parseable and non-terminal', () => {
    // Spot-check one puzzle per category to keep the test cheap; the
    // build script can produce hundreds of puzzles total and we don't
    // want the suite to balloon.
    for (const cat of LICHESS_CATEGORIES) {
      expect(cat.puzzles.length, `${cat.id} should have puzzles`).toBeGreaterThan(0);
      const p = cat.puzzles[0];
      const state = fromFen(p.fen);
      expect(outcome(state), `${cat.id}/${p.id} should be ongoing`).toBe('ongoing');
      expect(legalMoves(state).length).toBeGreaterThan(0);
    }
  });

  it('category labels and themes are populated', () => {
    for (const cat of LICHESS_CATEGORIES) {
      expect(cat.id).toMatch(/^[a-z0-9-]+$/);
      expect(cat.label.length).toBeGreaterThan(0);
      expect(cat.themes.length).toBeGreaterThan(0);
    }
  });
});
