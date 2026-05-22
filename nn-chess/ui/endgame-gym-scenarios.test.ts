import { describe, expect, it } from 'vitest';
import { isTerminal, sideToMove } from '../core/rules/index.js';
import { GYM_CATEGORIES, GYM_CATEGORIES_BY_ID } from './endgame-gym-scenarios.js';

describe('endgame gym scenarios', () => {
  it('every scenario is non-terminal and starts on the AI\'s move', () => {
    // The gym hands `playOneGame` `startingState` and an `opponentColor`.
    // If the opponent doesn't own the move at ply 0, the imitator gets
    // the first (winning) move — silently inverting the whole gym.
    // Guard statically.
    for (const cat of GYM_CATEGORIES) {
      for (const s of cat.scenarios) {
        expect(isTerminal(s.startingState), `${s.id} should not start terminal`).toBe(false);
        expect(sideToMove(s.startingState), `${s.id} side-to-move should match opponentColor`).toBe(s.opponentColor);
      }
    }
  });

  it('scenario IDs are unique within each category', () => {
    for (const cat of GYM_CATEGORIES) {
      const ids = cat.scenarios.map(s => s.id);
      expect(new Set(ids).size, `category ${cat.id}: duplicate scenario id`).toBe(ids.length);
    }
  });

  it('the rolled-up "all-decided" category equals the union of the other categories', () => {
    const all = GYM_CATEGORIES_BY_ID.get('all-decided');
    expect(all).toBeDefined();
    const allIds = new Set(all!.scenarios.map(s => s.id));
    let union = 0;
    for (const cat of GYM_CATEGORIES) {
      if (cat.id === 'all-decided') continue;
      union += cat.scenarios.length;
      for (const s of cat.scenarios) {
        expect(allIds.has(s.id)).toBe(true);
      }
    }
    expect(all!.scenarios.length).toBe(union);
  });

  it('exposes both textbook presets and at least one Lichess mate category', () => {
    expect(GYM_CATEGORIES_BY_ID.get('presets')).toBeDefined();
    expect(GYM_CATEGORIES_BY_ID.get('lichess:mate-1-2')).toBeDefined();
  });
});
