import { describe, it, expect } from 'vitest';
import { fromFen, sideToMove, legalMoves, outcome } from '../core/rules/index.js';
import { ENDGAME_PRESETS, ENDGAME_PRESETS_BY_ID } from './endgame-presets.js';

describe('endgame presets', () => {
  it('all FENs parse and are non-terminal', () => {
    for (const preset of ENDGAME_PRESETS) {
      const state = fromFen(preset.fen);
      expect(outcome(state), `${preset.id} should be ongoing`).toBe('ongoing');
      expect(legalMoves(state).length, `${preset.id} should have legal moves`).toBeGreaterThan(0);
    }
  });

  it('side-to-move matches the winning side', () => {
    // For the curated presets the winning side moves first — otherwise the
    // human (playing the winning side) would have to wait for an agent move
    // from a position where the agent has no idea what to do.
    for (const preset of ENDGAME_PRESETS) {
      const state = fromFen(preset.fen);
      expect(sideToMove(state), `${preset.id}: winning side should move first`).toBe(
        preset.winningSide,
      );
    }
  });

  it('ids are unique and the lookup map covers them', () => {
    const ids = ENDGAME_PRESETS.map(p => p.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const preset of ENDGAME_PRESETS) {
      expect(ENDGAME_PRESETS_BY_ID.get(preset.id)).toBe(preset);
    }
  });
});
