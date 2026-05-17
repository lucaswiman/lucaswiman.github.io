import { describe, expect, it } from 'vitest';
import {
  applyMove,
  fromFen,
  initialState,
} from '../rules/index.js';
import {
  BOARD_SIZE,
  FEATURE_COUNT,
  PLANE_COUNT,
  PLANE_INDEX,
  encodeState,
  flatIndex,
  readPlane,
} from './index.js';

describe('encoding: shape', () => {
  it('produces FEATURE_COUNT values', () => {
    expect(encodeState(initialState())).toHaveLength(FEATURE_COUNT);
    expect(FEATURE_COUNT).toBe(PLANE_COUNT * BOARD_SIZE * BOARD_SIZE);
  });

  it('returns a fresh Float32Array each call (no aliasing)', () => {
    const s = initialState();
    const a = encodeState(s);
    const b = encodeState(s);
    expect(a).not.toBe(b);
    a[0] = 42;
    expect(b[0]).not.toBe(42);
  });
});

describe('encoding: piece planes', () => {
  it('places white pawns on rank 2 and black pawns on rank 7', () => {
    const t = encodeState(initialState());
    for (let file = 0; file < 8; file++) {
      expect(readPlane(t, PLANE_INDEX.WP, 1, file)).toBe(1);
      expect(readPlane(t, PLANE_INDEX.BP, 6, file)).toBe(1);
    }
  });

  it('places kings on e1 and e8', () => {
    const t = encodeState(initialState());
    // e-file = file index 4
    expect(readPlane(t, PLANE_INDEX.WK, 0, 4)).toBe(1);
    expect(readPlane(t, PLANE_INDEX.BK, 7, 4)).toBe(1);
  });

  it('total piece-plane mass is 32 in the starting position', () => {
    const t = encodeState(initialState());
    let total = 0;
    for (let plane = 0; plane < 12; plane++) {
      for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
          total += readPlane(t, plane, r, f);
        }
      }
    }
    expect(total).toBe(32);
  });

  it('exactly one piece type occupies any given square (piece planes are disjoint)', () => {
    const t = encodeState(initialState());
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        let count = 0;
        for (let plane = 0; plane < 12; plane++) {
          count += readPlane(t, plane, r, f);
        }
        expect(count).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('encoding: side-to-move plane', () => {
  it('is all 1s when white moves', () => {
    const t = encodeState(initialState());
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        expect(readPlane(t, PLANE_INDEX.SIDE_TO_MOVE, r, f)).toBe(1);
      }
    }
  });

  it('is all 0s when black moves', () => {
    const t = encodeState(applyMove(initialState(), 'e2e4'));
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        expect(readPlane(t, PLANE_INDEX.SIDE_TO_MOVE, r, f)).toBe(0);
      }
    }
  });
});

describe('encoding: castling planes', () => {
  it('all four are set at the start', () => {
    const t = encodeState(initialState());
    for (const plane of [
      PLANE_INDEX.W_OO,
      PLANE_INDEX.W_OOO,
      PLANE_INDEX.B_OO,
      PLANE_INDEX.B_OOO,
    ]) {
      expect(readPlane(t, plane, 0, 0)).toBe(1);
      expect(readPlane(t, plane, 7, 7)).toBe(1);
    }
  });

  it('clears the right plane after castling', () => {
    const s = fromFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
    const t = encodeState(applyMove(s, 'e1g1'));
    // White lost both castling rights after moving the king.
    expect(readPlane(t, PLANE_INDEX.W_OO, 0, 0)).toBe(0);
    expect(readPlane(t, PLANE_INDEX.W_OOO, 0, 0)).toBe(0);
    // Black still has both.
    expect(readPlane(t, PLANE_INDEX.B_OO, 0, 0)).toBe(1);
    expect(readPlane(t, PLANE_INDEX.B_OOO, 0, 0)).toBe(1);
  });
});

describe('encoding: en-passant plane', () => {
  it('is empty when no ep target', () => {
    const t = encodeState(initialState());
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        expect(readPlane(t, PLANE_INDEX.EN_PASSANT, r, f)).toBe(0);
      }
    }
  });

  it('marks the ep target after a double pawn push when capture is possible', () => {
    // chess.js (correctly, per current FEN conventions) only records an
    // en-passant target square when an opposing pawn could actually
    // capture there. Set up a black pawn on d4 so white's e2-e4 creates
    // a real ep opportunity.
    const setup = fromFen('4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1');
    const t = encodeState(applyMove(setup, 'e2e4'));
    // ep target is e3 — rank index 2, file index 4
    expect(readPlane(t, PLANE_INDEX.EN_PASSANT, 2, 4)).toBe(1);
    // Exactly one square marked.
    let total = 0;
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        total += readPlane(t, PLANE_INDEX.EN_PASSANT, r, f);
      }
    }
    expect(total).toBe(1);
  });
});

describe('encoding: flatIndex bounds', () => {
  it('round-trips through (plane, rank, file)', () => {
    expect(flatIndex(0, 0, 0)).toBe(0);
    expect(flatIndex(0, 0, 7)).toBe(7);
    expect(flatIndex(0, 1, 0)).toBe(8);
    expect(flatIndex(1, 0, 0)).toBe(64);
    expect(flatIndex(PLANE_COUNT - 1, 7, 7)).toBe(FEATURE_COUNT - 1);
  });
});
