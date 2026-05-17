import { describe, expect, it } from 'vitest';
import {
  applyMove,
  fromFen,
  inCheck,
  initialState,
  isTerminal,
  legalMoves,
  outcome,
  parseUci,
  pieceAt,
  repetitionKey,
  sideToMove,
  STARTING_FEN,
  terminalValueForMover,
  toFen,
} from './index.js';

describe('rules: starting position', () => {
  it('has the standard starting FEN', () => {
    expect(toFen(initialState())).toBe(STARTING_FEN);
  });

  it('white moves first', () => {
    expect(sideToMove(initialState())).toBe('w');
  });

  it('generates exactly 20 legal first moves', () => {
    expect(legalMoves(initialState())).toHaveLength(20);
  });

  it('is not terminal', () => {
    expect(isTerminal(initialState())).toBe(false);
    expect(outcome(initialState())).toBe('ongoing');
  });

  it('is not in check', () => {
    expect(inCheck(initialState())).toBe(false);
  });
});

describe('rules: applyMove', () => {
  it('does not mutate its input', () => {
    const s0 = initialState();
    const fen0 = toFen(s0);
    const s1 = applyMove(s0, 'e2e4');
    expect(toFen(s0)).toBe(fen0);
    expect(toFen(s1)).not.toBe(fen0);
    expect(sideToMove(s1)).toBe('b');
  });

  it('throws on illegal moves', () => {
    expect(() => applyMove(initialState(), 'e2e5')).toThrow();
    expect(() => applyMove(initialState(), 'a1a8')).toThrow();
  });

  it('throws on malformed UCI', () => {
    expect(() => applyMove(initialState(), 'xyz')).toThrow();
  });

  it('handles promotion', () => {
    // White pawn on e7 about to promote (black king out of the way on a8).
    const s = fromFen('k7/4P3/8/8/8/8/8/4K3 w - - 0 1');
    const promoted = applyMove(s, 'e7e8q');
    expect(pieceAt(promoted, 'e8')).toEqual({ color: 'w', type: 'q' });
  });

  it('handles castling encoded as king two-square move', () => {
    const s = fromFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
    const castled = applyMove(s, 'e1g1');
    expect(pieceAt(castled, 'g1')).toEqual({ color: 'w', type: 'k' });
    expect(pieceAt(castled, 'f1')).toEqual({ color: 'w', type: 'r' });
  });
});

describe('rules: terminal detection', () => {
  it('detects fool\'s mate (black wins)', () => {
    let s = initialState();
    s = applyMove(s, 'f2f3');
    s = applyMove(s, 'e7e5');
    s = applyMove(s, 'g2g4');
    s = applyMove(s, 'd8h4');
    expect(isTerminal(s)).toBe(true);
    expect(outcome(s)).toBe('black-wins');
    // Black (the mover) won.
    expect(terminalValueForMover(s)).toBe(1);
  });

  it('detects scholar\'s mate (white wins)', () => {
    let s = initialState();
    s = applyMove(s, 'e2e4');
    s = applyMove(s, 'e7e5');
    s = applyMove(s, 'd1h5');
    s = applyMove(s, 'b8c6');
    s = applyMove(s, 'f1c4');
    s = applyMove(s, 'g8f6');
    s = applyMove(s, 'h5f7');
    expect(isTerminal(s)).toBe(true);
    expect(outcome(s)).toBe('white-wins');
    expect(terminalValueForMover(s)).toBe(1);
  });

  it('detects stalemate as a draw', () => {
    // Classic K+Q vs K stalemate position.
    const s = fromFen('7k/8/6Q1/6K1/8/8/8/8 b - - 0 1');
    expect(isTerminal(s)).toBe(true);
    expect(outcome(s)).toBe('draw');
    expect(terminalValueForMover(s)).toBe(0);
  });

  it('detects insufficient material', () => {
    // King vs king.
    const s = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    expect(isTerminal(s)).toBe(true);
    expect(outcome(s)).toBe('draw');
  });

  it('terminalValueForMover throws on non-terminal state', () => {
    expect(() => terminalValueForMover(initialState())).toThrow();
  });
});

describe('rules: parseUci', () => {
  it('parses 4-char moves', () => {
    expect(parseUci('e2e4')).toEqual({ from: 'e2', to: 'e4', promotion: undefined });
  });
  it('parses 5-char promotion moves', () => {
    expect(parseUci('e7e8q')).toEqual({ from: 'e7', to: 'e8', promotion: 'q' });
  });
  it('rejects malformed moves', () => {
    expect(() => parseUci('e2')).toThrow();
    expect(() => parseUci('e2e4q5')).toThrow();
  });
});

describe('rules: repetitionKey', () => {
  it('ignores move clocks', () => {
    const a = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    const b = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 42');
    expect(repetitionKey(a)).toBe(repetitionKey(b));
  });

  it('differs when castling rights or side to move differ', () => {
    const a = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    const b = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1');
    const c = fromFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1');
    expect(repetitionKey(a)).not.toBe(repetitionKey(b));
    expect(repetitionKey(a)).not.toBe(repetitionKey(c));
  });
});
