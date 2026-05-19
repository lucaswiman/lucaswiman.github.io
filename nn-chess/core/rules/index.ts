// Thin wrapper around `chess.js` exposing only the subset of the rules
// API the agent + UI + encoder need. Keeping this layer tight means we
// can swap chess.js out later (e.g. for a wasm engine for speed) without
// touching MCTS, the NN, or the UI.
//
// `GameState` is treated as immutable from the outside: `applyMove`
// returns a new state and never mutates its input.

import { Chess } from 'chess.js';

export type Color = 'w' | 'b';
export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
export type Square =
  | `${'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h'}${
      | '1'
      | '2'
      | '3'
      | '4'
      | '5'
      | '6'
      | '7'
      | '8'}`;

export interface Piece {
  color: Color;
  type: PieceType;
}

/**
 * A move encoded in long-algebraic / UCI form: "e2e4", "e7e8q", "e1g1"
 * (castle), etc. This is the canonical move representation everywhere
 * outside this module — chess.js's richer move objects do not leak.
 */
export type Move = string;

export type Outcome =
  | 'white-wins'
  | 'black-wins'
  | 'draw'
  | 'ongoing';

export interface GameState {
  // The wrapped engine. Treated as opaque by callers — they go through
  // the exported helpers below. Held by reference (not cloned on every
  // read) for performance.
  readonly _chess: Chess;
}

export const STARTING_FEN =
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

export function initialState(): GameState {
  return { _chess: new Chess() };
}

export function fromFen(fen: string): GameState {
  return { _chess: new Chess(fen) };
}

export function toFen(state: GameState): string {
  return state._chess.fen();
}

export function sideToMove(state: GameState): Color {
  return state._chess.turn();
}

export function pieceAt(state: GameState, square: Square): Piece | null {
  const p = state._chess.get(square);
  return p ? { color: p.color as Color, type: p.type as PieceType } : null;
}

/** Returns true iff the side to move is currently in check. */
export function inCheck(state: GameState): boolean {
  return state._chess.inCheck();
}

/**
 * All legal moves in the current position, as UCI strings. Order is
 * deterministic (whatever chess.js returns, which is itself
 * deterministic for a given FEN).
 */
export function legalMoves(state: GameState): Move[] {
  return state._chess.moves({ verbose: true }).map(moveToUci);
}

/**
 * Apply a move to `state`, returning a new state. Throws if `move` is
 * not legal. Never mutates the input.
 */
export function applyMove(state: GameState, move: Move): GameState {
  const next = new Chess(state._chess.fen());
  const parsed = parseUci(move);
  // chess.js's .move() throws on illegal moves in v1.x, which is what
  // we want.
  next.move(parsed);
  return { _chess: next };
}

export function isTerminal(state: GameState): boolean {
  return state._chess.isGameOver();
}

export function outcome(state: GameState): Outcome {
  const c = state._chess;
  if (!c.isGameOver()) return 'ongoing';
  if (c.isCheckmate()) {
    // The side whose turn it is has been checkmated.
    return c.turn() === 'w' ? 'black-wins' : 'white-wins';
  }
  return 'draw';
}

/**
 * The terminal value of a state, from the perspective of the side to
 * move BEFORE the terminal position was reached — i.e. the side that
 * just moved. Convention:
 *   +1 if the just-moved side won (delivered checkmate)
 *   -1 if the just-moved side lost (got checkmated — impossible since
 *       you can't move into checkmate of yourself, but kept for
 *       completeness)
 *    0 for any draw
 *
 * Throws if the position is not terminal.
 */
export function terminalValueForMover(state: GameState): number {
  if (!isTerminal(state)) {
    throw new Error('terminalValueForMover called on non-terminal state');
  }
  const o = outcome(state);
  if (o === 'draw') return 0;
  // After the move, sideToMove(state) is the OTHER side. If that side
  // is checkmated, the side that just moved won.
  const mover: Color = sideToMove(state) === 'w' ? 'b' : 'w';
  if (o === 'white-wins') return mover === 'w' ? 1 : -1;
  return mover === 'b' ? 1 : -1;
}

/**
 * A short, position-only repetition key (FEN without the move clocks).
 * Two positions with the same key are equivalent for threefold-
 * repetition purposes and — more usefully here — for caching NN
 * evaluations.
 */
export function repetitionKey(state: GameState): string {
  // FEN fields: piece-placement side castling ep halfmove fullmove
  // Drop the halfmove and fullmove counters.
  return toFen(state).split(' ').slice(0, 4).join(' ');
}

// --- internal helpers ---------------------------------------------------

interface VerboseMove {
  from: string;
  to: string;
  promotion?: string;
}

function moveToUci(m: VerboseMove): Move {
  return `${m.from}${m.to}${m.promotion ?? ''}`;
}

interface ParsedMove {
  from: string;
  to: string;
  promotion?: string;
}

export function parseUci(uci: Move): ParsedMove {
  if (uci.length !== 4 && uci.length !== 5) {
    throw new Error(`Invalid UCI move: ${uci}`);
  }
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promotion = uci.length === 5 ? uci[4] : undefined;
  return { from, to, promotion };
}
