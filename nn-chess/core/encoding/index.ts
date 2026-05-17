// Board ↔ tensor encoding. The agent's neural network sees boards
// through this module and only through this module.
//
// Design notes:
// - We encode raw game state (which squares hold which pieces, side to
//   move, castling rights, en-passant target). We do NOT encode any
//   evaluation features (material count, mobility, king safety, etc.)
//   — those would be the "heuristics" the project is explicitly trying
//   to avoid baking in.
// - We keep the tensor flat (Float32Array of length 8*8*PLANE_COUNT)
//   and let the NN module reshape into the layout its backend prefers.
//   This module is therefore framework-agnostic and trivially testable.
// - Plane order is documented and version-tagged via ENCODING_VERSION
//   so cached/saved weights can be invalidated when the encoding
//   changes.

import {
  pieceAt,
  sideToMove,
  toFen,
  type Color,
  type GameState,
  type PieceType,
  type Square,
} from '../rules/index.js';

export const ENCODING_VERSION = 1;

/**
 * Plane indices. Each plane is an 8×8 binary (or near-binary) grid.
 *
 *  0–5   : white piece presence — P, N, B, R, Q, K
 *  6–11  : black piece presence — p, n, b, r, q, k
 *  12    : side-to-move (all 1s if white, all 0s if black)
 *  13    : white can castle kingside
 *  14    : white can castle queenside
 *  15    : black can castle kingside
 *  16    : black can castle queenside
 *  17    : en-passant target square (1 on the target, 0 elsewhere)
 */
export const PLANE_INDEX = {
  WP: 0,
  WN: 1,
  WB: 2,
  WR: 3,
  WQ: 4,
  WK: 5,
  BP: 6,
  BN: 7,
  BB: 8,
  BR: 9,
  BQ: 10,
  BK: 11,
  SIDE_TO_MOVE: 12,
  W_OO: 13,
  W_OOO: 14,
  B_OO: 15,
  B_OOO: 16,
  EN_PASSANT: 17,
} as const;

export const PLANE_COUNT = 18;
export const BOARD_SIZE = 8;
export const FEATURE_COUNT = PLANE_COUNT * BOARD_SIZE * BOARD_SIZE;

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;

/**
 * Linear index into a flat [PLANE_COUNT, 8, 8] tensor laid out in
 * (plane, rank, file) order, with rank 0 = rank 1 of the board
 * (white's back rank).
 */
export function flatIndex(plane: number, rank: number, file: number): number {
  return plane * BOARD_SIZE * BOARD_SIZE + rank * BOARD_SIZE + file;
}

function pieceTypeToPlane(color: Color, type: PieceType): number {
  const offset = color === 'w' ? 0 : 6;
  switch (type) {
    case 'p': return offset + 0;
    case 'n': return offset + 1;
    case 'b': return offset + 2;
    case 'r': return offset + 3;
    case 'q': return offset + 4;
    case 'k': return offset + 5;
  }
}

function squareToCoord(sq: Square): { rank: number; file: number } {
  const file = sq.charCodeAt(0) - 'a'.charCodeAt(0);
  const rank = parseInt(sq[1], 10) - 1;
  return { rank, file };
}

/**
 * Encode a game state as a flat Float32Array of length FEATURE_COUNT.
 *
 * The returned array is a fresh allocation — callers may mutate it
 * (e.g. for batching) without worrying about aliasing.
 */
export function encodeState(state: GameState): Float32Array {
  const out = new Float32Array(FEATURE_COUNT);

  // Piece planes.
  for (let rank = 0; rank < BOARD_SIZE; rank++) {
    for (let file = 0; file < BOARD_SIZE; file++) {
      const sq = `${FILES[file]}${rank + 1}` as Square;
      const piece = pieceAt(state, sq);
      if (piece === null) continue;
      const plane = pieceTypeToPlane(piece.color, piece.type);
      out[flatIndex(plane, rank, file)] = 1;
    }
  }

  // Side-to-move plane.
  if (sideToMove(state) === 'w') {
    const base = flatIndex(PLANE_INDEX.SIDE_TO_MOVE, 0, 0);
    for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) out[base + i] = 1;
  }

  // Castling rights and en-passant come from the FEN.
  // FEN fields: piece-placement side castling ep halfmove fullmove
  const fen = toFen(state);
  const fields = fen.split(' ');
  const castling = fields[2];
  const ep = fields[3];

  const fillPlane = (planeIdx: number) => {
    const base = flatIndex(planeIdx, 0, 0);
    for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) out[base + i] = 1;
  };
  if (castling.includes('K')) fillPlane(PLANE_INDEX.W_OO);
  if (castling.includes('Q')) fillPlane(PLANE_INDEX.W_OOO);
  if (castling.includes('k')) fillPlane(PLANE_INDEX.B_OO);
  if (castling.includes('q')) fillPlane(PLANE_INDEX.B_OOO);

  if (ep !== '-' && ep.length === 2) {
    const { rank, file } = squareToCoord(ep as Square);
    out[flatIndex(PLANE_INDEX.EN_PASSANT, rank, file)] = 1;
  }

  return out;
}

/**
 * Read a single value out of an encoded tensor. Mostly a convenience
 * for tests — production code can index `flatIndex` directly.
 */
export function readPlane(
  features: Float32Array,
  plane: number,
  rank: number,
  file: number,
): number {
  return features[flatIndex(plane, rank, file)];
}
