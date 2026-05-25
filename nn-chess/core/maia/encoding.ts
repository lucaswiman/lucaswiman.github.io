// Maia-2 board encoding. NOT to be confused with `core/encoding/index.ts`
// (the project's own 18-channel encoder, which has a different plane
// order). The two encoders are kept separate on purpose; Maia-2 was
// trained with its own encoding and we must match it bit-for-bit to
// reproduce the published inference.
//
// Maia-2 layout (from maia2/utils.py:board_to_tensor):
//   18 channels × 8 rows × 8 cols, row 0 = rank 1 (white's back rank).
//   Channels 0..11   piece presence (P, N, B, R, Q, K) for white then black.
//   Channel 12       player-to-move: 1.0 if white to move, else 0.0.
//   Channels 13..16  castling rights: WK, WQ, BK, BQ (kingside/queenside).
//   Channel 17       en-passant target square (1 on the target, 0 else).
//
// Mirroring: Maia-2 was trained on positions where the side to move
// always plays as white. Black-to-move boards are mirrored vertically
// (chess.Board(fen).mirror()) before being tensorized, and the move
// list is mirrored back at the output. That's the same trick the
// reference uses and we mirror here.

import { Chess } from 'chess.js';
import type { GameState } from '../rules/index.js';
import { toFen } from '../rules/index.js';

/** Number of input channels Maia-2's CNN backbone consumes. */
export const MAIA_INPUT_CHANNELS = 18;
/** Square edge length (8). */
export const BOARD_SIZE = 8;
/** Flat tensor size: 18 * 8 * 8 = 1152. */
export const MAIA_FEATURE_COUNT = MAIA_INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;

/**
 * The 11 ELO buckets Maia-2 was trained against. Index `i` is the
 * bucket ID, the string is the human-readable label.
 *
 * Generated from `maia2.utils.create_elo_dict`:
 *   "<1100", "1100-1199", "1200-1299", ..., "1900-1999", ">=2000".
 */
export const MAIA_ELO_BUCKETS: ReadonlyArray<{ id: number; label: string; midpoint: number }> = [
  { id: 0,  label: '<1100',     midpoint: 1050 },
  { id: 1,  label: '1100–1199', midpoint: 1150 },
  { id: 2,  label: '1200–1299', midpoint: 1250 },
  { id: 3,  label: '1300–1399', midpoint: 1350 },
  { id: 4,  label: '1400–1499', midpoint: 1450 },
  { id: 5,  label: '1500–1599', midpoint: 1550 },
  { id: 6,  label: '1600–1699', midpoint: 1650 },
  { id: 7,  label: '1700–1799', midpoint: 1750 },
  { id: 8,  label: '1800–1899', midpoint: 1850 },
  { id: 9,  label: '1900–1999', midpoint: 1950 },
  { id: 10, label: '≥2000',     midpoint: 2050 },
];

/**
 * Map a raw ELO integer to the [0, 10] bucket Maia-2 was trained against.
 * Replicates `maia2.utils.map_to_category`.
 */
export function eloToBucket(elo: number): number {
  if (elo < 1100) return 0;
  if (elo >= 2000) return 10;
  // 1100 → 1, 1200 → 2, …, 1900 → 9
  return Math.floor((elo - 1100) / 100) + 1;
}

/**
 * Encode a game state as Maia-2's 18-channel tensor.
 *
 * Returns the canonical (always-from-white-to-move) tensor along with
 * a `mirrored` flag.  If the side to move in `state` is black, the
 * board is vertically mirrored before encoding and `mirrored` is set
 * to `true`; callers that want to read move probabilities out of the
 * policy head must then mirror move strings back via `mirrorUci`.
 *
 * Layout is flat `[channel, row, col]` with the same indexing as
 * `core/encoding/flatIndex` but with Maia-2's plane order.
 */
export function encodeForMaia(state: GameState): { features: Float32Array; mirrored: boolean } {
  const fen = toFen(state);
  const sideToMove = fen.split(' ')[1];
  const mirrored = sideToMove === 'b';
  // chess.js: load FEN, optionally mirror so it's always white-to-move.
  let chess = new Chess(fen);
  if (mirrored) {
    chess = mirrorChess(chess);
  }
  return { features: tensorFromChess(chess), mirrored };
}

// ── Internals ────────────────────────────────────────────────────────────────

/**
 * Mirror a chess.js board the way python-chess's `Board.mirror()` does:
 * swap colors of all pieces, mirror them vertically, swap castling
 * rights, and flip the side-to-move. en-passant target gets mirrored
 * vertically too.
 *
 * We re-build the FEN by hand because chess.js doesn't expose a built-
 * in `mirror()`.
 */
function mirrorChess(chess: Chess): Chess {
  const fen = chess.fen();
  const [placement, side, castling, ep, half, full] = fen.split(' ');
  // Mirror placement: reverse ranks, swap piece case (white↔black).
  const ranks = placement.split('/');
  const mirroredRanks = [...ranks].reverse().map(rank =>
    rank.split('').map(c => {
      if (c >= 'A' && c <= 'Z') return c.toLowerCase();
      if (c >= 'a' && c <= 'z') return c.toUpperCase();
      return c;
    }).join(''),
  );
  const newPlacement = mirroredRanks.join('/');
  const newSide = side === 'w' ? 'b' : 'w';
  // Castling: swap case.
  const newCastling = castling === '-' ? '-' : castling.split('').map(c => {
    if (c === 'K') return 'k';
    if (c === 'Q') return 'q';
    if (c === 'k') return 'K';
    if (c === 'q') return 'Q';
    return c;
  }).join('');
  // en-passant: mirror rank.
  let newEp = ep;
  if (ep !== '-' && ep.length === 2) {
    const file = ep[0];
    const rank = String(9 - parseInt(ep[1], 10));
    newEp = `${file}${rank}`;
  }
  const newFen = `${newPlacement} ${newSide} ${newCastling} ${newEp} ${half} ${full}`;
  return new Chess(newFen);
}

/**
 * Replicate `maia2.utils.board_to_tensor` exactly.  Channel order and
 * row indexing must match maia2's `divmod(square, 8)` (row first), and
 * chess.js square notation maps `a1=0, b1=1, …, h1=7, a2=8, …`.
 */
function tensorFromChess(chess: Chess): Float32Array {
  const out = new Float32Array(MAIA_FEATURE_COUNT);
  const board = chess.board();
  // chess.js board() returns rank-from-8 down to rank-1, with index 0 = rank 8.
  // We need the python-chess convention: row 0 = rank 1.
  const TYPE_TO_IDX: Record<string, number> = {
    p: 0, n: 1, b: 2, r: 3, q: 4, k: 5,
  };
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const piece = board[7 - r][f]; // rank 1 → board[7]
      if (piece === null) continue;
      const baseIdx = TYPE_TO_IDX[piece.type];
      const channel = baseIdx + (piece.color === 'w' ? 0 : 6);
      out[channel * 64 + r * 8 + f] = 1;
    }
  }
  // Turn channel.
  if (chess.turn() === 'w') {
    for (let i = 0; i < 64; i++) out[12 * 64 + i] = 1;
  }
  // Castling rights.
  const fen = chess.fen();
  const castling = fen.split(' ')[2];
  if (castling.includes('K')) { for (let i = 0; i < 64; i++) out[13 * 64 + i] = 1; }
  if (castling.includes('Q')) { for (let i = 0; i < 64; i++) out[14 * 64 + i] = 1; }
  if (castling.includes('k')) { for (let i = 0; i < 64; i++) out[15 * 64 + i] = 1; }
  if (castling.includes('q')) { for (let i = 0; i < 64; i++) out[16 * 64 + i] = 1; }
  // en-passant.
  const ep = fen.split(' ')[3];
  if (ep !== '-' && ep.length === 2) {
    const file = ep.charCodeAt(0) - 'a'.charCodeAt(0);
    const rank = parseInt(ep[1], 10) - 1;
    out[17 * 64 + rank * 8 + file] = 1;
  }
  return out;
}

/**
 * Mirror a UCI move string vertically (rank r ↔ rank 9-r).  Matches
 * `maia2.utils.mirror_move`. Promotion suffix is left untouched.
 */
export function mirrorUci(uci: string): string {
  if (uci.length !== 4 && uci.length !== 5) {
    throw new Error(`mirrorUci: invalid UCI move ${uci}`);
  }
  const mirrorSquare = (sq: string): string => {
    const file = sq[0];
    const rank = String(9 - parseInt(sq[1], 10));
    return `${file}${rank}`;
  };
  const start = mirrorSquare(uci.slice(0, 2));
  const end = mirrorSquare(uci.slice(2, 4));
  const promo = uci.length === 5 ? uci[4] : '';
  return `${start}${end}${promo}`;
}
