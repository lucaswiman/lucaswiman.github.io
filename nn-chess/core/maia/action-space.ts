// Maia-2's 1880-entry move action space.
//
// We re-build the exact list `maia2.utils.get_all_possible_moves()`
// emits: for every square, all legal moves of a lone queen and lone
// knight on that square (white-side, empty board), plus white-side
// promotion moves (e7e8 / e7d8 / e7f8 with q/r/b/n) covering all files.
// 1858 queen+knight moves + 22*4 promotion entries × correct shape
// = 1880 entries.  Order matters: we must match the Python list 1:1
// so the policy-head logits land in the right slot.
//
// The generator is exported as a module-level constant so the cost is
// paid once. There's a parity test in `action-space.test.ts` that
// compares this to the dumped fixture.

// ── Square helpers (file 0-7 = a-h, rank 0-7 = 1-8) ─────────────────────────

function uci(fromFile: number, fromRank: number, toFile: number, toRank: number): string {
  return (
    String.fromCharCode(97 + fromFile) +
    String.fromCharCode(49 + fromRank) +
    String.fromCharCode(97 + toFile) +
    String.fromCharCode(49 + toRank)
  );
}

function inBounds(file: number, rank: number): boolean {
  return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

// Replicates python-chess's deterministic move enumeration for a lone
// queen on a square, then a lone knight on the same square. The
// ordering matters; the reference iterates over `board.legal_moves`
// which is a square-by-square scan emitting moves in
// (from_square_ascending, to_square_ascending) order.
function queenMovesFrom(file: number, rank: number): string[] {
  const out: string[] = [];
  // The 8 queen directions, swept outward.
  const dirs: Array<[number, number]> = [
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1],          [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1],
  ];
  // python-chess emits moves in `scan_reversed(attacks)` order, which
  // is destination-square-index DESCENDING.
  const moves: Array<{ toFile: number; toRank: number }> = [];
  for (const [df, dr] of dirs) {
    let f = file + df;
    let r = rank + dr;
    while (inBounds(f, r)) {
      moves.push({ toFile: f, toRank: r });
      f += df;
      r += dr;
    }
  }
  // Sort by destination square = toRank * 8 + toFile.
  // python-chess emits moves in `scan_reversed(attacks)` order, which
  // is destination-square-index DESCENDING. We sort accordingly.
  moves.sort((a, b) => (b.toRank * 8 + b.toFile) - (a.toRank * 8 + a.toFile));
  for (const m of moves) out.push(uci(file, rank, m.toFile, m.toRank));
  return out;
}

function knightMovesFrom(file: number, rank: number): string[] {
  const deltas: Array<[number, number]> = [
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [ 1, -2], [ 1, 2], [ 2, -1], [ 2, 1],
  ];
  const moves: Array<{ toFile: number; toRank: number }> = [];
  for (const [df, dr] of deltas) {
    const f = file + df;
    const r = rank + dr;
    if (inBounds(f, r)) moves.push({ toFile: f, toRank: r });
  }
  // python-chess emits moves in `scan_reversed(attacks)` order, which
  // is destination-square-index DESCENDING. We sort accordingly.
  moves.sort((a, b) => (b.toRank * 8 + b.toFile) - (a.toRank * 8 + a.toFile));
  return moves.map(m => uci(file, rank, m.toFile, m.toRank));
}

function buildAllMoves(): string[] {
  const out: string[] = [];
  // Reference iterates squares as `for rank in range(8): for file in range(8)`,
  // with `square = chess.square(file, rank)` = rank * 8 + file.
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      out.push(...queenMovesFrom(file, rank));
      out.push(...knightMovesFrom(file, rank));
    }
  }
  // White-side pawn promotions: starting on rank 7 (index 6), targets
  // on rank 8 (index 7). For each file: forward, capture-left,
  // capture-right (when on board), times {q, r, b, n}.
  const promos = ['q', 'r', 'b', 'n'];
  for (let f = 0; f < 8; f++) {
    const startFile = String.fromCharCode(97 + f);
    for (const p of promos) {
      out.push(`${startFile}7${startFile}8${p}`);
    }
    if (f > 0) {
      const leftFile = String.fromCharCode(97 + f - 1);
      for (const p of promos) {
        out.push(`${startFile}7${leftFile}8${p}`);
      }
    }
    if (f < 7) {
      const rightFile = String.fromCharCode(97 + f + 1);
      for (const p of promos) {
        out.push(`${startFile}7${rightFile}8${p}`);
      }
    }
  }
  return out;
}

/** All 1880 UCI moves in the order Maia-2 was trained on. */
export const MAIA_MOVES: ReadonlyArray<string> = buildAllMoves();

/** Size of the policy head's output (and of `MAIA_MOVES`). */
export const MAIA_POLICY_SIZE = MAIA_MOVES.length;

/**
 * UCI → policy index lookup. Built lazily from `MAIA_MOVES`.
 * Returns -1 if the move isn't in Maia-2's action space (shouldn't
 * happen for legal moves; underpromotions and white-side promotions
 * are all covered).
 */
const _moveIndex = new Map<string, number>();
for (let i = 0; i < MAIA_MOVES.length; i++) _moveIndex.set(MAIA_MOVES[i], i);

export function maiaPolicyIndex(uciMove: string): number {
  const idx = _moveIndex.get(uciMove);
  return idx === undefined ? -1 : idx;
}
