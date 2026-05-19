#!/usr/bin/env node
// Build-time filter for the Lichess puzzle CSV (CC0). Reads the
// decompressed CSV and writes a small JSON file with a curated sample
// of endgame puzzles per category. Output is ~80-200 KB and is checked
// into the repo; the 1 GB CSV is NOT.
//
// Usage:
//   curl -L https://database.lichess.org/lichess_db_puzzle.csv.zst -o puzzles.csv.zst
//   zstd -d puzzles.csv.zst
//   node scripts/build-lichess-endgames.mjs --input puzzles.csv
//
// Output defaults to nn-chess/ui/lichess-endgames.json.
//
// Each Lichess row's `FEN` is the position BEFORE the first move of
// `Moves`; the first move is played by the side that just blundered,
// and the remaining moves are the solver's win. So the position our UI
// actually starts from is FEN + apply(Moves[0]). The side to move at
// that position is the side the human should play.

import fs from 'node:fs';
import readline from 'node:readline';
import { Chess } from 'chess.js';

const CATEGORIES = [
  { id: 'mate-1-2',     label: 'Mate in 1-2',           themes: ['mateIn1', 'mateIn2'],               per: 60  },
  { id: 'mate-3-5',     label: 'Mate in 3-5',           themes: ['mateIn3', 'mateIn4', 'mateIn5'],     per: 100 },
  { id: 'rook-endgame', label: 'Rook endgame',          themes: ['rookEndgame'],                       per: 100 },
  { id: 'queen-endgame',label: 'Queen endgame',         themes: ['queenEndgame'],                      per: 60  },
  { id: 'pawn-endgame', label: 'Pawn endgame',          themes: ['pawnEndgame'],                       per: 100 },
  { id: 'minor-endgame',label: 'Bishop/knight endgame', themes: ['bishopEndgame', 'knightEndgame'],    per: 80  },
];

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) out[a.slice(2)] = argv[++i];
  }
  return out;
}

// Mulberry32 — small deterministic PRNG so the output JSON is
// reproducible across runs with the same --seed.
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Splits a CSV line that has no embedded quotes/commas-in-fields. The
// Lichess puzzle CSV has neither — themes are space-separated within
// one column — so a plain split is safe.
function splitCsv(line) {
  return line.split(',');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const input = args.input;
  const output = args.output ?? 'nn-chess/ui/lichess-endgames.json';
  const minRating = parseInt(args['min-rating'] ?? '1000', 10);
  const maxRating = parseInt(args['max-rating'] ?? '1800', 10);
  const seed = parseInt(args.seed ?? '1', 10);
  if (!input) {
    console.error('--input <decompressed CSV> is required');
    process.exit(1);
  }

  const rng = mulberry32(seed);
  const themeToCat = new Map();
  for (const c of CATEGORIES) for (const t of c.themes) themeToCat.set(t, c.id);
  const catById = new Map(CATEGORIES.map(c => [c.id, c]));

  const buf = new Map(CATEGORIES.map(c => [c.id, []]));
  const seenCount = new Map(CATEGORIES.map(c => [c.id, 0]));

  const stream = fs.createReadStream(input);
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  let header = null;
  let processed = 0;
  let kept = 0;

  for await (const line of rl) {
    if (header === null) { header = line; continue; }
    processed++;
    if (processed % 500000 === 0) console.error(`  read ${processed.toLocaleString()} rows…`);

    const cols = splitCsv(line);
    if (cols.length < 8) continue;
    const id = cols[0];
    const fen = cols[1];
    const moves = cols[2];
    const rating = parseInt(cols[3], 10);
    const themesStr = cols[7];
    if (!fen || !moves || isNaN(rating)) continue;
    if (rating < minRating || rating > maxRating) continue;

    const themes = themesStr.split(' ');
    let catId = null;
    for (const t of themes) {
      const c = themeToCat.get(t);
      if (c) { catId = c; break; }
    }
    if (!catId) continue;

    let puzzleFen;
    try {
      const ch = new Chess(fen);
      const first = moves.split(' ')[0];
      ch.move({
        from: first.slice(0, 2),
        to: first.slice(2, 4),
        promotion: first.length === 5 ? first[4] : undefined,
      });
      puzzleFen = ch.fen();
    } catch {
      continue;
    }

    const cat = catById.get(catId);
    const items = buf.get(catId);
    const seen = seenCount.get(catId) + 1;
    seenCount.set(catId, seen);

    const entry = { id, fen: puzzleFen, themes: themesStr, rating };
    if (items.length < cat.per) {
      items.push(entry);
    } else {
      const j = Math.floor(rng() * seen);
      if (j < cat.per) items[j] = entry;
    }
    kept++;
  }

  const result = {
    source: 'https://database.lichess.org/#puzzles',
    license: 'CC0-1.0',
    note: 'Curated sample of the Lichess puzzle database (CC0). Generated by scripts/build-lichess-endgames.mjs.',
    generatedAt: new Date().toISOString(),
    minRating,
    maxRating,
    seed,
    categories: CATEGORIES.map(c => ({
      id: c.id,
      label: c.label,
      themes: c.themes,
      puzzles: buf.get(c.id).sort((a, b) => a.rating - b.rating),
    })),
  };

  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  console.error(`processed ${processed.toLocaleString()} rows, considered ${kept.toLocaleString()} matches`);
  for (const c of result.categories) console.error(`  ${c.id.padEnd(14)} ${String(c.puzzles.length).padStart(4)}`);
  console.error(`wrote ${output}`);
}

main();
