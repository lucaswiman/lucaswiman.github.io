import rawData from './lichess-endgames.json' with { type: 'json' };

export interface LichessPuzzle {
  id: string;
  fen: string;
  themes: string;
  rating: number;
}

export interface LichessCategory {
  id: string;
  label: string;
  themes: string[];
  puzzles: LichessPuzzle[];
}

interface LichessData {
  source: string;
  license: string;
  generatedAt: string;
  categories: LichessCategory[];
}

export const LICHESS_DATA: LichessData = rawData as LichessData;

export const LICHESS_CATEGORIES: readonly LichessCategory[] = LICHESS_DATA.categories.filter(
  c => c.puzzles.length > 0,
);

export const LICHESS_CATEGORIES_BY_ID: ReadonlyMap<string, LichessCategory> = new Map(
  LICHESS_CATEGORIES.map(c => [c.id, c] as const),
);

export function randomPuzzle(category: LichessCategory): LichessPuzzle {
  return category.puzzles[Math.floor(Math.random() * category.puzzles.length)];
}

export function lichessPuzzleUrl(id: string): string {
  return `https://lichess.org/training/${id}`;
}
