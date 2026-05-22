// Builds the pool of scenarios the endgame gym grinds through.
//
// "Scenario" here means a starting FEN together with the side that owns
// the forced win — the side the opponent is assigned to play. We pull from
// two sources: the small textbook preset set (KQ-K, KR-K, etc.) and
// the Lichess CC0 mate-in-N puzzle categories.
//
// Why these two sources?
//   - Textbook presets give the gym a guaranteed, short, won endgame
//     for each category. They're the warm-up.
//   - Lichess mate puzzles are decided by construction: the side to
//     move has a forced mate in 1–5. Mating themes are the only
//     Lichess category where "opponent on the winning side" is well-defined,
//     so the gym ignores the other categories (rook/queen/pawn/minor
//     endgames — those have a winner-by-rating-or-by-engine, not a
//     forced one).
//
// Long-running endgame categories like KQ-K are easy to recognise but
// hard for a cold-start net to win inside our ply cap; we keep them in
// anyway because the maxGamesPerScenario knob caps the wasted compute.

import { fromFen, sideToMove } from '../core/rules/index.js';
import type { Color } from '../core/rules/index.js';
import type { EndgameGymScenario } from '../core/selfplay/index.js';
import { ENDGAME_PRESETS } from './endgame-presets.js';
import { LICHESS_CATEGORIES_BY_ID } from './lichess-endgames.js';

/** Top-level grouping the user picks from in the UI. */
export interface GymCategory {
  /** Stable id used by the UI dropdown. */
  id: string;
  label: string;
  /** Short blurb for the user. */
  description: string;
  /** Resolved scenarios in this category. */
  scenarios: EndgameGymScenario[];
}

/**
 * Source IDs of the Lichess categories the gym should pull from.
 *
 * Only mate-in-N categories qualify: every puzzle in those has the
 * winning side to move, so we can hand the opponent the FEN's side-to-move
 * and trust that it's the side with the forced win. Other categories
 * (rookEndgame, queenEndgame, pawnEndgame, minorEndgame) describe
 * positions where the side to move may be winning, losing, or drawn —
 * not a clean "AI on the winning side" assignment.
 */
const MATE_LICHESS_IDS = ['mate-1-2', 'mate-3-5'] as const;

function presetScenarios(): EndgameGymScenario[] {
  return ENDGAME_PRESETS.map(p => ({
    id: `preset:${p.id}`,
    startingState: fromFen(p.fen),
    opponentColor: p.winningSide,
  }));
}

function lichessMateScenarios(catId: string): EndgameGymScenario[] {
  const cat = LICHESS_CATEGORIES_BY_ID.get(catId);
  if (!cat) return [];
  return cat.puzzles.map(p => {
    const startingState = fromFen(p.fen);
    return {
      id: `lichess:${catId}:${p.id}`,
      startingState,
      opponentColor: sideToMove(startingState) as Color,
    };
  });
}

function buildCategories(): GymCategory[] {
  const presets = presetScenarios();
  const lichessCats: GymCategory[] = MATE_LICHESS_IDS.flatMap(id => {
    const cat = LICHESS_CATEGORIES_BY_ID.get(id);
    if (!cat) return [];
    const scenarios = lichessMateScenarios(id);
    return [{
      id: `lichess:${id}`,
      label: cat.label,
      description: `${scenarios.length} Lichess CC0 puzzles. Opponent plays the side with the forced mate.`,
      scenarios,
    }];
  });
  // "Everything decided" rolls up all the above so the user can grind
  // through the full pool in one click. We materialize it once at module
  // load so the dropdown can show its size; the cost is tiny (one
  // chess.js Chess() per scenario).
  const everything: EndgameGymScenario[] = [...presets];
  for (const c of lichessCats) everything.push(...c.scenarios);

  return [
    {
      id: 'presets',
      label: `Endgame textbook presets (${presets.length})`,
      description:
        'Hand-picked short forced wins (ladder mate, K+Q vs K, K+R vs K, ' +
        'won K+P vs K). The gym\'s warm-up — most cold-start nets crack ' +
        'these within a handful of attempts.',
      scenarios: presets,
    },
    ...lichessCats,
    {
      id: 'all-decided',
      label: `All decided endgames (${everything.length})`,
      description:
        'Textbook presets followed by every Lichess mate-in-N puzzle the ' +
        'app ships with. Long-running; set a low maxGamesPerScenario.',
      scenarios: everything,
    },
  ];
}

export const GYM_CATEGORIES: readonly GymCategory[] = buildCategories();
export const GYM_CATEGORIES_BY_ID: ReadonlyMap<string, GymCategory> = new Map(
  GYM_CATEGORIES.map(c => [c.id, c] as const),
);
