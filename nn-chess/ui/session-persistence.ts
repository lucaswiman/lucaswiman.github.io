// Persists the in-progress user-vs-opponent game across page reloads.
//
// The persisted blob is intentionally tiny: a starting FEN plus the
// list of UCI moves played from there, alongside the UI's setup/color/
// puzzle metadata. On restore we replay the moves through the rules
// module to rebuild a `GameState` and re-encode the per-move features
// the opponent would have captured live — i.e. we never persist the
// Float32Array tensors, which would balloon the blob and tie the
// persistence format to the encoding version.
//
// Lives under `ui/` (not `core/`) because the persisted shape references
// the Lichess-puzzle struct, which is a UI concern; core stays unaware
// that the user can be sitting on a particular puzzle.
//
// Compatibility: if the saved sequence fails to replay (corrupt blob,
// rules change, illegal move) we drop it and let the UI start fresh
// rather than wedging the page.

import {
  applyMove,
  fromFen,
  outcome,
  sideToMove,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { GameMoveRecord } from '../core/opponent/index.js';
import { legalMoveIndicesFor, mateInOneIndicesFor } from '../core/opponent/index.js';
import {
  getJson,
  putJson,
  type BlobStorage,
} from '../core/storage/index.js';
import type { LichessPuzzle } from './lichess-endgames.js';

export const LAST_GAME_KEY = 'ui/last-game';

export type PlayerColor = 'w' | 'b';
export type ColorPreference = 'w' | 'b' | 'random';

export interface PersistedSession {
  playerColor: PlayerColor;
  colorPreference?: ColorPreference;
  setupId: string;
  customFen: string;
  currentPuzzle: LichessPuzzle | null;
  startingFen: string;
  moves: Move[];
}

export interface RestoredSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
  startingFen: string;
  playerColor: PlayerColor;
  colorPreference: ColorPreference;
  setupId: string;
  customFen: string;
  currentPuzzle: LichessPuzzle | null;
}

export function restoreSession(saved: PersistedSession): RestoredSession {
  let state = fromFen(saved.startingFen);
  const history: GameMoveRecord[] = [];
  for (const move of saved.moves) {
    history.push({
      features: encodeState(state),
      move,
      sideToMove: sideToMove(state),
      legalMoveIndices: legalMoveIndicesFor(state),
      mateMoveIndices: mateInOneIndicesFor(state),
    });
    state = applyMove(state, move);
  }
  const o = outcome(state);
  const over = o !== 'ongoing';
  return {
    state,
    history,
    over,
    finalOutcome: over ? o : null,
    startingFen: saved.startingFen,
    playerColor: saved.playerColor,
    colorPreference: saved.colorPreference ?? 'random',
    setupId: saved.setupId,
    customFen: saved.customFen,
    currentPuzzle: saved.currentPuzzle,
  };
}

export async function loadPersistedSession(
  storage: BlobStorage,
): Promise<RestoredSession | null> {
  const saved = await getJson<PersistedSession>(storage, LAST_GAME_KEY);
  if (saved === null) return null;
  try {
    return restoreSession(saved);
  } catch (err) {
    console.warn('nn-chess: failed to restore persisted session, ignoring', err);
    return null;
  }
}

export function savePersistedSession(
  storage: BlobStorage,
  data: PersistedSession,
): Promise<void> {
  return putJson(storage, LAST_GAME_KEY, data);
}
