import { describe, it, expect } from 'vitest';
import {
  LAST_GAME_KEY,
  loadPersistedSession,
  restoreSession,
  savePersistedSession,
  type PersistedSession,
} from './session-persistence.js';
import { STARTING_FEN, initialState, toFen } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import { createMemoryStorage } from '../adapters/storage-memory.js';

function fresh(overrides: Partial<PersistedSession> = {}): PersistedSession {
  return {
    playerColor: 'w',
    setupId: 'standard',
    customFen: '',
    currentPuzzle: null,
    startingFen: STARTING_FEN,
    moves: [],
    ...overrides,
  };
}

describe('restoreSession', () => {
  it('reconstructs a fresh game with no moves', () => {
    const restored = restoreSession(fresh());
    expect(restored.history).toHaveLength(0);
    expect(restored.over).toBe(false);
    expect(restored.finalOutcome).toBeNull();
    expect(toFen(restored.state)).toBe(STARTING_FEN);
  });

  it('replays moves and re-encodes features for each step', () => {
    const restored = restoreSession(fresh({ moves: ['e2e4', 'e7e5'] }));
    expect(restored.history).toHaveLength(2);
    expect(restored.history[0]?.move).toBe('e2e4');
    expect(restored.history[0]?.sideToMove).toBe('w');
    expect(restored.history[1]?.move).toBe('e7e5');
    expect(restored.history[1]?.sideToMove).toBe('b');
    expect(restored.history[0]?.features).toBeInstanceOf(Float32Array);
    expect(restored.over).toBe(false);
  });

  it('captures features from the position BEFORE each move (not after)', () => {
    // The first record's encoded state must match the starting
    // position, not the post-e2e4 position.
    const restored = restoreSession(fresh({ moves: ['e2e4'] }));
    const expected = encodeState(initialState());
    expect(Array.from(restored.history[0]!.features)).toEqual(Array.from(expected));
  });

  it('detects a terminal outcome (Fool\'s mate)', () => {
    const restored = restoreSession(
      fresh({ moves: ['f2f3', 'e7e5', 'g2g4', 'd8h4'] }),
    );
    expect(restored.over).toBe(true);
    expect(restored.finalOutcome).toBe('black-wins');
  });

  it('round-trips through storage', async () => {
    const storage = createMemoryStorage();
    const data = fresh({
      playerColor: 'b',
      setupId: 'lichess-mate-in-2',
      customFen: 'foo',
      currentPuzzle: { id: 'abc', fen: STARTING_FEN, themes: 'mateIn2', rating: 1500 },
      moves: ['e2e4', 'e7e5'],
    });
    await savePersistedSession(storage, data);
    const restored = await loadPersistedSession(storage);
    expect(restored).not.toBeNull();
    expect(restored?.playerColor).toBe('b');
    expect(restored?.setupId).toBe('lichess-mate-in-2');
    expect(restored?.customFen).toBe('foo');
    expect(restored?.currentPuzzle?.id).toBe('abc');
    expect(restored?.history.map(r => r.move)).toEqual(['e2e4', 'e7e5']);
  });

  it('returns null when nothing is persisted', async () => {
    const storage = createMemoryStorage();
    expect(await loadPersistedSession(storage)).toBeNull();
  });

  it('drops the saved blob rather than throwing on corrupt move sequences', async () => {
    const storage = createMemoryStorage();
    // Write a blob with an illegal move directly so loadPersistedSession
    // has to hit the catch branch.
    const corrupt: PersistedSession = fresh({ moves: ['z9z9'] });
    await savePersistedSession(storage, corrupt);
    expect(await loadPersistedSession(storage)).toBeNull();
  });
});

describe('LAST_GAME_KEY', () => {
  it('lives under the ui/ namespace so it can be shared with other UI prefs', () => {
    expect(LAST_GAME_KEY.startsWith('ui/')).toBe(true);
  });
});
