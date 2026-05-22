// Shared opponent wiring for the human-play page and the Stockfish
// sparring page. Both routes save/load to the same IndexedDB namespace
// so weights persist across pages — diverging the config here would
// silently shard the opponent in two.
//
// As of the two-network refactor the project loads BOTH an `Opponent`
// (the network we play against) and a `PlayerImitator` (a second net the
// opponent uses as a self-play stand-in for the user). Their states are
// stored under disjoint key prefixes inside the same `BlobStorage`.
//
// Storage backend: IndexedDB. Earlier builds used localStorage, which
// caps at ~5–10 MB per origin and produces QuotaExceededError once the
// replay buffer accumulates a few hundred examples. On first run with
// this build we migrate any existing localStorage entries into IDB and
// then free the localStorage quota.

import { ChessNet } from '../core/nn/index.js';
import {
  loadOpponent,
  type Opponent,
  type OpponentConfig,
} from '../core/opponent/index.js';
import {
  loadPlayerImitator,
  type PlayerImitator,
} from '../core/imitator/index.js';
import { createLocalStorageAdapter } from '../adapters/storage-localstorage.js';
import { createIndexedDBAdapter } from '../adapters/storage-indexeddb.js';
import { migrateBlobStorage } from '../adapters/migrate.js';
import type { BlobStorage } from '../core/storage/index.js';

export const STORAGE_PREFIX = 'nn-chess-v1/';
export const REPLAY_CAPACITY = 10_000;

// simulations=64 keeps a move under ~5s on a CPU; temperature=0 is greedy.
export const OPPONENT_CONFIG: OpponentConfig = {
  simulations: 64,
  cPuct: 1.5,
  temperature: 0,
};

/**
 * MCTS hyperparameters used during self-play between the opponent and
 * the imitator. The simulation count is sourced from the in-game MCTS
 * sims slider at call sites — `simulations` here is only a fallback
 * default for any code path that doesn't have access to the slider.
 *
 * Dirichlet noise is mixed into the root priors of every search so that
 * a cold-start net — which has near-uniform priors and value ≈ 0
 * everywhere — explores a variety of root moves instead of collapsing
 * onto whichever move has a microscopically larger prior. α=0.3,
 * ε=0.25 are AlphaZero's chess values; the spec is unchanged, only the
 * data-generation knob.
 */
export const SELFPLAY_SEARCH = {
  simulations: 16,
  cPuct: 1.5,
  dirichletAlpha: 0.3,
  dirichletEpsilon: 0.25,
};

/** Default games per self-play batch on a post-game training pass. */
export const SELFPLAY_BATCH_SIZE = 3;

/** Ply cap for any one self-play game (well under the 75-move rule). */
export const SELFPLAY_MAX_PLIES = 160;

/**
 * For the first N plies of each self-play game we sample from the MCTS
 * visit-count distribution instead of taking the argmax. AlphaZero
 * used 30 plies for chess; we match that, scaled down implicitly by
 * the lower simulation count (the relative shape of the distribution
 * is what matters here, not its absolute counts). Combined with
 * Dirichlet root noise this is the standard cure for "two pieces
 * shuffle to a 3-fold draw" in cold-start training games.
 */
export const SELFPLAY_TEMPERATURE_PLIES = 30;

/** Visit-count sampling temperature. 1.0 = raw visit counts (AlphaZero). */
export const SELFPLAY_TEMPERATURE = 1.0;

/** Gradient steps to run on the opponent net immediately after a self-play batch. */
export const SELFPLAY_GRADIENT_STEPS = 10;

/** Minibatch size for those post-batch gradient steps. */
export const SELFPLAY_GRADIENT_BATCH = 32;

/** Storage key for the "auto-run self-play after each game" user preference. */
export const AUTO_SELFPLAY_PREF_KEY = 'ui/auto-selfplay';

export function createOpponentStorage(): BlobStorage {
  return createIndexedDBAdapter({ prefix: STORAGE_PREFIX });
}

// Module-level guard so two concurrent callers (e.g. Game.tsx and a
// follow-up reload in StrictMode) don't both kick off the migration.
let migrationPromise: Promise<void> | null = null;

async function runMigrationIfNeeded(target: BlobStorage): Promise<void> {
  if (typeof window === 'undefined' || !('localStorage' in window)) return;
  const legacy = createLocalStorageAdapter({ prefix: STORAGE_PREFIX });
  try {
    const result = await migrateBlobStorage(legacy, target);
    if (result.migrated > 0) {
      console.info(`nn-chess: migrated ${result.migrated} key(s) from localStorage to IndexedDB`);
    }
  } catch (err) {
    // Migration failure is non-fatal: a fresh opponent simply starts empty.
    console.warn('nn-chess: migration from localStorage failed', err);
  }
}

export function loadOpponentFromStorage(storage: BlobStorage): Promise<Opponent> {
  if (migrationPromise === null) {
    migrationPromise = runMigrationIfNeeded(storage);
  }
  return migrationPromise.then(() =>
    loadOpponent(storage, {
      net: ChessNet.create({ seed: Date.now() }),
      config: OPPONENT_CONFIG,
      replayCapacity: REPLAY_CAPACITY,
    }),
  );
}

export function loadPlayerImitatorFromStorage(
  storage: BlobStorage,
): Promise<PlayerImitator> {
  if (migrationPromise === null) {
    migrationPromise = runMigrationIfNeeded(storage);
  }
  return migrationPromise.then(() =>
    loadPlayerImitator(storage, {
      // Seed offset so the imitator is initialized differently from the
      // opponent; otherwise two identically-initialized nets give MCTS
      // the same priors and self-play would converge on identical lines.
      net: ChessNet.create({ seed: Date.now() ^ 0x5A5A5A5A }),
      replayCapacity: REPLAY_CAPACITY,
    }),
  );
}
