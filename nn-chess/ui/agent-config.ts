// Shared agent wiring for the human-play page and the Stockfish sparring
// page. Both routes save/load to the same IndexedDB namespace so weights
// persist across pages — diverging the config here would silently shard
// the agent in two.
//
// Storage backend: IndexedDB. Earlier builds used localStorage, which
// caps at ~5–10 MB per origin and produces QuotaExceededError once the
// replay buffer accumulates a few hundred examples. On first run with
// this build we migrate any existing localStorage entries into IDB and
// then free the localStorage quota.

import { ChessNet } from '../core/nn/index.js';
import {
  loadAgent,
  type Agent,
  type AgentConfig,
} from '../core/agent/index.js';
import { createLocalStorageAdapter } from '../adapters/storage-localstorage.js';
import { createIndexedDBAdapter } from '../adapters/storage-indexeddb.js';
import { migrateBlobStorage } from '../adapters/migrate.js';
import type { BlobStorage } from '../core/storage/index.js';

export const STORAGE_PREFIX = 'nn-chess-v1/';
export const REPLAY_CAPACITY = 10_000;

// simulations=64 keeps a move under ~5s on a CPU; temperature=0 is greedy.
export const AGENT_CONFIG: AgentConfig = {
  simulations: 64,
  cPuct: 1.5,
  temperature: 0,
};

export function createAgentStorage(): BlobStorage {
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
    // Migration failure is non-fatal: a fresh agent simply starts empty.
    console.warn('nn-chess: migration from localStorage failed', err);
  }
}

export function loadAgentFromStorage(storage: BlobStorage): Promise<Agent> {
  if (migrationPromise === null) {
    migrationPromise = runMigrationIfNeeded(storage);
  }
  return migrationPromise.then(() =>
    loadAgent(storage, {
      net: ChessNet.create({ seed: Date.now() }),
      config: AGENT_CONFIG,
      replayCapacity: REPLAY_CAPACITY,
    }),
  );
}
