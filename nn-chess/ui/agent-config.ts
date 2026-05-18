// Shared agent wiring for the human-play page and the Stockfish sparring
// page. Both routes save/load to the same localStorage namespace so weights
// persist across pages — diverging the config here would silently shard
// the agent in two.

import { ChessNet } from '../core/nn/index.js';
import {
  loadAgent,
  type Agent,
  type AgentConfig,
} from '../core/agent/index.js';
import { createLocalStorageAdapter } from '../adapters/storage-localstorage.js';
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
  return createLocalStorageAdapter({ prefix: STORAGE_PREFIX });
}

export function loadAgentFromStorage(storage: BlobStorage): Promise<Agent> {
  return loadAgent(storage, {
    net: ChessNet.create({ seed: Date.now() }),
    config: AGENT_CONFIG,
    replayCapacity: REPLAY_CAPACITY,
  });
}
