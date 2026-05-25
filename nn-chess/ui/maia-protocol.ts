/**
 * Shared protocol for the Maia-2 inference Web Worker.
 *
 * The Maia-2 forward pass is ~2-3 conv blocks + transformer attention
 * over 1024-d × 8 tokens; on a CPU backend this takes hundreds of
 * milliseconds per move. Running it in a worker keeps the main thread
 * (and therefore the board UI) responsive between moves.
 *
 * Both `maia-worker.ts` (worker side) and `maia-client.ts` (main-thread
 * wrapper) import from here. No DOM / React imports — this file must
 * be safe to evaluate inside a Worker.
 */

import type { MaiaManifest } from '../core/maia/index.js';

/** Messages sent FROM the main thread TO the worker. */
export type MaiaRequest =
  | {
      kind: 'init';
      /** Raw bytes of `weights.bin` produced by convert-maia2.py. */
      blob: ArrayBuffer;
      manifest: MaiaManifest;
    }
  | {
      kind: 'select-move';
      requestId: number;
      /** Forsyth-Edwards Notation of the position to move in. */
      fen: string;
      /** Skill rating (raw ELO; bucketed inside the worker). */
      elo: number;
      /** Opponent skill (defaults to `elo`). */
      oppoElo?: number;
      /** 0 = greedy argmax; >0 = temperature sampling. */
      temperature?: number;
    }
  | {
      // Like `select-move`, but routes through `core/mcts.search` with
      // `maiaPolicy` providing priors + value. Each simulation costs one
      // full Maia forward pass, which is the same WASM/CPU bottleneck
      // the `select-move` path hits — so the UI caps the simulation
      // budget at a small number to keep per-move latency bounded.
      kind: 'search-move';
      requestId: number;
      fen: string;
      elo: number;
      oppoElo?: number;
      /** MCTS simulation budget. */
      simulations: number;
      /** PUCT exploration constant; mirrors `OPPONENT_CONFIG.cPuct`. */
      cPuct: number;
      /**
       * Visit-count sampling temperature.
       * 0 = argmax over visits; >0 samples ∝ N^(1/T). Mirrors the
       * temperature semantics of `selectMoveFromNet` with `search` set.
       */
      temperature?: number;
    };

/** Messages sent FROM the worker TO the main thread. */
export type MaiaResponse =
  | { kind: 'ready' }
  | {
      kind: 'move';
      requestId: number;
      move: string;
      value: number;
      /** Probability mass on each legal move, keyed by UCI string. */
      moveProbs: Record<string, number>;
      /**
       * Set when this response is the result of a `search-move` request.
       * Carries MCTS visit counts so the UI can render a parallel
       * inspector readout if desired.
       */
      visitCounts?: Record<string, number>;
    }
  | { kind: 'error'; requestId?: number; message: string };
