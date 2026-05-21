/**
 * Shared message types for the self-play Web Worker protocol.
 *
 * Both `selfplay-worker.ts` (the worker side) and `selfplay-client.ts`
 * (the main-thread wrapper) import from here. No React, no DOM, no UI
 * imports — this file must stay safe to use in a Worker context.
 *
 * A self-play run is three phases inside one worker invocation:
 *   1. Run `gradientSteps` gradient steps on the imitator from its own
 *      replay buffer, so the AI's self-play opponent reflects the
 *      user's latest moves before games are generated.
 *   2. Play `count` MCTS games between the AI net and the (just-updated)
 *      imitator net, collecting AlphaZero-style training examples for
 *      the AI's side.
 *   3. Run `gradientSteps` gradient steps on the AI net using the
 *      enlarged replay buffer.
 * Folding all phases into one worker call avoids extra rounds of weight
 * / buffer serialization across the worker boundary.
 */

/** Lightweight mirror of MCTS SearchConfig (only the fields self-play uses). */
export interface SelfPlaySearchConfig {
  simulations: number;
  cPuct: number;
}

/**
 * Messages sent FROM the main thread TO the worker.
 *
 * `start` kicks off a self-play batch sandwiched between gradient steps:
 *   - aiWeights:           serialized AI ChessNet bytes
 *   - imitatorWeights:     serialized imitator ChessNet bytes
 *   - replay:              serialized AI ReplayBuffer bytes (may be empty)
 *   - imitatorReplay:      serialized imitator ReplayBuffer bytes (may be empty)
 *   - replayCapacity:      capacity for both replay buffers
 *   - search:              MCTS budget applied to both sides
 *   - count:               number of self-play games to play
 *   - maxPlies:            per-game ply cap
 *   - gradientSteps:       number of gradient steps to run on each net
 *   - gradientBatchSize:   examples per gradient step
 *
 * `stop` requests a graceful early stop. The worker finishes the current
 * game / step, serializes, and posts a 'done' message.
 */
export type SelfPlayRequest =
  | {
      kind: 'start';
      aiWeights: Uint8Array;
      imitatorWeights: Uint8Array;
      replay: Uint8Array;
      imitatorReplay: Uint8Array;
      replayCapacity: number;
      search: SelfPlaySearchConfig;
      count: number;
      maxPlies: number;
      gradientSteps: number;
      gradientBatchSize: number;
    }
  | { kind: 'stop' };

/**
 * Messages sent FROM the worker TO the main thread.
 *
 * `ply`   — emitted once per board state inside a self-play game (one for
 *           the starting position, then one per move). Used by the UI to
 *           render a live board; pure background runs ignore it.
 * `game`  — one self-play game finished.
 * `step`  — periodic gradient-step progress report. `phase` distinguishes
 *           the imitator pre-train pass from the AI post-train pass.
 * `done`  — the whole batch finished (or stop was requested). Carries
 *           fresh AI + imitator weights and both replay buffers.
 * `error` — something went wrong; the worker is now idle.
 */
export type SelfPlayResponse =
  | {
      kind: 'ply';
      gameIndex: number;
      aiColor: 'w' | 'b';
      plyIndex: number;
      fen: string;
      lastMove: string | null;
    }
  | {
      kind: 'game';
      index: number;
      aiColor: 'w' | 'b';
      aiResult: 'win' | 'loss' | 'draw';
      plies: number;
    }
  | {
      kind: 'step';
      phase: 'imitator' | 'ai';
      stepIndex: number;
      valueLoss: number;
      policyLoss: number;
      totalLoss: number;
    }
  | {
      kind: 'done';
      aiWeights: Uint8Array;
      imitatorWeights: Uint8Array;
      replay: Uint8Array;
      imitatorReplay: Uint8Array;
      games: number;
      wins: number;
      losses: number;
      draws: number;
      examplesAdded: number;
      finalStep: number;
      imitatorFinalStep: number;
    }
  | { kind: 'error'; message: string };
