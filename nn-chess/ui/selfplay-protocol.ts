/**
 * Shared message types for the self-play Web Worker protocol.
 *
 * Both `selfplay-worker.ts` (the worker side) and `selfplay-client.ts`
 * (the main-thread wrapper) import from here. No React, no DOM, no UI
 * imports — this file must stay safe to use in a Worker context.
 *
 * A self-play run is two phases inside one worker invocation:
 *   1. Play `count` MCTS games between the AI net and the imitator net,
 *      collecting AlphaZero-style training examples for the AI's side.
 *   2. Run `gradientSteps` gradient steps on the AI net using the
 *      enlarged replay buffer.
 * Folding both phases into one worker call avoids two rounds of weight /
 * buffer serialization across the worker boundary.
 */

/** Lightweight mirror of MCTS SearchConfig (only the fields self-play uses). */
export interface SelfPlaySearchConfig {
  simulations: number;
  cPuct: number;
}

/**
 * Messages sent FROM the main thread TO the worker.
 *
 * `start` kicks off a self-play batch followed by gradient steps:
 *   - aiWeights:        serialized AI ChessNet bytes
 *   - imitatorWeights:  serialized imitator ChessNet bytes (read-only in worker)
 *   - replay:           serialized AI ReplayBuffer bytes (may be empty)
 *   - replayCapacity:   capacity for the replay buffer
 *   - search:           MCTS budget applied to both sides
 *   - count:            number of self-play games to play
 *   - maxPlies:         per-game ply cap
 *   - gradientSteps:    number of gradient steps to run after self-play
 *   - gradientBatchSize:examples per gradient step
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
 * `game`  — one self-play game finished.
 * `step`  — periodic gradient-step progress report.
 * `done`  — the whole batch finished (or stop was requested). Carries
 *           fresh AI weights and replay buffer; the imitator weights do
 *           not come back because the imitator is read-only here.
 * `error` — something went wrong; the worker is now idle.
 */
export type SelfPlayResponse =
  | {
      kind: 'game';
      index: number;
      aiColor: 'w' | 'b';
      aiResult: 'win' | 'loss' | 'draw';
      plies: number;
    }
  | {
      kind: 'step';
      stepIndex: number;
      valueLoss: number;
      policyLoss: number;
      totalLoss: number;
    }
  | {
      kind: 'done';
      aiWeights: Uint8Array;
      replay: Uint8Array;
      games: number;
      wins: number;
      losses: number;
      draws: number;
      examplesAdded: number;
      finalStep: number;
    }
  | { kind: 'error'; message: string };
