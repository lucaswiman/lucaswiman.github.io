/**
 * Shared message types for the training Web Worker protocol.
 *
 * Both `training-worker.ts` (the worker side) and `training-client.ts`
 * (the main-thread wrapper) import from here. No React, no DOM, no UI
 * imports — this file must stay safe to use in a Worker context.
 */

/**
 * Messages sent FROM the main thread TO the worker.
 *
 * `start` kicks off a training run:
 *   - weights:        serialized ChessNet bytes (from ChessNet.serialize())
 *   - replay:         serialized ReplayBuffer bytes (from ReplayBuffer.serialize())
 *   - replayCapacity: capacity to pass to deserializeReplayBuffer
 *   - batchSize:      examples per gradient step
 *   - totalSteps:     number of gradient steps to run
 *   - reportEvery:    how often to post a 'step' progress message
 *
 * `stop` requests a graceful early stop. The worker will finish the
 * current step, serialize, and post a 'done' message.
 */
export type TrainRequest =
  | {
      kind: 'start';
      weights: Uint8Array;
      replay: Uint8Array;
      replayCapacity: number;
      batchSize: number;
      totalSteps: number;
      reportEvery: number;
    }
  | { kind: 'stop' };

/**
 * Messages sent FROM the worker TO the main thread.
 *
 * `step`  — periodic progress report during training.
 * `done`  — training has finished (budget exhausted or stop requested).
 *            Carries fresh serialized weights and replay buffer.
 * `error` — something went wrong; the worker is now idle.
 */
export type TrainResponse =
  | {
      kind: 'step';
      stepIndex: number;
      valueLoss: number;
      policyLoss: number;
      totalLoss: number;
    }
  | {
      kind: 'done';
      weights: Uint8Array;
      replay: Uint8Array;
      finalStep: number;
    }
  | { kind: 'error'; message: string };
