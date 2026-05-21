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
 * `start` kicks off a training run on BOTH networks. Each net trains
 * against its own replay buffer with the same loss / batch-size / step
 * budget. The imitator is skipped silently when its buffer is too small.
 *   - weights:           serialized AI ChessNet bytes
 *   - replay:            serialized AI ReplayBuffer bytes
 *   - imitatorWeights:   serialized imitator ChessNet bytes
 *   - imitatorReplay:    serialized imitator ReplayBuffer bytes
 *   - replayCapacity:    capacity for both replay buffers
 *   - batchSize:         examples per gradient step
 *   - totalSteps:        number of gradient steps to run on each net
 *   - reportEvery:       how often to post a 'step' progress message
 *
 * `stop` requests a graceful early stop. The worker will finish the
 * current step, serialize, and post a 'done' message.
 */
export type TrainRequest =
  | {
      kind: 'start';
      weights: Uint8Array;
      replay: Uint8Array;
      imitatorWeights: Uint8Array;
      imitatorReplay: Uint8Array;
      replayCapacity: number;
      batchSize: number;
      totalSteps: number;
      reportEvery: number;
    }
  | { kind: 'stop' };

/**
 * Messages sent FROM the worker TO the main thread.
 *
 * `step`  — periodic progress report during training. `phase` distinguishes
 *           the AI pass from the imitator pass.
 * `done`  — training has finished (budget exhausted or stop requested).
 *            Carries fresh serialized weights and replay buffers for
 *            both nets.
 * `error` — something went wrong; the worker is now idle.
 */
export type TrainResponse =
  | {
      kind: 'step';
      phase: 'ai' | 'imitator';
      stepIndex: number;
      valueLoss: number;
      policyLoss: number;
      totalLoss: number;
    }
  | {
      kind: 'done';
      weights: Uint8Array;
      replay: Uint8Array;
      imitatorWeights: Uint8Array;
      imitatorReplay: Uint8Array;
      finalStep: number;
      imitatorFinalStep: number;
    }
  | { kind: 'error'; message: string };
