/**
 * Main-thread wrapper for the training Web Worker.
 *
 * `TrainingClient` encapsulates the raw Worker API so the rest of the
 * UI (`Game.tsx`, `TrainingPanel`) only deals with typed callbacks and
 * never touches `postMessage` / `onmessage` directly.
 *
 * Usage:
 *   const client = new TrainingClient();
 *   client.start(
 *     { weights, replay, replayCapacity, batchSize, totalSteps, reportEvery },
 *     (progress) => console.log(progress),
 *     (result)   => applyNewWeights(result),
 *     (err)      => console.error(err),
 *   );
 *   // Later:
 *   client.stop();
 */

import type { TrainRequest, TrainResponse } from './training-protocol.js';

export type TrainStartParams = Extract<TrainRequest, { kind: 'start' }>;

export type ProgressCallback = (progress: Extract<TrainResponse, { kind: 'step' }>) => void;
export type DoneCallback = (result: Extract<TrainResponse, { kind: 'done' }>) => void;
export type ErrorCallback = (message: string) => void;

export class TrainingClient {
  private worker: Worker | null = null;

  /**
   * Spawns (or re-uses) the Worker and starts a training run.
   *
   * If a run is already in progress, it is stopped first. The new run
   * uses freshly serialized weights and replay bytes passed in by the
   * caller — the worker never reads from storage directly.
   *
   * Vite/Astro bundlers recognize the `new URL('./training-worker.ts',
   * import.meta.url)` pattern and emit the worker as a separate chunk
   * with its own module graph. The `{ type: 'module' }` option tells the
   * browser to parse it as an ES module (required for top-level imports).
   */
  start(
    params: Omit<TrainStartParams, 'kind'>,
    onProgress: ProgressCallback,
    onDone: DoneCallback,
    onError: ErrorCallback,
  ): void {
    // Tear down any existing worker before starting a new one.
    this.stop();

    this.worker = new Worker(
      new URL('./training-worker.ts', import.meta.url),
      { type: 'module' },
    );

    this.worker.onmessage = (event: MessageEvent<TrainResponse>) => {
      const msg = event.data;
      if (msg.kind === 'step') {
        onProgress(msg);
      } else if (msg.kind === 'done') {
        onDone(msg);
        this._cleanup();
      } else if (msg.kind === 'error') {
        onError(msg.message);
        this._cleanup();
      }
    };

    this.worker.onerror = (event) => {
      onError(`Worker error: ${event.message}`);
      this._cleanup();
    };

    const req: TrainRequest = { kind: 'start', ...params };
    // Transfer ownership of the byte buffers to avoid a copy.
    this.worker.postMessage(req, [params.weights.buffer, params.replay.buffer]);
  }

  /** Request a graceful stop. The worker will post 'done' before exiting. */
  stop(): void {
    if (this.worker) {
      const req: TrainRequest = { kind: 'stop' };
      this.worker.postMessage(req);
    }
  }

  /** Forcefully terminate and clean up (used internally after done/error). */
  private _cleanup(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  /** Returns true if a training run is currently in progress. */
  get isRunning(): boolean {
    return this.worker !== null;
  }
}
