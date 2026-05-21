/**
 * Main-thread wrapper for the self-play Web Worker.
 *
 * `SelfPlayClient` encapsulates the raw Worker API so the rest of the
 * UI (`SelfPlayPanel`, `Game.tsx`) only deals with typed callbacks and
 * never touches `postMessage` / `onmessage` directly.
 *
 * Usage:
 *   const client = new SelfPlayClient();
 *   client.start(
 *     { aiWeights, imitatorWeights, replay, replayCapacity, search,
 *       count, maxPlies, gradientSteps, gradientBatchSize },
 *     (game)   => updateBatchCounter(game),
 *     (step)   => showLoss(step),
 *     (result) => persistWeights(result),
 *     (err)    => console.error(err),
 *   );
 *   // Later:
 *   client.stop();
 */

import type { SelfPlayRequest, SelfPlayResponse } from './selfplay-protocol.js';

export type SelfPlayStartParams = Extract<SelfPlayRequest, { kind: 'start' }>;

export type GameCallback = (game: Extract<SelfPlayResponse, { kind: 'game' }>) => void;
export type StepCallback = (step: Extract<SelfPlayResponse, { kind: 'step' }>) => void;
export type DoneCallback = (result: Extract<SelfPlayResponse, { kind: 'done' }>) => void;
export type ErrorCallback = (message: string) => void;
export type PlyCallback = (ply: Extract<SelfPlayResponse, { kind: 'ply' }>) => void;

export class SelfPlayClient {
  private worker: Worker | null = null;

  /**
   * Spawns (or re-uses) the Worker and starts a self-play batch.
   *
   * If a run is already in progress, it is stopped first. The new run
   * uses freshly serialized bytes passed in by the caller — the worker
   * never reads from storage directly.
   *
   * Vite/Astro bundlers recognize the `new URL('./selfplay-worker.ts',
   * import.meta.url)` pattern and emit the worker as a separate chunk
   * with its own module graph. The `{ type: 'module' }` option tells the
   * browser to parse it as an ES module (required for top-level imports).
   */
  start(
    params: Omit<SelfPlayStartParams, 'kind'>,
    onGame: GameCallback,
    onStep: StepCallback,
    onDone: DoneCallback,
    onError: ErrorCallback,
    onPly?: PlyCallback,
  ): void {
    // Tear down any existing worker before starting a new one.
    this.stop();

    this.worker = new Worker(
      new URL('./selfplay-worker.ts', import.meta.url),
      { type: 'module' },
    );

    this.worker.onmessage = (event: MessageEvent<SelfPlayResponse>) => {
      const msg = event.data;
      if (msg.kind === 'game') {
        onGame(msg);
      } else if (msg.kind === 'step') {
        onStep(msg);
      } else if (msg.kind === 'ply') {
        onPly?.(msg);
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

    const req: SelfPlayRequest = { kind: 'start', ...params };
    // Transfer ownership of the byte buffers to avoid a copy.
    this.worker.postMessage(req, [
      params.aiWeights.buffer,
      params.imitatorWeights.buffer,
      params.replay.buffer,
      params.imitatorReplay.buffer,
    ]);
  }

  /** Request a graceful stop. The worker will post 'done' before exiting. */
  stop(): void {
    if (this.worker) {
      const req: SelfPlayRequest = { kind: 'stop' };
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

  /** Returns true if a self-play run is currently in progress. */
  get isRunning(): boolean {
    return this.worker !== null;
  }
}
