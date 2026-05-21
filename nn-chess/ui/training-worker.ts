/**
 * Training Web Worker.
 *
 * Runs gradient steps in a loop off the main thread so training does
 * not block the UI. Communicates with the main thread via the typed
 * TrainRequest / TrainResponse protocol defined in training-protocol.ts.
 *
 * Trains BOTH the AI net and the player-imitator net in one worker
 * invocation. Each net is trained against its own replay buffer with
 * the same `value_loss + policy_loss` loop. If either buffer is smaller
 * than `batchSize`, that net is skipped (this is expected on a fresh
 * install where the imitator has zero recorded moves) — only an empty
 * AI buffer is reported as an error, since the user explicitly asked
 * to train.
 *
 * Imports ONLY from core/ modules and the shared protocol file — no
 * React, no DOM, no UI-only code. This keeps the worker safe to spin
 * up in any environment.
 */

import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

// Load the WASM binaries from CDN so they don't bloat the app bundle.
// Fall back to CPU if WASM isn't available (e.g. older browsers).
setWasmPaths(
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/',
);
try {
  await tf.setBackend('wasm');
} catch {
  await tf.setBackend('cpu');
}
import { ChessNet, deserialize } from '../core/nn/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
  type ReplayBuffer,
} from '../core/training/index.js';
import type { TrainRequest, TrainResponse } from './training-protocol.js';

// ── Worker state ──────────────────────────────────────────────────────────────

let stopRequested = false;

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Run `totalSteps` gradient steps on `net` from `buffer`. Returns the
 * number of steps actually completed (0 if the buffer is too small).
 * Posts `step` progress messages with the given `phase`.
 */
async function trainNet(
  net: ChessNet,
  buffer: ReplayBuffer,
  phase: 'ai' | 'imitator',
  batchSize: number,
  totalSteps: number,
  reportEvery: number,
): Promise<number> {
  if (buffer.size() < batchSize) return 0;

  let finalStep = 0;
  for (let step = 0; step < totalSteps; step++) {
    if (stopRequested) break;

    const batch = buffer.sample(batchSize);
    const losses = await net.trainBatch(batch);
    finalStep = step + 1;

    if ((step + 1) % reportEvery === 0 || step === totalSteps - 1) {
      const msg: TrainResponse = {
        kind: 'step',
        phase,
        stepIndex: step + 1,
        valueLoss: losses.valueLoss,
        policyLoss: losses.policyLoss,
        totalLoss: losses.totalLoss,
      };
      self.postMessage(msg);
    }
  }
  return finalStep;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<TrainRequest>) => {
  const req = event.data;

  if (req.kind === 'stop') {
    stopRequested = true;
    return;
  }

  if (req.kind !== 'start') return;

  stopRequested = false;

  const {
    weights,
    replay,
    imitatorWeights,
    imitatorReplay,
    replayCapacity,
    batchSize,
    totalSteps,
    reportEvery,
  } = req;

  // Deserialize both networks.
  let aiNet: ChessNet;
  let imitatorNet: ChessNet;
  try {
    aiNet = await deserialize(weights);
  } catch (err) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `Failed to deserialize AI weights: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    return;
  }
  try {
    imitatorNet = await deserialize(imitatorWeights);
  } catch (err) {
    aiNet.dispose();
    const msg: TrainResponse = {
      kind: 'error',
      message: `Failed to deserialize imitator weights: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    return;
  }

  const aiBuffer =
    replay.length > 0
      ? deserializeReplayBuffer(replay, replayCapacity)
      : createReplayBuffer(replayCapacity);
  const imitatorBuffer =
    imitatorReplay.length > 0
      ? deserializeReplayBuffer(imitatorReplay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  // The user clicked "Train" — empty AI buffer is a hard error, but an
  // empty imitator buffer is normal on a fresh profile and just gets
  // silently skipped below.
  if (aiBuffer.size() < batchSize) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `AI replay buffer has ${aiBuffer.size()} examples, need at least ${batchSize}`,
    };
    self.postMessage(msg);
    aiNet.dispose();
    imitatorNet.dispose();
    return;
  }

  let aiFinalStep = 0;
  let imitatorFinalStep = 0;
  try {
    aiFinalStep = await trainNet(aiNet, aiBuffer, 'ai', batchSize, totalSteps, reportEvery);
    if (!stopRequested) {
      imitatorFinalStep = await trainNet(
        imitatorNet,
        imitatorBuffer,
        'imitator',
        batchSize,
        totalSteps,
        reportEvery,
      );
    }
  } catch (err) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `Training failed: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    aiNet.dispose();
    imitatorNet.dispose();
    return;
  }

  // Serialize and send back.
  let finalWeights: Uint8Array;
  let finalImitatorWeights: Uint8Array;
  try {
    finalWeights = await aiNet.serialize();
    finalImitatorWeights = await imitatorNet.serialize();
  } catch (err) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `Failed to serialize weights after training: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    aiNet.dispose();
    imitatorNet.dispose();
    return;
  }

  const finalReplay = aiBuffer.serialize();
  const finalImitatorReplay = imitatorBuffer.serialize();

  const doneMsg: TrainResponse = {
    kind: 'done',
    weights: finalWeights,
    replay: finalReplay,
    imitatorWeights: finalImitatorWeights,
    imitatorReplay: finalImitatorReplay,
    finalStep: aiFinalStep,
    imitatorFinalStep,
  };

  // Transfer ownership of the underlying ArrayBuffers for zero-copy transfer.
  // The WindowPostMessageOptions form `{ transfer: [...] }` is the form that
  // TypeScript's DedicatedWorkerGlobalScope types accept here.
  self.postMessage(doneMsg, {
    transfer: [
      finalWeights.buffer,
      finalReplay.buffer,
      finalImitatorWeights.buffer,
      finalImitatorReplay.buffer,
    ],
  });

  aiNet.dispose();
  imitatorNet.dispose();
};
