/**
 * Training Web Worker.
 *
 * Runs gradient steps in a loop off the main thread so training does
 * not block the UI. Communicates with the main thread via the typed
 * TrainRequest / TrainResponse protocol defined in training-protocol.ts.
 *
 * Imports ONLY from core/ modules and the shared protocol file — no
 * React, no DOM, no UI-only code. This keeps the worker safe to spin
 * up in any environment.
 */

import '@tensorflow/tfjs'; // registers CPU backend
import { ChessNet, deserialize } from '../core/nn/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
} from '../core/training/index.js';
import type { TrainRequest, TrainResponse } from './training-protocol.js';

// ── Worker state ──────────────────────────────────────────────────────────────

let stopRequested = false;

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<TrainRequest>) => {
  const req = event.data;

  if (req.kind === 'stop') {
    stopRequested = true;
    return;
  }

  if (req.kind !== 'start') return;

  stopRequested = false;

  const { weights, replay, replayCapacity, batchSize, totalSteps, reportEvery } = req;

  // Deserialize network.
  let net: ChessNet;
  try {
    net = await deserialize(weights);
  } catch (err) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `Failed to deserialize weights: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    return;
  }

  // Deserialize replay buffer.
  const replayBuffer =
    replay.length > 0
      ? deserializeReplayBuffer(replay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  // Run gradient steps.
  let finalStep = 0;

  for (let step = 0; step < totalSteps; step++) {
    if (stopRequested) break;

    // Skip if not enough data.
    if (replayBuffer.size() < batchSize) {
      // Nothing to train on yet — report and finish.
      const msg: TrainResponse = {
        kind: 'error',
        message: `Replay buffer has ${replayBuffer.size()} examples, need at least ${batchSize}`,
      };
      self.postMessage(msg);
      net.dispose();
      return;
    }

    let losses: { valueLoss: number; policyLoss: number; totalLoss: number };
    try {
      const batch = replayBuffer.sample(batchSize);
      losses = await net.trainBatch(batch);
    } catch (err) {
      const msg: TrainResponse = {
        kind: 'error',
        message: `Training step ${step} failed: ${err instanceof Error ? err.message : String(err)}`,
      };
      self.postMessage(msg);
      net.dispose();
      return;
    }

    finalStep = step + 1;

    // Post progress every reportEvery steps (and on the last step).
    if ((step + 1) % reportEvery === 0 || step === totalSteps - 1) {
      const msg: TrainResponse = {
        kind: 'step',
        stepIndex: step + 1,
        valueLoss: losses.valueLoss,
        policyLoss: losses.policyLoss,
        totalLoss: losses.totalLoss,
      };
      self.postMessage(msg);
    }
  }

  // Serialize and send back.
  let finalWeights: Uint8Array;
  try {
    finalWeights = await net.serialize();
  } catch (err) {
    const msg: TrainResponse = {
      kind: 'error',
      message: `Failed to serialize weights after training: ${err instanceof Error ? err.message : String(err)}`,
    };
    self.postMessage(msg);
    net.dispose();
    return;
  }

  const finalReplay = replayBuffer.serialize();

  const doneMsg: TrainResponse = {
    kind: 'done',
    weights: finalWeights,
    replay: finalReplay,
    finalStep,
  };

  // Transfer ownership of the underlying ArrayBuffers for zero-copy transfer.
  // The WindowPostMessageOptions form `{ transfer: [...] }` is the form that
  // TypeScript's DedicatedWorkerGlobalScope types accept here.
  self.postMessage(doneMsg, { transfer: [finalWeights.buffer, finalReplay.buffer] });

  net.dispose();
};
