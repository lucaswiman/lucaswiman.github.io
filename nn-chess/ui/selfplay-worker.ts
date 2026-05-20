/**
 * Self-play Web Worker.
 *
 * Runs a batch of MCTS self-play games (AI net vs. imitator net) and
 * then a handful of gradient steps on the AI net off the main thread.
 * On a phone, doing this on the UI thread freezes the page for tens of
 * seconds; moving it here keeps the board interactive.
 *
 * Communicates with the main thread via the typed SelfPlayRequest /
 * SelfPlayResponse protocol defined in selfplay-protocol.ts.
 *
 * Imports ONLY from core/ modules and the shared protocol file — no
 * React, no DOM, no UI-only code. The worker reads no storage; the
 * caller hands it serialized weights + replay bytes and gets fresh
 * bytes back when the batch is done.
 */

import '@tensorflow/tfjs'; // registers CPU backend
import { ChessNet, deserialize } from '../core/nn/index.js';
import { policyFromNet } from '../core/agent/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
} from '../core/training/index.js';
import { runSelfPlayBatch } from '../core/selfplay/index.js';
import type { SelfPlayRequest, SelfPlayResponse } from './selfplay-protocol.js';

// ── Worker state ──────────────────────────────────────────────────────────────

let stopRequested = false;

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<SelfPlayRequest>) => {
  const req = event.data;

  if (req.kind === 'stop') {
    stopRequested = true;
    return;
  }

  if (req.kind !== 'start') return;

  stopRequested = false;

  const {
    aiWeights,
    imitatorWeights,
    replay,
    replayCapacity,
    search,
    count,
    maxPlies,
    gradientSteps,
    gradientBatchSize,
  } = req;

  // Deserialize both networks.
  let aiNet: ChessNet;
  let imitatorNet: ChessNet;
  try {
    aiNet = await deserialize(aiWeights);
  } catch (err) {
    postError(`Failed to deserialize AI weights: ${errMessage(err)}`);
    return;
  }
  try {
    imitatorNet = await deserialize(imitatorWeights);
  } catch (err) {
    aiNet.dispose();
    postError(`Failed to deserialize imitator weights: ${errMessage(err)}`);
    return;
  }

  const aiPolicy = policyFromNet(aiNet);
  const playerPolicy = policyFromNet(imitatorNet);

  const replayBuffer =
    replay.length > 0
      ? deserializeReplayBuffer(replay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  // Bridge the worker's `stop` message to runSelfPlayBatch's AbortSignal.
  const controller = new AbortController();
  const stopPoll = setInterval(() => {
    if (stopRequested) controller.abort();
  }, 50);

  let wins = 0;
  let losses = 0;
  let draws = 0;
  let games = 0;
  let examplesAdded = 0;
  let finalStep = 0;

  try {
    const batchResult = await runSelfPlayBatch({
      aiPolicy,
      playerPolicy,
      search,
      count,
      maxPlies,
      signal: controller.signal,
      onGame: (idx, info) => {
        const msg: SelfPlayResponse = {
          kind: 'game',
          index: idx,
          aiColor: info.aiColor,
          aiResult: info.aiResult,
          plies: info.plies,
        };
        self.postMessage(msg);
      },
    });

    games = batchResult.games;
    wins = batchResult.wins;
    losses = batchResult.losses;
    draws = batchResult.draws;
    examplesAdded = batchResult.aiTrainingExamples.length;

    for (const ex of batchResult.aiTrainingExamples) {
      replayBuffer.add(ex);
    }

    // Run gradient steps on the (now enlarged) replay buffer. We are
    // intentionally lenient: if the buffer is still smaller than
    // gradientBatchSize, just report 0 steps. The user may have very
    // few examples at first and the self-play games themselves are
    // worth keeping regardless.
    if (replayBuffer.size() >= gradientBatchSize && gradientSteps > 0) {
      const reportEvery = Math.max(1, Math.floor(gradientSteps / 10));
      for (let step = 0; step < gradientSteps; step++) {
        if (stopRequested) break;

        const batch = replayBuffer.sample(gradientBatchSize);
        const stepLosses = await aiNet.trainBatch(batch);
        finalStep = step + 1;

        if ((step + 1) % reportEvery === 0 || step === gradientSteps - 1) {
          const msg: SelfPlayResponse = {
            kind: 'step',
            stepIndex: step + 1,
            valueLoss: stepLosses.valueLoss,
            policyLoss: stepLosses.policyLoss,
            totalLoss: stepLosses.totalLoss,
          };
          self.postMessage(msg);
        }
      }
    }
  } catch (err) {
    clearInterval(stopPoll);
    aiNet.dispose();
    imitatorNet.dispose();
    postError(`Self-play failed: ${errMessage(err)}`);
    return;
  }

  clearInterval(stopPoll);

  // Serialize fresh AI weights + replay buffer for the round-trip back.
  let finalWeights: Uint8Array;
  try {
    finalWeights = await aiNet.serialize();
  } catch (err) {
    aiNet.dispose();
    imitatorNet.dispose();
    postError(`Failed to serialize AI weights after self-play: ${errMessage(err)}`);
    return;
  }

  const finalReplay = replayBuffer.serialize();

  const doneMsg: SelfPlayResponse = {
    kind: 'done',
    aiWeights: finalWeights,
    replay: finalReplay,
    games,
    wins,
    losses,
    draws,
    examplesAdded,
    finalStep,
  };

  // Transfer ownership of the underlying ArrayBuffers for zero-copy transfer.
  self.postMessage(doneMsg, { transfer: [finalWeights.buffer, finalReplay.buffer] });

  aiNet.dispose();
  imitatorNet.dispose();
};

function postError(message: string): void {
  const msg: SelfPlayResponse = { kind: 'error', message };
  self.postMessage(msg);
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
