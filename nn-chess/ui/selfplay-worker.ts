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
    imitatorReplay,
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

  const aiReplayBuffer =
    replay.length > 0
      ? deserializeReplayBuffer(replay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  const imitatorReplayBuffer =
    imitatorReplay.length > 0
      ? deserializeReplayBuffer(imitatorReplay, replayCapacity)
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
  let imitatorFinalStep = 0;

  try {
    // Phase 1: train the imitator on its own replay buffer so the AI's
    // self-play opponent reflects the user's most recent moves before any
    // games are generated. Skip silently when the buffer is too small —
    // mirrors PlayerImitator.trainStep's own contract.
    if (imitatorReplayBuffer.size() >= gradientBatchSize && gradientSteps > 0) {
      const reportEvery = Math.max(1, Math.floor(gradientSteps / 10));
      for (let step = 0; step < gradientSteps; step++) {
        if (stopRequested) break;

        const batch = imitatorReplayBuffer.sample(gradientBatchSize);
        const stepLosses = await imitatorNet.trainBatch(batch);
        imitatorFinalStep = step + 1;

        if ((step + 1) % reportEvery === 0 || step === gradientSteps - 1) {
          const msg: SelfPlayResponse = {
            kind: 'step',
            phase: 'imitator',
            stepIndex: step + 1,
            valueLoss: stepLosses.valueLoss,
            policyLoss: stepLosses.policyLoss,
            totalLoss: stepLosses.totalLoss,
          };
          self.postMessage(msg);
        }
      }
    }

    // Build MCTS policies AFTER imitator training so the games use the
    // updated weights.
    const aiPolicy = policyFromNet(aiNet);
    const playerPolicy = policyFromNet(imitatorNet);

    // Phase 2: run self-play games.
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
      onPly: (gameIndex, info) => {
        const msg: SelfPlayResponse = {
          kind: 'ply',
          gameIndex,
          aiColor: info.aiColor,
          plyIndex: info.plyIndex,
          fen: info.fen,
          lastMove: info.lastMove,
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
      aiReplayBuffer.add(ex);
    }

    // Phase 3: run gradient steps on the AI's (now enlarged) replay
    // buffer. Lenient when the buffer is still small — same rationale
    // as the imitator pass above.
    if (aiReplayBuffer.size() >= gradientBatchSize && gradientSteps > 0) {
      const reportEvery = Math.max(1, Math.floor(gradientSteps / 10));
      for (let step = 0; step < gradientSteps; step++) {
        if (stopRequested) break;

        const batch = aiReplayBuffer.sample(gradientBatchSize);
        const stepLosses = await aiNet.trainBatch(batch);
        finalStep = step + 1;

        if ((step + 1) % reportEvery === 0 || step === gradientSteps - 1) {
          const msg: SelfPlayResponse = {
            kind: 'step',
            phase: 'ai',
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

  // Serialize fresh weights + replay buffers for the round-trip back.
  let finalAiWeights: Uint8Array;
  let finalImitatorWeights: Uint8Array;
  try {
    finalAiWeights = await aiNet.serialize();
    finalImitatorWeights = await imitatorNet.serialize();
  } catch (err) {
    aiNet.dispose();
    imitatorNet.dispose();
    postError(`Failed to serialize weights after self-play: ${errMessage(err)}`);
    return;
  }

  const finalReplay = aiReplayBuffer.serialize();
  const finalImitatorReplay = imitatorReplayBuffer.serialize();

  const doneMsg: SelfPlayResponse = {
    kind: 'done',
    aiWeights: finalAiWeights,
    imitatorWeights: finalImitatorWeights,
    replay: finalReplay,
    imitatorReplay: finalImitatorReplay,
    games,
    wins,
    losses,
    draws,
    examplesAdded,
    finalStep,
    imitatorFinalStep,
  };

  // Transfer ownership of the underlying ArrayBuffers for zero-copy transfer.
  self.postMessage(doneMsg, {
    transfer: [
      finalAiWeights.buffer,
      finalImitatorWeights.buffer,
      finalReplay.buffer,
      finalImitatorReplay.buffer,
    ],
  });

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
