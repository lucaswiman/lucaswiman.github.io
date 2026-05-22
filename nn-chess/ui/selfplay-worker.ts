/**
 * Self-play Web Worker.
 *
 * Runs a batch of MCTS self-play games (opponent net vs. imitator net)
 * and then a handful of gradient steps on the opponent net off the main
 * thread. On a phone, doing this on the UI thread freezes the page for
 * tens of seconds; moving it here keeps the board interactive.
 *
 * Communicates with the main thread via the typed SelfPlayRequest /
 * SelfPlayResponse protocol defined in selfplay-protocol.ts.
 *
 * Imports ONLY from core/ modules and the shared protocol file — no
 * React, no DOM, no UI-only code. The worker reads no storage; the
 * caller hands it serialized weights + replay bytes and gets fresh
 * bytes back when the batch is done.
 */

import { ChessNet, deserialize } from '../core/nn/index.js';
import { policyFromNet } from '../core/opponent/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
} from '../core/training/index.js';
import {
  parseGymScenarios,
  runEndgameGym,
  runSelfPlayBatch,
} from '../core/selfplay/index.js';
import { fromFen, type GameState } from '../core/rules/index.js';
import type { SelfPlayRequest, SelfPlayResponse } from './selfplay-protocol.js';
import { installWorkerErrorHandlers, initBackend } from './worker-init.js';

// Forward any top-level error or unhandled rejection back through the
// message channel — `worker.onerror` on the parent side is often empty
// (especially on iOS Safari), so going via postMessage is the only way
// to surface a useful diagnostic.
installWorkerErrorHandlers();

// Load the WASM binaries from CDN so they don't bloat the app bundle.
// Fall back to CPU if WASM isn't available; if both fail, the worker
// still tries to run with the default tfjs backend.
const backendReady: Promise<void> = initBackend();

// ── Worker state ──────────────────────────────────────────────────────────────

let stopRequested = false;

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<SelfPlayRequest>) => {
  await backendReady;
  const req = event.data;

  if (req.kind === 'stop') {
    stopRequested = true;
    return;
  }

  if (req.kind === 'start-gym') {
    stopRequested = false;
    await handleGym(req);
    return;
  }

  if (req.kind !== 'start') return;

  stopRequested = false;

  const {
    opponentWeights,
    imitatorWeights,
    replay,
    imitatorReplay,
    replayCapacity,
    search,
    count,
    maxPlies,
    gradientSteps,
    gradientBatchSize,
    temperaturePlies,
    temperature,
    startingFens,
  } = req;

  // Pre-parse the curriculum pool once so we don't re-tokenize the same
  // FEN every game. A bad FEN here aborts the batch — there's no
  // graceful "skip this position" because we'd be sampling something
  // the UI was confident enough to send us.
  let startingStates: GameState[] | null = null;
  if (startingFens && startingFens.length > 0) {
    try {
      startingStates = startingFens.map(fromFen);
    } catch (err) {
      postError(`Failed to parse curriculum FEN: ${errMessage(err)}`);
      return;
    }
  }

  // Deserialize both networks.
  let opponentNet: ChessNet;
  let imitatorNet: ChessNet;
  try {
    opponentNet = await deserialize(opponentWeights);
  } catch (err) {
    postError(`Failed to deserialize opponent weights: ${errMessage(err)}`);
    return;
  }
  try {
    imitatorNet = await deserialize(imitatorWeights);
  } catch (err) {
    opponentNet.dispose();
    postError(`Failed to deserialize imitator weights: ${errMessage(err)}`);
    return;
  }

  const opponentReplayBuffer =
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
    // Phase 1: train the imitator on its own replay buffer so the
    // opponent's self-play stand-in reflects the user's most recent
    // moves before any games are generated. Skip silently when the
    // buffer is too small — mirrors PlayerImitator.trainStep's own
    // contract.
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
            illegalLoss: stepLosses.illegalLoss,
            totalLoss: stepLosses.totalLoss,
          };
          self.postMessage(msg);
        }
      }
    }

    // Build MCTS policies AFTER imitator training so the games use the
    // updated weights.
    const opponentPolicy = policyFromNet(opponentNet);
    const playerPolicy = policyFromNet(imitatorNet);

    // Phase 2: run self-play games.
    const batchResult = await runSelfPlayBatch({
      opponentPolicy,
      playerPolicy,
      search,
      count,
      maxPlies,
      temperaturePlies,
      temperature,
      startingStateForGame: startingStates
        ? () => startingStates![Math.floor(Math.random() * startingStates!.length)]
        : undefined,
      signal: controller.signal,
      onGame: (idx, info) => {
        const msg: SelfPlayResponse = {
          kind: 'game',
          index: idx,
          opponentColor: info.opponentColor,
          opponentResult: info.opponentResult,
          plies: info.plies,
        };
        self.postMessage(msg);
      },
      onPly: (gameIndex, info) => {
        const msg: SelfPlayResponse = {
          kind: 'ply',
          gameIndex,
          opponentColor: info.opponentColor,
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
    examplesAdded = batchResult.opponentTrainingExamples.length;

    for (const ex of batchResult.opponentTrainingExamples) {
      opponentReplayBuffer.add(ex);
    }

    // Phase 3: run gradient steps on the opponent's (now enlarged)
    // replay buffer. Lenient when the buffer is still small — same
    // rationale as the imitator pass above.
    if (opponentReplayBuffer.size() >= gradientBatchSize && gradientSteps > 0) {
      const reportEvery = Math.max(1, Math.floor(gradientSteps / 10));
      for (let step = 0; step < gradientSteps; step++) {
        if (stopRequested) break;

        const batch = opponentReplayBuffer.sample(gradientBatchSize);
        const stepLosses = await opponentNet.trainBatch(batch);
        finalStep = step + 1;

        if ((step + 1) % reportEvery === 0 || step === gradientSteps - 1) {
          const msg: SelfPlayResponse = {
            kind: 'step',
            phase: 'opponent',
            stepIndex: step + 1,
            valueLoss: stepLosses.valueLoss,
            policyLoss: stepLosses.policyLoss,
            illegalLoss: stepLosses.illegalLoss,
            totalLoss: stepLosses.totalLoss,
          };
          self.postMessage(msg);
        }
      }
    }
  } catch (err) {
    clearInterval(stopPoll);
    opponentNet.dispose();
    imitatorNet.dispose();
    postError(`Self-play failed: ${errMessage(err)}`);
    return;
  }

  clearInterval(stopPoll);

  // Serialize fresh weights + replay buffers for the round-trip back.
  let finalOpponentWeights: Uint8Array;
  let finalImitatorWeights: Uint8Array;
  try {
    finalOpponentWeights = await opponentNet.serialize();
    finalImitatorWeights = await imitatorNet.serialize();
  } catch (err) {
    opponentNet.dispose();
    imitatorNet.dispose();
    postError(`Failed to serialize weights after self-play: ${errMessage(err)}`);
    return;
  }

  const finalReplay = opponentReplayBuffer.serialize();
  const finalImitatorReplay = imitatorReplayBuffer.serialize();

  const doneMsg: SelfPlayResponse = {
    kind: 'done',
    opponentWeights: finalOpponentWeights,
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
      finalOpponentWeights.buffer,
      finalImitatorWeights.buffer,
      finalReplay.buffer,
      finalImitatorReplay.buffer,
    ],
  });

  opponentNet.dispose();
  imitatorNet.dispose();
};

function postError(message: string): void {
  const msg: SelfPlayResponse = { kind: 'error', message };
  self.postMessage(msg);
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ── Gym handler ──────────────────────────────────────────────────────────────
//
// Mirrors `start` (deserialize → run → reserialize) but routes through
// `runEndgameGym` instead of `runSelfPlayBatch`. Two notable differences
// from the regular self-play path:
//
//   1. The imitator pre-training pass is skipped. Its current weights
//      are good enough to defend a forced-mate position; spending
//      compute on the imitator before each gym run delays the gradient
//      steps the opponent actually needs.
//   2. We track scenario-level outcomes (`scenario-done` messages and
//      `scenariosConverted` in the `done` payload) so the UI can show
//      progress per scenario instead of per individual game.

async function handleGym(
  req: Extract<SelfPlayRequest, { kind: 'start-gym' }>,
): Promise<void> {
  const {
    opponentWeights,
    imitatorWeights,
    replay,
    imitatorReplay,
    replayCapacity,
    search,
    scenarios,
    maxGamesPerScenario,
    maxPlies,
    gradientSteps,
    gradientBatchSize,
    temperaturePlies,
    temperature,
  } = req;

  if (scenarios.length === 0) {
    postError('Gym run requested with an empty scenario list.');
    return;
  }

  // Validate + parse scenarios up front so we fail fast on bad FENs.
  let parsedScenarios: ReturnType<typeof parseGymScenarios>;
  try {
    parsedScenarios = parseGymScenarios(scenarios);
  } catch (err) {
    postError(`Gym scenario parse failed: ${errMessage(err)}`);
    return;
  }

  let opponentNet: ChessNet;
  let imitatorNet: ChessNet;
  try {
    opponentNet = await deserialize(opponentWeights);
  } catch (err) {
    postError(`Failed to deserialize opponent weights: ${errMessage(err)}`);
    return;
  }
  try {
    imitatorNet = await deserialize(imitatorWeights);
  } catch (err) {
    opponentNet.dispose();
    postError(`Failed to deserialize imitator weights: ${errMessage(err)}`);
    return;
  }

  const opponentReplayBuffer =
    replay.length > 0
      ? deserializeReplayBuffer(replay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  // The imitator's replay buffer is round-tripped untouched — the gym
  // doesn't train the imitator, but the buffer still has to come back
  // to the main thread so the UI's "imitator buffer" counter stays
  // accurate after the run.
  const imitatorReplayBuffer =
    imitatorReplay.length > 0
      ? deserializeReplayBuffer(imitatorReplay, replayCapacity)
      : createReplayBuffer(replayCapacity);

  const controller = new AbortController();
  const stopPoll = setInterval(() => {
    if (stopRequested) controller.abort();
  }, 50);

  let scenariosConverted = 0;
  let scenariosAttempted = 0;
  let totalWins = 0;
  let totalLosses = 0;
  let totalDraws = 0;
  let totalGames = 0;
  let examplesAdded = 0;
  let finalStep = 0;

  try {
    const opponentPolicy = policyFromNet(opponentNet);
    const playerPolicy = policyFromNet(imitatorNet);

    const gymResult = await runEndgameGym({
      scenarios: parsedScenarios,
      opponentPolicy,
      playerPolicy,
      search,
      maxGamesPerScenario,
      maxPlies,
      temperaturePlies,
      temperature,
      signal: controller.signal,
      onGame: (info) => {
        const msg: SelfPlayResponse = {
          kind: 'game',
          index: info.scenarioIdx * maxGamesPerScenario + info.gameInScenario,
          opponentColor: info.opponentColor,
          opponentResult: info.opponentResult,
          plies: info.plies,
          scenarioIdx: info.scenarioIdx,
          gameInScenario: info.gameInScenario,
        };
        self.postMessage(msg);
      },
      onScenarioDone: (info) => {
        const msg: SelfPlayResponse = {
          kind: 'scenario-done',
          scenarioIdx: info.scenarioIdx,
          scenarioId: parsedScenarios[info.scenarioIdx].id,
          converted: info.converted,
          gamesPlayed: info.gamesPlayed,
        };
        self.postMessage(msg);
      },
      onPly: (info) => {
        const msg: SelfPlayResponse = {
          kind: 'ply',
          gameIndex: info.scenarioIdx * maxGamesPerScenario + info.gameInScenario,
          opponentColor: info.opponentColor,
          plyIndex: info.plyIndex,
          fen: info.fen,
          lastMove: info.lastMove,
        };
        self.postMessage(msg);
      },
    });

    scenariosConverted = gymResult.scenariosConverted;
    scenariosAttempted = gymResult.scenariosAttempted;
    totalWins = gymResult.totalWins;
    totalLosses = gymResult.totalLosses;
    totalDraws = gymResult.totalDraws;
    totalGames = gymResult.totalGames;
    examplesAdded = gymResult.opponentTrainingExamples.length;

    for (const ex of gymResult.opponentTrainingExamples) opponentReplayBuffer.add(ex);

    // Gradient steps on the opponent net using the (now enlarged) replay
    // buffer. Same loop as the regular self-play flow's phase 3.
    if (opponentReplayBuffer.size() >= gradientBatchSize && gradientSteps > 0) {
      const reportEvery = Math.max(1, Math.floor(gradientSteps / 10));
      for (let step = 0; step < gradientSteps; step++) {
        if (stopRequested) break;
        const batch = opponentReplayBuffer.sample(gradientBatchSize);
        const stepLosses = await opponentNet.trainBatch(batch);
        finalStep = step + 1;
        if ((step + 1) % reportEvery === 0 || step === gradientSteps - 1) {
          const msg: SelfPlayResponse = {
            kind: 'step',
            phase: 'opponent',
            stepIndex: step + 1,
            valueLoss: stepLosses.valueLoss,
            policyLoss: stepLosses.policyLoss,
            illegalLoss: stepLosses.illegalLoss,
            totalLoss: stepLosses.totalLoss,
          };
          self.postMessage(msg);
        }
      }
    }
  } catch (err) {
    clearInterval(stopPoll);
    opponentNet.dispose();
    imitatorNet.dispose();
    postError(`Gym run failed: ${errMessage(err)}`);
    return;
  }

  clearInterval(stopPoll);

  let finalOpponentWeights: Uint8Array;
  let finalImitatorWeights: Uint8Array;
  try {
    finalOpponentWeights = await opponentNet.serialize();
    finalImitatorWeights = await imitatorNet.serialize();
  } catch (err) {
    opponentNet.dispose();
    imitatorNet.dispose();
    postError(`Failed to serialize weights after gym run: ${errMessage(err)}`);
    return;
  }

  const finalReplay = opponentReplayBuffer.serialize();
  const finalImitatorReplay = imitatorReplayBuffer.serialize();

  const doneMsg: SelfPlayResponse = {
    kind: 'done',
    opponentWeights: finalOpponentWeights,
    imitatorWeights: finalImitatorWeights,
    replay: finalReplay,
    imitatorReplay: finalImitatorReplay,
    games: totalGames,
    wins: totalWins,
    losses: totalLosses,
    draws: totalDraws,
    examplesAdded,
    finalStep,
    imitatorFinalStep: 0,
    scenariosConverted,
    scenariosAttempted,
  };

  self.postMessage(doneMsg, {
    transfer: [
      finalOpponentWeights.buffer,
      finalImitatorWeights.buffer,
      finalReplay.buffer,
      finalImitatorReplay.buffer,
    ],
  });

  opponentNet.dispose();
  imitatorNet.dispose();
}
