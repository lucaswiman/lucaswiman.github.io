/**
 * Maia-2 inference Web Worker.
 *
 * Loads the Maia-2 weights once (passed in via `init`), wraps them in a
 * `MaiaEngine`, and replies to `select-move` requests with the chosen
 * move plus its probability distribution + value-head readout.
 *
 * Errors are forwarded as `{kind: 'error', message}` so the main thread
 * can render them; some platforms (notably iOS Safari) strip the
 * `ErrorEvent` that `worker.onerror` receives, so going through the
 * message channel is the only reliable diagnostic path.
 */

import { MaiaEngine, loadMaiaWeights, maiaPolicy } from '../core/maia/index.js';
import { fromFen } from '../core/rules/index.js';
import { search } from '../core/mcts/index.js';
import { installWorkerErrorHandlers, initBackend } from './worker-init.js';
import type { MaiaRequest, MaiaResponse } from './maia-protocol.js';

installWorkerErrorHandlers();

const backendReady = initBackend();

let engine: MaiaEngine | null = null;

function send(msg: MaiaResponse): void {
  self.postMessage(msg);
}

self.addEventListener('message', async (event: MessageEvent<MaiaRequest>) => {
  const msg = event.data;
  try {
    if (msg.kind === 'init') {
      await backendReady;
      // Dispose any previous engine so we don't leak tensors on
      // re-initialization (e.g. a manifest version bump).
      engine?.dispose();
      const blobView = new Uint8Array(msg.blob);
      const weights = loadMaiaWeights(blobView, msg.manifest);
      engine = new MaiaEngine(weights);
      send({ kind: 'ready' });
      return;
    }
    if (msg.kind === 'select-move') {
      if (engine === null) {
        send({
          kind: 'error',
          requestId: msg.requestId,
          message: 'Maia worker not initialized — send {kind: "init"} first.',
        });
        return;
      }
      await backendReady;
      const state = fromFen(msg.fen);
      const result = engine.selectMove(state, {
        elo: msg.elo,
        oppoElo: msg.oppoElo,
        temperature: msg.temperature ?? 0,
      });
      const moveProbs: Record<string, number> = {};
      for (const [m, p] of result.moveProbs) moveProbs[m] = p;
      send({
        kind: 'move',
        requestId: msg.requestId,
        move: result.move,
        value: result.rootValue,
        moveProbs,
      });
      return;
    }
    if (msg.kind === 'search-move') {
      if (engine === null) {
        send({
          kind: 'error',
          requestId: msg.requestId,
          message: 'Maia worker not initialized — send {kind: "init"} first.',
        });
        return;
      }
      await backendReady;
      const state = fromFen(msg.fen);
      // Each MCTS sim costs one Maia forward pass. We run search()
      // inside the worker so the main thread isn't blocked round-
      // tripping per-sim through postMessage, and so we don't pay the
      // serialization cost of crossing the worker boundary 16+ times
      // per move.
      const policy = maiaPolicy(engine, msg.elo, msg.oppoElo);
      const result = await search(state, policy, {
        simulations: msg.simulations,
        cPuct: msg.cPuct,
      });
      const moveProbs: Record<string, number> = {};
      for (const [m, p] of result.priorPolicy) moveProbs[m] = p;
      const visitCounts: Record<string, number> = {};
      for (const [m, n] of result.visitCounts) visitCounts[m] = n;
      send({
        kind: 'move',
        requestId: msg.requestId,
        move: result.bestMove,
        value: result.rootValue,
        moveProbs,
        visitCounts,
      });
      return;
    }
  } catch (err) {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    const requestId =
      msg.kind === 'select-move' || msg.kind === 'search-move'
        ? msg.requestId
        : undefined;
    send({ kind: 'error', requestId, message });
  }
});
