// SelfPlayPanel — controls for running self-play games between the AI
// and the player imitator. Each batch generates AlphaZero-style
// training examples that are added to the AI's replay buffer, then a
// few gradient steps are taken so the new data immediately influences
// the AI's evaluation.
//
// The heavy lifting (MCTS + NN forward/backward) runs in a Web Worker
// via SelfPlayClient; this panel is pure UI plumbing. The same client
// is also driven by Game.tsx after each finished real game (auto-batch)
// — Game.tsx pokes us through the imperative `triggerAutoBatch` handle
// so there's exactly one worker code path in the app.
//
// The Stop button aborts between games via the worker's `stop`
// message; abort between MCTS plies is handled inside the worker.

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import type { Agent } from '../core/agent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { AGENT_STORAGE_KEYS } from '../core/agent/index.js';
import { SelfPlayClient } from './selfplay-client.js';
import {
  SELFPLAY_SEARCH,
  SELFPLAY_BATCH_SIZE,
  SELFPLAY_MAX_PLIES,
  SELFPLAY_GRADIENT_STEPS,
  SELFPLAY_GRADIENT_BATCH,
  REPLAY_CAPACITY,
} from './agent-config.js';

export interface SelfPlayPanelProps {
  agent: Agent | null;
  imitator: PlayerImitator | null;
  storage: BlobStorage;
  /** Reload both agents from storage after a batch updates their weights. */
  onAgentRefresh: () => Promise<void>;
  autoEnabled: boolean;
  onAutoEnabledChange: (enabled: boolean) => void;
}

export interface SelfPlayPanelHandle {
  /**
   * Kick off a self-play batch using the panel's current settings. Returns
   * a promise that resolves once the batch is done (or rejects on error).
   * No-op if a batch is already running or the nets aren't loaded yet.
   */
  triggerAutoBatch(): Promise<void>;
  /** True while a batch is in flight. */
  readonly isRunning: boolean;
}

interface BatchStatus {
  played: number;
  wins: number;
  losses: number;
  draws: number;
  examples: number;
}

interface StepStatus {
  stepIndex: number;
  totalLoss: number;
}

export const SelfPlayPanel = forwardRef<SelfPlayPanelHandle, SelfPlayPanelProps>(
  function SelfPlayPanel(
    { agent, imitator, storage, onAgentRefresh, autoEnabled, onAutoEnabledChange },
    ref,
  ) {
    const [batchSize, setBatchSize] = useState(SELFPLAY_BATCH_SIZE);
    const [running, setRunning] = useState(false);
    const [statusMsg, setStatusMsg] = useState<string | null>(null);
    const [batch, setBatch] = useState<BatchStatus | null>(null);
    const [stepStatus, setStepStatus] = useState<StepStatus | null>(null);

    const clientRef = useRef<SelfPlayClient | null>(null);
    // Kept in a ref so the imperative handle reads the live value without
    // re-binding the handle on every render.
    const runningRef = useRef(false);
    runningRef.current = running;

    useEffect(() => {
      return () => {
        clientRef.current?.stop();
        clientRef.current = null;
      };
    }, []);

    const runBatch = useCallback(
      async (gamesToPlay: number): Promise<void> => {
        if (runningRef.current || !agent || !imitator) return;

        setRunning(true);
        runningRef.current = true;
        setStatusMsg(null);
        setStepStatus(null);
        setBatch({ played: 0, wins: 0, losses: 0, draws: 0, examples: 0 });

        let aiWeights: Uint8Array;
        let imitatorWeights: Uint8Array;
        let replay: Uint8Array;
        try {
          [aiWeights, imitatorWeights, replay] = await Promise.all([
            agent.net.serialize(),
            imitator.net.serialize(),
            Promise.resolve(agent.replayBuffer.serialize()),
          ]);
        } catch (err) {
          setStatusMsg(`Self-play setup failed: ${err instanceof Error ? err.message : String(err)}`);
          setRunning(false);
          runningRef.current = false;
          return;
        }

        if (!clientRef.current) {
          clientRef.current = new SelfPlayClient();
        }

        return new Promise<void>((resolve) => {
          clientRef.current!.start(
            {
              aiWeights,
              imitatorWeights,
              replay,
              replayCapacity: REPLAY_CAPACITY,
              search: SELFPLAY_SEARCH,
              count: gamesToPlay,
              maxPlies: SELFPLAY_MAX_PLIES,
              gradientSteps: SELFPLAY_GRADIENT_STEPS,
              gradientBatchSize: SELFPLAY_GRADIENT_BATCH,
            },
            (game) => {
              setBatch(prev => {
                const base = prev ?? { played: 0, wins: 0, losses: 0, draws: 0, examples: 0 };
                return {
                  played: game.index + 1,
                  wins: base.wins + (game.aiResult === 'win' ? 1 : 0),
                  losses: base.losses + (game.aiResult === 'loss' ? 1 : 0),
                  draws: base.draws + (game.aiResult === 'draw' ? 1 : 0),
                  examples: base.examples,
                };
              });
            },
            (step) => {
              setStepStatus({ stepIndex: step.stepIndex, totalLoss: step.totalLoss });
            },
            async (result) => {
              setBatch(prev =>
                prev === null ? null : { ...prev, examples: result.examplesAdded },
              );
              try {
                await Promise.all([
                  storage.put(AGENT_STORAGE_KEYS.WEIGHTS, result.aiWeights),
                  storage.put(AGENT_STORAGE_KEYS.REPLAY, result.replay),
                ]);
                await onAgentRefresh();
                setStatusMsg(
                  `Self-play done — ${result.games} game(s): ` +
                  `AI ${result.wins}W ${result.losses}L ${result.draws}D, ` +
                  `${result.examplesAdded} new examples, ${result.finalStep} grad step(s).`,
                );
              } catch (err) {
                setStatusMsg(
                  `Self-play done but save failed: ${err instanceof Error ? err.message : String(err)}`,
                );
              } finally {
                setRunning(false);
                runningRef.current = false;
                resolve();
              }
            },
            (msg) => {
              setStatusMsg(`Self-play failed: ${msg}`);
              setRunning(false);
              runningRef.current = false;
              resolve();
            },
          );
        });
      },
      [agent, imitator, storage, onAgentRefresh],
    );

    const handleRun = useCallback(() => {
      void runBatch(batchSize);
    }, [runBatch, batchSize]);

    const handleStop = useCallback(() => {
      clientRef.current?.stop();
    }, []);

    useImperativeHandle(ref, () => ({
      triggerAutoBatch: () => runBatch(SELFPLAY_BATCH_SIZE),
      get isRunning() {
        return runningRef.current;
      },
    }), [runBatch]);

    const bufferSize = imitator?.replayBuffer.size() ?? 0;

    return (
      <details
        style={{
          border: '1px solid #ccc',
          borderRadius: '6px',
          padding: '10px 14px',
          background: '#f9f9f9',
          color: '#1a1a1a',
          marginTop: '12px',
        }}
      >
        <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none' }}>
          Self-play training (AI vs. player imitator)
        </summary>

        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ fontSize: '0.9em', color: '#444' }}>
            The AI plays the player-imitator under MCTS and trains on the
            outcomes. The imitator is whatever pattern your moves have
            taught it so far — <strong>{bufferSize.toLocaleString()}</strong>{' '}
            user moves recorded. Runs in a background worker so the board
            stays responsive.
          </div>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
            <label>
              Games:{' '}
              <input
                type="number"
                min={1}
                max={50}
                value={batchSize}
                onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
                style={{ width: '70px' }}
                disabled={running}
              />
            </label>
            <span style={{ fontSize: '0.85em', color: '#666' }}>
              MCTS sims/move: {SELFPLAY_SEARCH.simulations}
            </span>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9em' }}>
            <input
              type="checkbox"
              checked={autoEnabled}
              onChange={e => onAutoEnabledChange(e.target.checked)}
            />
            Auto-run after each game
          </label>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button
              type="button"
              onClick={handleRun}
              disabled={running || agent === null || imitator === null}
              style={{ padding: '5px 14px', cursor: running ? 'not-allowed' : 'pointer' }}
            >
              Run self-play batch
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={!running}
              style={{ padding: '5px 14px', cursor: running ? 'pointer' : 'not-allowed' }}
            >
              Stop
            </button>
          </div>

          {batch !== null && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Game {batch.played} / {batchSize} —{' '}
              AI: {batch.wins}W {batch.losses}L {batch.draws}D
              {batch.examples > 0 && ` — ${batch.examples} examples`}
            </div>
          )}

          {running && stepStatus && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Grad step {stepStatus.stepIndex} / {SELFPLAY_GRADIENT_STEPS} —{' '}
              loss: {stepStatus.totalLoss.toFixed(4)}
            </div>
          )}

          {statusMsg && (
            <div style={{ fontSize: '0.9em', color: '#333' }}>{statusMsg}</div>
          )}
        </div>
      </details>
    );
  },
);
