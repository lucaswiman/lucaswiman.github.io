// SelfPlayPanel — controls for running self-play games between the AI
// and the player imitator. Each batch trains the imitator on the user's
// recorded moves, generates AlphaZero-style examples by playing the
// updated imitator, then takes gradient steps on the AI net.
//
// The heavy lifting (MCTS + NN forward/backward) runs in a Web Worker
// via SelfPlayClient; this panel is pure UI plumbing. The same client
// is also driven by Game.tsx after each finished real game (auto-batch)
// — Game.tsx pokes us through the imperative `triggerAutoBatch` handle
// so there's exactly one worker code path in the app.
//
// "Until AI beats imitator" mode keeps firing batches in a loop and
// checks the AI's non-loss rate against the imitator over a sliding
// window after each batch; the per-batch worker call is unchanged.
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
import { IMITATOR_STORAGE_KEYS } from '../core/imitator/index.js';
import { SelfPlayClient } from './selfplay-client.js';
import {
  SELFPLAY_SEARCH,
  SELFPLAY_BATCH_SIZE,
  SELFPLAY_MAX_PLIES,
  SELFPLAY_GRADIENT_STEPS,
  SELFPLAY_GRADIENT_BATCH,
  REPLAY_CAPACITY,
} from './agent-config.js';

/**
 * Hard cap on the "Until AI beats imitator" loop so a stuck run can't
 * keep training forever in the background. Tuned for one user-session's
 * worth of patience: at ~5s per game and 3 games per batch that's
 * roughly 15 minutes of wall time before we give up.
 */
const UNTIL_BEATS_HARD_CAP = 200;

type Mode = 'fixed' | 'until-beats';

type AiResult = 'win' | 'loss' | 'draw';

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
  phase: 'imitator' | 'ai';
  stepIndex: number;
  totalLoss: number;
}

interface RunBatchOutcome {
  results: AiResult[];
}

export const SelfPlayPanel = forwardRef<SelfPlayPanelHandle, SelfPlayPanelProps>(
  function SelfPlayPanel(
    { agent, imitator, storage, onAgentRefresh, autoEnabled, onAutoEnabledChange },
    ref,
  ) {
    const [mode, setMode] = useState<Mode>('fixed');
    const [batchSize, setBatchSize] = useState(SELFPLAY_BATCH_SIZE);
    const [thresholdPct, setThresholdPct] = useState(60);
    const [windowSize, setWindowSize] = useState(20);
    const [running, setRunning] = useState(false);
    const [statusMsg, setStatusMsg] = useState<string | null>(null);
    const [batch, setBatch] = useState<BatchStatus | null>(null);
    const [stepStatus, setStepStatus] = useState<StepStatus | null>(null);
    // Sliding-window readout for the "Until AI beats imitator" loop.
    // Lives in component state so the panel re-renders as each batch
    // lands; the loop itself reads from `loopRef` to avoid stale closures.
    const [loopStats, setLoopStats] = useState<{
      total: number;
      windowResults: AiResult[];
    } | null>(null);
    // Mirrors loopActiveRef so the buttons can stay disabled (Run) and
    // enabled (Stop) for the full duration of the loop, not just each
    // individual batch.
    const [loopActive, setLoopActive] = useState(false);

    const clientRef = useRef<SelfPlayClient | null>(null);
    // Kept in a ref so the imperative handle reads the live value without
    // re-binding the handle on every render.
    const runningRef = useRef(false);
    runningRef.current = running;
    // Loop-level cancel flag: distinct from the worker's per-batch stop
    // because the loop must also short-circuit between batches.
    const loopStopRef = useRef(false);
    // True for the lifetime of the "Until" loop, including the windows
    // between individual batches. See runUntilBeatsLoop for rationale.
    const loopActiveRef = useRef(false);
    // Live ring buffer of AI outcomes seen during the current "Until"
    // loop. Lives in a ref so the loop body can read the up-to-date
    // window without depending on React state batching.
    const loopWindowRef = useRef<AiResult[]>([]);
    const loopTotalRef = useRef(0);

    useEffect(() => {
      return () => {
        clientRef.current?.stop();
        clientRef.current = null;
      };
    }, []);

    const runBatch = useCallback(
      async (gamesToPlay: number): Promise<RunBatchOutcome | null> => {
        if (runningRef.current || !agent || !imitator) return null;

        setRunning(true);
        runningRef.current = true;
        setStatusMsg(null);
        setStepStatus(null);
        setBatch({ played: 0, wins: 0, losses: 0, draws: 0, examples: 0 });

        let aiWeights: Uint8Array;
        let imitatorWeights: Uint8Array;
        let replay: Uint8Array;
        let imitatorReplay: Uint8Array;
        try {
          [aiWeights, imitatorWeights, replay, imitatorReplay] = await Promise.all([
            agent.net.serialize(),
            imitator.net.serialize(),
            Promise.resolve(agent.replayBuffer.serialize()),
            Promise.resolve(imitator.replayBuffer.serialize()),
          ]);
        } catch (err) {
          setStatusMsg(`Self-play setup failed: ${err instanceof Error ? err.message : String(err)}`);
          setRunning(false);
          runningRef.current = false;
          return null;
        }

        if (!clientRef.current) {
          clientRef.current = new SelfPlayClient();
        }

        const collectedResults: AiResult[] = [];

        return new Promise<RunBatchOutcome | null>((resolve) => {
          clientRef.current!.start(
            {
              aiWeights,
              imitatorWeights,
              replay,
              imitatorReplay,
              replayCapacity: REPLAY_CAPACITY,
              search: SELFPLAY_SEARCH,
              count: gamesToPlay,
              maxPlies: SELFPLAY_MAX_PLIES,
              gradientSteps: SELFPLAY_GRADIENT_STEPS,
              gradientBatchSize: SELFPLAY_GRADIENT_BATCH,
            },
            (game) => {
              collectedResults.push(game.aiResult);
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
              setStepStatus({ phase: step.phase, stepIndex: step.stepIndex, totalLoss: step.totalLoss });
            },
            async (result) => {
              setBatch(prev =>
                prev === null ? null : { ...prev, examples: result.examplesAdded },
              );
              try {
                await Promise.all([
                  storage.put(AGENT_STORAGE_KEYS.WEIGHTS, result.aiWeights),
                  storage.put(AGENT_STORAGE_KEYS.REPLAY, result.replay),
                  storage.put(IMITATOR_STORAGE_KEYS.WEIGHTS, result.imitatorWeights),
                  storage.put(IMITATOR_STORAGE_KEYS.REPLAY, result.imitatorReplay),
                ]);
                await onAgentRefresh();
                setStatusMsg(
                  `Self-play done — ${result.games} game(s): ` +
                  `AI ${result.wins}W ${result.losses}L ${result.draws}D, ` +
                  `${result.examplesAdded} new examples, ` +
                  `AI ${result.finalStep} / imitator ${result.imitatorFinalStep} grad step(s).`,
                );
              } catch (err) {
                setStatusMsg(
                  `Self-play done but save failed: ${err instanceof Error ? err.message : String(err)}`,
                );
              } finally {
                setRunning(false);
                runningRef.current = false;
                resolve({ results: collectedResults });
              }
            },
            (msg) => {
              setStatusMsg(`Self-play failed: ${msg}`);
              setRunning(false);
              runningRef.current = false;
              resolve(null);
            },
          );
        });
      },
      [agent, imitator, storage, onAgentRefresh],
    );

    // "Until AI beats imitator" loop: fire batches back-to-back and stop
    // when the rolling non-loss rate clears the threshold, the user hits
    // Stop, or we hit the hard safety cap.
    //
    // We park `loopActiveRef` = true around the whole loop (not just
    // each batch) so the imperative `isRunning` getter — which Game.tsx
    // checks before firing a post-game auto-batch — stays truthy even
    // in the brief windows between batches when runningRef has flipped
    // back to false.
    const runUntilBeatsLoop = useCallback(async () => {
      if (!agent || !imitator) return;
      if (loopActiveRef.current) return;
      loopActiveRef.current = true;
      setLoopActive(true);
      loopStopRef.current = false;
      loopWindowRef.current = [];
      loopTotalRef.current = 0;
      setLoopStats({ total: 0, windowResults: [] });

      const threshold = thresholdPct / 100;
      const win = windowSize;

      try {
        while (!loopStopRef.current) {
          if (loopTotalRef.current >= UNTIL_BEATS_HARD_CAP) {
            setStatusMsg(
              `Stopped: AI did not reach ${thresholdPct}% non-loss over ${win} games ` +
              `within the ${UNTIL_BEATS_HARD_CAP}-game cap.`,
            );
            break;
          }

          // Cap the batch so we don't blow past the hard cap.
          const remaining = UNTIL_BEATS_HARD_CAP - loopTotalRef.current;
          const gamesThisBatch = Math.max(1, Math.min(batchSize, remaining));

          const outcome = await runBatch(gamesThisBatch);
          if (outcome === null) {
            // runBatch already wrote a status message; just exit the loop.
            break;
          }

          for (const r of outcome.results) {
            loopWindowRef.current.push(r);
            if (loopWindowRef.current.length > win) loopWindowRef.current.shift();
            loopTotalRef.current += 1;
          }
          setLoopStats({
            total: loopTotalRef.current,
            windowResults: loopWindowRef.current.slice(),
          });

          if (loopStopRef.current) break;

          if (loopWindowRef.current.length >= win) {
            const nonLoss = loopWindowRef.current.filter(r => r !== 'loss').length;
            if (nonLoss / win >= threshold) {
              setStatusMsg(
                `Converged: AI non-loss ${nonLoss}/${win} (≥${thresholdPct}%) ` +
                `after ${loopTotalRef.current} game(s).`,
              );
              break;
            }
          }
        }
      } finally {
        loopStopRef.current = false;
        loopActiveRef.current = false;
        setLoopActive(false);
      }
    }, [agent, imitator, batchSize, thresholdPct, windowSize, runBatch]);

    const handleRun = useCallback(() => {
      if (mode === 'until-beats') {
        void runUntilBeatsLoop();
      } else {
        void runBatch(batchSize);
      }
    }, [mode, runBatch, runUntilBeatsLoop, batchSize]);

    const handleStop = useCallback(() => {
      // Both flags so the worker tears down its current batch AND the
      // outer loop short-circuits as soon as that batch resolves.
      loopStopRef.current = true;
      clientRef.current?.stop();
    }, []);

    useImperativeHandle(ref, () => ({
      triggerAutoBatch: async () => {
        // Skip the auto-batch entirely while the "Until" loop is active
        // — the loop is already pumping batches and a parallel one
        // would race for the worker.
        if (loopActiveRef.current) return;
        await runBatch(SELFPLAY_BATCH_SIZE);
      },
      get isRunning() {
        return runningRef.current || loopActiveRef.current;
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

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center', fontSize: '0.9em' }}>
            <strong>Mode:</strong>
            <label style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                name="selfplay-mode"
                value="fixed"
                checked={mode === 'fixed'}
                onChange={() => setMode('fixed')}
                disabled={running || loopActive}
              />
              {' '}Fixed batch size
            </label>
            <label style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                name="selfplay-mode"
                value="until-beats"
                checked={mode === 'until-beats'}
                onChange={() => setMode('until-beats')}
                disabled={running || loopActive}
              />
              {' '}Until AI beats imitator
            </label>
          </div>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
            {mode === 'fixed' ? (
              <label>
                Games:{' '}
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={batchSize}
                  onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
                  style={{ width: '70px' }}
                  disabled={running || loopActive}
                />
              </label>
            ) : (
              <>
                <label>
                  Target ≥{' '}
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={thresholdPct}
                    onChange={e => setThresholdPct(Math.max(1, Math.min(100, parseInt(e.target.value, 10) || 1)))}
                    style={{ width: '60px' }}
                    disabled={running || loopActive}
                  />
                  %
                </label>
                <label>
                  over last{' '}
                  <input
                    type="number"
                    min={1}
                    max={UNTIL_BEATS_HARD_CAP}
                    value={windowSize}
                    onChange={e => setWindowSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
                    style={{ width: '60px' }}
                    disabled={running || loopActive}
                  />
                  {' '}games
                </label>
                <span style={{ fontSize: '0.85em', color: '#666' }}>
                  (cap: {UNTIL_BEATS_HARD_CAP})
                </span>
              </>
            )}
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
              disabled={running || loopActive || agent === null || imitator === null}
              style={{ padding: '5px 14px', cursor: (running || loopActive) ? 'not-allowed' : 'pointer' }}
            >
              {mode === 'until-beats' ? 'Run until AI beats imitator' : 'Run self-play batch'}
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={!running && !loopActive}
              style={{ padding: '5px 14px', cursor: (running || loopActive) ? 'pointer' : 'not-allowed' }}
            >
              Stop
            </button>
          </div>

          {batch !== null && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Game {batch.played} —{' '}
              AI: {batch.wins}W {batch.losses}L {batch.draws}D
              {batch.examples > 0 && ` — ${batch.examples} examples`}
            </div>
          )}

          {mode === 'until-beats' && loopStats !== null && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Loop: {loopStats.total} game(s) total, last {loopStats.windowResults.length} / {windowSize} —{' '}
              non-loss {loopStats.windowResults.filter(r => r !== 'loss').length}/{loopStats.windowResults.length}
            </div>
          )}

          {running && stepStatus && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              {stepStatus.phase === 'imitator' ? 'Imitator' : 'AI'} grad step{' '}
              {stepStatus.stepIndex} / {SELFPLAY_GRADIENT_STEPS} —{' '}
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
