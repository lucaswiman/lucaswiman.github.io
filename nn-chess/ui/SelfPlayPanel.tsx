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
// "Until AI wins a game" mode keeps firing batches in a loop and stops
// as soon as the AI wins at least one game against the imitator (or the
// user clicks Stop, or we hit the hard cap). The per-batch worker call
// is unchanged; the loop logic lives in this panel.
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
import { fromFen, type GameState } from '../core/rules/index.js';
import Board from './Board.js';
import { SelfPlayClient } from './selfplay-client.js';
import {
  SELFPLAY_SEARCH,
  SELFPLAY_BATCH_SIZE,
  SELFPLAY_MAX_PLIES,
  SELFPLAY_GRADIENT_STEPS,
  SELFPLAY_GRADIENT_BATCH,
  SELFPLAY_TEMPERATURE_PLIES,
  SELFPLAY_TEMPERATURE,
  REPLAY_CAPACITY,
} from './agent-config.js';
import { ENDGAME_PRESETS } from './endgame-presets.js';
import { LICHESS_CATEGORIES } from './lichess-endgames.js';

/**
 * Hard cap on the "Until AI wins a game" loop so a stuck run can't keep
 * training forever in the background. Tuned for one user-session's
 * worth of patience: at ~5s per game and 3 games per batch that's
 * roughly 15 minutes of wall time before we give up.
 */
const UNTIL_WIN_HARD_CAP = 200;

type Mode = 'fixed' | 'until-win';

type AiResult = 'win' | 'loss' | 'draw';

/**
 * Curriculum dropdown options. `default` keeps the existing
 * full-game-from-startpos behaviour; the other entries each describe a
 * pool of starting FENs that the worker samples uniformly at random
 * per game.
 *
 * `endgame-presets` is the small textbook set (ladder mate, KQ-K, etc.)
 * — short, forcing, easy ±1 outcomes for the value head. Each
 * `lichess:<id>` option is one Lichess CC0 category (mateIn1-5, rook
 * endgames, pawn endgames, ...), all of which give the value head
 * decisive games inside the ply cap that pure startpos play almost
 * never produces with a cold-start net.
 */
interface CurriculumOption {
  id: string;
  label: string;
  fens: readonly string[];
}

function buildCurriculumOptions(): CurriculumOption[] {
  const presets: CurriculumOption = {
    id: 'endgame-presets',
    label: `Endgame textbook presets (${ENDGAME_PRESETS.length})`,
    fens: ENDGAME_PRESETS.map(p => p.fen),
  };
  const lichess: CurriculumOption[] = LICHESS_CATEGORIES.map(c => ({
    id: `lichess:${c.id}`,
    label: `Lichess: ${c.label} (${c.puzzles.length})`,
    fens: c.puzzles.map(p => p.fen),
  }));
  return [presets, ...lichess];
}

const CURRICULUM_OPTIONS = buildCurriculumOptions();
const CURRICULUM_OPTIONS_BY_ID = new Map(CURRICULUM_OPTIONS.map(o => [o.id, o] as const));

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

interface LivePly {
  gameIndex: number;
  aiColor: 'w' | 'b';
  plyIndex: number;
  state: GameState;
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
    // 'default' means "play full chess from startpos"; any other value
    // selects a curriculum pool (see CURRICULUM_OPTIONS).
    const [curriculumId, setCurriculumId] = useState<string>('default');
    const [running, setRunning] = useState(false);
    const [statusMsg, setStatusMsg] = useState<string | null>(null);
    const [batch, setBatch] = useState<BatchStatus | null>(null);
    const [stepStatus, setStepStatus] = useState<StepStatus | null>(null);
    // Live position rendered by <Board> while a batch is in flight. Set
    // by the worker's `ply` callback after every move; cleared when a
    // batch fully finishes so the board doesn't linger on a stale game.
    const [livePly, setLivePly] = useState<LivePly | null>(null);
    // Running totals for the "Until AI wins" loop. Lives in component
    // state so the panel re-renders as each batch lands; the loop itself
    // reads from refs to avoid stale closures.
    const [loopStats, setLoopStats] = useState<{
      total: number;
      wins: number;
      losses: number;
      draws: number;
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
    // between individual batches. See runUntilWinLoop for rationale.
    const loopActiveRef = useRef(false);
    const loopTotalRef = useRef(0);
    const loopWinsRef = useRef(0);
    const loopLossesRef = useRef(0);
    const loopDrawsRef = useRef(0);

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
        setLivePly(null);
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

        // Resolve the curriculum selection to a starting-FEN pool. We
        // snapshot the array here (not inside the worker callback) so
        // toggling the dropdown mid-batch can't accidentally swap
        // pools partway through.
        const curriculum = CURRICULUM_OPTIONS_BY_ID.get(curriculumId);
        const startingFens = curriculum ? Array.from(curriculum.fens) : undefined;

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
              temperaturePlies: SELFPLAY_TEMPERATURE_PLIES,
              temperature: SELFPLAY_TEMPERATURE,
              startingFens,
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
                setLivePly(null);
                resolve({ results: collectedResults });
              }
            },
            (msg) => {
              setStatusMsg(`Self-play failed: ${msg}`);
              setRunning(false);
              runningRef.current = false;
              setLivePly(null);
              resolve(null);
            },
            (ply) => {
              // Parse the FEN once per ply on the main thread. With ~80–120
              // plies per game and batches of ~3 games this is negligible
              // compared to MCTS, and keeps the Board's controlled state
              // type the same as everywhere else in the app.
              try {
                setLivePly({
                  gameIndex: ply.gameIndex,
                  aiColor: ply.aiColor,
                  plyIndex: ply.plyIndex,
                  state: fromFen(ply.fen),
                });
              } catch {
                // A bad FEN here would only break the live preview; the
                // training side already has its own copy of the state.
                // Swallow rather than abort the batch.
              }
            },
          );
        });
      },
      [agent, imitator, storage, onAgentRefresh, curriculumId],
    );

    // "Until AI wins a game" loop: fire batches back-to-back and stop
    // the moment the AI wins any game, the user hits Stop, or we hit
    // the hard safety cap.
    //
    // We park `loopActiveRef` = true around the whole loop (not just
    // each batch) so the imperative `isRunning` getter — which Game.tsx
    // checks before firing a post-game auto-batch — stays truthy even
    // in the brief windows between batches when runningRef has flipped
    // back to false.
    const runUntilWinLoop = useCallback(async () => {
      if (!agent || !imitator) return;
      if (loopActiveRef.current) return;
      loopActiveRef.current = true;
      setLoopActive(true);
      loopStopRef.current = false;
      loopTotalRef.current = 0;
      loopWinsRef.current = 0;
      loopLossesRef.current = 0;
      loopDrawsRef.current = 0;
      setLoopStats({ total: 0, wins: 0, losses: 0, draws: 0 });

      try {
        while (!loopStopRef.current) {
          if (loopTotalRef.current >= UNTIL_WIN_HARD_CAP) {
            setStatusMsg(
              `Stopped: AI did not win a game within the ${UNTIL_WIN_HARD_CAP}-game cap.`,
            );
            break;
          }

          // Cap the batch so we don't blow past the hard cap.
          const remaining = UNTIL_WIN_HARD_CAP - loopTotalRef.current;
          const gamesThisBatch = Math.max(1, Math.min(batchSize, remaining));

          const outcome = await runBatch(gamesThisBatch);
          if (outcome === null) {
            // runBatch already wrote a status message; just exit the loop.
            break;
          }

          let wonThisBatch = false;
          for (const r of outcome.results) {
            loopTotalRef.current += 1;
            if (r === 'win') {
              loopWinsRef.current += 1;
              wonThisBatch = true;
            } else if (r === 'loss') {
              loopLossesRef.current += 1;
            } else {
              loopDrawsRef.current += 1;
            }
          }
          setLoopStats({
            total: loopTotalRef.current,
            wins: loopWinsRef.current,
            losses: loopLossesRef.current,
            draws: loopDrawsRef.current,
          });

          if (loopStopRef.current) break;

          if (wonThisBatch) {
            setStatusMsg(
              `AI won after ${loopTotalRef.current} game(s) — ` +
              `${loopWinsRef.current}W ${loopLossesRef.current}L ${loopDrawsRef.current}D.`,
            );
            break;
          }
        }
      } finally {
        loopStopRef.current = false;
        loopActiveRef.current = false;
        setLoopActive(false);
      }
    }, [agent, imitator, batchSize, runBatch]);

    const handleRun = useCallback(() => {
      if (mode === 'until-win') {
        void runUntilWinLoop();
      } else {
        void runBatch(batchSize);
      }
    }, [mode, runBatch, runUntilWinLoop, batchSize]);

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
                value="until-win"
                checked={mode === 'until-win'}
                onChange={() => setMode('until-win')}
                disabled={running || loopActive}
              />
              {' '}Until AI wins a game
            </label>
          </div>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
            <label>
              {mode === 'fixed' ? 'Games:' : 'Games per batch:'}{' '}
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
            {mode === 'until-win' && (
              <span style={{ fontSize: '0.85em', color: '#666' }}>
                (cap: {UNTIL_WIN_HARD_CAP} games)
              </span>
            )}
            <span style={{ fontSize: '0.85em', color: '#666' }}>
              MCTS sims/move: {SELFPLAY_SEARCH.simulations}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', fontSize: '0.9em' }}>
            <label htmlFor="selfplay-curriculum">Starting position:</label>
            <select
              id="selfplay-curriculum"
              value={curriculumId}
              onChange={e => setCurriculumId(e.target.value)}
              disabled={running || loopActive}
              style={{ padding: '3px 6px' }}
            >
              <option value="default">Default (full chess game)</option>
              {CURRICULUM_OPTIONS.map(opt => (
                <option key={opt.id} value={opt.id}>{opt.label}</option>
              ))}
            </select>
            <span style={{ color: '#666', fontSize: '0.85em' }}>
              {curriculumId === 'default'
                ? 'Each game starts from the standard initial position.'
                : 'Each game starts from a random position in the selected pool.'}
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
              {mode === 'until-win' ? 'Run until AI wins' : 'Run self-play batch'}
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

          {livePly !== null && (
            <LiveBoard livePly={livePly} />
          )}

          {batch !== null && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Game {batch.played} —{' '}
              AI: {batch.wins}W {batch.losses}L {batch.draws}D
              {batch.examples > 0 && ` — ${batch.examples} examples`}
            </div>
          )}

          {mode === 'until-win' && loopStats !== null && (
            <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
              Loop: {loopStats.total} / {UNTIL_WIN_HARD_CAP} —{' '}
              AI {loopStats.wins}W {loopStats.losses}L {loopStats.draws}D
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

/**
 * Compact live-board view for the current self-play game.
 *
 * Orientation tracks the AI's color so the AI is always at the bottom
 * of the rendered board — this keeps the side labels stable across
 * games regardless of which color the AI was assigned this round.
 */
function LiveBoard({ livePly }: { livePly: LivePly }) {
  const aiColorLabel = livePly.aiColor === 'w' ? 'White' : 'Black';
  const imitatorColorLabel = livePly.aiColor === 'w' ? 'Black' : 'White';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
      <div style={{ fontSize: '0.85em', color: '#555' }}>
        Game {livePly.gameIndex + 1} — ply {livePly.plyIndex}
      </div>
      <div style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#333' }}>
        Imitator ({imitatorColorLabel})
      </div>
      <div style={{ width: '100%', maxWidth: '320px' }}>
        <Board
          state={livePly.state}
          interactive={false}
          boardOrientation={livePly.aiColor === 'w' ? 'white' : 'black'}
        />
      </div>
      <div style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#333' }}>
        AI ({aiColorLabel})
      </div>
    </div>
  );
}
