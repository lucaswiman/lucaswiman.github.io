// EndgameGymPanel — drives the endgame curriculum gym.
//
// For each scenario in the picked category the opponent plays the
// winning side against the imitator. The worker stops attempting a
// scenario as soon as the opponent wins one game and only the winning
// game's moves feed the opponent's replay buffer (the deliberate "no
// penalty for optimal moves in a losing attempt" rule from the project
// README). When the worker reports `done`, the panel saves new weights
// + replay through the shared storage and refreshes the in-memory
// opponent.
//
// Shares the SelfPlayClient with SelfPlayPanel — that client already
// owns the Worker lifecycle and runs one task at a time, so spawning
// the gym implicitly cancels any other self-play run and vice-versa.
// That coupling is exactly what we want: only one CPU-heavy job can
// hit the opponent net at once anyway, and forcing them through one
// client avoids races over weight bytes.

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { Opponent } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { OPPONENT_STORAGE_KEYS } from '../core/opponent/index.js';
import { IMITATOR_STORAGE_KEYS } from '../core/imitator/index.js';
import { fromFen, toFen, type GameState } from '../core/rules/index.js';
import Board from './Board.js';
import { SelfPlayClient } from './selfplay-client.js';
import type { GymScenarioRequest } from './selfplay-protocol.js';
import {
  REPLAY_CAPACITY,
  SELFPLAY_GRADIENT_BATCH,
  SELFPLAY_GRADIENT_STEPS,
  SELFPLAY_MAX_PLIES,
  SELFPLAY_SEARCH,
  SELFPLAY_TEMPERATURE,
  SELFPLAY_TEMPERATURE_PLIES,
} from './opponent-config.js';
import { GYM_CATEGORIES, GYM_CATEGORIES_BY_ID } from './endgame-gym-scenarios.js';

/**
 * Default cap on attempts per scenario. Set low so the gym moves
 * forward briskly even when the opponent can't crack a position — the
 * user can run the gym multiple times to keep grinding.
 */
const DEFAULT_MAX_GAMES_PER_SCENARIO = 4;

export interface EndgameGymPanelProps {
  opponent: Opponent | null;
  imitator: PlayerImitator | null;
  storage: BlobStorage;
  /** Reload both nets from storage after the worker writes new weights. */
  onOpponentRefresh: () => Promise<void>;
  /**
   * When true the worker emits per-ply messages and the panel renders a
   * live board for the in-flight game. When false the worker skips the
   * per-ply postMessage and no board is rendered — saves compute on
   * long gym runs.
   */
  previewEnabled: boolean;
  onPreviewEnabledChange: (enabled: boolean) => void;
  /** MCTS sims/move — sourced from the in-game slider. */
  simulations: number;
}

interface LiveState {
  scenarioIdx: number;
  scenarioId: string;
  gameInScenario: number;
  opponentColor: 'w' | 'b';
  plyIndex: number;
  state: GameState;
}

interface ScenarioOutcome {
  scenarioIdx: number;
  scenarioId: string;
  converted: boolean;
  gamesPlayed: number;
}

interface RunStats {
  totalScenarios: number;
  scenariosDone: number;
  scenariosConverted: number;
  totalGames: number;
  wins: number;
  losses: number;
  draws: number;
}

export function EndgameGymPanel({
  opponent,
  imitator,
  storage,
  onOpponentRefresh,
  previewEnabled,
  onPreviewEnabledChange,
  simulations,
}: EndgameGymPanelProps) {
  const [categoryId, setCategoryId] = useState<string>('presets');
  const [maxGamesPerScenario, setMaxGamesPerScenario] = useState(
    DEFAULT_MAX_GAMES_PER_SCENARIO,
  );
  const [running, setRunning] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [stats, setStats] = useState<RunStats | null>(null);
  const [recentOutcomes, setRecentOutcomes] = useState<ScenarioOutcome[]>([]);
  const [live, setLive] = useState<LiveState | null>(null);

  const clientRef = useRef<SelfPlayClient | null>(null);
  const runningRef = useRef(false);
  runningRef.current = running;

  useEffect(() => {
    return () => {
      clientRef.current?.stop();
      clientRef.current = null;
    };
  }, []);

  const category = GYM_CATEGORIES_BY_ID.get(categoryId) ?? GYM_CATEGORIES[0];

  const handleRun = useCallback(async () => {
    if (runningRef.current || !opponent || !imitator) return;

    const scenarios = category.scenarios;
    if (scenarios.length === 0) {
      setStatusMsg('Selected category has no scenarios.');
      return;
    }

    setRunning(true);
    runningRef.current = true;
    setStatusMsg(null);
    setRecentOutcomes([]);
    setLive(null);
    setStats({
      totalScenarios: scenarios.length,
      scenariosDone: 0,
      scenariosConverted: 0,
      totalGames: 0,
      wins: 0,
      losses: 0,
      draws: 0,
    });

    let opponentWeights: Uint8Array;
    let imitatorWeights: Uint8Array;
    let replay: Uint8Array;
    let imitatorReplay: Uint8Array;
    try {
      [opponentWeights, imitatorWeights, replay, imitatorReplay] = await Promise.all([
        opponent.net.serialize(),
        imitator.net.serialize(),
        Promise.resolve(opponent.replayBuffer.serialize()),
        Promise.resolve(imitator.replayBuffer.serialize()),
      ]);
    } catch (err) {
      setStatusMsg(`Gym setup failed: ${err instanceof Error ? err.message : String(err)}`);
      setRunning(false);
      runningRef.current = false;
      return;
    }

    if (!clientRef.current) clientRef.current = new SelfPlayClient();

    // Marshal scenarios for the worker. `toFen` is preferable to keeping
    // the in-memory startingState (which holds a chess.js Chess
    // instance) — the worker re-parses FENs on receipt so we don't
    // depend on structured-clone for chess.js objects.
    const gymScenarios: GymScenarioRequest[] = scenarios.map(s => ({
      id: s.id,
      fen: toFen(s.startingState),
      opponentColor: s.opponentColor,
    }));

    clientRef.current.startGym(
      {
        opponentWeights,
        imitatorWeights,
        replay,
        imitatorReplay,
        replayCapacity: REPLAY_CAPACITY,
        search: { ...SELFPLAY_SEARCH, simulations },
        scenarios: gymScenarios,
        maxGamesPerScenario,
        maxPlies: SELFPLAY_MAX_PLIES,
        gradientSteps: SELFPLAY_GRADIENT_STEPS,
        gradientBatchSize: SELFPLAY_GRADIENT_BATCH,
        temperaturePlies: SELFPLAY_TEMPERATURE_PLIES,
        temperature: SELFPLAY_TEMPERATURE,
        previewEnabled,
      },
      (game) => {
        setStats(prev => prev === null ? prev : {
          ...prev,
          totalGames: prev.totalGames + 1,
          wins: prev.wins + (game.opponentResult === 'win' ? 1 : 0),
          losses: prev.losses + (game.opponentResult === 'loss' ? 1 : 0),
          draws: prev.draws + (game.opponentResult === 'draw' ? 1 : 0),
        });
      },
      (scenarioDone) => {
        setStats(prev => prev === null ? prev : {
          ...prev,
          scenariosDone: prev.scenariosDone + 1,
          scenariosConverted: prev.scenariosConverted + (scenarioDone.converted ? 1 : 0),
        });
        setRecentOutcomes(prev => {
          const next = [...prev, {
            scenarioIdx: scenarioDone.scenarioIdx,
            scenarioId: scenarioDone.scenarioId,
            converted: scenarioDone.converted,
            gamesPlayed: scenarioDone.gamesPlayed,
          }];
          // Bound the visible log so a 160-scenario "all decided" run
          // doesn't grow the DOM without bound.
          return next.length > 8 ? next.slice(-8) : next;
        });
      },
      (_step) => {
        // Gradient-step progress isn't shown here — the gym's own
        // scenario-level progress is the more useful signal. The
        // protocol still emits step messages so the SelfPlayPanel-style
        // UI gets parity if we ever fold the gym into that panel.
      },
      async (done) => {
        try {
          await Promise.all([
            storage.put(OPPONENT_STORAGE_KEYS.WEIGHTS, done.opponentWeights),
            storage.put(OPPONENT_STORAGE_KEYS.REPLAY, done.replay),
            storage.put(IMITATOR_STORAGE_KEYS.WEIGHTS, done.imitatorWeights),
            storage.put(IMITATOR_STORAGE_KEYS.REPLAY, done.imitatorReplay),
          ]);
          await onOpponentRefresh();
          const conv = done.scenariosConverted ?? 0;
          const att = done.scenariosAttempted ?? 0;
          setStatusMsg(
            `Gym done — ${conv} / ${att} scenarios converted, ` +
            `opponent ${done.wins}W ${done.losses}L ${done.draws}D across ${done.games} game(s), ` +
            `${done.examplesAdded} new winning-side example(s), ` +
            `opponent ${done.finalStep} grad step(s).`,
          );
        } catch (err) {
          setStatusMsg(`Gym done but save failed: ${err instanceof Error ? err.message : String(err)}`);
        } finally {
          setRunning(false);
          runningRef.current = false;
          setLive(null);
        }
      },
      (msg) => {
        setStatusMsg(`Gym failed: ${msg}`);
        setRunning(false);
        runningRef.current = false;
        setLive(null);
      },
      (ply) => {
        // gameIndex = scenarioIdx * maxGamesPerScenario + gameInScenario
        const scenarioIdx = Math.floor(ply.gameIndex / maxGamesPerScenario);
        const gameInScenario = ply.gameIndex % maxGamesPerScenario;
        const sId = gymScenarios[scenarioIdx]?.id ?? '?';
        try {
          setLive({
            scenarioIdx,
            scenarioId: sId,
            gameInScenario,
            opponentColor: ply.opponentColor,
            plyIndex: ply.plyIndex,
            state: fromFen(ply.fen),
          });
        } catch {
          // Bad FEN here would only break the live preview.
        }
      },
    );
  }, [opponent, imitator, storage, onOpponentRefresh, simulations, category, maxGamesPerScenario, previewEnabled]);

  const handleStop = useCallback(() => {
    clientRef.current?.stop();
  }, []);

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
        Endgame gym (opponent on the winning side)
      </summary>

      <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ fontSize: '0.9em', color: '#444' }}>
          For each scenario in the picked category the opponent plays
          the side with the forced win; the player-imitator defends.
          The gym retries each position up to <code>max attempts</code>
          times and stops as soon as the opponent converts — only the
          winning game's moves feed the opponent's replay buffer.
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap', fontSize: '0.9em' }}>
          <label>
            Scenarios:{' '}
            <select
              value={categoryId}
              onChange={e => setCategoryId(e.target.value)}
              disabled={running}
              style={{ padding: '3px 6px' }}
            >
              {GYM_CATEGORIES.map(c => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </label>
          <label>
            Max attempts:{' '}
            <input
              type="number"
              min={1}
              max={50}
              value={maxGamesPerScenario}
              onChange={e => setMaxGamesPerScenario(
                Math.max(1, parseInt(e.target.value, 10) || 1),
              )}
              style={{ width: '70px' }}
              disabled={running}
            />
          </label>
          <span style={{ color: '#666' }}>
            MCTS sims/move: {simulations}
          </span>
        </div>

        <div style={{ fontSize: '0.85em', color: '#666' }}>
          {category.description}
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9em' }}>
          <input
            type="checkbox"
            checked={previewEnabled}
            onChange={e => onPreviewEnabledChange(e.target.checked)}
          />
          Preview training (render live board — turn off to save compute)
        </label>

        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={handleRun}
            disabled={running || opponent === null || imitator === null}
            style={{ padding: '5px 14px', cursor: running ? 'not-allowed' : 'pointer' }}
          >
            Run gym
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

        {stats && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
            Scenario {stats.scenariosDone} / {stats.totalScenarios}{' '}
            — converted {stats.scenariosConverted}{' '}
            | Opponent {stats.wins}W {stats.losses}L {stats.draws}D
            {' '}
            ({stats.totalGames} game{stats.totalGames === 1 ? '' : 's'})
          </div>
        )}

        {previewEnabled && live !== null && <LiveBoard live={live} />}

        {recentOutcomes.length > 0 && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
            <div style={{ color: '#666', marginBottom: '4px' }}>
              Recent scenarios:
            </div>
            {recentOutcomes.slice().reverse().map(o => (
              <div key={o.scenarioIdx}>
                #{o.scenarioIdx + 1} {o.scenarioId} —{' '}
                <span style={{ color: o.converted ? '#080' : '#888' }}>
                  {o.converted ? `converted in ${o.gamesPlayed}` : `gave up after ${o.gamesPlayed}`}
                </span>
              </div>
            ))}
          </div>
        )}

        {statusMsg && (
          <div style={{ fontSize: '0.9em', color: '#333' }}>{statusMsg}</div>
        )}
      </div>
    </details>
  );
}

function LiveBoard({ live }: { live: LiveState }) {
  const opponentColorLabel = live.opponentColor === 'w' ? 'White' : 'Black';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
      <div style={{ fontSize: '0.85em', color: '#555' }}>
        Scenario {live.scenarioIdx + 1} ({live.scenarioId}) —{' '}
        attempt {live.gameInScenario + 1}, ply {live.plyIndex}
      </div>
      <div style={{ fontSize: '0.85em', fontWeight: 'bold', color: '#333' }}>
        Opponent ({opponentColorLabel}, winning side)
      </div>
      <div style={{ width: '100%', maxWidth: '320px' }}>
        <Board
          state={live.state}
          interactive={false}
          boardOrientation={live.opponentColor === 'w' ? 'white' : 'black'}
        />
      </div>
    </div>
  );
}
