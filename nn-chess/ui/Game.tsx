import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import { InspectorPanel } from './InspectorPanel.js';
import { SelfPlayPanel } from './SelfPlayPanel.js';
import type { SelfPlayPanelHandle } from './SelfPlayPanel.js';
import { EndgameGymPanel } from './EndgameGymPanel.js';
import { TrainingLog, useTrainingLog } from './TrainingLog.js';
import { TrainingStatsPanel } from './TrainingStats.js';
import { LastMoveMcts } from './LastMoveMcts.js';
import {
  type NetStats,
  emptyNetStats,
  loadNetStats,
  saveNetStats,
  addTrainingPass,
  OPPONENT_TRAIN_STATS_KEY,
  IMITATOR_TRAIN_STATS_KEY,
} from './training-stats.js';
import {
  initialState,
  applyMove,
  fromFen,
  sideToMove,
  outcome,
  toFen,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { Opponent, GameMoveRecord } from '../core/opponent/index.js';
import { legalMoveIndicesFor, OPPONENT_STORAGE_KEYS } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import { IMITATOR_STORAGE_KEYS } from '../core/imitator/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { getJson, putJson } from '../core/storage/index.js';
import {
  OPPONENT_CONFIG,
  createOpponentStorage,
  loadOpponentFromStorage,
  loadPlayerImitatorFromStorage,
  AUTO_SELFPLAY_PREF_KEY,
  PREVIEW_TRAINING_PREF_KEY,
} from './opponent-config.js';
import { ENDGAME_PRESETS, ENDGAME_PRESETS_BY_ID } from './endgame-presets.js';
import {
  LICHESS_CATEGORIES,
  LICHESS_CATEGORIES_BY_ID,
  randomPuzzle,
  lichessPuzzleUrl,
} from './lichess-endgames.js';
import type { LichessPuzzle } from './lichess-endgames.js';
import {
  loadPersistedSession,
  savePersistedSession,
  type PersistedSession,
  type ColorPreference,
} from './session-persistence.js';

type PlayerColor = 'w' | 'b';
type SetupId = 'standard' | 'custom' | string;

interface GameSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
  // FEN of the position the game started from. Kept so the persisted
  // session can store `startingFen + moves` instead of every per-move
  // feature tensor.
  startingFen: string;
}

function sessionFromState(state: GameState): GameSession {
  return {
    state,
    history: [],
    over: false,
    finalOutcome: null,
    startingFen: toFen(state),
  };
}

export default function Game() {
  const [opponent, setOpponent] = useState<Opponent | null>(null);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);
  const [storage] = useState<BlobStorage>(createOpponentStorage);
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  const [colorPreference, setColorPreference] = useState<ColorPreference>('random');
  const [setupId, setSetupId] = useState<SetupId>('standard');
  const [customFen, setCustomFen] = useState<string>('');
  const [customFenError, setCustomFenError] = useState<string | null>(null);
  const [currentPuzzle, setCurrentPuzzle] = useState<LichessPuzzle | null>(null);
  const [session, setSession] = useState<GameSession>(() => sessionFromState(initialState()));
  const [opponentThinking, setOpponentThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opponentVersion, setOpponentVersion] = useState(0);
  const [lastSearch, setLastSearch] = useState<SearchResult | null>(null);
  // Default true: the brief asks for auto-batching on by default; the
  // persisted preference (loaded asynchronously below) overrides this
  // once it lands.
  const [autoSelfPlay, setAutoSelfPlay] = useState(true);
  // Default true to match the prior behaviour of always rendering the
  // live training board; the persisted value (loaded below) overrides
  // this once it lands.
  const [previewTraining, setPreviewTraining] = useState(true);
  const [simulations, setSimulations] = useState(OPPONENT_CONFIG.simulations);
  const [cPuct, setCPuct] = useState(OPPONENT_CONFIG.cPuct);
  // Shared log fed by every training-side panel. Lets the user see
  // gradient-descent vs MCTS phase transitions and storage saves
  // without staring at the console.
  const trainingLog = useTrainingLog();
  // Cumulative gradient-step / examples-seen counters per net, loaded
  // from storage on mount and bumped after every training pass.
  const [opponentStats, setOpponentStats] = useState<NetStats>(emptyNetStats);
  const [imitatorStats, setImitatorStats] = useState<NetStats>(emptyNetStats);
  // Gates the "persist session on change" effect: we only start writing
  // after the load attempt finishes, so the initial in-memory state
  // can't overwrite a saved session before we've had a chance to read it.
  const [sessionLoaded, setSessionLoaded] = useState(false);

  const opponentRef = useRef<Opponent | null>(null);
  opponentRef.current = opponent;

  const selfPlayPanelRef = useRef<SelfPlayPanelHandle | null>(null);

  // Prevents double-fire of the opponent move under React Strict Mode.
  const opponentMoveInFlight = useRef(false);
  const recordedRef = useRef<GameSession | null>(null);
  // Tracks which finished session has already kicked off auto-self-play.
  // Distinct from `recordedRef` because the auto-batch is gated on
  // both the save having been initiated AND the user's preference.
  const autoBatchedRef = useRef<GameSession | null>(null);

  const applyStartingState = useCallback((state: GameState) => {
    opponentMoveInFlight.current = false;
    recordedRef.current = null;
    setOpponentThinking(false);
    setSession(sessionFromState(state));
    setLastSearch(null);
    setError(null);
  }, []);

  // Builds the starting state for `id` without sampling — used when the
  // color picker changes mid-game and we want to keep the current puzzle
  // / preset rather than picking a new one.
  const startingStateForCurrent = useCallback((): GameState | null => {
    if (setupId === 'standard') return initialState();
    if (setupId === 'custom') {
      const trimmed = customFen.trim();
      if (trimmed === '') return null;
      try { return fromFen(trimmed); } catch { return null; }
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(setupId);
    if (preset) return fromFen(preset.fen);
    if (currentPuzzle) return fromFen(currentPuzzle.fen);
    return initialState();
  }, [setupId, customFen, currentPuzzle]);

  const pickColor = useCallback((preference: ColorPreference): PlayerColor =>
    preference === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : preference,
  []);

  const startNewGame = useCallback(() => {
    if (setupId === 'standard') {
      setCurrentPuzzle(null);
      setPlayerColor(pickColor(colorPreference));
      applyStartingState(initialState());
      return;
    }
    if (setupId === 'custom') {
      const trimmed = customFen.trim();
      if (trimmed === '') return;
      try {
        applyStartingState(fromFen(trimmed));
      } catch (err) {
        setCustomFenError(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`);
      }
      return;
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(setupId);
    if (preset) {
      setCurrentPuzzle(null);
      applyStartingState(fromFen(preset.fen));
      return;
    }
    const cat = LICHESS_CATEGORIES_BY_ID.get(setupId);
    if (cat) {
      const puzzle = randomPuzzle(cat);
      setCurrentPuzzle(puzzle);
      const state = fromFen(puzzle.fen);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    }
  }, [setupId, customFen, colorPreference, applyStartingState, pickColor]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadOpponentFromStorage(storage),
      loadPlayerImitatorFromStorage(storage),
    ])
      .then(([loadedOpponent, loadedImitator]) => {
        if (cancelled) return;
        setOpponent(loadedOpponent);
        setImitator(loadedImitator);
      })
      .catch(err => {
        if (!cancelled) setError(`Failed to load opponent: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => { cancelled = true; };
  }, [storage]);

  // Load the auto-self-play preference lazily — if it isn't present
  // yet (first-ever visit) we keep the default of `true`.
  useEffect(() => {
    let cancelled = false;
    getJson<{ enabled: boolean }>(storage, AUTO_SELFPLAY_PREF_KEY)
      .then(saved => {
        if (cancelled || saved === null) return;
        setAutoSelfPlay(saved.enabled);
      })
      .catch(() => { /* preference missing or corrupt — keep default */ });
    return () => { cancelled = true; };
  }, [storage]);

  const handleAutoSelfPlayChange = useCallback((enabled: boolean) => {
    setAutoSelfPlay(enabled);
    void putJson(storage, AUTO_SELFPLAY_PREF_KEY, { enabled });
  }, [storage]);

  useEffect(() => {
    let cancelled = false;
    getJson<{ enabled: boolean }>(storage, PREVIEW_TRAINING_PREF_KEY)
      .then(saved => {
        if (cancelled || saved === null) return;
        setPreviewTraining(saved.enabled);
      })
      .catch(() => { /* preference missing or corrupt — keep default */ });
    return () => { cancelled = true; };
  }, [storage]);

  const handlePreviewTrainingChange = useCallback((enabled: boolean) => {
    setPreviewTraining(enabled);
    void putJson(storage, PREVIEW_TRAINING_PREF_KEY, { enabled });
  }, [storage]);

  // Hydrate cumulative training stats from storage. Failures fall back
  // to zeros — the worst case is a fresh-looking counter, not data
  // corruption, since stats are independent of weights.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadNetStats(storage, OPPONENT_TRAIN_STATS_KEY),
      loadNetStats(storage, IMITATOR_TRAIN_STATS_KEY),
    ])
      .then(([opp, imi]) => {
        if (cancelled) return;
        setOpponentStats(opp);
        setImitatorStats(imi);
      })
      .catch(() => { /* keep defaults */ });
    return () => { cancelled = true; };
  }, [storage]);

  /**
   * Record a finished training pass. Called by each training-side panel
   * after the worker reports `done`. `gradientSteps` is the worker's
   * `finalStep` / `imitatorFinalStep`; `batchSize` is whatever the panel
   * configured the worker to use. Passes with zero steps are no-ops so
   * panels can call this unconditionally on `done`.
   */
  const recordTrainingPass = useCallback((
    net: 'opponent' | 'imitator',
    delta: { gradientSteps: number; batchSize: number },
  ) => {
    if (delta.gradientSteps <= 0) return;
    const bufferSize =
      net === 'opponent'
        ? opponentRef.current?.replayBuffer.size() ?? 0
        : imitator?.replayBuffer.size() ?? 0;
    const setter = net === 'opponent' ? setOpponentStats : setImitatorStats;
    const storageKey = net === 'opponent' ? OPPONENT_TRAIN_STATS_KEY : IMITATOR_TRAIN_STATS_KEY;
    setter(prev => {
      const next = addTrainingPass(prev, delta);
      void saveNetStats(storage, storageKey, next);
      const epochs = bufferSize > 0 ? next.examplesSeen / bufferSize : 0;
      const label = net === 'opponent' ? 'Opponent' : 'Imitator';
      trainingLog.append({
        source: 'training',
        kind: 'info',
        message:
          `${label} lifetime: ${next.gradientSteps.toLocaleString()} step(s), ` +
          `${next.examplesSeen.toLocaleString()} example(s) seen ` +
          `(~${epochs.toFixed(2)} effective epoch(s) over the current ` +
          `${bufferSize.toLocaleString()}-example buffer).`,
      });
      return next;
    });
  }, [storage, imitator, trainingLog]);

  useEffect(() => {
    if (opponent) opponent.setSimulations(simulations);
  }, [opponent, simulations]);

  useEffect(() => {
    if (opponent) opponent.setCPuct(cPuct);
  }, [opponent, cPuct]);

  // Restore the last in-progress (or just-finished) game from storage.
  // Runs once on mount. If the saved session is already terminal we
  // park its reference in `recordedRef` so the game-over effect below
  // doesn't double-record it into the opponent's replay buffer.
  useEffect(() => {
    let cancelled = false;
    loadPersistedSession(storage)
      .then(restored => {
        if (cancelled) return;
        if (restored !== null) {
          const newSession: GameSession = {
            state: restored.state,
            history: restored.history,
            over: restored.over,
            finalOutcome: restored.finalOutcome,
            startingFen: restored.startingFen,
          };
          setSession(newSession);
          setPlayerColor(restored.playerColor);
          setColorPreference(restored.colorPreference);
          setSetupId(restored.setupId);
          setCustomFen(restored.customFen);
          setCurrentPuzzle(restored.currentPuzzle);
          if (restored.over) {
            // The game already ended — either it was recorded in the
            // earlier session, or the tab was closed before the save
            // landed. Either way, don't re-run training on it.
            recordedRef.current = newSession;
          }
        }
        setSessionLoaded(true);
      })
      .catch(() => {
        if (!cancelled) setSessionLoaded(true);
      });
    return () => { cancelled = true; };
  }, [storage]);

  // Mirror the current session + setup back into storage. Gated on
  // `sessionLoaded` so we don't clobber a saved session before reading
  // it. Writes are fire-and-forget; on failure the worst case is that
  // the next reload starts from an earlier checkpoint.
  useEffect(() => {
    if (!sessionLoaded) return;
    const data: PersistedSession = {
      playerColor,
      colorPreference,
      setupId,
      customFen,
      currentPuzzle,
      startingFen: session.startingFen,
      moves: session.history.map(r => r.move),
    };
    void savePersistedSession(storage, data);
  }, [session, playerColor, colorPreference, setupId, customFen, currentPuzzle, sessionLoaded, storage]);

  useEffect(() => {
    if (opponent === null) return;
    if (session.over) return;
    if (opponentMoveInFlight.current) return;

    const currentSide = sideToMove(session.state);
    if (currentSide === playerColor) return;

    opponentMoveInFlight.current = true;
    setOpponentThinking(true);

    const capturedState = session.state;

    opponent.selectMove(capturedState).then(({ move, rootValue, visitCounts, priorPolicy }) => {
      opponentMoveInFlight.current = false;

      setLastSearch({ bestMove: move, rootValue, visitCounts, priorPolicy });

      const record: GameMoveRecord = {
        features: encodeState(capturedState),
        move,
        sideToMove: sideToMove(capturedState),
        legalMoveIndices: legalMoveIndicesFor(capturedState),
      };

      const nextState = applyMove(capturedState, move);
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';

      setSession(prev => {
        if (prev.state !== capturedState) return prev;
        return {
          state: nextState,
          history: [...prev.history, record],
          over: gameOver,
          finalOutcome: gameOver ? o : null,
          startingFen: prev.startingFen,
        };
      });

      setOpponentThinking(false);
    }).catch(err => {
      opponentMoveInFlight.current = false;
      setOpponentThinking(false);
      setError(`Opponent move failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [opponent, session, playerColor]);

  useEffect(() => {
    if (!session.over) return;
    if (!opponent) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;
    opponent.recordGameResult(session.history, session.finalOutcome);

    // Teach the player-imitator from the user's moves only. The
    // opponent's moves don't belong in the imitator's training set —
    // the imitator's whole purpose is to model the *user*.
    if (imitator) {
      const userMoves = session.history.filter(r => r.sideToMove === playerColor);
      imitator.recordUserMoves(userMoves, session.finalOutcome);
    }

    const savePromises: Promise<void>[] = [opponent.saveTo(storage)];
    if (imitator) savePromises.push(imitator.saveTo(storage));
    const sessionSnapshot = session;
    Promise.all(savePromises)
      .then(() => {
        // Only kick the auto-batch AFTER the post-game save lands: the
        // worker reads weights+replay we just persisted via
        // opponent.saveTo(...) through the serialized handles passed in,
        // so we want the in-memory `opponent` to reflect every recorded
        // example before we serialize it.
        if (!autoSelfPlay) return;
        if (autoBatchedRef.current === sessionSnapshot) return;
        const handle = selfPlayPanelRef.current;
        if (!handle || handle.isRunning) return;
        autoBatchedRef.current = sessionSnapshot;
        // Fire and forget: the user may start a new game while this
        // runs. onOpponentRefresh re-reads from storage when it finishes,
        // and that doesn't disturb the in-progress game's session state.
        void handle.triggerAutoBatch();
      })
      .catch(err => {
        setError(`Failed to save after game: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [session, opponent, imitator, playerColor, storage, autoSelfPlay]);

  const handleMove = useCallback((nextState: GameState, move: Move) => {
    if (!opponentRef.current) return;
    setSession(prev => {
      if (prev.over) return prev;
      const record: GameMoveRecord = {
        features: encodeState(prev.state),
        move,
        sideToMove: sideToMove(prev.state),
        legalMoveIndices: legalMoveIndicesFor(prev.state),
      };
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';
      return {
        state: nextState,
        history: [...prev.history, record],
        over: gameOver,
        finalOutcome: gameOver ? o : null,
        startingFen: prev.startingFen,
      };
    });
  }, []);

  const handleColorChange = useCallback((preference: ColorPreference) => {
    setColorPreference(preference);
    setPlayerColor(pickColor(preference));
    const starting = startingStateForCurrent();
    if (starting) applyStartingState(starting);
  }, [startingStateForCurrent, applyStartingState, pickColor]);

  const handleSetupChange = useCallback((id: SetupId) => {
    setSetupId(id);
    setCustomFenError(null);
    if (id === 'standard') {
      setCurrentPuzzle(null);
      applyStartingState(initialState());
      return;
    }
    if (id === 'custom') {
      setCurrentPuzzle(null);
      return; // wait for Apply
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(id);
    if (preset) {
      setCurrentPuzzle(null);
      setPlayerColor(preset.winningSide);
      applyStartingState(fromFen(preset.fen));
      return;
    }
    const cat = LICHESS_CATEGORIES_BY_ID.get(id);
    if (cat) {
      const puzzle = randomPuzzle(cat);
      setCurrentPuzzle(puzzle);
      const state = fromFen(puzzle.fen);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    }
  }, [applyStartingState]);

  const handleApplyCustomFen = useCallback(() => {
    const trimmed = customFen.trim();
    if (trimmed === '') {
      setCustomFenError('Enter a FEN string.');
      return;
    }
    try {
      const state = fromFen(trimmed);
      setCustomFenError(null);
      setCurrentPuzzle(null);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    } catch (err) {
      setCustomFenError(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [customFen, applyStartingState]);

  const handleOpponentRefresh = useCallback(async () => {
    try {
      const [reloadedOpponent, reloadedImitator] = await Promise.all([
        loadOpponentFromStorage(storage),
        loadPlayerImitatorFromStorage(storage),
      ]);
      setOpponent(reloadedOpponent);
      setImitator(reloadedImitator);
      setOpponentVersion(v => v + 1);
    } catch (err) {
      setError(`Failed to reload opponent: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage]);

  // Danger zone: delete one of {weights, replay} for one of {opponent,
  // imitator} and reload. Deleting weights+meta forces `loadOpponent` /
  // `loadPlayerImitator` to fall through their version checks and
  // return a fresh randomly-initialized net. Deleting replay drops the
  // recorded training examples without touching the weights. Both
  // operations are confirmed with a native dialog before they run.
  const handleClearOpponentHistory = useCallback(async () => {
    if (!window.confirm(
      'Clear ALL recorded opponent training examples? This cannot be undone. '
      + 'The opponent\'s weights will be kept.',
    )) return;
    try {
      await storage.delete(OPPONENT_STORAGE_KEYS.REPLAY);
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to clear opponent history: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleClearImitatorHistory = useCallback(async () => {
    if (!window.confirm(
      'Clear ALL recorded user moves (the imitator\'s training examples)? '
      + 'This cannot be undone. The imitator\'s weights will be kept.',
    )) return;
    try {
      await storage.delete(IMITATOR_STORAGE_KEYS.REPLAY);
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to clear imitator history: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleResetOpponentModel = useCallback(async () => {
    if (!window.confirm(
      'Reset the opponent model to a freshly-initialized network? '
      + 'This deletes the trained weights and the recorded training examples. '
      + 'This cannot be undone.',
    )) return;
    try {
      await Promise.all([
        storage.delete(OPPONENT_STORAGE_KEYS.WEIGHTS),
        storage.delete(OPPONENT_STORAGE_KEYS.META),
        storage.delete(OPPONENT_STORAGE_KEYS.REPLAY),
        storage.delete(OPPONENT_TRAIN_STATS_KEY),
      ]);
      setOpponentStats(emptyNetStats());
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to reset opponent model: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleResetImitatorModel = useCallback(async () => {
    if (!window.confirm(
      'Reset the imitator model to a freshly-initialized network? '
      + 'This deletes the trained weights and the recorded user moves. '
      + 'This cannot be undone.',
    )) return;
    try {
      await Promise.all([
        storage.delete(IMITATOR_STORAGE_KEYS.WEIGHTS),
        storage.delete(IMITATOR_STORAGE_KEYS.META),
        storage.delete(IMITATOR_STORAGE_KEYS.REPLAY),
        storage.delete(IMITATOR_TRAIN_STATS_KEY),
      ]);
      setImitatorStats(emptyNetStats());
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to reset imitator model: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  if (opponent === null && error === null) {
    return <div style={{ fontFamily: 'sans-serif', padding: '16px' }}>Loading opponent…</div>;
  }

  if (error && opponent === null) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px', color: '#c00' }}>
        {error}
      </div>
    );
  }

  const humanIsWhite = playerColor === 'w';
  const currentSide = sideToMove(session.state);
  const isHumanTurn = currentSide === playerColor && !session.over;
  const interactive = isHumanTurn && !opponentThinking;

  const o = session.finalOutcome;
  let outcomeText = '';
  if (o === 'white-wins') outcomeText = playerColor === 'w' ? 'You win!' : 'Opponent wins!';
  else if (o === 'black-wins') outcomeText = playerColor === 'b' ? 'You win!' : 'Opponent wins!';
  else if (o === 'draw') outcomeText = 'Draw!';

  const activePreset = ENDGAME_PRESETS_BY_ID.get(setupId);
  const activeLichessCat = LICHESS_CATEGORIES_BY_ID.get(setupId);

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>Setup:</strong>
        <select
          value={setupId}
          onChange={e => handleSetupChange(e.target.value)}
          style={{ padding: '3px 6px' }}
        >
          <option value="standard">Standard game</option>
          <optgroup label="Endgame training">
            {ENDGAME_PRESETS.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </optgroup>
          {LICHESS_CATEGORIES.length > 0 && (
            <optgroup label="Lichess endgames (random)">
              {LICHESS_CATEGORIES.map(c => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </optgroup>
          )}
          <option value="custom">Custom FEN…</option>
        </select>
        {(activeLichessCat || activePreset || setupId === 'standard') && (
          <button
            type="button"
            onClick={startNewGame}
            style={{ padding: '3px 10px', cursor: 'pointer' }}
            title={activeLichessCat ? 'Pick a different random puzzle' : 'Restart this position'}
          >
            {activeLichessCat ? 'Next puzzle' : 'Restart'}
          </button>
        )}
      </div>

      {activePreset && (
        <div
          style={{
            marginBottom: '12px',
            padding: '8px 12px',
            background: '#fffbe6',
            border: '1px solid #e6d97a',
            borderRadius: '6px',
            fontSize: '0.9em',
            color: '#444',
          }}
        >
          <div style={{ marginBottom: '4px' }}>{activePreset.description}</div>
          <div style={{ color: '#666' }}>
            You're set to play{' '}
            <strong>{activePreset.winningSide === 'w' ? 'White' : 'Black'}</strong>{' '}
            (the winning side) — that side's moves teach the opponent the mating pattern.
          </div>
        </div>
      )}

      {activeLichessCat && currentPuzzle && (
        <div
          style={{
            marginBottom: '12px',
            padding: '8px 12px',
            background: '#fffbe6',
            border: '1px solid #e6d97a',
            borderRadius: '6px',
            fontSize: '0.9em',
            color: '#444',
          }}
        >
          <div style={{ marginBottom: '4px' }}>
            <strong>{activeLichessCat.label}</strong>{' '}
            — puzzle{' '}
            <a href={lichessPuzzleUrl(currentPuzzle.id)} target="_blank" rel="noopener noreferrer">
              {currentPuzzle.id}
            </a>{' '}
            (rating {currentPuzzle.rating})
          </div>
          <div style={{ color: '#666', fontSize: '0.85em' }}>
            Themes: {currentPuzzle.themes}. Source: Lichess puzzle database (CC0).
          </div>
        </div>
      )}

      {setupId === 'custom' && (
        <div style={{ marginBottom: '12px', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            value={customFen}
            onChange={e => setCustomFen(e.target.value)}
            placeholder="paste a FEN…"
            style={{ flex: '1 1 360px', padding: '4px 8px', fontFamily: 'monospace', fontSize: '0.9em' }}
          />
          <button type="button" onClick={handleApplyCustomFen} style={{ padding: '4px 12px', cursor: 'pointer' }}>
            Apply
          </button>
          {customFenError && (
            <span style={{ color: '#c00', fontSize: '0.85em' }}>{customFenError}</span>
          )}
        </div>
      )}

      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center' }}>
        <strong>You play as:</strong>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="w"
            checked={colorPreference === 'w'}
            onChange={() => handleColorChange('w')}
          />
          {' '}White
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="b"
            checked={colorPreference === 'b'}
            onChange={() => handleColorChange('b')}
          />
          {' '}Black
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="random"
            checked={colorPreference === 'random'}
            onChange={() => handleColorChange('random')}
          />
          {' '}Random
        </label>
      </div>

      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>Strength:</strong>
        <input
          type="range"
          min={16}
          max={400}
          step={16}
          value={simulations}
          onChange={e => setSimulations(Number(e.target.value))}
          style={{ width: '120px' }}
        />
        <span style={{ fontSize: '0.9em', color: '#555' }}>{simulations} sims/move</span>
        <span style={{ fontSize: '0.9em', color: '#888' }}>|</span>
        <label style={{ fontSize: '0.9em', color: '#555', display: 'flex', gap: '8px', alignItems: 'center' }}
          title="PUCT exploration constant. Lower = search goes deeper on the best line; higher = broader, more exploratory tree.">
          explore (cPuct):
          <input
            type="range"
            min={0.5}
            max={4.0}
            step={0.25}
            value={cPuct}
            onChange={e => setCPuct(Number(e.target.value))}
            style={{ width: '100px' }}
          />
          <span style={{ fontFamily: 'monospace' }}>{cPuct.toFixed(2)}</span>
        </label>
      </div>

      <div style={{ marginBottom: '8px', minHeight: '24px' }}>
        {opponentThinking && (
          <span style={{ color: '#666', fontStyle: 'italic' }}>Opponent thinking…</span>
        )}
        {!opponentThinking && !session.over && (
          <span>
            {isHumanTurn ? 'Your turn' : 'Opponent to move'}
            {' — '}{currentSide === 'w' ? 'White' : 'Black'} to move
          </span>
        )}
      </div>

      <Board
        state={session.state}
        interactive={interactive}
        onMove={handleMove}
        boardOrientation={humanIsWhite ? 'white' : 'black'}
      />

      <LastMoveMcts lastSearch={lastSearch} />

      {session.over && (
        <div
          style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: '#f0f4ff',
            border: '1px solid #aac',
            borderRadius: '6px',
            color: '#1a1a1a',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
          }}
        >
          <strong style={{ fontSize: '1.1em' }}>{outcomeText}</strong>
          <button
            type="button"
            onClick={startNewGame}
            style={{ padding: '6px 16px', cursor: 'pointer' }}
          >
            {activeLichessCat ? 'Next puzzle' : 'New game'}
          </button>
        </div>
      )}

      {error && (
        <div style={{ marginTop: '8px', color: '#c00', fontSize: '0.9em' }}>
          {error}
        </div>
      )}

      {opponent && (
        <TrainingStatsPanel
          opponentStats={opponentStats}
          imitatorStats={imitatorStats}
          opponentBufferSize={opponent.replayBuffer.size()}
          imitatorBufferSize={imitator?.replayBuffer.size() ?? 0}
          opponentBufferCapacity={opponent.replayBuffer.capacity()}
          imitatorBufferCapacity={imitator?.replayBuffer.capacity() ?? opponent.replayBuffer.capacity()}
        />
      )}

      {opponent && (
        <div style={{ marginTop: '24px' }}>
          <TrainingPanel
            opponent={opponent}
            imitator={imitator}
            storage={storage}
            onOpponentRefresh={handleOpponentRefresh}
            onLog={trainingLog.append}
            onTrainingPass={recordTrainingPass}
            key={opponentVersion}
          />
        </div>
      )}

      <SelfPlayPanel
        ref={selfPlayPanelRef}
        opponent={opponent}
        imitator={imitator}
        storage={storage}
        onOpponentRefresh={handleOpponentRefresh}
        autoEnabled={autoSelfPlay}
        onAutoEnabledChange={handleAutoSelfPlayChange}
        previewEnabled={previewTraining}
        onPreviewEnabledChange={handlePreviewTrainingChange}
        simulations={simulations}
        cPuct={cPuct}
        onLog={trainingLog.append}
        onTrainingPass={recordTrainingPass}
      />

      <EndgameGymPanel
        opponent={opponent}
        imitator={imitator}
        storage={storage}
        onOpponentRefresh={handleOpponentRefresh}
        previewEnabled={previewTraining}
        onPreviewEnabledChange={handlePreviewTrainingChange}
        simulations={simulations}
        cPuct={cPuct}
        onLog={trainingLog.append}
        onTrainingPass={recordTrainingPass}
      />

      <TrainingLog entries={trainingLog.entries} onClear={trainingLog.clear} />

      <InspectorPanel
        opponent={opponent}
        state={session.state}
        lastSearch={lastSearch}
        opponentVersion={opponentVersion}
      />

      <details
        style={{
          marginTop: '24px',
          border: '1px solid #d99',
          borderRadius: '6px',
          padding: '10px 14px',
          background: '#fff5f5',
          color: '#1a1a1a',
        }}
      >
        <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none', color: '#a33' }}>
          Danger zone
        </summary>
        <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#444' }}>
          Each action is destructive and applies only to your browser's local
          storage. Trained weights and recorded games cannot be recovered
          after deletion.
        </div>
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-start' }}>
          <button
            type="button"
            onClick={handleClearOpponentHistory}
            disabled={opponent === null}
            style={{ padding: '5px 14px', cursor: opponent ? 'pointer' : 'not-allowed' }}
          >
            Clear opponent move history
          </button>
          <button
            type="button"
            onClick={handleClearImitatorHistory}
            disabled={imitator === null}
            style={{ padding: '5px 14px', cursor: imitator ? 'pointer' : 'not-allowed' }}
          >
            Clear user (imitator) move history
          </button>
          <button
            type="button"
            onClick={handleResetOpponentModel}
            disabled={opponent === null}
            style={{ padding: '5px 14px', cursor: opponent ? 'pointer' : 'not-allowed', color: '#a33' }}
          >
            Reset opponent model
          </button>
          <button
            type="button"
            onClick={handleResetImitatorModel}
            disabled={imitator === null}
            style={{ padding: '5px 14px', cursor: imitator ? 'pointer' : 'not-allowed', color: '#a33' }}
          >
            Reset imitator model
          </button>
        </div>
      </details>
    </div>
  );
}
