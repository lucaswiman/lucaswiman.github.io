import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import { InspectorPanel } from './InspectorPanel.js';
import { SelfPlayPanel } from './SelfPlayPanel.js';
import type { SelfPlayPanelHandle } from './SelfPlayPanel.js';
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
import type { Agent, GameMoveRecord } from '../core/agent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { getJson, putJson } from '../core/storage/index.js';
import {
  createAgentStorage,
  loadAgentFromStorage,
  loadPlayerImitatorFromStorage,
  AUTO_SELFPLAY_PREF_KEY,
} from './agent-config.js';
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
  const [agent, setAgent] = useState<Agent | null>(null);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);
  const [storage] = useState<BlobStorage>(createAgentStorage);
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  const [setupId, setSetupId] = useState<SetupId>('standard');
  const [customFen, setCustomFen] = useState<string>('');
  const [customFenError, setCustomFenError] = useState<string | null>(null);
  const [currentPuzzle, setCurrentPuzzle] = useState<LichessPuzzle | null>(null);
  const [session, setSession] = useState<GameSession>(() => sessionFromState(initialState()));
  const [agentThinking, setAgentThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentVersion, setAgentVersion] = useState(0);
  const [lastSearch, setLastSearch] = useState<SearchResult | null>(null);
  // Default true: the brief asks for auto-batching on by default; the
  // persisted preference (loaded asynchronously below) overrides this
  // once it lands.
  const [autoSelfPlay, setAutoSelfPlay] = useState(true);
  // Gates the "persist session on change" effect: we only start writing
  // after the load attempt finishes, so the initial in-memory state
  // can't overwrite a saved session before we've had a chance to read it.
  const [sessionLoaded, setSessionLoaded] = useState(false);

  const agentRef = useRef<Agent | null>(null);
  agentRef.current = agent;

  const selfPlayPanelRef = useRef<SelfPlayPanelHandle | null>(null);

  // Prevents double-fire of the agent move under React Strict Mode.
  const agentMoveInFlight = useRef(false);
  const recordedRef = useRef<GameSession | null>(null);
  // Tracks which finished session has already kicked off auto-self-play.
  // Distinct from `recordedRef` because the auto-batch is gated on
  // both the save having been initiated AND the user's preference.
  const autoBatchedRef = useRef<GameSession | null>(null);

  const applyStartingState = useCallback((state: GameState) => {
    agentMoveInFlight.current = false;
    recordedRef.current = null;
    setAgentThinking(false);
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

  const startNewGame = useCallback(() => {
    if (setupId === 'standard') {
      setCurrentPuzzle(null);
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
  }, [setupId, customFen, applyStartingState]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadAgentFromStorage(storage),
      loadPlayerImitatorFromStorage(storage),
    ])
      .then(([loadedAgent, loadedImitator]) => {
        if (cancelled) return;
        setAgent(loadedAgent);
        setImitator(loadedImitator);
      })
      .catch(err => {
        if (!cancelled) setError(`Failed to load agent: ${err instanceof Error ? err.message : String(err)}`);
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

  // Restore the last in-progress (or just-finished) game from storage.
  // Runs once on mount. If the saved session is already terminal we
  // park its reference in `recordedRef` so the game-over effect below
  // doesn't double-record it into the agent's replay buffer.
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
      setupId,
      customFen,
      currentPuzzle,
      startingFen: session.startingFen,
      moves: session.history.map(r => r.move),
    };
    void savePersistedSession(storage, data);
  }, [session, playerColor, setupId, customFen, currentPuzzle, sessionLoaded, storage]);

  useEffect(() => {
    if (agent === null) return;
    if (session.over) return;
    if (agentMoveInFlight.current) return;

    const currentSide = sideToMove(session.state);
    if (currentSide === playerColor) return;

    agentMoveInFlight.current = true;
    setAgentThinking(true);

    const capturedState = session.state;

    agent.selectMove(capturedState).then(({ move, rootValue, visitCounts }) => {
      agentMoveInFlight.current = false;

      // priorPolicy is not exposed by Agent.selectMove; leave it empty so the
      // inspector's prior% column reads 0 until core/ is extended.
      setLastSearch({ bestMove: move, rootValue, visitCounts, priorPolicy: new Map() });

      const record: GameMoveRecord = {
        features: encodeState(capturedState),
        move,
        sideToMove: sideToMove(capturedState),
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

      setAgentThinking(false);
    }).catch(err => {
      agentMoveInFlight.current = false;
      setAgentThinking(false);
      setError(`Agent move failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [agent, session, playerColor]);

  useEffect(() => {
    if (!session.over) return;
    if (!agent) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;
    agent.recordGameResult(session.history, session.finalOutcome);

    // Teach the player-imitator from the user's moves only. The
    // agent's moves don't belong in the imitator's training set —
    // the imitator's whole purpose is to model the *user*.
    if (imitator) {
      const userMoves = session.history.filter(r => r.sideToMove === playerColor);
      imitator.recordUserMoves(userMoves, session.finalOutcome);
    }

    const savePromises: Promise<void>[] = [agent.saveTo(storage)];
    if (imitator) savePromises.push(imitator.saveTo(storage));
    const sessionSnapshot = session;
    Promise.all(savePromises)
      .then(() => {
        // Only kick the auto-batch AFTER the post-game save lands: the
        // worker reads weights+replay we just persisted via
        // agent.saveTo(...) through the serialized handles passed in,
        // so we want the in-memory `agent` to reflect every recorded
        // example before we serialize it.
        if (!autoSelfPlay) return;
        if (autoBatchedRef.current === sessionSnapshot) return;
        const handle = selfPlayPanelRef.current;
        if (!handle || handle.isRunning) return;
        autoBatchedRef.current = sessionSnapshot;
        // Fire and forget: the user may start a new game while this
        // runs. onAgentRefresh re-reads from storage when it finishes,
        // and that doesn't disturb the in-progress game's session state.
        void handle.triggerAutoBatch();
      })
      .catch(err => {
        setError(`Failed to save after game: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [session, agent, imitator, playerColor, storage, autoSelfPlay]);

  const handleMove = useCallback((nextState: GameState, move: Move) => {
    if (!agentRef.current) return;
    setSession(prev => {
      if (prev.over) return prev;
      const record: GameMoveRecord = {
        features: encodeState(prev.state),
        move,
        sideToMove: sideToMove(prev.state),
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

  const handleColorChange = useCallback((color: PlayerColor) => {
    setPlayerColor(color);
    const starting = startingStateForCurrent();
    if (starting) applyStartingState(starting);
  }, [startingStateForCurrent, applyStartingState]);

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

  const handleAgentRefresh = useCallback(async () => {
    try {
      const [reloadedAgent, reloadedImitator] = await Promise.all([
        loadAgentFromStorage(storage),
        loadPlayerImitatorFromStorage(storage),
      ]);
      setAgent(reloadedAgent);
      setImitator(reloadedImitator);
      setAgentVersion(v => v + 1);
    } catch (err) {
      setError(`Failed to reload agent: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage]);

  if (agent === null && error === null) {
    return <div style={{ fontFamily: 'sans-serif', padding: '16px' }}>Loading agent…</div>;
  }

  if (error && agent === null) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px', color: '#c00' }}>
        {error}
      </div>
    );
  }

  const humanIsWhite = playerColor === 'w';
  const currentSide = sideToMove(session.state);
  const isHumanTurn = currentSide === playerColor && !session.over;
  const interactive = isHumanTurn && !agentThinking;

  const o = session.finalOutcome;
  let outcomeText = '';
  if (o === 'white-wins') outcomeText = playerColor === 'w' ? 'You win!' : 'Agent wins!';
  else if (o === 'black-wins') outcomeText = playerColor === 'b' ? 'You win!' : 'Agent wins!';
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
            (the winning side) — that side's moves teach the agent the mating pattern.
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
            checked={playerColor === 'w'}
            onChange={() => handleColorChange('w')}
          />
          {' '}White
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="b"
            checked={playerColor === 'b'}
            onChange={() => handleColorChange('b')}
          />
          {' '}Black
        </label>
      </div>

      <div style={{ marginBottom: '8px', minHeight: '24px' }}>
        {agentThinking && (
          <span style={{ color: '#666', fontStyle: 'italic' }}>AI thinking…</span>
        )}
        {!agentThinking && !session.over && (
          <span>
            {isHumanTurn ? 'Your turn' : 'Agent to move'}
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

      {agent && (
        <div style={{ marginTop: '24px' }}>
          <TrainingPanel
            agent={agent}
            storage={storage}
            onAgentRefresh={handleAgentRefresh}
            key={agentVersion}
          />
        </div>
      )}

      <SelfPlayPanel
        ref={selfPlayPanelRef}
        agent={agent}
        imitator={imitator}
        storage={storage}
        onAgentRefresh={handleAgentRefresh}
        autoEnabled={autoSelfPlay}
        onAutoEnabledChange={handleAutoSelfPlayChange}
      />

      <InspectorPanel
        agent={agent}
        state={session.state}
        lastSearch={lastSearch}
        agentVersion={agentVersion}
      />
    </div>
  );
}
