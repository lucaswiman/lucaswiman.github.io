import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import { InspectorPanel } from './InspectorPanel.js';
import {
  initialState,
  applyMove,
  fromFen,
  sideToMove,
  outcome,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { Agent, GameMoveRecord } from '../core/agent/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { createAgentStorage, loadAgentFromStorage } from './agent-config.js';
import { ENDGAME_PRESETS, ENDGAME_PRESETS_BY_ID } from './endgame-presets.js';
import {
  LICHESS_CATEGORIES,
  LICHESS_CATEGORIES_BY_ID,
  randomPuzzle,
  lichessPuzzleUrl,
} from './lichess-endgames.js';
import type { LichessPuzzle } from './lichess-endgames.js';

type PlayerColor = 'w' | 'b';
type SetupId = 'standard' | 'custom' | string;

interface GameSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
}

function sessionFromState(state: GameState): GameSession {
  return { state, history: [], over: false, finalOutcome: null };
}

export default function Game() {
  const [agent, setAgent] = useState<Agent | null>(null);
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

  const agentRef = useRef<Agent | null>(null);
  agentRef.current = agent;

  // Prevents double-fire of the agent move under React Strict Mode.
  const agentMoveInFlight = useRef(false);
  const recordedRef = useRef<GameSession | null>(null);

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
    loadAgentFromStorage(storage)
      .then(loaded => { if (!cancelled) setAgent(loaded); })
      .catch(err => {
        if (!cancelled) setError(`Failed to load agent: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => { cancelled = true; };
  }, [storage]);

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
    agent.saveTo(storage).catch(err => {
      setError(`Failed to save after game: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [session, agent, storage]);

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
      const reloaded = await loadAgentFromStorage(storage);
      setAgent(reloaded);
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

      <InspectorPanel
        agent={agent}
        state={session.state}
        lastSearch={lastSearch}
        agentVersion={agentVersion}
      />
    </div>
  );
}
