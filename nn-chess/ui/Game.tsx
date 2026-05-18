import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import { InspectorPanel } from './InspectorPanel.js';
import {
  initialState,
  applyMove,
  sideToMove,
  outcome,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { Agent, GameMoveRecord } from '../core/agent/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { createAgentStorage, loadAgentFromStorage } from './agent-config.js';

type PlayerColor = 'w' | 'b';

interface GameSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
}

function freshSession(): GameSession {
  return { state: initialState(), history: [], over: false, finalOutcome: null };
}

export default function Game() {
  const [agent, setAgent] = useState<Agent | null>(null);
  const [storage] = useState<BlobStorage>(createAgentStorage);
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  const [session, setSession] = useState<GameSession>(freshSession);
  const [agentThinking, setAgentThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentVersion, setAgentVersion] = useState(0);
  const [lastSearch, setLastSearch] = useState<SearchResult | null>(null);

  const agentRef = useRef<Agent | null>(null);
  agentRef.current = agent;

  // Prevents double-fire of the agent move under React Strict Mode.
  const agentMoveInFlight = useRef(false);
  const recordedRef = useRef<GameSession | null>(null);

  const resetSession = useCallback(() => {
    agentMoveInFlight.current = false;
    recordedRef.current = null;
    setAgentThinking(false);
    setSession(freshSession());
    setLastSearch(null);
    setError(null);
  }, []);

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
    resetSession();
  }, [resetSession]);

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

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
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
            onClick={resetSession}
            style={{ padding: '6px 16px', cursor: 'pointer' }}
          >
            New game
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
