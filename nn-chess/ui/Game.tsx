/**
 * Game.tsx — top-level React component for nn-chess human-vs-agent gameplay.
 *
 * Responsibilities:
 *   - Owns one Agent instance, loaded from/saved to localStorage via the
 *     namespaced adapter (prefix 'nn-chess-v1/').
 *   - Lets the user pick which color they play (white/black).
 *   - Renders <Board /> in controlled mode, applying agent moves when it's
 *     the agent's turn.
 *   - Calls agent.recordGameResult() on game end, then agent.saveTo().
 *   - Includes <TrainingPanel /> for background training controls.
 *
 * Agent config:
 *   simulations: 64  — kept low for real-time CPU play; a move should
 *                       come back in ≲ 5 seconds on a typical laptop.
 *   cPuct:       1.5 — standard PUCT exploration constant.
 *   temperature: 0   — greedy play (argmax on visit counts) for consistency.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import {
  initialState,
  applyMove,
  sideToMove,
  outcome,
  isTerminal,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import {
  loadAgent,
  type Agent,
  type AgentConfig,
  type GameMoveRecord,
} from '../core/agent/index.js';
import { ChessNet } from '../core/nn/index.js';
import { createReplayBuffer } from '../core/training/index.js';
import { createLocalStorageAdapter } from '../adapters/storage-localstorage.js';
import type { BlobStorage } from '../core/storage/index.js';

// ── Constants ─────────────────────────────────────────────────────────────────

const STORAGE_PREFIX = 'nn-chess-v1/';
const REPLAY_CAPACITY = 10_000;

/**
 * Default agent config. simulations=64 keeps move time ≲ 5 seconds on a
 * CPU. Raise for stronger (slower) play once baseline is working.
 */
const AGENT_CONFIG: AgentConfig = {
  simulations: 64,
  cPuct: 1.5,
  temperature: 0, // greedy argmax — deterministic, no rng needed
};

// ── Helper: build the namespaced localStorage adapter ─────────────────────────

function makeStorage(): BlobStorage {
  return createLocalStorageAdapter({ prefix: STORAGE_PREFIX });
}

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── Component ─────────────────────────────────────────────────────────────────

export default function Game() {
  // Agent — null while loading from storage.
  const [agent, setAgent] = useState<Agent | null>(null);
  const [storage] = useState<BlobStorage>(makeStorage);

  // Which color the human plays.
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');

  // Current game session.
  const [session, setSession] = useState<GameSession>(freshSession);

  // True while the agent is computing a move.
  const [agentThinking, setAgentThinking] = useState(false);

  // Error message for any unhandled async failures.
  const [error, setError] = useState<string | null>(null);

  // Track agent refresh trigger so TrainingPanel can signal "agent updated".
  const [agentVersion, setAgentVersion] = useState(0);

  // Ref to hold the latest agent so async callbacks don't close over stale values.
  const agentRef = useRef<Agent | null>(null);
  agentRef.current = agent;

  // Ref to prevent double-triggering agent moves in React Strict Mode.
  const agentMoveInFlight = useRef(false);

  // ── Load agent on mount ──────────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const freshNet = ChessNet.create({ seed: Date.now() });
        const loaded = await loadAgent(storage, {
          net: freshNet,
          config: AGENT_CONFIG,
          replayCapacity: REPLAY_CAPACITY,
        });
        if (!cancelled) {
          setAgent(loaded);
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Failed to load agent: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [storage]);

  // ── Agent move trigger ────────────────────────────────────────────────────────

  useEffect(() => {
    if (agent === null) return;
    if (session.over) return;
    if (agentThinking) return;

    const currentSide = sideToMove(session.state);
    if (currentSide === playerColor) return; // Human's turn.

    // Agent's turn — fire off a move.
    if (agentMoveInFlight.current) return;
    agentMoveInFlight.current = true;
    setAgentThinking(true);

    const capturedState = session.state;

    agent.selectMove(capturedState).then(({ move }) => {
      agentMoveInFlight.current = false;

      // Record the move in history before applying.
      const record: GameMoveRecord = {
        features: encodeState(capturedState),
        move,
        sideToMove: sideToMove(capturedState),
      };

      const nextState = applyMove(capturedState, move);
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';

      setSession(prev => {
        // Guard against stale closure — only apply if session hasn't changed.
        if (prev.state !== capturedState) return prev;
        const newHistory = [...prev.history, record];
        return {
          state: nextState,
          history: newHistory,
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
  }, [agent, session, playerColor, agentThinking]);

  // ── Record completed game ─────────────────────────────────────────────────────

  // Ref to avoid double-recording in Strict Mode.
  const recordedRef = useRef<GameSession | null>(null);

  useEffect(() => {
    if (!session.over) return;
    if (!agent) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;

    agent.recordGameResult(session.history, session.finalOutcome);
    // Persist asynchronously — don't block UI.
    agent.saveTo(storage).catch(err => {
      setError(`Failed to save after game: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [session, agent, storage]);

  // ── Human move handler ────────────────────────────────────────────────────────

  const handleMove = useCallback((nextState: GameState, move: Move) => {
    if (!agentRef.current) return;

    // Reconstruct the state before the move from our session state.
    // Board calls onMove with the state AFTER the move, so we need
    // to re-derive the pre-move state. We capture session.state at the
    // time the callback fires.
    setSession(prev => {
      if (prev.over) return prev;

      // Encode the pre-move state (prev.state) for the history record.
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

  // We don't use Board's onGameOver — we detect terminal state ourselves in
  // handleMove / the agent move path and set session.over. This avoids a
  // double-record if Board fires onGameOver.

  // ── New game ──────────────────────────────────────────────────────────────────

  const startNewGame = useCallback(() => {
    agentMoveInFlight.current = false;
    recordedRef.current = null;
    setAgentThinking(false);
    setSession(freshSession());
    setError(null);
  }, []);

  // ── Color picker ──────────────────────────────────────────────────────────────

  const handleColorChange = useCallback((color: PlayerColor) => {
    setPlayerColor(color);
    agentMoveInFlight.current = false;
    recordedRef.current = null;
    setAgentThinking(false);
    setSession(freshSession());
    setError(null);
  }, []);

  // ── Agent refresh (called by TrainingPanel after training) ────────────────────

  const handleAgentRefresh = useCallback(async () => {
    try {
      const freshNet = ChessNet.create({ seed: Date.now() });
      const reloaded = await loadAgent(storage, {
        net: freshNet,
        config: AGENT_CONFIG,
        replayCapacity: REPLAY_CAPACITY,
      });
      setAgent(reloaded);
      setAgentVersion(v => v + 1);
    } catch (err) {
      setError(`Failed to reload agent: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage]);

  // ── Rendering ─────────────────────────────────────────────────────────────────

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
  const agentColor = humanIsWhite ? 'b' : 'w';
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
      {/* Color picker */}
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

      {/* Status bar */}
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

      {/* Board */}
      <Board
        state={session.state}
        interactive={interactive}
        onMove={handleMove}
        boardOrientation={humanIsWhite ? 'white' : 'black'}
      />

      {/* Game-over banner */}
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
            New game
          </button>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div style={{ marginTop: '8px', color: '#c00', fontSize: '0.9em' }}>
          {error}
        </div>
      )}

      {/* Training panel */}
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
    </div>
  );
}
