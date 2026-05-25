// Play directly against the player-imitator network.
//
// This page lets the user play a normal game against the imitator (the
// behaviour-clone of the user). It exists as both
//   - an interpretability tool — "does the imitator actually play like me?"
//   - a way to collect more imitator training data per session, since
//     every finished game here still feeds `imitator.recordUserMoves`.
//
// Differences from the main `Game.tsx`:
//   - The opponent for each move is the imitator (no MCTS, no value
//     search). Moves come straight from the imitator's policy head via
//     `imitator.selectMove(state)`. See core/imitator/index.ts for why
//     that's still consistent with the README's "no heuristics" rule.
//   - There's no training UI here: no opponent self-play, no Stockfish,
//     no gradient-step controls. Training continues to happen on the
//     main page; this route's only side-effect is appending to the
//     imitator's replay buffer when a game ends, exactly like the
//     main page does.
//   - The opponent network is not touched at all (no recordGameResult,
//     no saveTo).

import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import {
  initialState,
  applyMove,
  sideToMove,
  outcome,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { GameMoveRecord } from '../core/opponent/index.js';
import { legalMoveIndicesFor, mateInOneIndicesFor } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import {
  createOpponentStorage,
  loadPlayerImitatorFromStorage,
} from './opponent-config.js';

type PlayerColor = 'w' | 'b';
type ColorPreference = PlayerColor | 'random';

interface Session {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
  /** True if the user resigned (vs the game ending by checkmate/stalemate/etc.). */
  resigned: boolean;
}

function freshSession(): Session {
  return {
    state: initialState(),
    history: [],
    over: false,
    finalOutcome: null,
    resigned: false,
  };
}

function pickColor(pref: ColorPreference): PlayerColor {
  if (pref === 'random') return Math.random() < 0.5 ? 'w' : 'b';
  return pref;
}

export default function PlayImitatorGame() {
  const [storage] = useState<BlobStorage>(createOpponentStorage);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);
  const [colorPreference, setColorPreference] = useState<ColorPreference>('w');
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  // Non-zero temperature lets the imitator vary its play across games so
  // the page is actually useful for "is the model multimodal?" inspection.
  // Default is 0 (greedy) to match the imitator's training objective; the
  // user can opt into sampling via the toggle below.
  const [temperature, setTemperature] = useState(0);
  const [session, setSession] = useState<Session>(freshSession);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Prevents double-fire under React Strict Mode (mirrors Game.tsx).
  const imitatorMoveInFlight = useRef(false);
  // Tracks which finished session has already been recorded so a
  // re-render of the game-over effect doesn't double-record.
  const recordedRef = useRef<Session | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadPlayerImitatorFromStorage(storage)
      .then(loaded => { if (!cancelled) setImitator(loaded); })
      .catch(err => {
        if (!cancelled) {
          setError(`Failed to load imitator: ${err instanceof Error ? err.message : String(err)}`);
        }
      });
    return () => { cancelled = true; };
  }, [storage]);

  const startNewGame = useCallback(() => {
    imitatorMoveInFlight.current = false;
    recordedRef.current = null;
    setThinking(false);
    setPlayerColor(pickColor(colorPreference));
    setSession(freshSession());
    setError(null);
  }, [colorPreference]);

  const handleColorChange = useCallback((pref: ColorPreference) => {
    setColorPreference(pref);
    setPlayerColor(pickColor(pref));
    setSession(freshSession());
    imitatorMoveInFlight.current = false;
    recordedRef.current = null;
    setThinking(false);
    setError(null);
  }, []);

  // Drive the imitator's move whenever it's its turn.
  useEffect(() => {
    if (imitator === null) return;
    if (session.over) return;
    if (imitatorMoveInFlight.current) return;

    const currentSide = sideToMove(session.state);
    if (currentSide === playerColor) return;

    imitatorMoveInFlight.current = true;
    setThinking(true);

    const capturedState = session.state;
    imitator.selectMove(capturedState, { temperature })
      .then(({ move }) => {
        imitatorMoveInFlight.current = false;

        const record: GameMoveRecord = {
          features: encodeState(capturedState),
          move,
          sideToMove: sideToMove(capturedState),
          legalMoveIndices: legalMoveIndicesFor(capturedState),
          mateMoveIndices: mateInOneIndicesFor(capturedState),
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
            resigned: false,
          };
        });

        setThinking(false);
      })
      .catch(err => {
        imitatorMoveInFlight.current = false;
        setThinking(false);
        setError(`Imitator move failed: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [imitator, session, playerColor, temperature]);

  // When the game ends, train the imitator on the user's own moves
  // (exactly like the main page does). Imitator-side moves are
  // deliberately excluded so the imitator never trains on its own output.
  useEffect(() => {
    if (!session.over) return;
    if (!imitator) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;
    const userMoves = session.history.filter(r => r.sideToMove === playerColor);
    imitator.recordUserMoves(userMoves, session.finalOutcome);
    imitator.saveTo(storage).catch(err => {
      setError(`Failed to save imitator after game: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [session, imitator, playerColor, storage]);

  const handleUserMove = useCallback((nextState: GameState, move: Move) => {
    setSession(prev => {
      if (prev.over) return prev;
      const record: GameMoveRecord = {
        features: encodeState(prev.state),
        move,
        sideToMove: sideToMove(prev.state),
        legalMoveIndices: legalMoveIndicesFor(prev.state),
        mateMoveIndices: mateInOneIndicesFor(prev.state),
      };
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';
      return {
        state: nextState,
        history: [...prev.history, record],
        over: gameOver,
        finalOutcome: gameOver ? o : null,
        resigned: false,
      };
    });
  }, []);

  const handleResign = useCallback(() => {
    if (session.over) return;
    // Resignation = the user loses. Mirrors the convention used by the
    // main Game.tsx: the user voluntarily ends the game and the imitator
    // still trains on the moves the user actually made up to this point.
    const losing: Outcome = playerColor === 'w' ? 'black-wins' : 'white-wins';
    setSession(prev => ({ ...prev, over: true, finalOutcome: losing, resigned: true }));
  }, [session.over, playerColor]);

  if (imitator === null && error === null) {
    return <div style={{ fontFamily: 'sans-serif', padding: '16px' }}>Loading imitator…</div>;
  }
  if (error && imitator === null) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px', color: '#c00' }}>
        {error}
      </div>
    );
  }

  const humanIsWhite = playerColor === 'w';
  const currentSide = sideToMove(session.state);
  const isHumanTurn = currentSide === playerColor && !session.over;
  const interactive = isHumanTurn && !thinking;

  const o = session.finalOutcome;
  let outcomeText = '';
  if (o === 'white-wins') outcomeText = playerColor === 'w' ? 'You win!' : 'Imitator wins!';
  else if (o === 'black-wins') outcomeText = playerColor === 'b' ? 'You win!' : 'Imitator wins!';
  else if (o === 'draw') outcomeText = 'Draw!';
  if (session.resigned) outcomeText += ' (you resigned)';

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>You play as:</strong>
        {(['w', 'b', 'random'] as ColorPreference[]).map(c => (
          <label key={c} style={{ cursor: 'pointer' }}>
            <input
              type="radio"
              name="playerColor"
              value={c}
              checked={colorPreference === c}
              onChange={() => handleColorChange(c)}
            />
            {' '}
            {c === 'w' ? 'White' : c === 'b' ? 'Black' : 'Random'}
          </label>
        ))}
      </div>

      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', gap: '8px', alignItems: 'center', cursor: 'pointer' }}>
          <strong>Imitator temperature:</strong>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={e => setTemperature(Number(e.target.value))}
            style={{ width: '140px' }}
          />
          <span style={{ fontFamily: 'monospace', minWidth: '2.5em' }}>{temperature.toFixed(1)}</span>
        </label>
        <span style={{ fontSize: '0.85em', color: '#666' }}>
          0 = greedy (deterministic); higher = more variety in the imitator's replies.
        </span>
      </div>

      <div style={{ marginBottom: '8px', minHeight: '24px' }}>
        {thinking && (
          <span style={{ color: '#666', fontStyle: 'italic' }}>Imitator thinking…</span>
        )}
        {!thinking && !session.over && (
          <span>
            {isHumanTurn ? 'Your turn' : 'Imitator to move'}
            {' — '}{currentSide === 'w' ? 'White' : 'Black'} to move
          </span>
        )}
      </div>

      <Board
        state={session.state}
        interactive={interactive}
        onMove={handleUserMove}
        boardOrientation={humanIsWhite ? 'white' : 'black'}
      />

      <div style={{ marginTop: '12px', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button
          type="button"
          onClick={startNewGame}
          style={{ padding: '6px 16px', cursor: 'pointer' }}
        >
          New game
        </button>
        <button
          type="button"
          onClick={handleResign}
          disabled={session.over}
          style={{ padding: '6px 16px', cursor: session.over ? 'not-allowed' : 'pointer' }}
        >
          Resign
        </button>
      </div>

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
            flexWrap: 'wrap',
          }}
        >
          <strong style={{ fontSize: '1.1em' }}>{outcomeText}</strong>
          <span style={{ fontSize: '0.85em', color: '#555' }}>
            Your moves from this game have been added to the imitator's training buffer.
          </span>
        </div>
      )}

      {error && (
        <div style={{ marginTop: '8px', color: '#c00', fontSize: '0.9em' }}>
          {error}
        </div>
      )}

      {session.history.length > 0 && (
        <details style={{ marginTop: '14px' }}>
          <summary style={{ cursor: 'pointer', userSelect: 'none' }}>
            Move history ({session.history.length})
          </summary>
          <ol
            style={{
              margin: '6px 0 0 1.5em',
              padding: 0,
              fontFamily: 'monospace',
              fontSize: '0.9em',
            }}
          >
            {session.history.map((r, i) => (
              <li key={i}>
                {r.move}
                <span style={{ color: '#888' }}>
                  {' '}({r.sideToMove === playerColor ? 'you' : 'imitator'})
                </span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
