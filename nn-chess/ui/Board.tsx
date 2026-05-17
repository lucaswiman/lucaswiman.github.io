/**
 * Interactive chess board React island.
 *
 * Owns game state via the `nn-chess/core/rules` API (never imports chess.js
 * directly). Renders a `react-chessboard` `<Chessboard />` and exposes
 * callback seams the future agent integration will wire into.
 *
 * Controlled vs uncontrolled mode
 * ────────────────────────────────
 * If the `state` prop is provided, `Board` runs in **controlled mode**:
 *   - It renders the position given by `state` rather than maintaining its
 *     own internal `GameState`.
 *   - Move handling still goes through the `onMove` callback; the parent is
 *     responsible for updating `state` in response.
 * If `state` is omitted, `Board` runs in **uncontrolled mode** (the original
 * behavior): it owns its own internal state and the "Reset" button works as
 * before. This preserves the standalone "play yourself" demo.
 *
 * The `interactive` prop (default `true`) disables drag-drop when `false`.
 * `Game.tsx` sets this to `false` while the agent is thinking.
 */

import { useState, useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import type { PieceDropHandlerArgs } from 'react-chessboard';
import {
  initialState,
  applyMove,
  legalMoves,
  sideToMove,
  inCheck,
  outcome,
  isTerminal,
  toFen,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';

export type { GameState, Move, Outcome };

export interface BoardProps {
  /**
   * Optional controlled state. When provided, the board renders this
   * position and does not maintain its own internal state.
   * When absent (default), Board owns its own state (uncontrolled mode).
   */
  state?: GameState;

  /**
   * When false, drag-drop is disabled regardless of the game state.
   * Defaults to true. Set to false while the agent is thinking.
   */
  interactive?: boolean;

  /**
   * Board orientation — 'white' shows white at the bottom (default),
   * 'black' flips the board so black is at the bottom.
   * Game.tsx sets this based on which color the human is playing.
   */
  boardOrientation?: 'white' | 'black';

  /**
   * Called after every successful move with the resulting state and the
   * UCI move string. The future Agent integration will use this to
   * trigger NN-driven replies.
   */
  onMove?: (state: GameState, move: Move) => void;
  /**
   * Called once the game reaches a terminal state. The future training
   * loop will hook in here to record the result.
   */
  onGameOver?: (state: GameState, outcome: Outcome) => void;
}

interface InternalState {
  game: GameState;
  /** Ordered list of UCI moves played so far. */
  moves: Move[];
}

function freshState(): InternalState {
  return { game: initialState(), moves: [] };
}

/** Maps an `Outcome` to a human-readable label. */
function outcomeLabel(o: Outcome): string {
  switch (o) {
    case 'white-wins': return 'White wins';
    case 'black-wins': return 'Black wins';
    case 'draw':       return 'Draw';
    case 'ongoing':    return 'Ongoing';
  }
}

/**
 * A self-contained chess board that lets two human players take turns.
 * Mount in an Astro page with `client:only="react"` — react-chessboard
 * accesses the DOM and is not SSR-safe.
 *
 * Accepts an optional `state` prop for controlled mode (used by Game.tsx).
 * Without it, Board owns its own state (original uncontrolled behavior).
 */
export default function Board({
  state: controlledState,
  interactive = true,
  boardOrientation = 'white',
  onMove,
  onGameOver,
}: BoardProps) {
  // Uncontrolled internal state — only used when `controlledState` is absent.
  const [internalState, setInternalState] = useState<InternalState>(freshState);

  const isControlled = controlledState !== undefined;

  // The "live" game is the controlled state if provided, otherwise the
  // component's own internal state.
  const game: GameState = isControlled ? controlledState : internalState.game;

  const handleReset = useCallback(() => {
    if (!isControlled) {
      setInternalState(freshState());
    }
  }, [isControlled]);

  const handlePieceDrop = useCallback(
    ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
      if (targetSquare === null) return false;
      if (isTerminal(game)) return false;
      if (!interactive) return false;

      const bareMove: Move = `${sourceSquare}${targetSquare}`;

      // Attempt the bare move; fall back to queen promotion when the
      // bare move is illegal but the queen-promotion variant is legal.
      // TODO: add a promotion-picker UI so the player can choose the piece.
      let appliedMove: Move = bareMove;
      let nextGame: GameState;
      try {
        nextGame = applyMove(game, bareMove);
      } catch {
        const promoMove: Move = `${bareMove}q`;
        if (legalMoves(game).includes(promoMove)) {
          try {
            nextGame = applyMove(game, promoMove);
            appliedMove = promoMove;
          } catch {
            return false;
          }
        } else {
          return false;
        }
      }

      // Update internal state in uncontrolled mode.
      if (!isControlled) {
        setInternalState(prev => ({
          game: nextGame,
          moves: [...prev.moves, appliedMove],
        }));
      }

      onMove?.(nextGame, appliedMove);

      const o = outcome(nextGame);
      if (o !== 'ongoing') {
        onGameOver?.(nextGame, o);
      }

      return true;
    },
    [game, interactive, isControlled, onMove, onGameOver],
  );

  const currentOutcome = outcome(game);
  const side = sideToMove(game);
  const checked = inCheck(game);
  const terminal = isTerminal(game);

  // In uncontrolled mode, show the internal move list.
  // In controlled mode, the parent owns the history — we don't show it here.
  const moves = isControlled ? [] : internalState.moves;

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <Chessboard
        options={{
          position: toFen(game),
          onPieceDrop: handlePieceDrop,
          allowDragging: interactive && !terminal,
          boardOrientation,
          boardStyle: {
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
          },
        }}
      />

      <div
        style={{
          marginTop: '12px',
          display: 'flex',
          gap: '12px',
          alignItems: 'center',
          flexWrap: 'wrap',
        }}
      >
        <span>
          <strong>To move:</strong>{' '}
          {terminal ? '—' : side === 'w' ? 'White' : 'Black'}
        </span>
        {checked && !terminal && (
          <span style={{ color: '#c00', fontWeight: 'bold' }}>In check</span>
        )}
        <span>
          <strong>Status:</strong> {outcomeLabel(currentOutcome)}
        </span>
        {/* Only show Reset in uncontrolled mode; Game.tsx provides its own controls. */}
        {!isControlled && (
          <button
            type="button"
            onClick={handleReset}
            style={{ padding: '4px 12px', cursor: 'pointer', marginLeft: 'auto' }}
          >
            Reset
          </button>
        )}
      </div>

      {moves.length > 0 && (
        <details style={{ marginTop: '10px' }}>
          <summary style={{ cursor: 'pointer', userSelect: 'none' }}>
            Move history ({moves.length})
          </summary>
          <ol
            style={{
              margin: '6px 0 0 1.5em',
              padding: 0,
              fontFamily: 'monospace',
              fontSize: '0.9em',
            }}
          >
            {moves.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
