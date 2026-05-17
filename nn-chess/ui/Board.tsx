/**
 * Interactive chess board React island.
 *
 * Owns game state via the `nn-chess/core/rules` API (never imports chess.js
 * directly). Renders a `react-chessboard` `<Chessboard />` and exposes
 * callback seams the future agent integration will wire into.
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
 */
export default function Board({ onMove, onGameOver }: BoardProps) {
  const [{ game, moves }, setState] = useState<InternalState>(freshState);

  const handleReset = useCallback(() => {
    setState(freshState());
  }, []);

  const handlePieceDrop = useCallback(
    ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
      if (targetSquare === null) return false;
      if (isTerminal(game)) return false;

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

      setState(prev => ({
        game: nextGame,
        moves: [...prev.moves, appliedMove],
      }));

      onMove?.(nextGame, appliedMove);

      const o = outcome(nextGame);
      if (o !== 'ongoing') {
        onGameOver?.(nextGame, o);
      }

      return true;
    },
    [game, onMove, onGameOver],
  );

  const currentOutcome = outcome(game);
  const side = sideToMove(game);
  const checked = inCheck(game);
  const terminal = isTerminal(game);

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <Chessboard
        options={{
          position: toFen(game),
          onPieceDrop: handlePieceDrop,
          allowDragging: !terminal,
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
        <button
          type="button"
          onClick={handleReset}
          style={{ padding: '4px 12px', cursor: 'pointer', marginLeft: 'auto' }}
        >
          Reset
        </button>
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
