// Play-against-Maia-2 game UI.
//
// Mirrors `Game.tsx`'s structure: load the player-imitator from
// storage, let the user pick a side + a strength bucket, then drive a
// human-vs-engine game. On game end we ONLY record the user's own
// moves into the imitator's replay buffer — the opponent network's
// replay buffer is untouched, and Maia's moves are never recorded
// anywhere (per the brief).
//
// The Maia engine is a frozen third-party network served as a static
// asset. We boot the worker the first time the user clicks "Start" so
// that visitors who never use this page never pay the ~40 MiB
// download cost.

import { useState, useEffect, useRef, useCallback } from 'react';
import Board from './Board.js';
import {
  initialState,
  applyMove,
  sideToMove,
  outcome,
  isTerminal,
  toFen,
  legalMoves,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { GameMoveRecord } from '../core/opponent/index.js';
import { legalMoveIndicesFor, mateInOneIndicesFor } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { MAIA_ELO_BUCKETS } from '../core/maia/index.js';
import {
  createOpponentStorage,
  loadPlayerImitatorFromStorage,
} from './opponent-config.js';
import {
  MaiaClient,
  fetchMaiaWeights,
  DEFAULT_MAIA_SOURCE,
  type MaiaLoadProgress,
} from './maia-client.js';

type PlayerColor = 'w' | 'b';

interface GameSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
  startingFen: string;
}

function freshSession(): GameSession {
  const state = initialState();
  return {
    state,
    history: [],
    over: false,
    finalOutcome: null,
    startingFen: toFen(state),
  };
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

type EngineStatus =
  | { phase: 'idle' }
  | { phase: 'loading'; progress: MaiaLoadProgress | null }
  | { phase: 'ready' }
  | { phase: 'error'; message: string };

export default function PlayMaiaGame() {
  const [storage] = useState<BlobStorage>(createOpponentStorage);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);

  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  const [eloBucket, setEloBucket] = useState<number>(5); // default 1500-1599
  const [session, setSession] = useState<GameSession>(freshSession);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>({ phase: 'idle' });
  const [maiaThinking, setMaiaThinking] = useState(false);
  const [lastMaiaReadout, setLastMaiaReadout] = useState<
    { move: Move; value: number; topMoves: Array<{ move: Move; prob: number }> } | null
  >(null);
  const [error, setError] = useState<string | null>(null);

  // Persisted resources whose lifetimes outlive a single React render.
  const clientRef = useRef<MaiaClient | null>(null);
  const recordedRef = useRef<GameSession | null>(null);
  const moveInFlight = useRef(false);

  // Load the imitator once on mount so we know we have somewhere to
  // route the user's moves once the game ends.
  useEffect(() => {
    let cancelled = false;
    loadPlayerImitatorFromStorage(storage)
      .then(loaded => {
        if (!cancelled) setImitator(loaded);
      })
      .catch(err => {
        if (!cancelled) {
          setError(
            `Failed to load player-imitator: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      });
    return () => { cancelled = true; };
  }, [storage]);

  // Tear down the worker when the page unmounts.
  useEffect(() => {
    return () => {
      clientRef.current?.dispose();
      clientRef.current = null;
    };
  }, []);

  const ensureEngine = useCallback(async (): Promise<MaiaClient> => {
    if (clientRef.current) return clientRef.current;
    setEngineStatus({ phase: 'loading', progress: null });
    try {
      const { blob, manifest } = await fetchMaiaWeights({
        source: DEFAULT_MAIA_SOURCE,
        storage,
        onProgress: p => setEngineStatus({ phase: 'loading', progress: p }),
      });
      const client = await MaiaClient.create(blob, manifest);
      clientRef.current = client;
      setEngineStatus({ phase: 'ready' });
      return client;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setEngineStatus({ phase: 'error', message });
      throw err;
    }
  }, [storage]);

  const newGame = useCallback(() => {
    moveInFlight.current = false;
    recordedRef.current = null;
    setMaiaThinking(false);
    setLastMaiaReadout(null);
    setError(null);
    setSession(freshSession());
  }, []);

  // Drive Maia's move whenever it's its turn.
  useEffect(() => {
    if (session.over) return;
    if (clientRef.current === null) return;
    if (moveInFlight.current) return;

    const current = sideToMove(session.state);
    if (current === playerColor) return;

    moveInFlight.current = true;
    setMaiaThinking(true);
    const fen = toFen(session.state);
    const elo = MAIA_ELO_BUCKETS[eloBucket].midpoint;
    const capturedState = session.state;
    clientRef.current
      .selectMove(fen, elo, { oppoElo: elo, temperature: 0 })
      .then(({ move, value, moveProbs }) => {
        moveInFlight.current = false;
        setMaiaThinking(false);

        const legal = legalMoves(capturedState);
        if (!legal.includes(move)) {
          // Belt and braces; engine.selectMove already guards.
          setError(`Maia returned illegal move ${move}`);
          return;
        }
        // Stash a top-3 readout for display.
        const top = Array.from(moveProbs.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([m, p]) => ({ move: m, prob: p }));
        setLastMaiaReadout({ move, value, topMoves: top });

        // Record Maia's move BUT only into the session history with
        // sideToMove = maia's side. The user-only routing in the
        // game-over effect filters it out before training data is built.
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
            ...prev,
            state: nextState,
            history: [...prev.history, record],
            over: gameOver,
            finalOutcome: gameOver ? o : null,
          };
        });
      })
      .catch(err => {
        moveInFlight.current = false;
        setMaiaThinking(false);
        setError(`Maia move failed: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [session, playerColor, eloBucket]);

  // On game end: route the USER's moves only into the imitator. Maia's
  // moves are never recorded anywhere — that's the whole point of this
  // page's training routing (see CLAUDE.md / brief).
  useEffect(() => {
    if (!session.over) return;
    if (!imitator) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;
    const userMoves = session.history.filter(r => r.sideToMove === playerColor);
    imitator.recordUserMoves(userMoves, session.finalOutcome);
    imitator.saveTo(storage).catch(err => {
      setError(
        `Failed to persist imitator after game: ${err instanceof Error ? err.message : String(err)}`,
      );
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
        ...prev,
        state: nextState,
        history: [...prev.history, record],
        over: gameOver,
        finalOutcome: gameOver ? o : null,
      };
    });
  }, []);

  const handleStart = useCallback(async () => {
    try {
      await ensureEngine();
    } catch {
      return;
    }
    newGame();
  }, [ensureEngine, newGame]);

  const handleResign = useCallback(() => {
    setSession(prev => {
      if (prev.over) return prev;
      // Resignation = the OTHER side wins. The imitator value head
      // sees this as a loss for the user-side moves, matching the
      // "user resigned" semantics.
      const o: Outcome = playerColor === 'w' ? 'black-wins' : 'white-wins';
      return { ...prev, over: true, finalOutcome: o };
    });
  }, [playerColor]);

  // ── Render ─────────────────────────────────────────────────────────────

  const currentSide = sideToMove(session.state);
  const isHumanTurn = !session.over && currentSide === playerColor;
  const interactive =
    engineStatus.phase === 'ready' && isHumanTurn && !maiaThinking;
  const boardOrientation = playerColor === 'w' ? 'white' : 'black';

  let outcomeText = '';
  if (session.finalOutcome === 'white-wins') {
    outcomeText = playerColor === 'w' ? 'You win!' : 'Maia wins!';
  } else if (session.finalOutcome === 'black-wins') {
    outcomeText = playerColor === 'b' ? 'You win!' : 'Maia wins!';
  } else if (session.finalOutcome === 'draw') {
    outcomeText = 'Draw!';
  }

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <fieldset
        disabled={engineStatus.phase === 'loading' || maiaThinking}
        style={{
          marginBottom: '16px',
          padding: '12px 16px',
          border: '1px solid #ccc',
          borderRadius: '6px',
        }}
      >
        <legend style={{ fontWeight: 'bold' }}>Game setup</legend>
        <div style={{ marginBottom: '10px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <strong>You play as:</strong>
          {(['w', 'b'] as PlayerColor[]).map(c => (
            <label key={c} style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                name="maiaPlayerColor"
                value={c}
                checked={playerColor === c}
                onChange={() => setPlayerColor(c)}
              />
              {' '}
              {c === 'w' ? 'White' : 'Black'}
            </label>
          ))}
        </div>
        <div style={{ marginBottom: '10px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <strong>Maia strength:</strong>
          <select
            value={eloBucket}
            onChange={e => setEloBucket(Number(e.target.value))}
            style={{ padding: '3px 6px' }}
          >
            {MAIA_ELO_BUCKETS.map(b => (
              <option key={b.id} value={b.id}>
                {b.label} (≈ {b.midpoint})
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            type="button"
            onClick={handleStart}
            disabled={engineStatus.phase === 'loading'}
            style={{ padding: '6px 16px', cursor: 'pointer' }}
          >
            {engineStatus.phase === 'ready' ? 'New game' : 'Start (download Maia weights)'}
          </button>
          {!session.over && session.history.length > 0 && (
            <button
              type="button"
              onClick={handleResign}
              style={{ padding: '6px 16px', cursor: 'pointer' }}
            >
              Resign
            </button>
          )}
        </div>
      </fieldset>

      {engineStatus.phase === 'loading' && (
        <div style={{
          marginBottom: '12px',
          padding: '10px 14px',
          background: '#f5f5f5',
          border: '1px solid #ddd',
          borderRadius: '6px',
          fontSize: '0.9em',
          color: '#1a1a1a',
        }}>
          Loading Maia-2 weights…{' '}
          {engineStatus.progress && (
            <>
              {fmtBytes(engineStatus.progress.loaded)}
              {engineStatus.progress.total > 0 && (
                <> / {fmtBytes(engineStatus.progress.total)}
                  {' '}
                  ({((engineStatus.progress.loaded / engineStatus.progress.total) * 100).toFixed(1)}%)
                </>
              )}
            </>
          )}
        </div>
      )}

      {engineStatus.phase === 'error' && (
        <div style={{
          marginBottom: '12px',
          padding: '10px 14px',
          background: '#fee',
          border: '1px solid #c66',
          borderRadius: '6px',
          color: '#900',
          fontSize: '0.9em',
        }}>
          Failed to load Maia: {engineStatus.message}
        </div>
      )}

      {(engineStatus.phase === 'ready' || session.history.length > 0) && (
        <>
          <div style={{ marginBottom: '8px', minHeight: '24px' }}>
            {maiaThinking
              ? <span style={{ color: '#666', fontStyle: 'italic' }}>Maia thinking…</span>
              : !session.over
                ? <span>{isHumanTurn ? 'Your turn' : 'Maia to move'} — {currentSide === 'w' ? 'White' : 'Black'} to move</span>
                : null}
          </div>

          <div style={{ maxWidth: '480px' }}>
            <Board
              state={session.state}
              interactive={interactive}
              onMove={handleUserMove}
              boardOrientation={boardOrientation}
            />
          </div>

          {lastMaiaReadout && (
            <div style={{
              marginTop: '12px',
              padding: '8px 12px',
              background: '#f5f5f5',
              border: '1px solid #ddd',
              borderRadius: '6px',
              fontSize: '0.85em',
              color: '#444',
            }}>
              <strong>Last Maia move:</strong> {lastMaiaReadout.move}{' '}
              <span style={{ color: '#666' }}>
                (value head: {lastMaiaReadout.value.toFixed(3)})
              </span>
              <br />
              <strong>Maia's top candidates:</strong>{' '}
              {lastMaiaReadout.topMoves
                .map(m => `${m.move} (${(m.prob * 100).toFixed(1)}%)`)
                .join(', ')}
            </div>
          )}

          {session.over && (
            <div style={{
              marginTop: '16px',
              padding: '12px 16px',
              background: '#f0f4ff',
              border: '1px solid #aac',
              borderRadius: '6px',
              color: '#1a1a1a',
              display: 'flex',
              alignItems: 'center',
              gap: '16px',
            }}>
              <strong style={{ fontSize: '1.1em' }}>{outcomeText}</strong>
              <button
                type="button"
                onClick={newGame}
                style={{ padding: '6px 16px', cursor: 'pointer' }}
              >
                New game
              </button>
            </div>
          )}
        </>
      )}

      <p style={{
        marginTop: '24px',
        fontSize: '0.85em',
        color: '#666',
        maxWidth: '620px',
      }}>
        <strong>About this page.</strong>{' '}
        <a href="https://github.com/CSSLab/maia2" target="_blank" rel="noopener noreferrer">Maia-2</a>{' '}
        is a third-party human-style chess engine (NeurIPS 2024,
        MIT-licensed). Picking a bucket selects a target rating Maia
        was trained to imitate. The weights file is ~40 MiB on first
        load and is cached locally afterward.{' '}
        <strong>Training routing:</strong>{' '}
        your moves train the player-imitator the same way games against the
        nn-chess opponent do. Maia's moves are NOT recorded anywhere — we
        treat Maia as a sparring partner, not as training data.
      </p>

      {error && (
        <div style={{ marginTop: '12px', color: '#c00', fontSize: '0.9em' }}>
          Error: {error}
        </div>
      )}
    </div>
  );
}
