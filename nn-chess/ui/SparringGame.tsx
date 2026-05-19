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
import type { Agent, GameMoveRecord } from '../core/agent/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { StockfishEngine } from './stockfish-protocol.js';
import type { StockfishConfig } from './stockfish-protocol.js';
import { createAgentStorage, loadAgentFromStorage } from './agent-config.js';

type AgentColor = 'w' | 'b' | 'alternate';
type RunState = 'idle' | 'running' | 'paused' | 'stopping';

interface Tally {
  wins: number;
  draws: number;
  losses: number;
}

interface SparringSession {
  /** Current board state. */
  state: GameState;
  /** Move history for the current game. */
  history: GameMoveRecord[];
  /** Which game number (1-indexed) we are on, within this run. */
  gameIndex: number;
  /** Which color the agent plays this game. */
  agentColor: 'w' | 'b';
  /** Move count in current game. */
  moveCount: number;
  /** Last UCI move played. */
  lastMove: Move | null;
  /** Outcome of the last completed game. */
  lastOutcome: Outcome | null;
}

function freshSession(agentColor: 'w' | 'b', gameIndex: number): SparringSession {
  return {
    state: initialState(),
    history: [],
    gameIndex,
    agentColor,
    moveCount: 0,
    lastMove: null,
    lastOutcome: null,
  };
}



function outcomeLabel(o: Outcome | null): string {
  if (!o) return '—';
  switch (o) {
    case 'white-wins': return 'White wins';
    case 'black-wins': return 'Black wins';
    case 'draw':       return 'Draw';
    case 'ongoing':    return 'Ongoing';
  }
}

function agentResultFromOutcome(agentColor: 'w' | 'b', o: Outcome): 'win' | 'draw' | 'loss' {
  if (o === 'draw') return 'draw';
  if (o === 'white-wins') return agentColor === 'w' ? 'win' : 'loss';
  return agentColor === 'b' ? 'win' : 'loss';
}



export default function SparringGame() {
  const [storage] = useState<BlobStorage>(createAgentStorage);
  const [agent, setAgent] = useState<Agent | null>(null);
  const agentRef = useRef<Agent | null>(null);
  agentRef.current = agent;

  const [agentColorChoice, setAgentColorChoice] = useState<AgentColor>('w');
  const [sfSkill, setSfSkill] = useState(5);
  const [sfMovetime, setSfMovetime] = useState(100);
  const [gamesToPlay, setGamesToPlay] = useState(10);

  const [runState, setRunState] = useState<RunState>('idle');
  const runStateRef = useRef<RunState>('idle');
  runStateRef.current = runState;

  const [session, setSession] = useState<SparringSession | null>(null);
  const sessionRef = useRef<SparringSession | null>(null);
  sessionRef.current = session;

  const [tally, setTally] = useState<Tally>({ wins: 0, draws: 0, losses: 0 });

  const sfRef = useRef<StockfishEngine | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    return () => {
      sfRef.current?.dispose();
      sfRef.current = null;
    };
  }, []);


  const gameLoopRunning = useRef(false);

  const runGameLoop = useCallback(
    async (
      totalGames: number,
      colorChoice: AgentColor,
      sfConfig: StockfishConfig,
    ) => {
      if (gameLoopRunning.current) return;
      gameLoopRunning.current = true;

      // Spin up Stockfish.
      let sf: StockfishEngine;
      try {
        sf = await StockfishEngine.create(sfConfig);
        sfRef.current = sf;
      } catch (err) {
        setError(
          `Failed to start Stockfish: ${err instanceof Error ? err.message : String(err)}`,
        );
        setRunState('idle');
        runStateRef.current = 'idle';
        gameLoopRunning.current = false;
        return;
      }

      // Helper: resolve agent color for game N (0-indexed).
      function agentColorForGame(idx: number): 'w' | 'b' {
        if (colorChoice === 'alternate') return idx % 2 === 0 ? 'w' : 'b';
        return colorChoice;
      }

      const infinite = totalGames === 0;
      let gameIdx = 0;

      while (
        (infinite || gameIdx < totalGames) &&
        runStateRef.current !== 'stopping'
      ) {
        // Handle pause — busy-wait with a small sleep.
        while (runStateRef.current === 'paused') {
          await new Promise<void>(r => setTimeout(r, 200));
        }
        if (runStateRef.current === 'stopping') break;

        const agentColor = agentColorForGame(gameIdx);
        let sess = freshSession(agentColor, gameIdx + 1);
        setSession(sess);
        sessionRef.current = sess;

        // Play one game.
        while (!isTerminal(sess.state) && runStateRef.current !== 'stopping') {
          // Handle pause mid-game.
          while (runStateRef.current === 'paused') {
            await new Promise<void>(r => setTimeout(r, 200));
          }
          if (runStateRef.current === 'stopping') break;

          const currentAgent = agentRef.current!;
          const currentSide = sideToMove(sess.state);
          const preMoveState = sess.state;
          let move: Move;

          if (currentSide === agentColor) {
            // Agent's turn.
            const result = await currentAgent.selectMove(preMoveState);
            move = result.move;
          } else {
            // Stockfish's turn.
            const fen = toFen(preMoveState);
            const raw = await sf.bestMove(fen);
            // Validate that Stockfish returned a legal move.
            const legal = legalMoves(preMoveState);
            move = legal.includes(raw)
              ? raw
              : legal[0]; // fallback to first legal move (should never happen)
          }

          // Record and apply.
          const record: GameMoveRecord = {
            features: encodeState(preMoveState),
            move,
            sideToMove: currentSide,
          };

          const nextState = applyMove(preMoveState, move);

          sess = {
            ...sess,
            state: nextState,
            history: [...sess.history, record],
            moveCount: sess.moveCount + 1,
            lastMove: move,
          };
          setSession(sess);
          sessionRef.current = sess;
        }

        // Game over (or stopped mid-game — skip recording if interrupted).
        if (runStateRef.current === 'stopping') break;

        const finalOutcome = outcome(sess.state);
        if (finalOutcome !== 'ongoing') {
          const currentAgent = agentRef.current!;
          currentAgent.recordGameResult(sess.history, finalOutcome);
          currentAgent.saveTo(storage).catch(err => {
            setError(
              `Failed to save after game: ${err instanceof Error ? err.message : String(err)}`,
            );
          });

          // Update tally.
          const agentResult = agentResultFromOutcome(agentColor, finalOutcome);
          setTally(prev => ({
            wins:   agentResult === 'win'  ? prev.wins + 1  : prev.wins,
            draws:  agentResult === 'draw' ? prev.draws + 1 : prev.draws,
            losses: agentResult === 'loss' ? prev.losses + 1 : prev.losses,
          }));

          // Update session with final outcome.
          sess = { ...sess, lastOutcome: finalOutcome };
          setSession(sess);
          sessionRef.current = sess;
        }

        gameIdx++;
      }

      // Tear down Stockfish.
      sf.dispose();
      sfRef.current = null;

      setRunState('idle');
      runStateRef.current = 'idle';
      gameLoopRunning.current = false;
    },
    [storage],
  );



  const handleStart = useCallback(() => {
    if (runState !== 'idle') return;
    if (!agent) return;

    setError(null);
    setTally({ wins: 0, draws: 0, losses: 0 });
    setSession(null);
    setRunState('running');
    runStateRef.current = 'running';

    const sfConfig: StockfishConfig = { skill: sfSkill, movetimeMs: sfMovetime };
    runGameLoop(gamesToPlay, agentColorChoice, sfConfig);
  }, [runState, agent, sfSkill, sfMovetime, gamesToPlay, agentColorChoice, runGameLoop]);

  const handlePause = useCallback(() => {
    if (runState !== 'running') return;
    setRunState('paused');
    runStateRef.current = 'paused';
  }, [runState]);

  const handleResume = useCallback(() => {
    if (runState !== 'paused') return;
    setRunState('running');
    runStateRef.current = 'running';
  }, [runState]);

  const handleStop = useCallback(() => {
    if (runState === 'idle') return;
    setRunState('stopping');
    runStateRef.current = 'stopping';
  }, [runState]);



  if (!agent && !error) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px' }}>
        Loading agent…
      </div>
    );
  }

  if (error && !agent) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px', color: '#c00' }}>
        {error}
      </div>
    );
  }

  const isRunning = runState === 'running';
  const isPaused = runState === 'paused';
  const isStopping = runState === 'stopping';
  const isIdle = runState === 'idle';

  const currentSide = session ? sideToMove(session.state) : null;
  const boardOrientation =
    session?.agentColor === 'b' ? 'black' : 'white';

  const totalGamesToDisplay = gamesToPlay === 0 ? '∞' : gamesToPlay;

  return (
    <div style={{ fontFamily: 'sans-serif' }}>

      {/* Config panel — disabled while running */}
      <fieldset
        disabled={!isIdle}
        style={{
          marginBottom: '16px',
          padding: '12px 16px',
          border: '1px solid #ccc',
          borderRadius: '6px',
        }}
      >
        <legend style={{ fontWeight: 'bold' }}>Sparring config</legend>

        {/* Agent color */}
        <div style={{ marginBottom: '10px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <strong>Agent plays as:</strong>
          {(['w', 'b', 'alternate'] as AgentColor[]).map(c => (
            <label key={c} style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                name="agentColor"
                value={c}
                checked={agentColorChoice === c}
                onChange={() => setAgentColorChoice(c)}
              />
              {' '}
              {c === 'w' ? 'White' : c === 'b' ? 'Black' : 'Alternate'}
            </label>
          ))}
        </div>

        {/* Stockfish skill */}
        <div style={{ marginBottom: '10px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <strong>Stockfish skill:</strong>
          <input
            type="range"
            min={0}
            max={20}
            value={sfSkill}
            onChange={e => setSfSkill(Number(e.target.value))}
            style={{ width: '160px' }}
          />
          <span>{sfSkill}</span>
        </div>

        {/* Movetime */}
        <div style={{ marginBottom: '10px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <strong>Movetime (ms):</strong>
          <input
            type="number"
            min={10}
            max={10000}
            value={sfMovetime}
            onChange={e => setSfMovetime(Number(e.target.value))}
            style={{ width: '80px' }}
          />
        </div>

        {/* Games to play */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <strong>Games to play:</strong>
          <input
            type="number"
            min={0}
            max={10000}
            value={gamesToPlay}
            onChange={e => setGamesToPlay(Number(e.target.value))}
            style={{ width: '80px' }}
          />
          <span style={{ color: '#666', fontSize: '0.85em' }}>
            (0 = run until stopped)
          </span>
        </div>
      </fieldset>

      {/* Control buttons */}
      <div style={{ marginBottom: '16px', display: 'flex', gap: '10px' }}>
        <button
          type="button"
          onClick={handleStart}
          disabled={!isIdle || !agent}
          style={{ padding: '8px 20px', cursor: isIdle && agent ? 'pointer' : 'default' }}
        >
          Start
        </button>
        {isRunning && (
          <button
            type="button"
            onClick={handlePause}
            style={{ padding: '8px 20px', cursor: 'pointer' }}
          >
            Pause
          </button>
        )}
        {isPaused && (
          <button
            type="button"
            onClick={handleResume}
            style={{ padding: '8px 20px', cursor: 'pointer' }}
          >
            Resume
          </button>
        )}
        {(isRunning || isPaused) && (
          <button
            type="button"
            onClick={handleStop}
            style={{ padding: '8px 20px', cursor: 'pointer' }}
          >
            Stop
          </button>
        )}
        {isStopping && (
          <span style={{ padding: '8px', color: '#888', fontStyle: 'italic' }}>
            Stopping after current move…
          </span>
        )}
      </div>

      {/* Status readout */}
      {(session || !isIdle) && (
        <div
          style={{
            marginBottom: '16px',
            padding: '10px 14px',
            background: '#f5f5f5',
            border: '1px solid #ddd',
            borderRadius: '6px',
            fontSize: '0.9em',
          }}
        >
          {session && (
            <>
              <div>
                <strong>Game:</strong>{' '}
                {session.gameIndex} of {totalGamesToDisplay}
                {' | '}
                <strong>Agent plays:</strong>{' '}
                {session.agentColor === 'w' ? 'White' : 'Black'}
                {' | '}
                <strong>Move:</strong> {session.moveCount}
                {currentSide && (
                  <>
                    {' | '}
                    <strong>To move:</strong>{' '}
                    {currentSide === 'w' ? 'White' : 'Black'}
                    {currentSide === session.agentColor ? ' (agent)' : ' (Stockfish)'}
                  </>
                )}
              </div>
              {session.lastMove && (
                <div>
                  <strong>Last move:</strong> {session.lastMove}
                </div>
              )}
              {session.lastOutcome && (
                <div>
                  <strong>Last game:</strong>{' '}
                  {outcomeLabel(session.lastOutcome)}
                  {' — agent '}
                  {agentResultFromOutcome(session.agentColor, session.lastOutcome)}
                </div>
              )}
            </>
          )}
          {isPaused && (
            <div style={{ color: '#888', fontStyle: 'italic' }}>Paused.</div>
          )}
        </div>
      )}

      {/* Running tally */}
      {(tally.wins + tally.draws + tally.losses > 0 || !isIdle) && (
        <div
          style={{
            marginBottom: '16px',
            padding: '10px 14px',
            background: '#eef4ff',
            border: '1px solid #aac',
            borderRadius: '6px',
            fontSize: '0.9em',
          }}
        >
          <strong>Session tally</strong> (agent){' '}
          — Wins: <strong>{tally.wins}</strong>
          {' | '}Draws: <strong>{tally.draws}</strong>
          {' | '}Losses: <strong>{tally.losses}</strong>
          {tally.wins + tally.draws + tally.losses > 0 && (
            <>
              {' | '}Win rate:{' '}
              <strong>
                {(
                  (tally.wins / (tally.wins + tally.draws + tally.losses)) *
                  100
                ).toFixed(1)}
                %
              </strong>
            </>
          )}
        </div>
      )}

      {/* Board (spectator-only) */}
      {session && (
        <div style={{ maxWidth: '480px' }}>
          <Board
            state={session.state}
            interactive={false}
            boardOrientation={boardOrientation}
          />
        </div>
      )}

      {/* Error display */}
      {error && (
        <div style={{ marginTop: '12px', color: '#c00', fontSize: '0.9em' }}>
          Error: {error}
        </div>
      )}
    </div>
  );
}
