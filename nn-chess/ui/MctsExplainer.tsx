// MctsExplainer — an interactive walk-through of an MCTS search on a
// hand-picked mate-in-2 position. Mounted by the Astro page at
// `/nn-chess/mcts` and intentionally not linked from the main game UI.
//
// What it shows
// ─────────────
// At each position in a (user-walked) line up to MAX_PLIES half-moves
// deep, MCTS runs with the user's stored networks. The candidate moves
// are listed with their PUCT-flavoured statistics: visit count N, prior
// P, and average backed-up value Q. Clicking a candidate advances the
// demo to the resulting position; if the picked move is checkmate the
// demo stops there.
//
// Why two networks?
// ─────────────────
// White (the winning side in our mate-in-2) plays with the opponent
// network — the one trained by the user playing games against it.
// Black plays with the player-imitator, which is the imitation-learning
// model of the user's own moves. Both come from the same `BlobStorage`
// the human-play page writes to, so what you see here reflects whatever
// state your trained networks are actually in. A freshly-initialized
// network will still find the mate provided sims/move is high enough —
// the terminal +1 signal propagates back regardless of policy priors.
//
// The MCTS module exposes `edgeValues: Map<Move, number>` on
// `SearchResult` (Q(s,a) at root for every legal move). We use it to
// render the exploitation column in the candidate table. The same
// mechanic powers the InspectorPanel; we just expose it explicitly so
// the demo can teach with it.

import { useEffect, useState, useCallback, useMemo } from 'react';
import { Chessboard } from 'react-chessboard';
import {
  fromFen, toFen, applyMove, isTerminal, outcome, sideToMove,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { search } from '../core/mcts/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import { policyFromNet } from '../core/opponent/index.js';
import type { Opponent } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import {
  createOpponentStorage,
  loadOpponentFromStorage,
  loadPlayerImitatorFromStorage,
} from './opponent-config.js';

// Position: White Ke6, Qd1, Black Ke8. White to move and mate in two.
//   1. Qd7+   Kf8 (only legal reply: Kxd7 protected, others attacked by Q)
//   2. Qf7#   king on f8 has no escape, queen defended by Ke6.
const MATE_IN_2_FEN = '4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1';

// The mate-in-2 mainline, indexed by step. We pass each ply's "correct"
// move down to the candidate table so it's always rendered even when
// cold-start MCTS doesn't visit it enough to land in the top-N by visit
// count. The check that the move is actually legal in the current state
// happens in the table, so deviating from the line just hides the
// downstream hints.
const MATE_LINE: readonly Move[] = ['d1d7', 'e8f8', 'd7f7'];

// Cap on user-clicked moves. The mate is 3 plies long, so 4 gives one
// ply of slack for offline explorations the user might click into.
// Counted as moves (clicks), not positions: with MAX_MOVES = 4 we show
// up to 5 boards (initial + 4 follow-ups).
const MAX_MOVES = 4;

// Generous defaults so the demo lands on the mate even with cold-start
// nets. The strength slider on the main page is capped at 400 sims; we
// allow up to 1024 here because users will be experimenting.
const DEFAULT_SIMS = 256;
const DEFAULT_CPUCT = 1.5;

type NetKind = 'opponent' | 'imitator';

interface Step {
  // Board state immediately before any move is taken on this ply.
  state: GameState;
  // Which net evaluates leaves expanded by MCTS at this position.
  net: NetKind;
  // MCTS result for this position, or null while the search is running.
  searchResult: SearchResult | null;
  // The move the user picked at this position, or null if not yet
  // chosen. Set => next step has been (or will be) appended.
  chosenMove: Move | null;
  // Set iff `state` is terminal (checkmate / stalemate / draw).
  finalOutcome: Outcome | null;
}

// White (winning side) uses the opponent net; black (defender) uses
// the imitator. Keeps the demo honest: both nets the user has trained
// actually play, and the imitator gets to defend the way it would in
// self-play data generation.
function netKindForSide(side: 'w' | 'b'): NetKind {
  return side === 'w' ? 'opponent' : 'imitator';
}

function netForKind(kind: NetKind, opp: Opponent, imi: PlayerImitator) {
  return kind === 'opponent' ? opp.net : imi.net;
}

function makeInitialStep(): Step {
  const state = fromFen(MATE_IN_2_FEN);
  return {
    state,
    net: netKindForSide(sideToMove(state)),
    searchResult: null,
    chosenMove: null,
    finalOutcome: null,
  };
}

export default function MctsExplainer() {
  const [opponent, setOpponent] = useState<Opponent | null>(null);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [simulations, setSimulations] = useState(DEFAULT_SIMS);
  const [cPuct, setCPuct] = useState(DEFAULT_CPUCT);
  const [error, setError] = useState<string | null>(null);

  // Load both networks from IndexedDB. The same storage backend backs the
  // main play page, so whatever weights the user has trained show up here.
  useEffect(() => {
    let cancelled = false;
    const storage = createOpponentStorage();
    Promise.all([
      loadOpponentFromStorage(storage),
      loadPlayerImitatorFromStorage(storage),
    ])
      .then(([opp, imi]) => {
        if (cancelled) return;
        setOpponent(opp);
        setImitator(imi);
      })
      .catch(err => {
        if (cancelled) return;
        setError(`Failed to load networks: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => { cancelled = true; };
  }, []);

  // Once both networks are loaded, seed the steps array with the start
  // position. Re-running this whenever the sims/cPuct change is handled
  // by `reset` below — that's the user-initiated path.
  useEffect(() => {
    if (opponent === null || imitator === null) return;
    if (steps.length > 0) return;
    setSteps([makeInitialStep()]);
  }, [opponent, imitator, steps.length]);

  // Drive the MCTS search for whichever step is currently unanalysed.
  // Each step appended by `chooseMove` starts with searchResult = null;
  // this effect notices and fills it in. Identity check on the captured
  // step object (not its state) so a slider-driven re-run — which keeps
  // the same state but replaces the step object — invalidates any
  // in-flight stale result.
  useEffect(() => {
    if (opponent === null || imitator === null) return;
    if (steps.length === 0) return;
    const idx = steps.length - 1;
    const last = steps[idx];
    if (last.searchResult !== null) return;
    if (last.finalOutcome !== null) return;

    let cancelled = false;
    const net = netForKind(last.net, opponent, imitator);
    // Depth-2 mate lookahead so the demo deterministically locks in the
    // mate-in-2 as soon as MCTS visits the first move of the mainline.
    // The default (depth-1) is fine for real play; we want the bigger
    // window here for pedagogy.
    search(last.state, policyFromNet(net), { simulations, cPuct, mateSearchDepth: 2 })
      .then(result => {
        if (cancelled) return;
        setSteps(prev => {
          if (prev.length === 0) return prev;
          const lastNow = prev[prev.length - 1];
          // Bail if the trailing step has been replaced (slider change
          // or user navigation); the freshly-invalidated step will get
          // its own search via this same effect.
          if (lastNow !== last) return prev;
          const updated = [...prev];
          updated[updated.length - 1] = { ...lastNow, searchResult: result };
          return updated;
        });
      })
      .catch(err => {
        if (cancelled) return;
        setError(`MCTS failed: ${err instanceof Error ? err.message : String(err)}`);
      });

    return () => { cancelled = true; };
  }, [steps, opponent, imitator, simulations, cPuct]);

  // When the user moves a slider, replace the trailing step with a
  // fresh object (same state, but searchResult cleared). The effect
  // above sees the null result and re-runs MCTS at the new sims/cPuct.
  // Replacing the object — rather than mutating it — is what trips the
  // in-flight identity check so a stale 256-sim completion can't
  // overwrite the new 1024-sim result.
  useEffect(() => {
    setSteps(prev => {
      if (prev.length === 0) return prev;
      const idx = prev.length - 1;
      const last = prev[idx];
      if (last.searchResult === null) return prev;  // nothing to invalidate
      if (last.finalOutcome !== null) return prev;  // terminal: no search runs here
      const updated = [...prev];
      updated[idx] = { ...last, searchResult: null };
      return updated;
    });
  }, [simulations, cPuct]);

  const chooseMove = useCallback((stepIndex: number, move: Move) => {
    setSteps(prev => {
      if (stepIndex >= prev.length) return prev;
      // Truncate so we drop any downstream exploration the user is now
      // overwriting; record the freshly-chosen move on the current step.
      const truncated = prev.slice(0, stepIndex + 1);
      const current = truncated[stepIndex];
      const updatedCurrent: Step = { ...current, chosenMove: move };
      truncated[stepIndex] = updatedCurrent;

      // Build the resulting state.
      let nextState: GameState;
      try {
        nextState = applyMove(current.state, move);
      } catch {
        // Move was illegal (shouldn't happen since we only offer
        // search-result keys, all of which are legal); bail silently.
        return prev;
      }
      const terminal = isTerminal(nextState);
      const finalOutcome = terminal ? outcome(nextState) : null;

      // Decide whether to append a follow-up step. The user has just
      // made move number `truncated.length` (initial position is move 0).
      // We always show terminal positions so the user can see the mating
      // board; otherwise stop appending once we'd exceed MAX_MOVES clicks.
      const movesAfterAppend = truncated.length;  // initial is index 0
      const overCap = movesAfterAppend > MAX_MOVES;
      if (overCap && !terminal) {
        return truncated;
      }
      const nextStep: Step = {
        state: nextState,
        net: netKindForSide(sideToMove(nextState)),
        searchResult: null,
        chosenMove: null,
        finalOutcome,
      };
      truncated.push(nextStep);
      return truncated;
    });
  }, []);

  const reset = useCallback(() => {
    if (opponent === null || imitator === null) return;
    setSteps([makeInitialStep()]);
  }, [opponent, imitator]);

  // ── Render ────────────────────────────────────────────────────────────────

  if (error !== null && opponent === null) {
    return <div style={{ color: '#c00', padding: '12px' }}>{error}</div>;
  }
  if (opponent === null || imitator === null) {
    return <div style={{ padding: '12px', color: '#666' }}>Loading networks…</div>;
  }

  const atCap = steps.length > MAX_MOVES && steps[steps.length - 1].finalOutcome === null;
  const lineEnded = steps.length > 0 && steps[steps.length - 1].finalOutcome !== null;

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <div
        style={{
          marginBottom: '14px',
          padding: '10px 12px',
          background: '#f5f7fb',
          border: '1px solid #d4dbe8',
          borderRadius: '6px',
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          alignItems: 'center',
          fontSize: '0.9em',
        }}
      >
        <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <strong>sims/move:</strong>
          <input
            type="range"
            min={16}
            max={1024}
            step={16}
            value={simulations}
            onChange={e => setSimulations(Number(e.target.value))}
            style={{ width: '140px' }}
          />
          <span style={{ fontFamily: 'monospace', minWidth: '38px' }}>{simulations}</span>
        </label>
        <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <strong>cPuct:</strong>
          <input
            type="range"
            min={0.25}
            max={4.0}
            step={0.25}
            value={cPuct}
            onChange={e => setCPuct(Number(e.target.value))}
            style={{ width: '100px' }}
          />
          <span style={{ fontFamily: 'monospace', minWidth: '34px' }}>{cPuct.toFixed(2)}</span>
        </label>
        <button
          type="button"
          onClick={reset}
          style={{ padding: '4px 12px', cursor: 'pointer' }}
        >
          Reset walkthrough
        </button>
        {error && <span style={{ color: '#c00' }}>{error}</span>}
      </div>

      <p style={{ marginTop: 0, color: '#444' }}>
        Mate-in-2 (white to move). White's queen is on d1; the
        kings face off on the e-file. The forced sequence is
        {' '}<code>1. Qd7+ Kf8 2. Qf7#</code>. Click any candidate move in
        the table to advance to the resulting position and trigger a new
        MCTS search there.
      </p>

      {steps.map((step, idx) => (
        <StepView
          key={idx}
          step={step}
          stepIndex={idx}
          isLast={idx === steps.length - 1}
          onChoose={chooseMove}
          simulations={simulations}
          cPuct={cPuct}
          featuredMove={MATE_LINE[idx]}
        />
      ))}

      {atCap && (
        <div style={{ marginTop: '10px', padding: '8px 12px', background: '#fffbe6', border: '1px solid #e6d97a', borderRadius: '6px', fontSize: '0.9em', color: '#555' }}>
          Reached the {MAX_MOVES}-move cap. Click <em>Reset walkthrough</em> to start over.
        </div>
      )}
      {lineEnded && (
        <div style={{ marginTop: '10px', padding: '8px 12px', background: '#ecf7ec', border: '1px solid #9ec99e', borderRadius: '6px', fontSize: '0.9em', color: '#333' }}>
          Line ended — {outcomeLabel(steps[steps.length - 1].finalOutcome!)}.
        </div>
      )}
    </div>
  );
}

// ── Per-step view ───────────────────────────────────────────────────────────

interface StepViewProps {
  step: Step;
  stepIndex: number;
  isLast: boolean;
  onChoose: (stepIndex: number, move: Move) => void;
  simulations: number;
  cPuct: number;
  // The mate-mainline move expected at this ply, when one is defined.
  // CandidateTable surfaces it as a pinned row even when it's outside
  // the visit-count top-N, so the user can always read its Q/N/U.
  featuredMove?: Move;
}

function StepView({ step, stepIndex, isLast, onChoose, simulations, cPuct, featuredMove }: StepViewProps) {
  const ply = stepIndex + 1;
  const side = sideToMove(step.state);
  const sideLabel = side === 'w' ? 'White' : 'Black';
  const isMate = step.finalOutcome === 'white-wins' || step.finalOutcome === 'black-wins';

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '12px 14px',
        border: '1px solid #ccc',
        borderRadius: '6px',
        background: isLast ? '#fbfcff' : '#fafafa',
      }}
    >
      <div style={{ marginBottom: '8px', display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'baseline' }}>
        <strong>Ply {ply}</strong>
        <span style={{ color: '#444' }}>
          {step.finalOutcome === null
            ? `${sideLabel} to move`
            : `Final position (${outcomeLabel(step.finalOutcome)})`}
        </span>
        {step.finalOutcome === null && (
          <span style={{ fontSize: '0.85em', color: '#777' }}>
            evaluator: <code>{step.net}</code> network
          </span>
        )}
        {step.chosenMove !== null && (
          <span style={{ fontSize: '0.85em', color: '#1e5fb4' }}>
            picked <code>{step.chosenMove}</code>
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ width: '260px', flexShrink: 0 }}>
          <Chessboard
            options={{
              position: toFen(step.state),
              allowDragging: false,
              boardStyle: { borderRadius: '4px', boxShadow: '0 1px 4px rgba(0,0,0,0.2)' },
            }}
          />
        </div>

        <div style={{ flex: '1 1 320px', minWidth: '300px' }}>
          {step.finalOutcome !== null ? (
            <div style={{ color: '#555' }}>
              {isMate
                ? 'Checkmate. The game is over — no further MCTS needed.'
                : 'Terminal position (draw). No further MCTS.'}
            </div>
          ) : step.searchResult === null ? (
            <div style={{ color: '#666' }}>
              Running MCTS ({simulations} simulations, cPuct={cPuct.toFixed(2)})…
            </div>
          ) : (
            <CandidateTable
              search={step.searchResult}
              chosen={step.chosenMove}
              onChoose={move => onChoose(stepIndex, move)}
              simulations={simulations}
              cPuct={cPuct}
              featuredMove={featuredMove}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Candidate-move table ────────────────────────────────────────────────────

interface CandidateTableProps {
  search: SearchResult;
  chosen: Move | null;
  onChoose: (move: Move) => void;
  simulations: number;
  cPuct: number;
  featuredMove?: Move;
}

function CandidateTable({ search, chosen, onChoose, simulations, cPuct, featuredMove }: CandidateTableProps) {
  const totalVisits = useMemo(
    () => Array.from(search.visitCounts.values()).reduce((a, b) => a + b, 0),
    [search.visitCounts],
  );
  const sqrtN = Math.sqrt(totalVisits);

  // Top-5 by visit count, plus the mate-mainline move appended if it
  // exists as a legal candidate in this position but wasn't already in
  // the top 5. The pinned row keeps the demo readable when the cold-
  // start prior buries d1d7 / d7f7 outside the visit-count leaders.
  const { rows, featuredPinned } = useMemo(() => {
    const items = Array.from(search.visitCounts.entries()).map(([move, visits]) => {
      const prior = search.priorPolicy.get(move) ?? 0;
      const q = search.edgeValues.get(move) ?? 0;
      const u = cPuct * prior * sqrtN / (1 + visits);
      return { move, visits, prior, q, u, puct: q + u };
    });
    items.sort((a, b) => b.visits - a.visits);
    const top = items.slice(0, 5);
    let pinned = false;
    if (featuredMove !== undefined && search.visitCounts.has(featuredMove)
        && !top.some(r => r.move === featuredMove)) {
      const featured = items.find(r => r.move === featuredMove);
      if (featured !== undefined) {
        top.push(featured);
        pinned = true;
      }
    }
    return { rows: top, featuredPinned: pinned };
  }, [search, cPuct, sqrtN, featuredMove]);

  return (
    <>
      <div style={{ fontSize: '0.85em', color: '#666', marginBottom: '6px' }}>
        Top {rows.length} candidates by visit count
        {' '}({totalVisits.toLocaleString()} sim{totalVisits === 1 ? '' : 's'} total).
        Click a row to play that move.
      </div>
      <table
        style={{
          borderCollapse: 'collapse',
          fontSize: '0.85em',
          fontFamily: 'monospace',
          width: '100%',
        }}
      >
        <thead>
          <tr style={{ color: '#666', textAlign: 'right', borderBottom: '1px solid #ddd' }}>
            <th style={{ padding: '4px 8px', textAlign: 'left' }}>move</th>
            <th style={{ padding: '4px 8px' }} title="visit count N(s,a)">N</th>
            <th style={{ padding: '4px 8px' }} title="visit fraction = N(s,a) / sum of N">N%</th>
            <th style={{ padding: '4px 8px' }} title="prior P(s,a) from policy net">P%</th>
            <th style={{ padding: '4px 8px' }} title="Q(s,a) = average backed-up value from this edge, in [-1, 1]">Q</th>
            <th style={{ padding: '4px 8px' }} title="exploration bonus = cPuct * P * sqrt(N_parent) / (1 + N(s,a))">U</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => {
            const isChosen = chosen === row.move;
            const isFeatured = featuredMove === row.move;
            return (
              <tr
                key={row.move}
                onClick={() => onChoose(row.move)}
                style={{
                  cursor: 'pointer',
                  background: isChosen ? '#e8f1ff' : 'transparent',
                }}
              >
                <td style={{ padding: '4px 8px' }}>
                  {row.move}
                  {isChosen && (
                    <span style={{ color: '#1e5fb4', marginLeft: '6px', fontWeight: 'bold' }}>← played</span>
                  )}
                  {isFeatured && !isChosen && (
                    <span style={{ color: '#7a5e1e', marginLeft: '6px', fontStyle: 'italic' }}>(mainline)</span>
                  )}
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{row.visits}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>
                  {totalVisits === 0 ? '—' : (row.visits / totalVisits * 100).toFixed(1)}
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'right', color: '#777' }}>
                  {(row.prior * 100).toFixed(1)}
                </td>
                <td style={{
                  padding: '4px 8px',
                  textAlign: 'right',
                  color: row.q >= 0 ? '#1f7a32' : '#a8231a',
                }}>
                  {row.q >= 0 ? '+' : ''}{row.q.toFixed(3)}
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'right', color: '#777' }}>
                  {row.u.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {search.visitCounts.size > rows.length && (
        <div style={{ marginTop: '6px', fontSize: '0.78em', color: '#888' }}>
          + {search.visitCounts.size - rows.length} other move{search.visitCounts.size - rows.length === 1 ? '' : 's'} not shown
          {featuredPinned ? ' (mainline pinned below visit-count top 5).' : '.'}
        </div>
      )}
    </>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function outcomeLabel(o: Outcome): string {
  switch (o) {
    case 'white-wins': return 'white wins';
    case 'black-wins': return 'black wins';
    case 'draw':       return 'draw';
    case 'ongoing':    return 'ongoing';
  }
}
