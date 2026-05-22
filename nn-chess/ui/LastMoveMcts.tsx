// LastMoveMcts — compact panel that lives next to the board on the
// human-play page and summarises the MCTS search the opponent just
// finished. Inspector has the full readout under a collapsible; this
// component answers the at-a-glance question "what was the opponent
// thinking on its last move?" without the user clicking anything.
//
// The data is the same `SearchResult` already stored as `lastSearch` in
// Game.tsx — we just pick the top 3 entries by visit count, mark which
// one was actually played, and show root value + total simulations.

import type { SearchResult } from '../core/mcts/index.js';
import type { Move } from '../core/rules/index.js';

export interface LastMoveMctsProps {
  lastSearch: SearchResult | null;
}

/**
 * Render `value` (in [-1, 1] from the side-to-move's POV) as a two-tone
 * horizontal bar centered at 0. Kept inline rather than importing the
 * Inspector's copy because that one lives behind a `<details>` and
 * would couple this small read-only view to the inspector internals.
 */
function ValueBar({ value }: { value: number }) {
  const pct = Math.min(50, Math.abs(value) * 50);
  const isPos = value >= 0;
  return (
    <div
      style={{
        position: 'relative',
        width: '120px',
        height: '12px',
        background: '#e8e8e8',
        borderRadius: '2px',
        overflow: 'hidden',
        display: 'inline-block',
        verticalAlign: 'middle',
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: 0,
          width: '1px',
          height: '100%',
          background: '#999',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: 0,
          height: '100%',
          width: `${pct}%`,
          left: isPos ? '50%' : `${50 - pct}%`,
          background: isPos ? '#2a9a3c' : '#c0392b',
        }}
      />
    </div>
  );
}

interface Row {
  move: Move;
  visits: number;
  prior: number;
  visitProb: number;
  isBest: boolean;
}

export function LastMoveMcts({ lastSearch }: LastMoveMctsProps) {
  if (lastSearch === null) return null;

  const { bestMove, rootValue, visitCounts, priorPolicy } = lastSearch;

  const totalVisits = Array.from(visitCounts.values()).reduce((a, b) => a + b, 0);
  const entries: Row[] = Array.from(visitCounts.entries()).map(([move, visits]) => ({
    move,
    visits,
    prior: priorPolicy.get(move) ?? 0,
    visitProb: totalVisits > 0 ? visits / totalVisits : 0,
    isBest: move === bestMove,
  }));
  entries.sort((a, b) => b.visits - a.visits);
  const top = entries.slice(0, 3);
  const remaining = entries.length - top.length;

  return (
    <div
      style={{
        marginTop: '12px',
        padding: '10px 14px',
        border: '1px solid #ccc',
        borderRadius: '6px',
        background: '#f9f9f9',
        color: '#1a1a1a',
        fontSize: '0.9em',
      }}
    >
      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
        <strong>Opponent&apos;s last MCTS search:</strong>
        <span style={{ fontFamily: 'monospace' }}>
          played <strong>{bestMove}</strong>
        </span>
        <span style={{ color: '#666' }}>
          ({totalVisits.toLocaleString()} sim{totalVisits === 1 ? '' : 's'})
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
          root value{' '}
          <span style={{ fontFamily: 'monospace' }}>
            {rootValue >= 0 ? '+' : ''}{rootValue.toFixed(3)}
          </span>
          <ValueBar value={rootValue} />
        </span>
      </div>

      <table style={{ borderCollapse: 'collapse', fontSize: '0.85em', fontFamily: 'monospace' }}>
        <thead>
          <tr style={{ color: '#666', textAlign: 'left' }}>
            <th style={{ paddingRight: '6px' }}>#</th>
            <th style={{ paddingRight: '10px' }}>Move</th>
            <th style={{ paddingRight: '10px', textAlign: 'right' }}>Visits</th>
            <th style={{ paddingRight: '10px', textAlign: 'right' }}>Visit %</th>
            <th style={{ paddingRight: '10px', textAlign: 'right' }}>Prior %</th>
          </tr>
        </thead>
        <tbody>
          {top.map((row, i) => (
            <tr key={row.move} style={{ background: row.isBest ? '#eaf3ff' : 'transparent' }}>
              <td style={{ paddingRight: '6px', color: '#888' }}>{i + 1}.</td>
              <td style={{ paddingRight: '10px' }}>
                {row.move}
                {row.isBest && (
                  <span style={{ color: '#1e5fb4', marginLeft: '6px', fontWeight: 'bold' }}>← played</span>
                )}
              </td>
              <td style={{ paddingRight: '10px', textAlign: 'right' }}>{row.visits}</td>
              <td style={{ paddingRight: '10px', textAlign: 'right' }}>
                {(row.visitProb * 100).toFixed(1)}%
              </td>
              <td style={{ paddingRight: '10px', textAlign: 'right', color: '#666' }}>
                {(row.prior * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {remaining > 0 && (
        <div style={{ marginTop: '6px', fontSize: '0.78em', color: '#888' }}>
          + {remaining} other move{remaining === 1 ? '' : 's'} considered. Full readout in the inspector below.
        </div>
      )}
    </div>
  );
}
