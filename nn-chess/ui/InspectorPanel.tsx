// NN calls (predict, getActivations) are memoized on a (stateKey, agentVersion)
// pair so unrelated re-renders (e.g. agentThinking toggling) don't reinvoke
// the network. stateKey is toFen(state) because GameState wraps a mutable
// chess.js instance.
//
// Per-layer weight stats are shown instead of full matrices: the policy head
// alone has 128 × 4096 = ~524k weights, which is useless and slow to render.

import { useMemo, useState, useCallback } from 'react';
import { softmax, type Agent } from '../core/agent/index.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { GameState } from '../core/rules/index.js';
import { legalMoves, sideToMove, toFen } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import { policyIndexFromUci } from '../core/nn/index.js';

export interface InspectorPanelProps {
  agent: Agent | null;
  state: GameState;
  lastSearch: SearchResult | null;
  agentVersion: number;
}

function arrayStats(arr: Float32Array): { min: number; max: number; mean: number } {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
    sum += arr[i];
  }
  return { min, max, mean: arr.length > 0 ? sum / arr.length : 0 };
}

/** Value bar: centered at 0. Green = positive, red = negative. */
function ValueBar({ value }: { value: number }) {
  const pct = Math.abs(value) * 50; // up to 50% of half-width
  const isPos = value >= 0;
  return (
    <div
      style={{
        position: 'relative',
        width: '200px',
        height: '16px',
        background: '#e8e8e8',
        borderRadius: '3px',
        overflow: 'hidden',
        display: 'inline-block',
        verticalAlign: 'middle',
      }}
    >
      {/* center tick */}
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
      {/* fill */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          height: '100%',
          width: `${pct}%`,
          left: isPos ? '50%' : `${50 - pct}%`,
          background: isPos ? '#2a9a3c' : '#c0392b',
          borderRadius: '2px',
        }}
      />
    </div>
  );
}

/** Tiny probability bar for policy/MCTS rows. */
function ProbBar({ prob, max }: { prob: number; max: number }) {
  const pct = max > 0 ? (prob / max) * 100 : 0;
  return (
    <div
      style={{
        display: 'inline-block',
        width: '80px',
        height: '10px',
        background: '#e8e8e8',
        borderRadius: '2px',
        overflow: 'hidden',
        verticalAlign: 'middle',
      }}
    >
      <div
        style={{
          width: `${pct}%`,
          height: '100%',
          background: '#3a6ed4',
        }}
      />
    </div>
  );
}

/** One row in a move table. */
function MoveRow({
  rank,
  move,
  prob,
  maxProb,
  extra,
}: {
  rank: number;
  move: string;
  prob: number;
  maxProb: number;
  extra?: React.ReactNode;
}) {
  return (
    <tr style={{ fontSize: '0.85em', fontFamily: 'monospace' }}>
      <td style={{ paddingRight: '6px', color: '#888' }}>{rank}.</td>
      <td style={{ paddingRight: '8px' }}>{move}</td>
      <td style={{ paddingRight: '6px', textAlign: 'right' }}>
        {(prob * 100).toFixed(1)}%
      </td>
      <td style={{ paddingRight: '8px' }}>
        <ProbBar prob={prob} max={maxProb} />
      </td>
      {extra}
    </tr>
  );
}

/**
 * Activation strip for a dense layer.
 * Clips values to [mean - 3σ, mean + 3σ], maps to a 0-255 brightness,
 * and renders each activation as a 4px-wide colored cell.
 * For large layers (>256 units) we show at most 256 cells to keep the
 * DOM manageable — a comment explains the skip.
 */
function ActivationStrip({ activations }: { activations: Float32Array }) {
  const MAX_CELLS = 256;
  const step = Math.max(1, Math.ceil(activations.length / MAX_CELLS));
  const cells: number[] = [];
  for (let i = 0; i < activations.length; i += step) cells.push(activations[i]);

  const stats = arrayStats(activations);
  // Compute std dev for clipping
  let variance = 0;
  for (let i = 0; i < activations.length; i++) {
    const d = activations[i] - stats.mean;
    variance += d * d;
  }
  const std = Math.sqrt(variance / Math.max(activations.length, 1));
  const lo = stats.mean - 3 * std;
  const hi = stats.mean + 3 * std;
  const range = hi - lo || 1;

  return (
    <div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1px',
          marginBottom: '4px',
        }}
      >
        {cells.map((v, i) => {
          const brightness = Math.round(((Math.max(lo, Math.min(hi, v)) - lo) / range) * 255);
          return (
            <div
              key={i}
              title={`activation[${i * step}] = ${v.toFixed(4)}`}
              style={{
                width: '4px',
                height: '12px',
                background: `rgb(${brightness},${Math.round(brightness * 0.5)},${255 - brightness})`,
                flexShrink: 0,
              }}
            />
          );
        })}
      </div>
      {step > 1 && (
        <div style={{ fontSize: '0.75em', color: '#888', marginBottom: '2px' }}>
          (showing 1 in {step} activations for {activations.length}-unit layer)
        </div>
      )}
      <div style={{ fontSize: '0.8em', fontFamily: 'monospace', color: '#555' }}>
        min {stats.min.toFixed(4)} / max {stats.max.toFixed(4)} / mean {stats.mean.toFixed(4)}
      </div>
    </div>
  );
}

interface WeightStats {
  totalBytes: number;
  layerStats: Array<{ name: string; min: number; max: number; mean: number; count: number }>;
}

export function InspectorPanel({
  agent,
  state,
  lastSearch,
  agentVersion,
}: InspectorPanelProps) {
  const [weightStats, setWeightStats] = useState<WeightStats | null>(null);
  const [loadingWeights, setLoadingWeights] = useState(false);

  const stateKey = useMemo(() => {
    if (!agent) return '';
    return `${toFen(state)}@v${agentVersion}`;
  }, [agent, state, agentVersion]);

  // stateKey already encodes agent (via agentVersion) and state, so the
  // exhaustive-deps lint is intentionally suppressed on the NN-call memos.
  const prediction = useMemo(() => {
    if (!agent) return null;
    try {
      return agent.net.predict(encodeState(state));
    } catch {
      return null;
    }
  }, [stateKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const activations = useMemo(() => {
    if (!agent) return null;
    try {
      const names = agent.net.layerNames();
      return agent.net.getActivations(encodeState(state), names);
    } catch {
      return null;
    }
  }, [stateKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const topPolicyMoves = useMemo(() => {
    if (!prediction || !agent) return [];
    const legal = legalMoves(state);
    if (legal.length === 0) return [];

    const probs = softmax(prediction.policy);
    const entries = legal.map(m => ({
      move: m,
      prob: probs[policyIndexFromUci(m)] ?? 0,
    }));
    entries.sort((a, b) => b.prob - a.prob);
    return entries.slice(0, 10);
  }, [stateKey, prediction]); // eslint-disable-line react-hooks/exhaustive-deps

  const topMctsMoves = useMemo(() => {
    if (!lastSearch) return [];
    const { visitCounts, priorPolicy } = lastSearch;
    const totalVisits = Array.from(visitCounts.values()).reduce((a, b) => a + b, 0);
    const entries = Array.from(visitCounts.entries()).map(([move, visits]) => ({
      move,
      visits,
      prior: priorPolicy.get(move) ?? 0,
      visitProb: totalVisits > 0 ? visits / totalVisits : 0,
    }));
    entries.sort((a, b) => b.visits - a.visits);
    return entries.slice(0, 10);
  }, [lastSearch]);


  const handleLoadWeights = useCallback(async () => {
    if (!agent || loadingWeights) return;
    setLoadingWeights(true);
    try {
      const bytes = await agent.net.serialize();
      // Parse the serialized blob to extract per-layer weight tensors.
      // Wire format: 4-byte LE uint32 header-length, header JSON, weight bytes.
      // See nn/index.ts for the full spec.
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const headerLen = view.getUint32(0, true);
      const headerJson = new TextDecoder().decode(bytes.slice(4, 4 + headerLen));
      const header = JSON.parse(headerJson) as {
        weightSpecs: Array<{ name: string; shape: number[]; dtype: string }>;
      };
      const rawWeights = bytes.slice(4 + headerLen);
      const weightsF32 = new Float32Array(rawWeights.buffer, rawWeights.byteOffset, rawWeights.byteLength / 4);

      // Walk through weight specs to compute per-layer stats.
      // NOTE: we do NOT render the full matrices — for a 128×4096 policy head
      // that is 524 288 floats, which is useless to display and expensive to
      // iterate in JS. Min/max/mean per tensor is sufficient for sanity-checking.
      const layerStats: WeightStats['layerStats'] = [];
      let offset = 0;
      for (const spec of header.weightSpecs) {
        const count = spec.shape.reduce((a, b) => a * b, 1);
        const slice = weightsF32.subarray(offset, offset + count);
        const stats = arrayStats(slice);
        layerStats.push({ name: spec.name, min: stats.min, max: stats.max, mean: stats.mean, count });
        offset += count;
      }

      setWeightStats({ totalBytes: bytes.byteLength, layerStats });
    } catch (err) {
      console.error('InspectorPanel: weight loading failed', err);
    } finally {
      setLoadingWeights(false);
    }
  }, [agent, loadingWeights]);


  const side = sideToMove(state);
  const sideLabel = side === 'w' ? 'White' : 'Black';


  return (
    <details
      style={{
        border: '1px solid #ccc',
        borderRadius: '6px',
        padding: '10px 14px',
        background: '#f9f9f9',
        color: '#1a1a1a',
        marginTop: '12px',
      }}
    >
      <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none' }}>
        Inspector (interpretability)
      </summary>

      {!agent ? (
        <div style={{ marginTop: '10px', color: '#888' }}>Agent not loaded.</div>
      ) : (
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* ── (a) Value head ──────────────────────────────────────────────── */}
          <section>
            <strong>Value head</strong>
            <div style={{ marginTop: '6px', fontSize: '0.9em' }}>
              {prediction === null ? (
                <span style={{ color: '#888' }}>—</span>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                  <span>
                    <span style={{ fontFamily: 'monospace' }}>
                      {prediction.value >= 0 ? '+' : ''}{prediction.value.toFixed(4)}
                    </span>
                    <span style={{ color: '#666', marginLeft: '6px' }}>
                      ({sideLabel} to move&apos;s POV)
                    </span>
                  </span>
                  <ValueBar value={prediction.value} />
                </div>
              )}
            </div>
          </section>

          {/* ── (b) Raw NN policy top-10 ────────────────────────────────────── */}
          <section>
            <strong>Raw NN policy — top legal moves</strong>
            <div style={{ fontSize: '0.78em', color: '#888', marginBottom: '4px' }}>
              Softmax over logits, masked to legal moves. Not MCTS visit counts.
            </div>
            {topPolicyMoves.length === 0 ? (
              <span style={{ color: '#888', fontSize: '0.9em' }}>No legal moves.</span>
            ) : (
              <table style={{ borderCollapse: 'collapse' }}>
                <tbody>
                  {topPolicyMoves.map(({ move, prob }, i) => (
                    <MoveRow
                      key={move}
                      rank={i + 1}
                      move={move}
                      prob={prob}
                      maxProb={topPolicyMoves[0].prob}
                    />
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* ── (c) MCTS readout ─────────────────────────────────────────────── */}
          <section>
            <strong>MCTS — last search</strong>
            {lastSearch === null ? (
              <div style={{ color: '#888', fontSize: '0.9em', marginTop: '4px' }}>
                No search yet (agent hasn&apos;t moved this game).
              </div>
            ) : (
              <>
                <div style={{ fontSize: '0.9em', marginTop: '4px', marginBottom: '6px' }}>
                  Root value (backed-up):{' '}
                  <span style={{ fontFamily: 'monospace' }}>
                    {lastSearch.rootValue >= 0 ? '+' : ''}{lastSearch.rootValue.toFixed(4)}
                  </span>
                  {'  '}
                  <ValueBar value={lastSearch.rootValue} />
                </div>
                <div style={{ fontSize: '0.78em', color: '#888', marginBottom: '4px' }}>
                  Visit counts, prior probabilities, and normalized visit probability.
                </div>
                {topMctsMoves.length === 0 ? (
                  <span style={{ color: '#888', fontSize: '0.9em' }}>No moves.</span>
                ) : (
                  <table style={{ borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ fontSize: '0.78em', color: '#666', textAlign: 'left' }}>
                        <th style={{ paddingRight: '6px' }}>#</th>
                        <th style={{ paddingRight: '8px' }}>Move</th>
                        <th style={{ paddingRight: '8px' }}>Visits</th>
                        <th style={{ paddingRight: '8px' }}>Visit%</th>
                        <th style={{ paddingRight: '8px' }}></th>
                        <th style={{ paddingRight: '8px' }}>Prior%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topMctsMoves.map(({ move, visits, prior, visitProb }, i) => (
                        <tr key={move} style={{ fontSize: '0.85em', fontFamily: 'monospace' }}>
                          <td style={{ paddingRight: '6px', color: '#888' }}>{i + 1}.</td>
                          <td style={{ paddingRight: '8px' }}>{move}</td>
                          <td style={{ paddingRight: '8px', textAlign: 'right' }}>{visits}</td>
                          <td style={{ paddingRight: '6px', textAlign: 'right' }}>
                            {(visitProb * 100).toFixed(1)}%
                          </td>
                          <td style={{ paddingRight: '8px' }}>
                            <ProbBar prob={visitProb} max={topMctsMoves[0].visitProb} />
                          </td>
                          <td style={{ paddingRight: '8px', color: '#888' }}>
                            {(prior * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            )}
          </section>

          {/* ── (d) Activation viewer ────────────────────────────────────────── */}
          <section>
            <strong>Activations</strong>
            <div style={{ fontSize: '0.78em', color: '#888', marginBottom: '6px' }}>
              Post-activation values for each layer on the current position.
              Color: blue = low, red/orange = high (clipped to ±3σ).
            </div>
            {activations === null ? (
              <span style={{ color: '#888', fontSize: '0.9em' }}>Not available.</span>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {Object.entries(activations).map(([name, arr]) => (
                  <details key={name} style={{ paddingLeft: '8px' }}>
                    <summary
                      style={{
                        cursor: 'pointer',
                        userSelect: 'none',
                        fontSize: '0.9em',
                        fontFamily: 'monospace',
                      }}
                    >
                      {name} ({arr.length} units)
                    </summary>
                    <div style={{ marginTop: '6px' }}>
                      <ActivationStrip activations={arr} />
                    </div>
                  </details>
                ))}
              </div>
            )}
          </section>

          {/* ── (e) Weight inspector ─────────────────────────────────────────── */}
          <section>
            <strong>Weight inspector</strong>
            <div style={{ fontSize: '0.78em', color: '#888', marginBottom: '6px' }}>
              Per-tensor statistics (min / max / mean / count). Full matrices
              are not shown — even a single 128×4096 head has 524 288 weights.
            </div>
            <button
              type="button"
              onClick={handleLoadWeights}
              disabled={loadingWeights}
              style={{ padding: '4px 12px', cursor: loadingWeights ? 'not-allowed' : 'pointer', marginBottom: '8px' }}
            >
              {loadingWeights ? 'Loading…' : 'Load weights'}
            </button>
            {weightStats !== null && (
              <div>
                <div style={{ fontSize: '0.85em', marginBottom: '6px' }}>
                  Total size:{' '}
                  <span style={{ fontFamily: 'monospace' }}>
                    {(weightStats.totalBytes / 1024).toFixed(1)} KiB
                  </span>
                </div>
                <table style={{ borderCollapse: 'collapse', fontSize: '0.8em', fontFamily: 'monospace' }}>
                  <thead>
                    <tr style={{ color: '#666', textAlign: 'left' }}>
                      <th style={{ paddingRight: '10px' }}>Tensor</th>
                      <th style={{ paddingRight: '8px', textAlign: 'right' }}>Count</th>
                      <th style={{ paddingRight: '8px', textAlign: 'right' }}>Min</th>
                      <th style={{ paddingRight: '8px', textAlign: 'right' }}>Max</th>
                      <th style={{ paddingRight: '8px', textAlign: 'right' }}>Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {weightStats.layerStats.map(({ name, count, min, max, mean }) => (
                      <tr key={name}>
                        <td style={{ paddingRight: '10px', color: '#333' }}>{name}</td>
                        <td style={{ paddingRight: '8px', textAlign: 'right', color: '#555' }}>
                          {count.toLocaleString()}
                        </td>
                        <td style={{ paddingRight: '8px', textAlign: 'right' }}>{min.toFixed(4)}</td>
                        <td style={{ paddingRight: '8px', textAlign: 'right' }}>{max.toFixed(4)}</td>
                        <td style={{ paddingRight: '8px', textAlign: 'right' }}>{mean.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

        </div>
      )}
    </details>
  );
}
