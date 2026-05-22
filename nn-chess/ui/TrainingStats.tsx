// TrainingStats — at-a-glance "how much training has actually happened
// to each net" panel. Reads from the cumulative `NetStats` blobs and
// turns them into something the user can interpret:
//
//   - Lifetime gradient steps
//   - Examples seen (steps × batch size, summed across every training pass)
//   - Effective epochs over the *current* buffer (examplesSeen / bufferSize)
//
// Effective epochs is the overfit heuristic: when this number rises into
// the tens against a small buffer, the net has revisited the same
// examples many times and is plausibly memorizing them. Without a
// validation split this is the cheapest available signal.

import type { NetStats } from './training-stats.js';
import { effectiveEpochs } from './training-stats.js';

export interface TrainingStatsPanelProps {
  opponentStats: NetStats;
  imitatorStats: NetStats;
  opponentBufferSize: number;
  imitatorBufferSize: number;
  opponentBufferCapacity: number;
  imitatorBufferCapacity: number;
}

function formatTimestamp(ts: number | null): string {
  if (ts === null) return 'never';
  const d = new Date(ts);
  const now = Date.now();
  const delta = now - ts;
  if (delta < 60_000) return `${Math.round(delta / 1000)}s ago`;
  if (delta < 3_600_000) return `${Math.round(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.round(delta / 3_600_000)}h ago`;
  return d.toLocaleString();
}

/**
 * Color the effective-epochs readout when it crosses common overfitting
 * thresholds. With pure behaviour cloning on a small replay buffer,
 * single-digit epochs are usually fine; double-digit is where memorization
 * starts to dominate generalization.
 */
function epochColor(epochs: number): string {
  if (epochs >= 20) return '#b40000';
  if (epochs >= 10) return '#c46a00';
  return '#1a1a1a';
}

interface NetRowProps {
  label: string;
  stats: NetStats;
  bufferSize: number;
  bufferCapacity: number;
  description: string;
}

function NetRow({ label, stats, bufferSize, bufferCapacity, description }: NetRowProps) {
  const epochs = effectiveEpochs(stats, bufferSize);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '0.9em' }}>
      <div style={{ fontWeight: 'bold' }}>{label}</div>
      <div style={{ color: '#444', fontSize: '0.85em' }}>{description}</div>
      <div style={{ fontFamily: 'monospace' }}>
        <span style={{ color: '#666' }}>Gradient steps:</span>{' '}
        {stats.gradientSteps.toLocaleString()}
        {'  '}
        <span style={{ color: '#666' }}>Examples seen:</span>{' '}
        {stats.examplesSeen.toLocaleString()}
      </div>
      <div style={{ fontFamily: 'monospace' }}>
        <span style={{ color: '#666' }}>Effective epochs over current buffer:</span>{' '}
        <span style={{ color: epochColor(epochs), fontWeight: epochs >= 10 ? 'bold' : 'normal' }}>
          {epochs.toFixed(2)}×
        </span>{' '}
        <span style={{ color: '#888' }}>
          ({bufferSize.toLocaleString()} / {bufferCapacity.toLocaleString()} examples in buffer)
        </span>
      </div>
      <div style={{ fontFamily: 'monospace', color: '#666' }}>
        Last trained: {formatTimestamp(stats.lastTrainedAt)}
      </div>
    </div>
  );
}

export function TrainingStatsPanel({
  opponentStats,
  imitatorStats,
  opponentBufferSize,
  imitatorBufferSize,
  opponentBufferCapacity,
  imitatorBufferCapacity,
}: TrainingStatsPanelProps) {
  return (
    <details
      open
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
        Training health
      </summary>

      <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ fontSize: '0.85em', color: '#555' }}>
          "Effective epochs" is examples-seen divided by the current buffer
          size — a rough sense of how many times the net has revisited
          each example on average. Single-digit values are normal; values
          past ~10 mean the imitator is plausibly memorizing the user's
          moves rather than generalizing.
        </div>

        <NetRow
          label="Opponent"
          description="The network the user plays against. Trained AlphaZero-style on terminal outcomes."
          stats={opponentStats}
          bufferSize={opponentBufferSize}
          bufferCapacity={opponentBufferCapacity}
        />
        <NetRow
          label="Imitator"
          description="Behaviour-clones the user's moves. Acts as the opponent's self-play stand-in."
          stats={imitatorStats}
          bufferSize={imitatorBufferSize}
          bufferCapacity={imitatorBufferCapacity}
        />
      </div>
    </details>
  );
}
