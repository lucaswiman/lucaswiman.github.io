// TrainingLog — a single rolling log shared by every training-side panel
// (manual Train, self-play, endgame gym) so the user can see at a glance
// which kind of work is happening right now: gradient descent on one of
// the nets, MCTS inference inside a self-play game, or a save to
// IndexedDB. Without this it's easy to mistake long stretches of MCTS
// for "stuck" and vice versa.
//
// The log itself is just an array of entries with a monotonic id; Game.tsx
// owns the state and hands `append` down to each panel. The panels call
// it at phase boundaries (not per-step) so the log stays scannable.

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Cap on retained entries. Tuned to fit a long gym run (hundreds of
 * scenarios) without blowing up DOM size — old entries are dropped from
 * the front when the cap is exceeded.
 */
const LOG_CAPACITY = 200;

export type LogSource = 'training' | 'selfplay' | 'gym';

/**
 * `phase` — gradient descent / MCTS / batch boundary started.
 * `save`  — bytes just landed in storage.
 * `info`  — neutral context (batch began, ended, etc.).
 * `error` — something failed; surfaced in red.
 */
export type LogKind = 'phase' | 'save' | 'info' | 'error';

export interface LogEntry {
  id: number;
  ts: number;
  source: LogSource;
  kind: LogKind;
  message: string;
}

/** Callback the panels use to push a new entry. */
export type LogAppender = (entry: Omit<LogEntry, 'id' | 'ts'>) => void;

let nextId = 0;

/**
 * State holder for the log. Returns a stable `append` callback so
 * children don't re-render on every tick.
 */
export function useTrainingLog(): {
  entries: LogEntry[];
  append: LogAppender;
  clear: () => void;
} {
  const [entries, setEntries] = useState<LogEntry[]>([]);

  const append = useCallback<LogAppender>((entry) => {
    setEntries(prev => {
      const next: LogEntry = { ...entry, id: nextId++, ts: Date.now() };
      const grown = [...prev, next];
      return grown.length > LOG_CAPACITY ? grown.slice(grown.length - LOG_CAPACITY) : grown;
    });
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, append, clear };
}

const SOURCE_LABEL: Record<LogSource, string> = {
  training: 'train',
  selfplay: 'selfplay',
  gym: 'gym',
};

const KIND_COLOR: Record<LogKind, string> = {
  phase: '#1e5fb4',
  save: '#1f8a3a',
  info: '#444',
  error: '#b40000',
};

function formatTime(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export interface TrainingLogProps {
  entries: LogEntry[];
  onClear: () => void;
}

export function TrainingLog({ entries, onClear }: TrainingLogProps) {
  // Auto-scroll the entry list to the bottom when new entries arrive so
  // the most recent activity stays visible. Guarded so the user can
  // scroll up to read history if they want — we only snap to bottom when
  // we were already near the bottom before the append.
  const listRef = useRef<HTMLDivElement | null>(null);
  const wasAtBottomRef = useRef(true);

  useEffect(() => {
    const el = listRef.current;
    if (el && wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [entries]);

  const handleScroll = useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    // "Near bottom" tolerance covers smooth-scroll wobble and sub-pixel rounding.
    wasAtBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }, []);

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
        Training log
      </summary>

      <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', fontSize: '0.85em', color: '#666' }}>
          <span>{entries.length} entr{entries.length === 1 ? 'y' : 'ies'}</span>
          <button
            type="button"
            onClick={onClear}
            disabled={entries.length === 0}
            style={{ padding: '2px 10px', cursor: entries.length === 0 ? 'not-allowed' : 'pointer' }}
          >
            Clear
          </button>
        </div>

        <div
          ref={listRef}
          onScroll={handleScroll}
          style={{
            maxHeight: '220px',
            overflowY: 'auto',
            border: '1px solid #ddd',
            borderRadius: '4px',
            background: '#fff',
            padding: '6px 10px',
            fontFamily: 'monospace',
            fontSize: '0.85em',
            lineHeight: 1.4,
          }}
        >
          {entries.length === 0 ? (
            <div style={{ color: '#999', fontStyle: 'italic' }}>
              No training activity yet. The log records phase transitions
              (gradient descent vs. MCTS) and saves to storage.
            </div>
          ) : (
            entries.map(e => (
              <div key={e.id} style={{ color: KIND_COLOR[e.kind] }}>
                <span style={{ color: '#888' }}>{formatTime(e.ts)}</span>{' '}
                <span style={{ color: '#888' }}>[{SOURCE_LABEL[e.source]}]</span>{' '}
                {e.message}
              </div>
            ))
          )}
        </div>
      </div>
    </details>
  );
}
