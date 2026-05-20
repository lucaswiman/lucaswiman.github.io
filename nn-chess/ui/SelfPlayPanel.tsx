// SelfPlayPanel — controls for running self-play games between the AI
// and the player imitator. Each batch generates AlphaZero-style
// training examples that are added to the AI's replay buffer, then a
// few gradient steps are taken so the new data immediately influences
// the AI's evaluation.
//
// Self-play runs on the main thread. With a low MCTS simulation count
// (see SELFPLAY_SEARCH in agent-config.ts) a small batch is tolerable
// even on a phone, but the UI thread will be busy. The panel exposes a
// "Stop" button that aborts between plies via an AbortSignal so the
// user can bail out at any point.

import { useCallback, useRef, useState } from 'react';
import type { Agent } from '../core/agent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { runSelfPlayBatch } from '../core/selfplay/index.js';
import { policyFromNet } from '../core/agent/index.js';
import {
  SELFPLAY_SEARCH,
  SELFPLAY_BATCH_SIZE,
  SELFPLAY_MAX_PLIES,
} from './agent-config.js';

export interface SelfPlayPanelProps {
  agent: Agent | null;
  imitator: PlayerImitator | null;
  storage: BlobStorage;
  /** Reload both agents from storage after a batch updates their weights. */
  onAgentRefresh: () => Promise<void>;
}

interface BatchStatus {
  played: number;
  wins: number;
  losses: number;
  draws: number;
  examples: number;
}

export function SelfPlayPanel({
  agent,
  imitator,
  storage,
  onAgentRefresh,
}: SelfPlayPanelProps) {
  const [batchSize, setBatchSize] = useState(SELFPLAY_BATCH_SIZE);
  const [running, setRunning] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchStatus | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const handleRun = useCallback(async () => {
    if (running || !agent || !imitator) return;

    setRunning(true);
    setStatusMsg(null);
    setBatch({ played: 0, wins: 0, losses: 0, draws: 0, examples: 0 });
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await runSelfPlayBatch({
        aiPolicy: policyFromNet(agent.net),
        playerPolicy: policyFromNet(imitator.net),
        search: SELFPLAY_SEARCH,
        count: batchSize,
        maxPlies: SELFPLAY_MAX_PLIES,
        signal: controller.signal,
        onGame: (idx, info) => {
          setBatch(prev => {
            const base = prev ?? { played: 0, wins: 0, losses: 0, draws: 0, examples: 0 };
            return {
              played: idx + 1,
              wins: base.wins + (info.aiResult === 'win' ? 1 : 0),
              losses: base.losses + (info.aiResult === 'loss' ? 1 : 0),
              draws: base.draws + (info.aiResult === 'draw' ? 1 : 0),
              examples: base.examples,
            };
          });
        },
      });

      // Add the AI's training examples into the buffer.
      for (const ex of result.aiTrainingExamples) {
        agent.replayBuffer.add(ex);
      }
      setBatch(prev => prev === null ? null : { ...prev, examples: result.aiTrainingExamples.length });

      // Take a few gradient steps on the new data so the AI's
      // evaluations actually shift before the next user game.
      const stepBatch = 32;
      const stepsToRun = 10;
      let steps = 0;
      for (let i = 0; i < stepsToRun; i++) {
        if (controller.signal.aborted) break;
        const r = await agent.trainStep(stepBatch);
        if (r === null) break;
        steps++;
      }

      await agent.saveTo(storage);
      await onAgentRefresh();

      setStatusMsg(
        `Self-play done — ${result.games} game(s): ` +
        `AI ${result.wins}W ${result.losses}L ${result.draws}D, ` +
        `${result.aiTrainingExamples.length} new examples, ${steps} grad step(s).`,
      );
    } catch (err) {
      setStatusMsg(`Self-play failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      abortRef.current = null;
      setRunning(false);
    }
  }, [running, agent, imitator, batchSize, storage, onAgentRefresh]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const bufferSize = imitator?.replayBuffer.size() ?? 0;

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
        Self-play training (AI vs. player imitator)
      </summary>

      <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ fontSize: '0.9em', color: '#444' }}>
          The AI plays the player-imitator under MCTS and trains on the
          outcomes. The imitator is whatever pattern your moves have
          taught it so far — <strong>{bufferSize.toLocaleString()}</strong>{' '}
          user moves recorded.
        </div>

        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label>
            Games:{' '}
            <input
              type="number"
              min={1}
              max={50}
              value={batchSize}
              onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
              style={{ width: '70px' }}
              disabled={running}
            />
          </label>
          <span style={{ fontSize: '0.85em', color: '#666' }}>
            MCTS sims/move: {SELFPLAY_SEARCH.simulations}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={handleRun}
            disabled={running || agent === null || imitator === null}
            style={{ padding: '5px 14px', cursor: running ? 'not-allowed' : 'pointer' }}
          >
            Run self-play batch
          </button>
          <button
            type="button"
            onClick={handleStop}
            disabled={!running}
            style={{ padding: '5px 14px', cursor: running ? 'pointer' : 'not-allowed' }}
          >
            Stop
          </button>
        </div>

        {batch !== null && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
            Game {batch.played} / {batchSize} —{' '}
            AI: {batch.wins}W {batch.losses}L {batch.draws}D
            {batch.examples > 0 && ` — ${batch.examples} examples`}
          </div>
        )}

        {statusMsg && (
          <div style={{ fontSize: '0.9em', color: '#333' }}>{statusMsg}</div>
        )}
      </div>
    </details>
  );
}
