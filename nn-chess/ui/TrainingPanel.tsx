// onAgentRefresh round-trips through storage rather than swapping ChessNet
// state in place — same code path as startup, one extra deserialize is cheap
// against the training cost.

import { useState, useRef, useCallback, useEffect } from 'react';
import type { Agent } from '../core/agent/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { TrainingClient } from './training-client.js';
import type { DoneCallback, ProgressCallback } from './training-client.js';
import { AGENT_STORAGE_KEYS } from '../core/agent/index.js';

export interface TrainingPanelProps {
  agent: Agent;
  storage: BlobStorage;
  onAgentRefresh: () => Promise<void>;
}

interface TrainProgress {
  stepIndex: number;
  valueLoss: number;
  policyLoss: number;
  totalLoss: number;
}

export function TrainingPanel({ agent, storage, onAgentRefresh }: TrainingPanelProps) {
  const [steps, setSteps] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const clientRef = useRef<TrainingClient | null>(null);

  useEffect(() => {
    return () => {
      clientRef.current?.stop();
      clientRef.current = null;
    };
  }, []);

  const handleTrain = useCallback(async () => {
    if (training) return;

    setTraining(true);
    setProgress(null);
    setStatusMsg(null);

    // Serialize current weights and replay buffer to send to the worker.
    let weights: Uint8Array;
    let replay: Uint8Array;
    try {
      weights = await agent.net.serialize();
      replay = agent.replayBuffer.serialize();
    } catch (err) {
      setStatusMsg(`Serialization failed: ${err instanceof Error ? err.message : String(err)}`);
      setTraining(false);
      return;
    }

    const onProgress: ProgressCallback = (msg) => {
      setProgress({
        stepIndex: msg.stepIndex,
        valueLoss: msg.valueLoss,
        policyLoss: msg.policyLoss,
        totalLoss: msg.totalLoss,
      });
    };

    const onDone: DoneCallback = async (result) => {
      setTraining(false);
      setStatusMsg(`Training done — ${result.finalStep} steps.`);

      // Persist the new weights and replay buffer to storage.
      setSaving(true);
      try {
        await Promise.all([
          storage.put(AGENT_STORAGE_KEYS.WEIGHTS, result.weights),
          storage.put(AGENT_STORAGE_KEYS.REPLAY, result.replay),
        ]);
        // Reload the agent from storage so it picks up the new weights.
        await onAgentRefresh();
        setStatusMsg(`Training done — ${result.finalStep} steps. Weights saved and reloaded.`);
      } catch (err) {
        setStatusMsg(`Training done but save failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setSaving(false);
      }
    };

    const onError = (msg: string) => {
      setTraining(false);
      setStatusMsg(`Error: ${msg}`);
    };

    if (!clientRef.current) {
      clientRef.current = new TrainingClient();
    }

    clientRef.current.start(
      {
        weights,
        replay,
        replayCapacity: agent.replayBuffer.capacity(),
        batchSize,
        totalSteps: steps,
        reportEvery: Math.max(1, Math.floor(steps / 10)),
      },
      onProgress,
      onDone,
      onError,
    );
  }, [training, agent, storage, batchSize, steps, onAgentRefresh]);

  const handleStop = useCallback(() => {
    clientRef.current?.stop();
    // The worker will post 'done' and onDone will set training = false.
  }, []);

  const handleSaveWeights = useCallback(async () => {
    setSaving(true);
    setStatusMsg(null);
    try {
      await agent.saveTo(storage);
      setStatusMsg('Weights saved.');
    } catch (err) {
      setStatusMsg(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSaving(false);
    }
  }, [agent, storage]);

  const bufferSize = agent.replayBuffer.size();
  const bufferCapacity = agent.replayBuffer.capacity();

  const canTrain = !training && bufferSize >= batchSize;

  return (
    <details
      style={{
        border: '1px solid #ccc',
        borderRadius: '6px',
        padding: '10px 14px',
        background: '#f9f9f9',
      }}
    >
      <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none' }}>
        Training controls
      </summary>

      <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Buffer readout */}
        <div>
          <strong>Replay buffer:</strong> {bufferSize.toLocaleString()} /{' '}
          {bufferCapacity.toLocaleString()} examples
          {bufferSize < batchSize && (
            <span style={{ color: '#888', marginLeft: '8px', fontSize: '0.9em' }}>
              (need at least {batchSize} to train)
            </span>
          )}
        </div>

        {/* Numeric inputs */}
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label>
            Steps:{' '}
            <input
              type="number"
              min={1}
              max={10000}
              value={steps}
              onChange={e => setSteps(Math.max(1, parseInt(e.target.value, 10) || 1))}
              style={{ width: '80px' }}
              disabled={training}
            />
          </label>
          <label>
            Batch size:{' '}
            <input
              type="number"
              min={1}
              max={1024}
              value={batchSize}
              onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value, 10) || 1))}
              style={{ width: '70px' }}
              disabled={training}
            />
          </label>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={handleTrain}
            disabled={!canTrain}
            style={{ padding: '5px 14px', cursor: canTrain ? 'pointer' : 'not-allowed' }}
          >
            Train
          </button>
          <button
            type="button"
            onClick={handleStop}
            disabled={!training}
            style={{ padding: '5px 14px', cursor: training ? 'pointer' : 'not-allowed' }}
          >
            Stop
          </button>
          <button
            type="button"
            onClick={handleSaveWeights}
            disabled={saving || training}
            style={{ padding: '5px 14px', cursor: (!saving && !training) ? 'pointer' : 'not-allowed' }}
          >
            Save weights
          </button>
        </div>

        {/* Progress */}
        {training && progress && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
            Step {progress.stepIndex} / {steps} —{' '}
            val: {progress.valueLoss.toFixed(4)},{' '}
            pol: {progress.policyLoss.toFixed(4)},{' '}
            total: {progress.totalLoss.toFixed(4)}
          </div>
        )}
        {training && !progress && (
          <div style={{ color: '#666', fontStyle: 'italic' }}>Training…</div>
        )}

        {/* Status message */}
        {statusMsg && (
          <div style={{ fontSize: '0.9em', color: saving ? '#666' : '#333' }}>
            {statusMsg}
          </div>
        )}
      </div>
    </details>
  );
}
