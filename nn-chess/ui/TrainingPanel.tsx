// onOpponentRefresh round-trips through storage rather than swapping ChessNet
// state in place — same code path as startup, one extra deserialize is cheap
// against the training cost.
//
// The Train button drives both nets in lockstep: opponent from its
// replay buffer, then imitator from its replay buffer. The AlphaZero-shaped
// loss is identical for both, and giving the user a single button keeps
// the framing simple ("train everything you've learned from"). The
// imitator pass is silently skipped when its buffer is empty.

import { useState, useRef, useCallback, useEffect } from 'react';
import type { Opponent } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { TrainingClient } from './training-client.js';
import type { DoneCallback, ProgressCallback } from './training-client.js';
import { OPPONENT_STORAGE_KEYS } from '../core/opponent/index.js';
import { IMITATOR_STORAGE_KEYS } from '../core/imitator/index.js';

export interface TrainingPanelProps {
  opponent: Opponent;
  imitator: PlayerImitator | null;
  storage: BlobStorage;
  onOpponentRefresh: () => Promise<void>;
}

interface TrainProgress {
  phase: 'opponent' | 'imitator';
  stepIndex: number;
  valueLoss: number;
  policyLoss: number;
  illegalLoss: number;
  totalLoss: number;
}

export function TrainingPanel({ opponent, imitator, storage, onOpponentRefresh }: TrainingPanelProps) {
  const [steps, setSteps] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const clientRef = useRef<TrainingClient | null>(null);

  useEffect(() => {
    return () => {
      clientRef.current?.stop();
      clientRef.current = null;
    };
  }, []);

  const handleTrain = useCallback(async () => {
    if (training) return;
    if (!imitator) {
      // The imitator is loaded alongside the opponent at startup, so this
      // null state is just the initial flash before useEffect resolves.
      setStatusMsg('Imitator not ready yet — try again in a moment.');
      return;
    }

    setTraining(true);
    setProgress(null);
    setStatusMsg(null);

    // Serialize current weights and replay buffers for both nets.
    let weights: Uint8Array;
    let replay: Uint8Array;
    let imitatorWeights: Uint8Array;
    let imitatorReplay: Uint8Array;
    try {
      [weights, imitatorWeights] = await Promise.all([
        opponent.net.serialize(),
        imitator.net.serialize(),
      ]);
      replay = opponent.replayBuffer.serialize();
      imitatorReplay = imitator.replayBuffer.serialize();
    } catch (err) {
      setStatusMsg(`Serialization failed: ${err instanceof Error ? err.message : String(err)}`);
      setTraining(false);
      return;
    }

    const onProgress: ProgressCallback = (msg) => {
      setProgress({
        phase: msg.phase,
        stepIndex: msg.stepIndex,
        valueLoss: msg.valueLoss,
        policyLoss: msg.policyLoss,
        illegalLoss: msg.illegalLoss,
        totalLoss: msg.totalLoss,
      });
    };

    const onDone: DoneCallback = async (result) => {
      // Persist both nets' new state, then trigger an opponent reload so
      // the in-memory handles match what's on disk.
      try {
        await Promise.all([
          storage.put(OPPONENT_STORAGE_KEYS.WEIGHTS, result.weights),
          storage.put(OPPONENT_STORAGE_KEYS.REPLAY, result.replay),
          storage.put(IMITATOR_STORAGE_KEYS.WEIGHTS, result.imitatorWeights),
          storage.put(IMITATOR_STORAGE_KEYS.REPLAY, result.imitatorReplay),
        ]);
        await onOpponentRefresh();
        setStatusMsg(
          `Training done — opponent ${result.finalStep} step(s), ` +
          `imitator ${result.imitatorFinalStep} step(s). Saved and reloaded.`,
        );
      } catch (err) {
        setStatusMsg(`Training done but save failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setTraining(false);
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
        imitatorWeights,
        imitatorReplay,
        replayCapacity: opponent.replayBuffer.capacity(),
        batchSize,
        totalSteps: steps,
        reportEvery: Math.max(1, Math.floor(steps / 10)),
      },
      onProgress,
      onDone,
      onError,
    );
  }, [training, opponent, imitator, storage, batchSize, steps, onOpponentRefresh]);

  const handleStop = useCallback(() => {
    clientRef.current?.stop();
    // The worker will post 'done' and onDone will set training = false.
  }, []);

  const bufferSize = opponent.replayBuffer.size();
  const bufferCapacity = opponent.replayBuffer.capacity();
  const imitatorBufferSize = imitator?.replayBuffer.size() ?? 0;

  const canTrain = !training && bufferSize >= batchSize && imitator !== null;

  return (
    <details
      style={{
        border: '1px solid #ccc',
        borderRadius: '6px',
        padding: '10px 14px',
        background: '#f9f9f9',
        color: '#1a1a1a',
      }}
    >
      <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none' }}>
        Training controls
      </summary>

      <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Buffer readout */}
        <div>
          <strong>Opponent replay buffer:</strong> {bufferSize.toLocaleString()} /{' '}
          {bufferCapacity.toLocaleString()} examples
          {bufferSize < batchSize && (
            <span style={{ color: '#888', marginLeft: '8px', fontSize: '0.9em' }}>
              (need at least {batchSize} to train)
            </span>
          )}
        </div>
        <div>
          <strong>Imitator replay buffer:</strong> {imitatorBufferSize.toLocaleString()} /{' '}
          {bufferCapacity.toLocaleString()} examples
          {imitatorBufferSize < batchSize && imitatorBufferSize > 0 && (
            <span style={{ color: '#888', marginLeft: '8px', fontSize: '0.9em' }}>
              (below batch size — imitator pass will be skipped)
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
        </div>

        {/* Progress */}
        {training && progress && (
          <div style={{ fontFamily: 'monospace', fontSize: '0.9em' }}>
            {progress.phase === 'opponent' ? 'Opponent' : 'Imitator'} step {progress.stepIndex} / {steps} —{' '}
            val: {progress.valueLoss.toFixed(4)},{' '}
            pol: {progress.policyLoss.toFixed(4)},{' '}
            ill: {progress.illegalLoss.toFixed(4)},{' '}
            total: {progress.totalLoss.toFixed(4)}
          </div>
        )}
        {training && !progress && (
          <div style={{ color: '#666', fontStyle: 'italic' }}>Training…</div>
        )}

        {/* Status message */}
        {statusMsg && (
          <div style={{ fontSize: '0.9em', color: '#333' }}>
            {statusMsg}
          </div>
        )}
      </div>
    </details>
  );
}
