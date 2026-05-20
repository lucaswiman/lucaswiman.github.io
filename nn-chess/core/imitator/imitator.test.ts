// Unit tests for the player-imitator behavioural contract.
// We test against a stub ChessNet so the suite stays tfjs-free.

import { describe, expect, it } from 'vitest';
import { createPlayerImitator, IMITATOR_STORAGE_KEYS } from './index.js';
import { createReplayBuffer } from '../training/index.js';
import { createMemoryStorage } from '../../adapters/storage-memory.js';
import { buildImitationExamples, type GameMoveRecord } from '../agent/index.js';
import {
  POLICY_SIZE,
  type ChessNet,
  type TrainResult,
  type TrainingExample,
} from '../nn/index.js';
import { FEATURE_COUNT } from '../encoding/index.js';
import { policyIndexFromUci } from '../nn/index.js';

// Minimal ChessNet stand-in. Only the methods the imitator actually
// calls (trainBatch, serialize) need to behave; everything else throws
// so we catch accidental dependencies.
function stubNet(): ChessNet & {
  trainCalls: TrainingExample[][];
  serializeCalls: number;
} {
  const trainCalls: TrainingExample[][] = [];
  let serializeCalls = 0;
  const net = {
    predict() { throw new Error('predict not used in these tests'); },
    async trainBatch(examples: TrainingExample[]): Promise<TrainResult> {
      trainCalls.push(examples);
      return { valueLoss: 0.1, policyLoss: 0.2, totalLoss: 0.3 };
    },
    async serialize(): Promise<Uint8Array> {
      serializeCalls++;
      return new Uint8Array([0xab, 0xcd]);
    },
    layerNames() { return []; },
    getActivations() { return {}; },
    dispose() { /* no-op */ },
  } as unknown as ChessNet & { trainCalls: TrainingExample[][]; serializeCalls: number };
  Object.defineProperty(net, 'trainCalls', { get: () => trainCalls });
  Object.defineProperty(net, 'serializeCalls', { get: () => serializeCalls });
  return net;
}

function moveRecord(side: 'w' | 'b', uci: string): GameMoveRecord {
  return {
    features: new Float32Array(FEATURE_COUNT),
    move: uci,
    sideToMove: side,
  };
}

describe('buildImitationExamples', () => {
  it('produces a positive policy target for every move (unlike buildTrainingExamples)', () => {
    const moves = [
      moveRecord('w', 'e2e4'),
      moveRecord('w', 'g1f3'),
    ];
    const examples = buildImitationExamples(moves, 'black-wins');
    expect(examples).toHaveLength(2);
    for (const ex of examples) {
      expect(ex.policyTarget).not.toBeNull();
      expect(ex.policyTarget!.length).toBe(POLICY_SIZE);
    }
    expect(examples[0].policyTarget![policyIndexFromUci('e2e4')]).toBe(1);
    expect(examples[1].policyTarget![policyIndexFromUci('g1f3')]).toBe(1);
  });

  it('encodes the value target from the user side\'s perspective', () => {
    const whiteMoves = [moveRecord('w', 'e2e4'), moveRecord('w', 'd2d4')];
    expect(buildImitationExamples(whiteMoves, 'white-wins').map(e => e.valueTarget)).toEqual([1, 1]);
    expect(buildImitationExamples(whiteMoves, 'black-wins').map(e => e.valueTarget)).toEqual([-1, -1]);
    expect(buildImitationExamples(whiteMoves, 'draw').map(e => e.valueTarget)).toEqual([0, 0]);
  });

  it('throws on an unfinished game', () => {
    expect(() => buildImitationExamples([moveRecord('w', 'e2e4')], 'ongoing')).toThrow();
  });
});

describe('PlayerImitator', () => {
  it('recordUserMoves adds one example per move (always policy-targeted)', () => {
    const buf = createReplayBuffer(100);
    const net = stubNet();
    const imitator = createPlayerImitator({ net, replayBuffer: buf });

    imitator.recordUserMoves(
      [moveRecord('w', 'e2e4'), moveRecord('w', 'g1f3')],
      'white-wins',
    );

    expect(buf.size()).toBe(2);
    const snap = buf.snapshot();
    for (const ex of snap) {
      expect(ex.policyTarget).not.toBeNull();
    }
  });

  it('recordUserMoves with no moves is a no-op', () => {
    const buf = createReplayBuffer(10);
    const imitator = createPlayerImitator({ net: stubNet(), replayBuffer: buf });
    imitator.recordUserMoves([], 'draw');
    expect(buf.size()).toBe(0);
  });

  it('trainStep returns null until enough examples are buffered', async () => {
    const buf = createReplayBuffer(50);
    const net = stubNet();
    const imitator = createPlayerImitator({ net, replayBuffer: buf });

    expect(await imitator.trainStep(4)).toBeNull();

    imitator.recordUserMoves(
      [
        moveRecord('w', 'e2e4'),
        moveRecord('w', 'd2d4'),
        moveRecord('w', 'g1f3'),
        moveRecord('w', 'b1c3'),
      ],
      'white-wins',
    );

    const result = await imitator.trainStep(4);
    expect(result).not.toBeNull();
    expect(result!.totalLoss).toBeCloseTo(0.3);
    // The stub net should have been called once with a 4-example batch.
    expect((net as ReturnType<typeof stubNet>).trainCalls).toHaveLength(1);
    expect((net as ReturnType<typeof stubNet>).trainCalls[0]).toHaveLength(4);
  });

  it('saveTo writes weights, replay, and meta to the imitator keyspace', async () => {
    const buf = createReplayBuffer(10);
    const net = stubNet();
    const imitator = createPlayerImitator({ net, replayBuffer: buf });
    imitator.recordUserMoves([moveRecord('w', 'e2e4')], 'draw');

    const storage = createMemoryStorage();
    await imitator.saveTo(storage);

    const keys = (await storage.list()).sort();
    expect(keys).toEqual(
      [
        IMITATOR_STORAGE_KEYS.META,
        IMITATOR_STORAGE_KEYS.REPLAY,
        IMITATOR_STORAGE_KEYS.WEIGHTS,
      ].sort(),
    );
  });
});
