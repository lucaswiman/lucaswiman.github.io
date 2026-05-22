// Unit tests for core/opponent/index.ts.

import { describe, expect, it } from 'vitest';
import {
  buildTrainingExamples,
  policyFromNet,
  createOpponent,
  loadOpponent,
  legalMoveIndicesFor,
  mateInOneIndicesFor,
  OPPONENT_STORAGE_KEYS,
  type GameMoveRecord,
} from './index.js';
import { ChessNet, POLICY_SIZE, NN_VERSION, policyIndexFromUci } from '../nn/index.js';
import { createReplayBuffer } from '../training/index.js';
import { createMemoryStorage } from '../../adapters/storage-memory.js';
import {
  initialState,
  fromFen,
  legalMoves,
  applyMove,
  sideToMove,
} from '../rules/index.js';
import { encodeState } from '../encoding/index.js';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Simple seeded PRNG (mulberry32). */
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Build a quick set of move records for a two-move game. */
function twoMoveGame(): GameMoveRecord[] {
  const state0 = initialState();
  const moves0 = legalMoves(state0);
  const move0 = moves0[0];
  const state1 = applyMove(state0, move0);
  const moves1 = legalMoves(state1);
  const move1 = moves1[0];

  return [
    {
      features: encodeState(state0),
      move: move0,
      sideToMove: 'w',
      legalMoveIndices: legalMoveIndicesFor(state0),
      mateMoveIndices: mateInOneIndicesFor(state0),
    },
    {
      features: encodeState(state1),
      move: move1,
      sideToMove: 'b',
      legalMoveIndices: legalMoveIndicesFor(state1),
      mateMoveIndices: mateInOneIndicesFor(state1),
    },
  ];
}

/** Compare two Float32Arrays byte-for-byte. */
function float32Equal(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// ── buildTrainingExamples ────────────────────────────────────────────────────

describe('buildTrainingExamples', () => {
  it('output length matches input length', () => {
    const records = twoMoveGame();
    const examples = buildTrainingExamples(records, 'white-wins');
    expect(examples).toHaveLength(records.length);
  });

  it('white wins: white positions get valueTarget=+1 and one-hot policy, black positions get -1 and null policy', () => {
    const records = twoMoveGame();
    // records[0] is white-to-move, records[1] is black-to-move.
    const examples = buildTrainingExamples(records, 'white-wins');

    // White-to-move position.
    expect(examples[0].valueTarget).toBe(1);
    expect(examples[0].policyTarget).not.toBeNull();
    expect(examples[0].policyTarget).toHaveLength(POLICY_SIZE);
    // One-hot: exactly one 1, the rest 0.
    const hotCount = Array.from(examples[0].policyTarget!).filter(v => v === 1).length;
    expect(hotCount).toBe(1);
    expect(Array.from(examples[0].policyTarget!).reduce((a, b) => a + b, 0)).toBe(1);

    // Black-to-move position.
    expect(examples[1].valueTarget).toBe(-1);
    expect(examples[1].policyTarget).toBeNull();
  });

  it('black wins: black positions get valueTarget=+1 and one-hot policy, white positions get -1 and null policy', () => {
    const records = twoMoveGame();
    const examples = buildTrainingExamples(records, 'black-wins');

    // White-to-move position.
    expect(examples[0].valueTarget).toBe(-1);
    expect(examples[0].policyTarget).toBeNull();

    // Black-to-move position.
    expect(examples[1].valueTarget).toBe(1);
    expect(examples[1].policyTarget).not.toBeNull();
    const hotCount = Array.from(examples[1].policyTarget!).filter(v => v === 1).length;
    expect(hotCount).toBe(1);
  });

  it('draw: all positions get valueTarget=0 and null policy', () => {
    const records = twoMoveGame();
    const examples = buildTrainingExamples(records, 'draw');

    for (const ex of examples) {
      expect(ex.valueTarget).toBe(0);
      expect(ex.policyTarget).toBeNull();
    }
  });

  it("ongoing outcome throws", () => {
    const records = twoMoveGame();
    expect(() => buildTrainingExamples(records, 'ongoing')).toThrow();
  });

  it('emits a corrective policy target when a mate-in-1 was on the board and was MISSED', () => {
    // White Ke6, Qd1, Black Kf8. The position is a forced mate-in-1
    // for White: Qd7! threatens Qf7# and Black has no escape — wait,
    // simpler: reuse the existing mate-in-1 fixture from mcts.test.ts:
    // K f6, Qg1, k g8.  g1g7 is the unique mate.  We record a NON-mate
    // move (g1a1, a quiet queen retreat) to simulate the case the user
    // is worried about: a checkmate left on the table.
    const state = fromFen('6k1/8/5K2/8/8/8/8/6Q1 w - - 0 1');
    const records: GameMoveRecord[] = [{
      features: encodeState(state),
      move: 'g1a1',  // legal non-mate retreat
      sideToMove: 'w',
      legalMoveIndices: legalMoveIndicesFor(state),
      mateMoveIndices: mateInOneIndicesFor(state),
    }];
    // The fixture is interesting only if there really IS a mate-in-1.
    expect(records[0].mateMoveIndices.length).toBeGreaterThan(0);

    // Both win and draw should still emit the corrective target — the
    // corrective behaviour is independent of game outcome.
    const examples = buildTrainingExamples(records, 'draw');
    const target = examples[0].policyTarget!;
    expect(target).not.toBeNull();

    // Every mate index should have positive weight.
    for (const i of records[0].mateMoveIndices) {
      expect(target[i]).toBeGreaterThan(0);
    }
    // Every legal non-mate index should have NEGATIVE weight (the played
    // move d1d2 is one of these, but the property must hold for every
    // non-mate alternative).
    const mateSet = new Set(records[0].mateMoveIndices);
    const nonMate = records[0].legalMoveIndices.filter(i => !mateSet.has(i));
    for (const i of nonMate) {
      expect(target[i]).toBeLessThan(0);
    }
    // Positive and negative sides each sum to ±1.
    const pos = records[0].mateMoveIndices.reduce((s, i) => s + target[i], 0);
    const neg = nonMate.reduce((s, i) => s + target[i], 0);
    expect(pos).toBeCloseTo(1, 5);
    expect(neg).toBeCloseTo(-1, 5);
  });

  it('policy target one-hot index matches the played move', async () => {
    const records = twoMoveGame();
    // records[0] is white-to-move; white wins → policy target is set.
    const examples = buildTrainingExamples(records, 'white-wins');
    const pt = examples[0].policyTarget!;
    // Find the hot index.
    let hotIdx = -1;
    for (let i = 0; i < pt.length; i++) {
      if (pt[i] === 1) { hotIdx = i; break; }
    }
    expect(hotIdx).toBeGreaterThanOrEqual(0);
    // Verify it encodes the played move.
    expect(hotIdx).toBe(policyIndexFromUci(records[0].move));
  });

  it('handles a longer game (6 moves)', () => {
    const records: GameMoveRecord[] = [];
    let state = initialState();
    for (let i = 0; i < 6; i++) {
      const moves = legalMoves(state);
      const move = moves[0];
      records.push({
        features: encodeState(state),
        move,
        sideToMove: sideToMove(state),
        legalMoveIndices: legalMoveIndicesFor(state),
        mateMoveIndices: mateInOneIndicesFor(state),
      });
      state = applyMove(state, move);
    }

    const examplesWW = buildTrainingExamples(records, 'white-wins');
    expect(examplesWW).toHaveLength(6);
    for (const ex of examplesWW) {
      expect(Math.abs(ex.valueTarget)).toBe(1);
    }

    const examplesDraw = buildTrainingExamples(records, 'draw');
    for (const ex of examplesDraw) {
      expect(ex.valueTarget).toBe(0);
      expect(ex.policyTarget).toBeNull();
    }
  });
});

// ── policyFromNet ─────────────────────────────────────────────────────────────

describe('policyFromNet', () => {
  it('returned priors are only over legal moves', async () => {
    const net = ChessNet.create({ seed: 42 });
    const policy = policyFromNet(net);
    const state = initialState();
    const legal = new Set(legalMoves(state));

    const { priors } = await policy.evaluate(state);

    for (const move of priors.keys()) {
      expect(legal.has(move)).toBe(true);
    }
    // All legal moves should be in the priors map.
    for (const move of legal) {
      expect(priors.has(move)).toBe(true);
    }
  });

  it('priors sum to ~1 (within 1e-5)', async () => {
    const net = ChessNet.create({ seed: 42 });
    const policy = policyFromNet(net);
    const state = initialState();

    const { priors } = await policy.evaluate(state);
    const sum = Array.from(priors.values()).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 4); // within 1e-4
  });

  it('returned value is in [-1, 1]', async () => {
    const net = ChessNet.create({ seed: 42 });
    const policy = policyFromNet(net);
    const state = initialState();

    const { value } = await policy.evaluate(state);
    expect(value).toBeGreaterThanOrEqual(-1);
    expect(value).toBeLessThanOrEqual(1);
  });

  it('terminal position: returns empty priors and value 0', async () => {
    // Fool's mate — checkmate in 2.
    // 1. f3 e5  2. g4 Qh4#
    const net = ChessNet.create({ seed: 42 });
    const policy = policyFromNet(net);
    const terminalFen = 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3';
    const state = fromFen(terminalFen);

    const { priors, value } = await policy.evaluate(state);
    expect(priors.size).toBe(0);
    expect(value).toBe(0);
  });
});

// ── createOpponent / selectMove ──────────────────────────────────────────────

describe('createOpponent / selectMove', () => {
  it('temperature=0 and deterministic rng: selectMove returns the highest-visit move', async () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const rng = makePrng(7);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: {
        simulations: 30,
        cPuct: 1.5,
        temperature: 0,
        rng,
      },
    });

    const state = initialState();
    const { move, visitCounts } = await opponent.selectMove(state);

    // The returned move must be the one with the most visits.
    const bestN = Math.max(...Array.from(visitCounts.values()));
    expect(visitCounts.get(move)).toBe(bestN);
  }, 30_000);

  it('temperature>0 and seeded rng: same seed → same move', async () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const state = initialState();

    const runWithSeed = async (seed: number) => {
      const opponent = createOpponent({
        net,
        replayBuffer: buf,
        config: {
          simulations: 30,
          cPuct: 1.5,
          temperature: 1.0,
          rng: makePrng(seed),
        },
      });
      const { move } = await opponent.selectMove(state);
      return move;
    };

    const move1 = await runWithSeed(123);
    const move2 = await runWithSeed(123);
    expect(move1).toBe(move2);
  }, 30_000);

  it('selectMove returns a legal move', async () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: {
        simulations: 20,
        cPuct: 1.5,
        temperature: 0,
        rng: makePrng(99),
      },
    });

    const state = initialState();
    const { move } = await opponent.selectMove(state);
    const legal = legalMoves(state);
    expect(legal).toContain(move);
  }, 30_000);

  it('recordGameResult adds examples to the replay buffer', () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });

    expect(opponent.replayBuffer.size()).toBe(0);
    opponent.recordGameResult(twoMoveGame(), 'white-wins');
    expect(opponent.replayBuffer.size()).toBe(2);
  });

  it('recordGameResult with ongoing outcome throws', () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });

    expect(() => opponent.recordGameResult(twoMoveGame(), 'ongoing')).toThrow();
  });

  it('trainStep returns null when buffer is smaller than batchSize', async () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });

    opponent.recordGameResult(twoMoveGame(), 'white-wins');
    // Buffer has 2 examples; requesting batch of 32 should return null.
    const result = await opponent.trainStep(32);
    expect(result).toBeNull();
  });

  it('trainStep returns losses when buffer is large enough', async () => {
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(1000);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });

    // Fill buffer with enough examples.
    for (let i = 0; i < 5; i++) {
      opponent.recordGameResult(twoMoveGame(), i % 2 === 0 ? 'white-wins' : 'black-wins');
    }
    // Buffer has 10 examples; request batch of 4.
    const result = await opponent.trainStep(4);
    expect(result).not.toBeNull();
    expect(typeof result!.valueLoss).toBe('number');
    expect(typeof result!.policyLoss).toBe('number');
    expect(typeof result!.totalLoss).toBe('number');
  });
});

// ── loadOpponent ─────────────────────────────────────────────────────────────

describe('loadOpponent', () => {
  it('fresh storage → fresh opponent with empty replay buffer', async () => {
    const storage = createMemoryStorage();
    const net = ChessNet.create({ seed: 42 });
    const opponent = await loadOpponent(storage, {
      net,
      config: { simulations: 10, cPuct: 1.5 },
      replayCapacity: 100,
    });

    expect(opponent.replayBuffer.size()).toBe(0);
    expect(opponent.replayBuffer.capacity()).toBe(100);
    // Net should produce the same result as the supplied net.
    const state = initialState();
    const features = encodeState(state);
    const pred1 = net.predict(features);
    const pred2 = opponent.net.predict(features);
    expect(pred1.value).toBeCloseTo(pred2.value, 5);
  });

  it('saveTo then loadOpponent: loaded opponent has same NN predictions and replay buffer contents', async () => {
    const storage = createMemoryStorage();
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(100);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });

    // Add some training data.
    opponent.recordGameResult(twoMoveGame(), 'white-wins');
    opponent.recordGameResult(twoMoveGame(), 'draw');
    expect(opponent.replayBuffer.size()).toBe(4);

    // Persist.
    await opponent.saveTo(storage);

    // Restore.
    const net2 = ChessNet.create({ seed: 999 }); // different seed → different weights initially
    const loaded = await loadOpponent(storage, {
      net: net2,
      config: { simulations: 10, cPuct: 1.5 },
      replayCapacity: 100,
    });

    // NN predictions should match the saved net, not net2.
    const features = encodeState(initialState());
    const origPred = net.predict(features);
    const loadedPred = loaded.net.predict(features);
    expect(loadedPred.value).toBeCloseTo(origPred.value, 3);

    // Replay buffer should have same contents.
    const origSnap = opponent.replayBuffer.snapshot();
    const loadedSnap = loaded.replayBuffer.snapshot();
    expect(loadedSnap).toHaveLength(origSnap.length);
    for (let i = 0; i < origSnap.length; i++) {
      expect(float32Equal(loadedSnap[i].features, origSnap[i].features)).toBe(true);
      expect(loadedSnap[i].valueTarget).toBe(origSnap[i].valueTarget);
      expect((loadedSnap[i].policyTarget === null)).toBe((origSnap[i].policyTarget === null));
    }
  });

  it('NN_VERSION mismatch → ignores stored state and starts fresh', async () => {
    const storage = createMemoryStorage();
    const net = ChessNet.create({ seed: 42 });
    const buf = createReplayBuffer(100);
    const opponent = createOpponent({
      net,
      replayBuffer: buf,
      config: { simulations: 10, cPuct: 1.5 },
    });
    opponent.recordGameResult(twoMoveGame(), 'white-wins');
    await opponent.saveTo(storage);

    // Tamper with the stored NN_VERSION.
    const { putJson: putJ } = await import('../storage/index.js');
    await putJ(storage, OPPONENT_STORAGE_KEYS.META, { NN_VERSION: NN_VERSION + 999 });

    const net2 = ChessNet.create({ seed: 7 });
    const loaded = await loadOpponent(storage, {
      net: net2,
      config: { simulations: 10, cPuct: 1.5 },
      replayCapacity: 100,
    });

    // Should start with an empty replay buffer (default net2, not the saved one).
    expect(loaded.replayBuffer.size()).toBe(0);
  });

  it('missing WEIGHTS key → returns fresh opponent', async () => {
    const storage = createMemoryStorage();
    // Write meta with correct version but no weights.
    const { putJson: putJ } = await import('../storage/index.js');
    await putJ(storage, OPPONENT_STORAGE_KEYS.META, { NN_VERSION });

    const net = ChessNet.create({ seed: 42 });
    const loaded = await loadOpponent(storage, {
      net,
      config: { simulations: 10, cPuct: 1.5 },
      replayCapacity: 50,
    });

    expect(loaded.replayBuffer.size()).toBe(0);
  });
});
