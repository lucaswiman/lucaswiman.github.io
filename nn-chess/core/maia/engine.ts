// MaiaEngine — high-level wrapper around the Maia-2 forward pass.
//
// Exposes a `selectMove` method shaped like `Opponent.selectMove` so the
// UI can swap engines without rewriting the game loop. Maia is a frozen
// foreign engine; we do not train it, do not run MCTS over it, and do
// not feed its moves into either of our own networks' training data
// (see CLAUDE.md / README.md). The engine class is therefore stateless
// apart from the loaded weights.
//
// The engine handles two important details on top of `maiaForward`:
//   1. Masking illegal moves before the softmax / argmax — Maia's
//      policy head outputs logits over all 1880 trained moves, only a
//      subset of which is legal in any position.
//   2. Mirroring: Maia was trained on white-to-move positions only, so
//      the encoder mirrors the board for black-to-move positions and
//      this engine has to mirror the chosen UCI move back.
//
// LICENSE: MIT, derived from https://github.com/CSSLab/maia2.

import {
  encodeForMaia,
  eloToBucket,
  mirrorUci,
  MAIA_ELO_BUCKETS,
} from './encoding.js';
import { maiaForward } from './model.js';
import type { MaiaWeights } from './weights.js';
import { MAIA_MOVES, maiaPolicyIndex } from './action-space.js';
import type { GameState, Move } from '../rules/index.js';
import { legalMoves } from '../rules/index.js';

/** Configuration for one `selectMove` call. */
export interface MaiaSelectMoveOptions {
  /** Skill rating in [<1100, >=2000]; mapped to one of 11 buckets. */
  elo: number;
  /** Estimated opponent skill (defaults to `elo`). */
  oppoElo?: number;
  /**
   * `0` → greedy argmax over legal moves.
   * `> 0` → sample proportional to `softmax(logits / temperature)`
   *   restricted to legal moves.
   */
  temperature?: number;
  /** Optional RNG for temperature sampling. Defaults to `Math.random`. */
  rng?: () => number;
}

export interface MaiaSelectMoveResult {
  /** Chosen UCI move. Always legal in the input position. */
  move: Move;
  /** Probability mass Maia assigned to each legal move (sums to 1). */
  moveProbs: Map<Move, number>;
  /** Maia's value-head output; ≈ tanh-scaled expected outcome [-1, +1]. */
  rootValue: number;
}

export class MaiaEngine {
  private readonly weights: MaiaWeights;

  constructor(weights: MaiaWeights) {
    this.weights = weights;
  }

  /** Read-only view of the buckets the user can pick from. */
  static readonly ELO_BUCKETS = MAIA_ELO_BUCKETS;

  /**
   * Pick a move from the given position at the requested skill level.
   * Throws if the position has no legal moves (caller should check
   * `isTerminal` first).
   */
  selectMove(state: GameState, options: MaiaSelectMoveOptions): MaiaSelectMoveResult {
    const legal = legalMoves(state);
    if (legal.length === 0) {
      throw new Error('MaiaEngine.selectMove called on a terminal position');
    }

    const { features, mirrored } = encodeForMaia(state);
    const eloSelf = eloToBucket(options.elo);
    const eloOppo = eloToBucket(options.oppoElo ?? options.elo);

    const { policyLogits, value } = maiaForward(this.weights, features, {
      eloSelf,
      eloOppo,
    });

    // Map every legal move to its Maia policy index. Underpromotions to
    // queen/knight/etc. are all distinct entries in the action space, so
    // we don't need to fold them.
    //
    // If the board was mirrored before encoding, the indices live in
    // the mirrored move's slot, not the original move's slot.
    const mirroredLegal = mirrored ? legal.map(m => mirrorUci(m)) : legal.slice();

    // Build a logits-per-legal-move array, then softmax-normalize.
    const logits = new Float32Array(legal.length);
    for (let i = 0; i < legal.length; i++) {
      const idx = maiaPolicyIndex(mirroredLegal[i]);
      logits[i] = idx >= 0 ? policyLogits[idx] : -Infinity;
    }

    // Numerically-stable softmax over the (possibly tiny) legal subset.
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    if (!Number.isFinite(max)) {
      // Pathological: none of the legal moves were in Maia's action
      // space. Shouldn't happen on standard chess; fall back to uniform.
      max = 0;
      for (let i = 0; i < logits.length; i++) logits[i] = 0;
    }
    const probs = new Float32Array(legal.length);
    let sum = 0;
    for (let i = 0; i < legal.length; i++) {
      const e = Math.exp(logits[i] - max);
      probs[i] = e;
      sum += e;
    }
    if (sum > 0) {
      for (let i = 0; i < probs.length; i++) probs[i] /= sum;
    } else {
      // Numeric underflow → uniform.
      const u = 1 / probs.length;
      for (let i = 0; i < probs.length; i++) probs[i] = u;
    }

    const moveProbs = new Map<Move, number>();
    for (let i = 0; i < legal.length; i++) moveProbs.set(legal[i], probs[i]);

    // Sample or argmax.
    const temperature = options.temperature ?? 0;
    let chosenIdx = 0;
    if (temperature === 0) {
      // Greedy: pick highest prob, break ties by lexicographic UCI
      // (same convention as Opponent.selectMove).
      let bestP = -Infinity;
      let best: Move = legal[0];
      for (let i = 0; i < legal.length; i++) {
        const p = probs[i];
        if (p > bestP || (p === bestP && legal[i] < best)) {
          bestP = p;
          best = legal[i];
          chosenIdx = i;
        }
      }
      return { move: best, moveProbs, rootValue: value };
    }

    // Temperature sampling.
    const rng = options.rng ?? Math.random;
    if (temperature !== 1) {
      // Re-softmax with the requested temperature.
      let s = 0;
      for (let i = 0; i < probs.length; i++) {
        probs[i] = Math.exp((logits[i] - max) / temperature);
        s += probs[i];
      }
      if (s > 0) {
        for (let i = 0; i < probs.length; i++) probs[i] /= s;
      }
    }
    const draw = rng();
    let cum = 0;
    for (let i = 0; i < legal.length; i++) {
      cum += probs[i];
      if (draw < cum) {
        chosenIdx = i;
        break;
      }
      chosenIdx = i;
    }
    const chosen = legal[chosenIdx];
    return { move: chosen, moveProbs, rootValue: value };
  }

  /** Tear down the weight tensors. The engine becomes unusable. */
  dispose(): void {
    this.weights.dispose();
  }
}

// Re-export for convenience so consumers can `import { MAIA_MOVES } from .../engine`.
export { MAIA_MOVES };
