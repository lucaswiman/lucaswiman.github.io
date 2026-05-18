// Agent — glues ChessNet + MCTS + ReplayBuffer into the single object
// the UI talks to.
//
// Responsibilities:
//   - Adapt ChessNet to MCTS's Policy interface (mask illegal moves,
//     renormalize, pass value through).
//   - Run MCTS with AgentConfig, then pick a move from the visit-count
//     distribution (temperature=0 → greedy; temperature>0 → sample).
//   - Convert finished games into TrainingExamples via the
//     winner-reinforcement rule and push them into the ReplayBuffer.
//   - Sample a minibatch from the buffer and drive one gradient step.
//   - Serialize / restore the whole agent (weights + buffer + meta)
//     through the BlobStorage port.
//
// What this module does NOT do:
//   - Touch the DOM, window, localStorage, React, or Astro.
//   - Implement any chess heuristic beyond the rules.
//   - Generate self-play data (data comes only from real games).

import { encodeState } from '../encoding/index.js';
import type { GameState, Move, Outcome } from '../rules/index.js';
import { legalMoves, isTerminal } from '../rules/index.js';
import {
  search,
  type Policy,
  type SearchConfig,
} from '../mcts/index.js';
import {
  ChessNet,
  deserialize,
  policyIndexFromUci,
  POLICY_SIZE,
  NN_VERSION,
  type TrainingExample,
} from '../nn/index.js';
import {
  type BlobStorage,
  putJson,
  getJson,
} from '../storage/index.js';
import {
  createReplayBuffer,
  deserializeReplayBuffer,
  type ReplayBuffer,
} from '../training/index.js';

// ── Public types ─────────────────────────────────────────────────────────────

/**
 * One position from one game, captured before the move was played.
 * Accumulate these during a game, then pass them all to
 * `recordGameResult` once the outcome is known.
 */
export interface GameMoveRecord {
  /** Encoded state BEFORE the move was played. */
  features: Float32Array;
  /** UCI move that was actually played from that state. */
  move: Move;
  /** Whose move it was. */
  sideToMove: 'w' | 'b';
}

export interface AgentConfig {
  /** MCTS simulations per selectMove call. */
  simulations: number;
  /** PUCT exploration constant. */
  cPuct: number;
  dirichletAlpha?: number;
  dirichletEpsilon?: number;
  /**
   * Optional deterministic rng for tests. Both MCTS and temperature
   * sampling use this rng so tests are fully reproducible.
   */
  rng?: () => number;
  /**
   * Temperature for move selection from visit counts.
   *   > 0  sample proportional to N^(1/T)
   *   = 0  greedy argmax (ties broken by prior, then by UCI string)
   * Defaults to 0 (greedy).
   */
  temperature?: number;
}

export interface Agent {
  /** Pick a move for the agent to play from `state`. */
  selectMove(
    state: GameState,
  ): Promise<{ move: Move; rootValue: number; visitCounts: Map<Move, number> }>;

  /**
   * Record a complete game. Converts move records into TrainingExamples
   * and pushes them into the replay buffer per the winner-reinforcement
   * rule (see README). Does NO NN work.
   */
  recordGameResult(moves: GameMoveRecord[], finalOutcome: Outcome): void;

  /** Persist current NN weights + replay buffer to storage. */
  saveTo(storage: BlobStorage): Promise<void>;

  /**
   * Sample one minibatch from the buffer and run one gradient step.
   * Returns the losses, or null if the buffer has fewer examples than
   * `batchSize`.
   */
  trainStep(
    batchSize: number,
  ): Promise<{ valueLoss: number; policyLoss: number; totalLoss: number } | null>;

  /** Read-only handles for the UI / interpretability viewer. */
  readonly net: ChessNet;
  readonly replayBuffer: ReplayBuffer;
}

export interface CreateAgentOptions {
  net: ChessNet;
  replayBuffer: ReplayBuffer;
  config: AgentConfig;
}

/** Storage keys used by loadAgent / saveTo. */
export const AGENT_STORAGE_KEYS = {
  WEIGHTS: 'agent/weights',
  REPLAY: 'agent/replay',
  META: 'agent/meta',
} as const;

// ── policyFromNet ────────────────────────────────────────────────────────────

/**
 * Adapt a ChessNet to MCTS's Policy interface.
 *
 * The NN returns raw logits over all 4096 from→to pairs, most of which
 * are illegal in any given position. We:
 *   1. Run softmax over the raw logits.
 *   2. Zero out all illegal moves.
 *   3. Renormalize so the surviving entries sum to 1.
 *   4. Fall back to a uniform prior if all legal moves had zero mass.
 *
 * For terminal positions the caller should not call this, but if it
 * does we return empty priors and value 0 (safe default).
 *
 * Exported for independent testing and for the future Stockfish-as-
 * policy harness.
 */
export function policyFromNet(net: ChessNet): Policy {
  return {
    async evaluate(
      state: GameState,
    ): Promise<{ priors: Map<Move, number>; value: number }> {
      const moves = legalMoves(state);

      // Safe default for terminal states (caller shouldn't reach here).
      if (moves.length === 0 || isTerminal(state)) {
        return { priors: new Map(), value: 0 };
      }

      const features = encodeState(state);
      const { value, policy: logits } = net.predict(features);

      // Softmax over the full logit vector.
      const probs = softmax(logits);

      // Mask illegal moves.
      const priorMap = new Map<Move, number>();
      let sum = 0;
      for (const move of moves) {
        const idx = policyIndexFromUci(move);
        const p = probs[idx];
        if (p > 0) {
          priorMap.set(move, p);
          sum += p;
        }
      }

      // Renormalize or fall back to uniform.
      let finalPriors: Map<Move, number>;
      if (sum === 0) {
        const u = 1 / moves.length;
        finalPriors = new Map(moves.map(m => [m, u]));
      } else {
        finalPriors = new Map(
          Array.from(priorMap.entries()).map(([m, p]) => [m, p / sum]),
        );
        // Legal moves not in priorMap get 0 (MCTS handles 0-prior edges).
        for (const m of moves) {
          if (!finalPriors.has(m)) finalPriors.set(m, 0);
        }
      }

      return { priors: finalPriors, value };
    },
  };
}

// ── buildTrainingExamples ────────────────────────────────────────────────────

/**
 * Convert a finished game's move records into TrainingExamples per the
 * winner-reinforcement rule documented in nn-chess/README.md.
 *
 * Value target (from the side-to-move's perspective at each position):
 *   white wins → white-to-move positions get +1, black-to-move get -1
 *   black wins → black-to-move positions get +1, white-to-move get -1
 *   draw       → all positions get 0
 *
 * Policy target (one-hot over POLICY_SIZE for the move played):
 *   Only produced for positions where the player who moved went on to
 *   WIN the game. Losing-side positions and all draw positions get
 *   policyTarget = null (policy loss skipped).
 *
 * Throws if `finalOutcome` is 'ongoing' (caller error — can't build
 * training examples from an unfinished game).
 *
 * Exported for unit testing in isolation.
 */
export function buildTrainingExamples(
  moves: GameMoveRecord[],
  finalOutcome: Outcome,
): TrainingExample[] {
  if (finalOutcome === 'ongoing') {
    throw new Error(
      'buildTrainingExamples: finalOutcome is "ongoing" — game is not finished',
    );
  }

  return moves.map(record => {
    // Value target from side-to-move's perspective.
    let valueTarget: number;
    if (finalOutcome === 'draw') {
      valueTarget = 0;
    } else if (finalOutcome === 'white-wins') {
      valueTarget = record.sideToMove === 'w' ? 1 : -1;
    } else {
      // black-wins
      valueTarget = record.sideToMove === 'b' ? 1 : -1;
    }

    // Policy target: one-hot only for the winning side's positions.
    // Draws → null on both sides (skip policy loss).
    let policyTarget: Float32Array | null = null;
    const isWinner =
      finalOutcome !== 'draw' &&
      ((finalOutcome === 'white-wins' && record.sideToMove === 'w') ||
        (finalOutcome === 'black-wins' && record.sideToMove === 'b'));

    if (isWinner) {
      policyTarget = new Float32Array(POLICY_SIZE);
      const idx = policyIndexFromUci(record.move);
      policyTarget[idx] = 1;
    }

    return { features: record.features, valueTarget, policyTarget };
  });
}

// ── createAgent ──────────────────────────────────────────────────────────────

export function createAgent(opts: CreateAgentOptions): Agent {
  const { net, replayBuffer, config } = opts;

  const selectMove = async (
    state: GameState,
  ): Promise<{ move: Move; rootValue: number; visitCounts: Map<Move, number> }> => {
    const policy = policyFromNet(net);

    const searchConfig: SearchConfig = {
      simulations: config.simulations,
      cPuct: config.cPuct,
      dirichletAlpha: config.dirichletAlpha,
      dirichletEpsilon: config.dirichletEpsilon,
      rng: config.rng,
    };

    const result = await search(state, policy, searchConfig);
    const { rootValue, visitCounts } = result;

    const temperature = config.temperature ?? 0;
    const move = pickMove(visitCounts, result.priorPolicy, temperature, config.rng);

    return { move, rootValue, visitCounts };
  };

  const recordGameResult = (
    gameMoves: GameMoveRecord[],
    finalOutcome: Outcome,
  ): void => {
    // buildTrainingExamples throws on 'ongoing' — let it propagate.
    const examples = buildTrainingExamples(gameMoves, finalOutcome);
    replayBuffer.addAll(examples);
  };

  const saveTo = async (storage: BlobStorage): Promise<void> => {
    const meta = { NN_VERSION };
    await Promise.all([
      net.serialize().then(bytes => storage.put(AGENT_STORAGE_KEYS.WEIGHTS, bytes)),
      storage.put(AGENT_STORAGE_KEYS.REPLAY, replayBuffer.serialize()),
      putJson(storage, AGENT_STORAGE_KEYS.META, meta),
    ]);
  };

  const trainStep = async (
    batchSize: number,
  ): Promise<{ valueLoss: number; policyLoss: number; totalLoss: number } | null> => {
    if (replayBuffer.size() < batchSize) return null;
    const batch = replayBuffer.sample(batchSize, config.rng);
    return net.trainBatch(batch);
  };

  return {
    selectMove,
    recordGameResult,
    saveTo,
    trainStep,
    get net() { return net; },
    get replayBuffer() { return replayBuffer; },
  };
}

// ── loadAgent ────────────────────────────────────────────────────────────────

/**
 * Restore weights + replay buffer from storage if present and compatible;
 * otherwise return a fresh agent built from `defaults`.
 *
 * Version mismatch (stored NN_VERSION differs from current NN_VERSION) is
 * treated as a cache miss — the stored state is ignored and a fresh agent
 * is returned. This ensures that an architecture or encoding change doesn't
 * silently corrupt training.
 */
export async function loadAgent(
  storage: BlobStorage,
  defaults: {
    net: ChessNet;
    config: AgentConfig;
    replayCapacity: number;
  },
): Promise<Agent> {
  const fresh = () =>
    createAgent({
      net: defaults.net,
      replayBuffer: createReplayBuffer(defaults.replayCapacity),
      config: defaults.config,
    });

  // Read meta first — check version before loading weights.
  const meta = await getJson<{ NN_VERSION: number }>(
    storage,
    AGENT_STORAGE_KEYS.META,
  );

  if (meta === null || meta.NN_VERSION !== NN_VERSION) {
    // Nothing stored yet, or stored weights are from a different architecture.
    return fresh();
  }

  // Try to load weights and replay buffer.
  const [weightBytes, replayBytes] = await Promise.all([
    storage.get(AGENT_STORAGE_KEYS.WEIGHTS),
    storage.get(AGENT_STORAGE_KEYS.REPLAY),
  ]);

  if (weightBytes === null) {
    return fresh();
  }

  let net: ChessNet;
  try {
    net = await deserialize(weightBytes);
  } catch {
    // Corrupt weights — start fresh.
    return fresh();
  }

  const replayBuffer =
    replayBytes !== null
      ? deserializeReplayBuffer(replayBytes, defaults.replayCapacity)
      : createReplayBuffer(defaults.replayCapacity);

  return createAgent({ net, replayBuffer, config: defaults.config });
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Numerically-stable softmax over a Float32Array.
 * Returns a new Float32Array of the same length with values summing to 1.
 */
export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  if (sum > 0) {
    for (let i = 0; i < out.length; i++) out[i] /= sum;
  }
  return out;
}

/**
 * Pick a move from the MCTS visit counts.
 *
 * temperature = 0: greedy argmax; ties broken by prior, then by UCI string
 *                  (stable, deterministic, matches MCTS's own bestMove
 *                   selection — we intentionally re-implement here so the
 *                   temperature=0 path uses no rng at all).
 * temperature > 0: sample proportional to N^(1/T) using the provided rng
 *                  (or Math.random if none is given).
 */
function pickMove(
  visitCounts: Map<Move, number>,
  priorPolicy: Map<Move, number>,
  temperature: number,
  rng: (() => number) | undefined,
): Move {
  const moves = Array.from(visitCounts.keys());

  if (temperature === 0 || moves.length === 1) {
    // Greedy with tie-breaking: most visits > highest prior > lexicographic.
    let bestMove = moves[0];
    let bestN = visitCounts.get(bestMove) ?? 0;
    let bestP = priorPolicy.get(bestMove) ?? 0;
    for (const m of moves) {
      const n = visitCounts.get(m) ?? 0;
      const p = priorPolicy.get(m) ?? 0;
      if (
        n > bestN ||
        (n === bestN && p > bestP) ||
        (n === bestN && p === bestP && m < bestMove)
      ) {
        bestN = n;
        bestP = p;
        bestMove = m;
      }
    }
    return bestMove;
  }

  // Temperature sampling: weight ∝ N^(1/T).
  const invT = 1 / temperature;
  const weights = moves.map(m => Math.pow(visitCounts.get(m) ?? 0, invT));
  const totalWeight = weights.reduce((a, b) => a + b, 0);

  if (totalWeight === 0) {
    // All visit counts are 0 (shouldn't happen after MCTS, but be safe).
    return moves[0];
  }

  const draw = (rng ?? Math.random)() * totalWeight;
  let cumulative = 0;
  for (let i = 0; i < moves.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return moves[i];
  }
  // Floating-point rounding fallback.
  return moves[moves.length - 1];
}
