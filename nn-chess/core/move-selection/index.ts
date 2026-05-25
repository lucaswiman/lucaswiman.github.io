// Generic move selection over a ChessNet. Two modes:
//   - MCTS (when `options.search` is set): runs the project's `search()`
//     with the supplied net as the policy/value source. The mate-in-1
//     override baked into `search` applies — that's the project's one
//     approved heuristic ("checkmate is terminal"), see `core/mcts`.
//   - Policy-head (when `options.search` is absent): masks illegal
//     moves and argmaxes (or P^(1/T)-samples) the legal subset of the
//     net's policy head directly. No value-based lookahead.
//
// This function is used in three places:
//   - PlayerImitator.selectMove — drives play-vs-imitator and is the
//     same selection function `core/selfplay` uses for the imitator.
//   - Game.tsx — the opponent network's no-MCTS branch when the
//     "Use MCTS" toggle is off; the MCTS-on branch still goes through
//     `Opponent.selectMove` for its richer return shape (visit counts
//     etc. needed by the inspector panel).
//   - Tests.
//
// History: this lived in `core/imitator/index.ts` and was named
// `selectImitatorMove`. It was always generic over any ChessNet; the
// name was misleading once `Game.tsx` started calling it for the
// opponent net. Moved here in the "MCTS toggle" cleanup.

import { policyFromNet } from '../opponent/index.js';
import type { ChessNet } from '../nn/index.js';
import type { GameState, Move } from '../rules/index.js';
import { isTerminal, legalMoves } from '../rules/index.js';
import { search, type SearchConfig } from '../mcts/index.js';

export interface MoveSelectionOptions {
  /**
   * When provided, drive selection with MCTS over the net. When absent,
   * pick straight from the policy head.
   */
  search?: SearchConfig;
  /**
   * Sampling temperature. With `search` set this applies to visit
   * counts (T=0 picks argmax; T>0 samples ∝ N^(1/T)). Without `search`
   * it applies to the raw priors (T=0 greedy; T>0 samples ∝ P^(1/T)).
   * Defaults to 0.
   */
  temperature?: number;
  rng?: () => number;
}

export interface MoveSelectionResult {
  move: Move;
  /** Legal-move priors out of the policy head. */
  priors: Map<Move, number>;
  /** Set only when `options.search` was provided. */
  search?: {
    rootValue: number;
    visitCounts: Map<Move, number>;
    edgeValues: Map<Move, number>;
    mateInOneMoves: Move[];
  };
}

export async function selectMoveFromNet(
  net: ChessNet,
  state: GameState,
  options: MoveSelectionOptions = {},
): Promise<MoveSelectionResult> {
  if (isTerminal(state)) {
    throw new Error('selectMoveFromNet: called on a terminal GameState');
  }
  const moves = legalMoves(state);
  if (moves.length === 0) {
    throw new Error('selectMoveFromNet: no legal moves at non-terminal state');
  }

  const policy = policyFromNet(net);
  const temperature = options.temperature ?? 0;

  if (options.search !== undefined) {
    const result = await search(state, policy, {
      ...options.search,
      rng: options.search.rng ?? options.rng,
    });
    const { rootValue, visitCounts, priorPolicy, edgeValues, mateInOneMoves } = result;
    const move =
      mateInOneMoves.length > 0
        ? mateInOneMoves[0]
        : pickFromVisits(visitCounts, priorPolicy, temperature, options.rng);
    return {
      move,
      priors: priorPolicy,
      search: { rootValue, visitCounts, edgeValues, mateInOneMoves },
    };
  }

  const { priors } = await policy.evaluate(state);

  if (temperature === 0 || moves.length === 1) {
    let bestMove = moves[0];
    let bestP = priors.get(bestMove) ?? 0;
    for (const m of moves) {
      const p = priors.get(m) ?? 0;
      if (p > bestP || (p === bestP && m < bestMove)) {
        bestP = p;
        bestMove = m;
      }
    }
    return { move: bestMove, priors };
  }

  const invT = 1 / temperature;
  const weights = moves.map(m => Math.pow(priors.get(m) ?? 0, invT));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) {
    const idx = Math.floor((options.rng ?? Math.random)() * moves.length);
    return { move: moves[Math.min(idx, moves.length - 1)], priors };
  }
  const draw = (options.rng ?? Math.random)() * total;
  let cumulative = 0;
  for (let i = 0; i < moves.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return { move: moves[i], priors };
  }
  return { move: moves[moves.length - 1], priors };
}

function pickFromVisits(
  visitCounts: Map<Move, number>,
  priors: Map<Move, number>,
  temperature: number,
  rng: (() => number) | undefined,
): Move {
  const entries = Array.from(visitCounts.entries());
  if (entries.length === 0) {
    throw new Error('pickFromVisits: empty visit-count map (root had no legal moves?)');
  }
  if (temperature === 0 || entries.length === 1) {
    // Argmax of visits; ties broken by prior, then lex by UCI for determinism.
    let bestMove = entries[0][0];
    let bestN = entries[0][1];
    let bestP = priors.get(bestMove) ?? 0;
    for (let i = 1; i < entries.length; i++) {
      const [m, n] = entries[i];
      const p = priors.get(m) ?? 0;
      if (n > bestN || (n === bestN && p > bestP) || (n === bestN && p === bestP && m < bestMove)) {
        bestMove = m;
        bestN = n;
        bestP = p;
      }
    }
    return bestMove;
  }
  const invT = 1 / temperature;
  const weights = entries.map(([, n]) => Math.pow(n, invT));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) {
    const idx = Math.floor((rng ?? Math.random)() * entries.length);
    return entries[Math.min(idx, entries.length - 1)][0];
  }
  const draw = (rng ?? Math.random)() * total;
  let cumulative = 0;
  for (let i = 0; i < entries.length; i++) {
    cumulative += weights[i];
    if (draw < cumulative) return entries[i][0];
  }
  return entries[entries.length - 1][0];
}
