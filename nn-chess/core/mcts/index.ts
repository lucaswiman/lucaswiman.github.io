// PUCT-style Monte Carlo Tree Search
//
// This module implements AlphaZero-style MCTS. It knows nothing about chess
// beyond what `GameState`, `Move`, and the rules helpers tell it. All chess
// knowledge (piece values, tactics, etc.) must come through the `Policy`
// interface — the NN (or any other evaluator) plugs in there.
//
// References:
//   Silver et al. "Mastering Chess and Shogi by Self-Play…" (AlphaZero)
//   Silver et al. "Mastering the game of Go without human knowledge" (AlphaGo Zero)

import type { GameState, Move } from '../rules/index.js';
import {
  isTerminal,
  legalMoves,
  applyMove,
  terminalValueForMover,
  repetitionKey,
} from '../rules/index.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * What MCTS asks of the world for each position it expands.
 * Implementations adapt the NN (or any other prior+value source) to this.
 */
export interface Policy {
  /**
   * Evaluate a non-terminal position.
   * - `priors`: a probability distribution over the legal moves in `state`
   *   (sums to ~1 after the caller has masked illegals and renormalized).
   * - `value`: predicted outcome from `state`'s side-to-move's perspective,
   *   in [-1, 1].
   */
  evaluate(state: GameState): Promise<{ priors: Map<Move, number>; value: number }>;
}

export interface SearchConfig {
  simulations: number;       // number of MCTS playouts per search
  cPuct: number;             // PUCT exploration constant (e.g. 1.5)
  dirichletAlpha?: number;   // optional root-noise alpha (default: none)
  dirichletEpsilon?: number; // weight of dirichlet noise at the root
  rng?: () => number;        // deterministic rng for tests; default Math.random
  // Plies of forced-line lookahead to run at each leaf expansion. When >0,
  // before falling back to the NN value head MCTS asks "is this position a
  // forced ±1/0 outcome within K plies under best play by both sides?"
  // If so, that certain value replaces the NN's noisy estimate. This is
  // pure rules-only propagation of "checkmate is terminal" — no chess
  // heuristic beyond the existing terminal handling.
  //
  // Defaults to 1 (mate-in-1 detection at leaves). Combined with the
  // mate-in-1 override in `selectEdge`, this is enough for the opponent
  // to actively seize every available checkmate while keeping per-leaf
  // cost bounded. Crank up to 2 to also detect mate-in-2 at leaves —
  // useful for pedagogy demos where you want a single visit to the
  // first move of a mating sequence to lock in immediately.
  mateSearchDepth?: number;
}

export interface SearchResult {
  bestMove: Move;                  // argmax visit count (ties broken by prior, then deterministically)
  rootValue: number;               // value backed up to the root after search
  visitCounts: Map<Move, number>;
  priorPolicy: Map<Move, number>;  // priors at the root (for inspection / training)
  // Q(root, a) for every root edge, in [-1, 1] from the root's side-to-move
  // perspective. Edges that were never visited (N=0) have Q=0. Useful for
  // inspection / pedagogical visualisations of the PUCT exploit term.
  edgeValues: Map<Move, number>;
  // Set when at least one legal move from the root state delivers immediate
  // checkmate. `bestMove` is guaranteed to be one of these moves, and any
  // move-picker downstream of search() should treat this as mandatory: a
  // known mate-in-1 is always strictly better than any uncertain
  // alternative, so we never want to leave it on the board.
  mateInOneMoves: Move[];
}

// ---------------------------------------------------------------------------
// Internal tree representation
// ---------------------------------------------------------------------------

// One edge in the tree = one move from a parent node.
// Stores the per-edge PUCT statistics.
interface Edge {
  move: Move;
  // P(s, a) — prior probability from the policy network
  P: number;
  // N(s, a) — visit count
  N: number;
  // W(s, a) — total value backed up through this edge
  W: number;
  // Cached child state (avoids re-computing applyMove on revisits)
  childState: GameState | null;
  // Pointer to the child node once it has been expanded
  childNode: Node | null;
  // True iff applying `move` ends the game with the mover winning
  // (i.e. delivers checkmate). Cached lazily on first inspection in
  // selectEdge so we only pay one applyMove per edge per tree.
  deliversCheckmate: boolean | null;
}

// One node in the tree = one game state.
interface Node {
  state: GameState;
  edges: Edge[];          // one per legal move; empty for terminal nodes
  isExpanded: boolean;    // true once policy.evaluate has been called here
  isTerminal: boolean;
}

// ---------------------------------------------------------------------------
// Dirichlet noise (Marsaglia/Tsang Gamma sampler)
// ---------------------------------------------------------------------------

// Why Gamma? Dirichlet(alpha) can be sampled as
//   x_i ~ Gamma(alpha, 1),  then normalize: d_i = x_i / sum(x).
// The Marsaglia-Tsang method is the standard fast Gamma sampler.
function sampleGamma(alpha: number, rng: () => number): number {
  if (alpha >= 1) {
    // Marsaglia & Tsang (2000), Algorithm 1
    const d = alpha - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = sampleNormal(rng);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = rng();
      const x2 = x * x;
      if (u < 1 - 0.0331 * x2 * x2) return d * v;
      if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  // alpha < 1: use the x^(1/alpha) trick
  return sampleGamma(alpha + 1, rng) * Math.pow(rng(), 1 / alpha);
}

// Box-Muller for standard normal samples (needed by Marsaglia-Tsang)
function sampleNormal(rng: () => number): number {
  // Simple Box-Muller; we discard the second variate for simplicity.
  const u1 = Math.max(rng(), 1e-15); // guard against log(0)
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Sample a Dirichlet(alpha) vector of length `n`.
function sampleDirichlet(n: number, alpha: number, rng: () => number): number[] {
  const gammas = Array.from({ length: n }, () => sampleGamma(alpha, rng));
  const sum = gammas.reduce((a, b) => a + b, 0);
  return sum === 0 ? Array(n).fill(1 / n) : gammas.map(g => g / sum);
}

// ---------------------------------------------------------------------------
// Tree helpers
// ---------------------------------------------------------------------------

function makeNode(state: GameState): Node {
  return {
    state,
    edges: [],
    isExpanded: false,
    isTerminal: isTerminal(state),
  };
}

// Create edges for every legal move; child states are computed lazily.
function expandEdges(node: Node, priors: Map<Move, number>): void {
  const moves = legalMoves(node.state);
  node.edges = moves.map(move => ({
    move,
    P: priors.get(move) ?? 0,
    N: 0,
    W: 0,
    childState: null,
    childNode: null,
    deliversCheckmate: null,
  }));
  node.isExpanded = true;
}

// Is the child state reached by this edge a terminal position where the
// mover (the side that just moved through this edge) won? Computes and
// caches the answer the first time it's asked.
function edgeDeliversCheckmate(edge: Edge, parentState: GameState): boolean {
  if (edge.deliversCheckmate !== null) return edge.deliversCheckmate;
  const child = childState(edge, parentState);
  const result = isTerminal(child) && terminalValueForMover(child) === 1;
  edge.deliversCheckmate = result;
  return result;
}

// PUCT score for one edge.
// Q(s,a) + cPuct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
//
// Why this formula? The second term shrinks as N(s,a) grows, pushing
// exploration toward high-prior, unvisited edges. As N grows large the
// Q term dominates and we exploit the best-found line.
function puctScore(edge: Edge, sqrtParentN: number, cPuct: number): number {
  const Q = edge.N === 0 ? 0 : edge.W / edge.N;
  return Q + cPuct * edge.P * sqrtParentN / (1 + edge.N);
}

function selectEdge(node: Node, cPuct: number): Edge {
  // Forced-line override 1 — only one legal move. There's nothing to
  // explore, so don't waste a PUCT computation: take the forced reply
  // and let the search continue into the resulting position.
  if (node.edges.length === 1) return node.edges[0];

  // Forced-line override 2 — a move that delivers immediate checkmate.
  // Checkmate is always strictly better than any uncertain alternative,
  // so we pull it out of the PUCT competition entirely. This is the
  // extension of "checkmate is terminal" the project's MCTS now relies
  // on so it actively seizes mates instead of bumbling into them when
  // sims happen to be lucky enough.
  for (const edge of node.edges) {
    if (edgeDeliversCheckmate(edge, node.state)) return edge;
  }

  const sqrtN = Math.sqrt(node.edges.reduce((s, e) => s + e.N, 0));
  let best = node.edges[0];
  let bestScore = -Infinity;
  for (const edge of node.edges) {
    const score = puctScore(edge, sqrtN, cPuct);
    if (score > bestScore) {
      bestScore = score;
      best = edge;
    }
  }
  return best;
}

// Get (or compute and cache) the child GameState for an edge.
function childState(edge: Edge, parentState: GameState): GameState {
  if (edge.childState === null) {
    edge.childState = applyMove(parentState, edge.move);
  }
  return edge.childState;
}

// ---------------------------------------------------------------------------
// Forced-mate lookahead
// ---------------------------------------------------------------------------

// Returns the value of `state` from its side-to-move's perspective, if every
// line within `depth` plies resolves to a terminal outcome under optimal
// play. Returns null when at least one continuation hits the depth horizon
// without terminating — i.e. the result is uncertain at this depth and the
// caller should fall back to the value-head estimate.
//
// This is the depth-K extension of MCTS's existing "checkmate is terminal"
// rule: rather than waiting for PUCT selection to randomly stumble into a
// terminal leaf, every expansion gets a small bounded minimax over the
// terminal outcomes. No piece values, no king safety, no chess heuristic
// beyond "is this state checkmated / stalemated / drawn under the rules?"
function forcedTerminalValue(state: GameState, depth: number): number | null {
  if (isTerminal(state)) {
    // terminalValueForMover is from the just-moved side's POV; side-to-
    // move at a terminal is the opponent of the mover, so flip the sign.
    return -terminalValueForMover(state);
  }
  if (depth <= 0) return null;

  const moves = legalMoves(state);
  // A non-terminal node has at least one legal move (no-legal-moves would
  // be checkmate or stalemate, both caught by isTerminal above).
  let best: number = -Infinity;
  let sawHorizon = false;
  for (const m of moves) {
    const v = forcedTerminalValue(applyMove(state, m), depth - 1);
    if (v === null) {
      sawHorizon = true;
      // Don't bail yet: we might still find a forced-win move that
      // dominates the horizon-uncertain branches.
      continue;
    }
    // v is from the child's side-to-move (the opponent). Flip for ours.
    const our = -v;
    if (our > best) best = our;
    if (our === 1) return 1;  // found a forced-win move — game over here.
  }
  // No forced win. If any branch ran off the depth horizon, we can't
  // claim a forced loss either — that branch might escape with deeper
  // search. Fall back to the value head.
  if (sawHorizon) return null;
  // Every branch resolved. `best` is the strongest guarantee we have:
  // -1 means every move loses (forced loss); 0 means our best is a
  // forced draw.
  return best === -Infinity ? null : best;
}

// ---------------------------------------------------------------------------
// Main search
// ---------------------------------------------------------------------------

export async function search(
  state: GameState,
  policy: Policy,
  config: SearchConfig,
): Promise<SearchResult> {
  if (isTerminal(state)) {
    throw new Error('search() called on a terminal GameState');
  }

  const { simulations, cPuct, dirichletAlpha, dirichletEpsilon } = config;
  const rng = config.rng ?? Math.random.bind(Math);
  const mateSearchDepth = config.mateSearchDepth ?? 1;

  // Cache: repetitionKey → { priors, value } so transpositions reuse the
  // same evaluation. (One search call only; we don't persist across calls.)
  const evalCache = new Map<string, { priors: Map<Move, number>; value: number }>();

  // Memoised policy.evaluate that masks to legal moves and renormalizes.
  async function evaluate(s: GameState): Promise<{ priors: Map<Move, number>; value: number }> {
    const key = repetitionKey(s);
    const cached = evalCache.get(key);
    if (cached !== undefined) return cached;

    const raw = await policy.evaluate(s);
    const legal = new Set(legalMoves(s));

    // Mask illegal moves and renormalize.
    // Why: the NN may assign probability to moves that are illegal in this
    // position (e.g. it never saw this exact state); we must drop those
    // and redistribute probability so priors still sum to 1.
    let sum = 0;
    const masked = new Map<Move, number>();
    for (const [m, p] of raw.priors) {
      if (legal.has(m) && p > 0) {
        masked.set(m, p);
        sum += p;
      }
    }

    // If the policy assigned zero mass to all legal moves (e.g. uniform
    // over illegal moves only), fall back to a uniform prior.
    let finalPriors: Map<Move, number>;
    if (sum === 0) {
      const uniform = 1 / legal.size;
      finalPriors = new Map(Array.from(legal).map(m => [m, uniform]));
    } else {
      finalPriors = new Map(Array.from(masked).map(([m, p]) => [m, p / sum]));
      // Fill in legal moves not covered by the policy with 0
      for (const m of legal) {
        if (!finalPriors.has(m)) finalPriors.set(m, 0);
      }
    }

    const result = { priors: finalPriors, value: raw.value };
    evalCache.set(key, result);
    return result;
  }

  // Build the root node.
  const root = makeNode(state);
  const rootEval = await evaluate(state);
  expandEdges(root, rootEval.priors);

  // Optionally mix Dirichlet noise into the root priors.
  // Why at the root only? We want exploration at the point we're deciding
  // which actual move to play; we don't want noise corrupting every
  // internal node (that would make the tree inconsistent).
  if (dirichletAlpha !== undefined && dirichletEpsilon !== undefined && root.edges.length > 0) {
    const noise = sampleDirichlet(root.edges.length, dirichletAlpha, rng);
    const eps = dirichletEpsilon;
    for (let i = 0; i < root.edges.length; i++) {
      root.edges[i].P = (1 - eps) * root.edges[i].P + eps * noise[i];
    }
  }

  // Run `simulations` MCTS playouts.
  for (let sim = 0; sim < simulations; sim++) {
    // --- Selection: walk from root to a leaf following PUCT ---
    const path: Array<{ node: Node; edge: Edge }> = [];
    let node = root;

    while (node.isExpanded && !node.isTerminal) {
      const edge = selectEdge(node, cPuct);
      path.push({ node, edge });
      const nextState = childState(edge, node.state);
      if (edge.childNode === null) {
        edge.childNode = makeNode(nextState);
      }
      node = edge.childNode;
    }

    // --- Expansion + evaluation ---
    // `node` is now a leaf: either unexpanded or terminal.
    let value: number;

    if (node.isTerminal) {
      // terminalValueForMover returns the value from the side that JUST MOVED
      // into this terminal state (i.e. the side that is NOT to move here).
      // Our backup convention accumulates value from the perspective of the
      // side TO MOVE at each node. Since the side to move at the terminal node
      // is the one that did NOT make the last move (and lost if it's checkmate),
      // we negate to get the value from the terminal node's side-to-move.
      value = -terminalValueForMover(node.state);
    } else {
      // First visit to this node: query the policy, expand edges.
      const ev = await evaluate(node.state);
      expandEdges(node, ev.priors);
      // value is from the current node's side-to-move perspective.
      value = ev.value;

      // Forced-line override: if a bounded minimax over the rules alone
      // proves this position is a ±1/0 outcome within `mateSearchDepth`
      // plies, replace the NN's value with that certainty. This is what
      // makes a mate-in-2 lock in from a single visit to the first move:
      // expanding the post-1st-move leaf does a depth-2 search, finds
      // the mate, and backs +1 up through the edge that selected it.
      if (mateSearchDepth > 0) {
        const forced = forcedTerminalValue(node.state, mateSearchDepth);
        if (forced !== null) value = forced;
      }
    }

    // --- Backup ---
    // Walk back up the path, alternating sign each step.
    //
    // Why alternate? Each edge was selected from the parent's perspective.
    // After the child is evaluated from its own side-to-move's perspective,
    // the parent sees the negation (a good position for the child is bad for
    // the parent, since they're opponents).
    for (let i = path.length - 1; i >= 0; i--) {
      const { edge } = path[i];
      // Flip perspective: this edge leads to a state where it's the
      // opponent's turn, so the value from that state's perspective is
      // the negative of the value from ours.
      value = -value;
      edge.N += 1;
      edge.W += value;
    }
  }

  // --- Extract result ---
  const visitCounts = new Map<Move, number>(root.edges.map(e => [e.move, e.N]));
  const priorPolicy = new Map<Move, number>(root.edges.map(e => [e.move, e.P]));
  const edgeValues = new Map<Move, number>(
    root.edges.map(e => [e.move, e.N === 0 ? 0 : e.W / e.N]),
  );

  // Collect every root edge that ends the game in our favour. The cache
  // populated by selectEdge above already knows the answer for any edge
  // that was ever inspected; for edges we never visited we compute it
  // here so the caller's "any mate available?" question is always
  // exhaustive at the root.
  const mateInOneMoves: Move[] = [];
  for (const edge of root.edges) {
    if (edgeDeliversCheckmate(edge, root.state)) {
      mateInOneMoves.push(edge.move);
    }
  }
  // Lexicographic order so ties are deterministic (matches how bestMove
  // breaks ties below).
  mateInOneMoves.sort();

  // bestMove: a mate-in-1 wins outright over everything else. Otherwise
  // argmax visit count; ties broken by prior, then by move string (stable).
  let bestMove: Move;
  if (mateInOneMoves.length > 0) {
    bestMove = mateInOneMoves[0];
  } else {
    bestMove = root.edges[0].move;
    let bestN = -1;
    let bestP = -1;
    for (const edge of root.edges) {
      if (
        edge.N > bestN ||
        (edge.N === bestN && edge.P > bestP) ||
        (edge.N === bestN && edge.P === bestP && edge.move < bestMove)
      ) {
        bestN = edge.N;
        bestP = edge.P;
        bestMove = edge.move;
      }
    }
  }

  // rootValue: the backed-up value at the root, from white's perspective
  // is ambiguous — we return the root node's Q (average value from root's
  // side-to-move perspective, which is the sum over edge W/N weighted by N).
  // Simplest: use the edge with the most visits.
  const bestEdge = root.edges.find(e => e.move === bestMove)!;
  const rootValue = bestEdge.N > 0 ? bestEdge.W / bestEdge.N : 0;

  return {
    bestMove,
    rootValue,
    visitCounts,
    priorPolicy,
    edgeValues,
    mateInOneMoves,
  };
}
