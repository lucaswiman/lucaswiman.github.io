# nn-chess

> *Candidate tagline for the eventual blog post:* **"Like AlphaZero,
> but shittier — because it's trained off your moves."**

A browser-based chess game that learns to play **solely from the win/loss
signal of games it plays against you**. No opening books, no piece-square
tables, no material counting, no hand-tuned evaluation. The only chess
knowledge baked into the agent is:

1. The rules of chess (legal moves, check, checkmate, stalemate, draw
   conditions).
2. **Avoid being checkmated / try to checkmate** — this is the *only*
   heuristic and serves as the terminal reward signal.

Everything else — piece values, tactics, positional understanding — must
be discovered by the neural network from **games the user plays against
it**. The agent does **not** self-play to bootstrap; the only training
signal available to it is the user's own games. Down the road we will
also let the user point the agent at a known-strong opponent (e.g.
Stockfish via a worker) and watch it learn from those games, but pure
self-play between two copies of the agent is deliberately out of scope —
the whole point of the project is "what happens when a small network
learns chess from one human's games."

The app lives at the route **`/nn-chess`** on this site, is built as part
of the normal Astro / GitHub Actions pipeline, and persists trained
network weights in the browser (via `localStorage` for now, but behind an
abstract `Storage` interface so we can swap in IndexedDB, a remote
backend, or a file download/upload later).

We may eventually split this into its own repository; the directory
layout and module boundaries are chosen so that move is mechanical.

## Goals (in priority order)

1. **Pedagogical clarity.** The code should be a readable reference for
   how MCTS + a value/policy network learns a game with zero hand-coded
   evaluation. Comments should explain *why*, names should explain
   *what*.
2. **Honest learning signal.** Absolutely no heuristics beyond
   "checkmate is terminal and good/bad." The agent has to be bad at
   first and that's the point. Resist every temptation to nudge it.
3. **Strong factoring.** A pure, framework-agnostic core (rules + MCTS +
   NN + training loop) with no DOM, no React, no `localStorage`
   references. The UI is one consumer; tests are another; a future
   "train against Stockfish" harness is a third.
4. **Interpretability-ready.** The NN module must expose intermediate
   activations and weights through a stable API so we can build
   visualizations and mechanistic-interpretability tooling on top
   without reaching into internals.
5. **Static deploy.** Everything compiles to static assets shipped by
   the existing GitHub Pages workflow. No server.

## Non-goals (for now)

- Beating Stockfish. Beating a beginner would already be a win.
- WebGPU / native acceleration. CPU tensors (e.g. `@tensorflow/tfjs`
  with the WASM or CPU backend) are fine; we can swap backends later.
- Multi-user accounts, cloud-synced weights, leaderboards.
- A polished UI. Clean, functional, debuggable beats pretty.

## Architecture sketch

```
nn-chess/
├── README.md           ← this file
├── core/               ← pure TS, no DOM, no framework. Heavily unit-tested.
│   ├── rules/          ← thin wrapper over `chess.js` exposing the
│   │                     subset of the rules API the agent + UI need.
│   ├── encoding/       ← board ↔ tensor encoding (planes per piece type,
│   │                     side to move, castling rights, etc.)
│   ├── nn/             ← value+policy network. Pluggable backend
│   │                     (tfjs to start). Exposes forward pass,
│   │                     training step, weight serialization, and
│   │                     activation introspection hooks.
│   ├── mcts/           ← PUCT-style MCTS that consults the NN for
│   │                     priors + value. Pure functions over a
│   │                     `GameState` + `Policy` abstraction; no I/O.
│   ├── training/       ← user-game recorder, replay buffer, training
│   │                     loop. Takes a `Storage` port (see below).
│   │                     No self-play loop — training data is the
│   │                     user's own games (and, later, games against
│   │                     a configured external opponent).
│   ├── storage/        ← `Storage` interface (get/put/list/delete of
│   │                     opaque blobs keyed by string). Adapters live
│   │                     outside `core/`.
│   └── agent/          ← glues MCTS + NN + (optional) training into a
│                         single "Agent" object the UI talks to.
├── adapters/           ← non-core, environment-specific glue.
│   ├── storage-localstorage.ts
│   ├── storage-memory.ts        ← used by tests
│   └── (future) storage-indexeddb.ts, storage-remote.ts
├── ui/                 ← React island mounted by an Astro page at
│                         `/nn-chess`. Renders the board (e.g.
│                         `react-chessboard`), training controls, and —
│                         later — weight / activation visualizations.
└── tests/              ← unit tests for everything in `core/`.
```

### Why this split

- **`core/` has no environment dependencies.** It can run in Node for
  tests, in a Web Worker for training without blocking the UI, in a
  CLI harness, or in a future standalone repo. The UI never imports
  anything outside `core/`'s public API plus an adapter.
- **`Storage` is a port, not `localStorage`.** The UI picks the
  adapter; the core just sees `get`/`put`. Swapping to IndexedDB or
  adding "export weights to a file" is one new adapter, no core
  changes.
- **The agent owns the MCTS+NN loop**, not the UI. The UI calls
  `agent.selectMove(state)` and `agent.recordGameResult(history,
  outcome)`. The same agent code drives human-play in the browser
  and (later) play against an external opponent like Stockfish; both
  are sources of *real* games, not self-play.

## Libraries we intend to use

Final choices made when implementing; this is the current shortlist.

- **`chess.js`** — move generation, legality, check / checkmate /
  stalemate detection, FEN/PGN. Battle-tested, no need to roll our own.
- **`react-chessboard`** — board rendering and drag-drop. Tiny API,
  works well with `chess.js`.
- **`@tensorflow/tfjs`** — NN forward/backward in-browser. CPU/WASM
  backend by default; WebGL/WebGPU optional later.
- **`vitest`** — unit tests for `core/`. Fast, ESM-native, works with
  the Astro/TS setup.
- **`@astrojs/react`** — adds React island support to the existing
  Astro site so the chess UI can live at `/nn-chess` without rewriting
  the rest of the blog.

## Reward & training (the only "heuristic")

After a game ends, every position from that game becomes a training
example for the NN, with the targets derived purely from the outcome:

**Value target** for a position is the eventual game outcome from the
side-to-move's perspective:

- `+1` if the side to move went on to win the game,
- `−1` if the side to move went on to lose the game,
- `0` for any draw (stalemate, threefold, 50-move, insufficient
  material).

**Policy target** for a position is the move that was actually played
from it — *but only if the player who played it won the game*. In
other words:

- **User wins** ⇒ the user's moves become positive examples for the
  policy head (the NN learns "in this kind of position, play what the
  user played"). The AI's moves from that game are not used as policy
  targets — only the value head sees them, with target `−1`.
- **AI wins** ⇒ the AI's moves become positive examples for the
  policy head (standard AlphaZero-style self-improvement, except the
  data comes from a real game against the user, not self-play). The
  user's moves from that game are not used as policy targets.
- **Draw** ⇒ value target is `0` for both sides' positions; we skip
  the policy loss on draws so we're not reinforcing moves that didn't
  decisively work.

This is the *only* learning signal. The NN is doing imitation
learning of the winner — whoever the winner happens to be — combined
with standard value-head bootstrapping from terminal outcomes.

No material count, no mobility, no king safety, no piece-square
tables. Training is AlphaZero-shaped on the loss side —
`value_loss + policy_loss` (+ small L2) over minibatches sampled
from a replay buffer — but the buffer is filled **only with games
the user (or a configured external opponent) actually played against
the agent**. There is no self-play data-generation loop.

Note one subtle departure from AlphaZero: AlphaZero's policy target
is the MCTS visit-count distribution at the root, which is
well-defined for the AI's moves but not for the user's. To keep the
training pipeline uniform across both kinds of moves, **we use the
actually-played move as a one-hot policy target** (for both AI and
human moves), and we only apply the policy loss to positions where
the player who moved went on to win. This makes the policy head a
behavior-cloner of the winner.

This is a hard constraint, not an oversight: AlphaZero gets away
with a tiny terminal signal because it generates millions of
self-play games. We don't, deliberately. The interesting research
question this project exists to play with is: **how does an MCTS+NN
agent that only ever sees one person's games against it actually
play, and what does its learned evaluation look like under
interpretability tools?** "Like AlphaZero but shittier" is the
load-bearing frame, not a self-deprecating joke.

Because the only signal is terminal and the data volume is tiny,
**early play will be terrible** and that is expected. Resist every
temptation to "help" by adding self-play, opening books, or a
material-aware initialization.

## Roadmap

1. **Scaffolding.** Add `@astrojs/react`, create `/nn-chess` route,
   stub out the `core/` directory structure, set up `vitest`, wire
   the GitHub Actions build to also run tests.
2. **Rules + encoding + tests.** Wrap `chess.js`, implement the
   board↔tensor encoding, exhaustive unit tests on both.
3. **NN skeleton.** Small value+policy net in tfjs. Tests for
   shape, determinism with a fixed seed, weight save/load
   round-trip through the `Storage` port.
4. **MCTS.** PUCT search over the `GameState` abstraction with the
   NN providing priors and value. Tests against trivial endgames
   (e.g. K+Q vs K mate-in-1) where the right move is unambiguous.
5. **Agent + UI.** React board, "new game" / "resign" controls,
   move history, status. Persist weights via the `localStorage`
   adapter.
6. **Training loop.** Replay buffer fed by completed user-vs-agent
   games, background gradient steps in a Web Worker so training
   doesn't block the UI, controls for "train for N steps on the
   games you've already played."
7. **Interpretability.** Weight + activation viewer; per-move
   visualization of MCTS priors and value head output.
8. **Stockfish sparring.** Optional opponent via `stockfish.js` in
   a Worker so we can watch the agent improve against a known
   yardstick.
9. **Endgame curriculum.** Setup picker that starts a game from a
   curated endgame FEN — hand-written textbook mates (KQ-K, KR-K,
   ladder, KP-K) plus a sample of the Lichess CC0 puzzle database
   filtered by endgame themes (`mateIn1-5`, `rookEndgame`,
   `pawnEndgame`, etc.). The Lichess sample is built offline by
   `scripts/build-lichess-endgames.mjs` and committed as a small
   JSON file under `nn-chess/ui/lichess-endgames.json`; the 1 GB
   source CSV is never checked in. The goal is to give the value
   head real ±1 outcomes within ~10 plies and put forcing mating
   sequences into the replay buffer, instead of letting a freshly-
   initialized net stumble around from startpos against the
   50-move rule.

## Things we will explicitly *not* do without a discussion

- Add any chess heuristic to the agent's evaluation, MCTS prior, or
  reward (including "tiny" ones like material count as a tiebreak).
- Couple `core/` to `window`, `localStorage`, React, or Astro.
- Skip tests on `core/` modules.
- Inline trained weights into the repo. Weights live in the user's
  browser; the repo ships only the (random-init) architecture.
