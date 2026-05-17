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
be discovered by the neural network from self-play and games against the
human user.

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
│   ├── training/       ← self-play / human-play game recorder,
│   │                     replay buffer, training loop. Takes a
│   │                     `Storage` port (see below).
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
  CLI harness for "play 10,000 games against Stockfish overnight," or
  in a future standalone repo. The UI never imports anything outside
  `core/`'s public API plus an adapter.
- **`Storage` is a port, not `localStorage`.** The UI picks the
  adapter; the core just sees `get`/`put`. Swapping to IndexedDB or
  adding "export weights to a file" is one new adapter, no core
  changes.
- **The agent owns the MCTS+NN loop**, not the UI. The UI calls
  `agent.selectMove(state)` and `agent.recordGameResult(history,
  outcome)`. This means the same agent code drives human-play,
  self-play, and engine-vs-engine harnesses.

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

The agent's value target for a position is the eventual game outcome
from the side-to-move's perspective:

- `+1` if that side delivered checkmate,
- `−1` if that side was checkmated,
- `0` for any draw (stalemate, threefold, 50-move, insufficient
  material).

That's it. No material count, no mobility, no king safety, no
piece-square tables. MCTS gives us the policy target (visit-count
distribution at the root). Training is standard AlphaZero-style:
self-play games + human-play games go into a replay buffer; we
periodically sample minibatches and do a gradient step on
`value_loss + policy_loss` (+ small L2).

Because the only signal is terminal, **early play will be terrible**
and that is expected and load-bearing. The interesting research
question this project exists to play with is: how quickly can it
bootstrap from nothing, and what does its learned evaluation look
like under interpretability tools?

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
6. **Training loop.** Replay buffer, background training in a Web
   Worker, controls for "train N self-play games."
7. **Interpretability.** Weight + activation viewer; per-move
   visualization of MCTS priors and value head output.
8. **Stockfish sparring.** Optional opponent via `stockfish.js` in
   a Worker so we can watch the agent improve against a known
   yardstick.

## Things we will explicitly *not* do without a discussion

- Add any chess heuristic to the agent's evaluation, MCTS prior, or
  reward (including "tiny" ones like material count as a tiebreak).
- Couple `core/` to `window`, `localStorage`, React, or Astro.
- Skip tests on `core/` modules.
- Inline trained weights into the repo. Weights live in the user's
  browser; the repo ships only the (random-init) architecture.
