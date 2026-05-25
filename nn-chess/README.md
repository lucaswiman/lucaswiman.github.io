# nn-chess

> *Candidate tagline for the eventual blog post:* **"Like AlphaZero,
> but shittier — because it's trained off your moves."**

A browser-based chess game that learns to play **only from games tied
to the user** — either games the user plays directly against the
opponent network, or self-play games against a second network that
imitates the user. No opening books, no piece-square tables, no
material counting, no hand-tuned evaluation. The only chess knowledge
baked into the opponent is:

1. The rules of chess (legal moves, check, checkmate, stalemate, draw
   conditions).
2. **Avoid being checkmated / try to checkmate** — this is the *only*
   heuristic and serves as the terminal reward signal.

Everything else — piece values, tactics, positional understanding — must
be discovered by the network from games whose lineage traces back to
the user's own moves.

### Two networks

The project trains **two** networks side by side:

- **Opponent network (`opponent`)** — the one the user plays against.
  Trained AlphaZero-style on terminal outcomes: positions of the
  winning side carry a one-hot policy target, positions of the losing
  side and draws carry only a value target.
- **Player-imitator network (`imitator`)** — a behaviour-cloning model
  of the user. Trained supervised on every move the user plays,
  regardless of whether the user won, drew, or lost.

After each real game two things happen automatically:

1. The imitator records the user's moves and trains toward "predict
   what this user would play here."
2. The user can run a batch of MCTS self-play games between the
   opponent and the imitator; the opponent trains on the outcomes the
   same way it trains on real games.

This is a deliberate departure from the project's original "no
self-play, ever" rule. The original framing was "what does a tiny net
look like with one human's worth of data" — but with one human's data,
the opponent never accumulates enough decisive games to learn anything
beyond move legality. Routing the opponent's self-play through a
user-imitator gives a tractable data-volume boost while keeping the
learning signal tied to the user: the imitator only knows what the
user has shown it, so the opponent is still effectively training
against patterns the user has produced. Pure self-play between two
copies of the opponent (i.e. AlphaZero-style bootstrapping without a
user-in-the-loop) remains out of scope.

Down the road we will also let the user point the opponent at a known-
strong external player (e.g. Stockfish via a worker) and watch it
learn from those games.

### Play-the-imitator mode

There is also a dedicated route, **`/nn-chess/play-imitator`**, where the
user plays directly against the player-imitator instead of the opponent.
It exists as an interpretability tool — "does the imitator actually play
like me?" — and as a way to harvest more imitator training examples per
session without having to grind through full games against the (early,
terrible) opponent network. Move selection here runs the same MCTS
`search()` the opponent and self-play use, with the imitator's net
supplying priors and value; the page exposes a sims slider and a
visit-count sampling temperature. This mirrors how `core/selfplay`
already drives the imitator and means the only project-approved
heuristic ("checkmate is terminal") behaves identically across all
surfaces. A raw-policy-head path is still available on
`selectImitatorMove` for tests and for future interpretability views
that want to see what the policy says before MCTS smooths it out.
Finished games on this page feed `imitator.recordUserMoves(...)` on the
user's own moves only — the imitator's own moves are never routed back
into its replay buffer, so the "user-only training signal" invariant
holds.

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
   "checkmate is terminal and good/bad." The opponent has to be bad at
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
│   │                     subset of the rules API the opponent + UI need.
│   ├── encoding/       ← board ↔ tensor encoding (planes per piece type,
│   │                     side to move, castling rights, etc.)
│   ├── nn/             ← value+policy network. Pluggable backend
│   │                     (tfjs to start). Exposes forward pass,
│   │                     training step, weight serialization, and
│   │                     activation introspection hooks.
│   ├── mcts/           ← PUCT-style MCTS that consults the NN for
│   │                     priors + value. Pure functions over a
│   │                     `GameState` + `Policy` abstraction; no I/O.
│   ├── training/       ← replay buffer + serialization. Takes a
│   │                     `Storage` port (see below).
│   ├── storage/        ← `Storage` interface (get/put/list/delete of
│   │                     opaque blobs keyed by string). Adapters live
│   │                     outside `core/`.
│   ├── imitator/       ← player-imitator network (behaviour cloning
│   │                     of the user). Same shape as the opponent's
│   │                     NN, trained on every user move regardless of
│   │                     outcome.
│   ├── selfplay/       ← MCTS-vs-MCTS game runner used to pit the
│   │                     opponent against the imitator and harvest
│   │                     AlphaZero-style training examples.
│   └── opponent/       ← glues MCTS + NN + (optional) training into a
│                         single "Opponent" object the UI talks to.
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
- **The opponent owns the MCTS+NN loop**, not the UI. The UI calls
  `opponent.selectMove(state)` and `opponent.recordGameResult(history,
  outcome)`. The same opponent code drives human-play in the browser
  and (later) play against an external player like Stockfish; both
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

There are two learners and they take training data from different
sources.

### Opponent network

The opponent is trained AlphaZero-style on the outcomes of finished
games. Two kinds of games feed its replay buffer:

- **Real games** — the user plays the opponent; every position from
  that game becomes a training example.
- **Self-play games** — the opponent plays the player-imitator under
  MCTS; positions from the opponent's side of those games become
  training examples too.

For each opponent training example:

**Value target** is the eventual game outcome from the side-to-move's
perspective:

- `+1` if the side to move went on to win the game,
- `−1` if the side to move went on to lose the game,
- `0` for any draw (stalemate, threefold, 50-move, insufficient
  material).

**Policy target** is the move that was actually played from that
position — *but only if the side that moved went on to win*. Losing-
side positions and draws contribute value loss only; policy loss is
skipped. This holds for both real and self-play games, so the
training pipeline is uniform across both data sources.

### Player-imitator network

The imitator is a pure behaviour-cloning model of the user. After
each real game, **every** user move from that game becomes a positive
example for the imitator's policy head, regardless of whether the
user won, drew, or lost. The imitator's value head sees the same
outcome-from-side-to-move signal the opponent uses, so it can act as
a serviceable MCTS leaf evaluator during self-play.

The imitator does **not** train on self-play games. Its only signal
is the user's own moves.

### Loss & optimization

Both networks share the same architecture and training step:
`value_loss + policy_loss` (+ small L2) over minibatches sampled
from each network's own replay buffer. Training is AlphaZero-shaped
on the loss side; what differs from AlphaZero is the data-generation
loop (no infinite self-play; everything is gated on real user
games).

### Why this is still not pure bootstrapping

The imitator only ever sees moves the user has actually played, so
the opponent's self-play data is downstream of the user's own play
style. The opponent is not improving against an arbitrary strong
player — it is improving against a learned proxy for *this user*.
Strength is bounded by what the imitator can squeeze out of the user's
moves.

Because the only ground-truth signal is terminal and the data volume
is tiny, **early play will be terrible** and that is expected. Resist
every temptation to "help" by adding opening books or a material-
aware initialization.

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
5. **Opponent + UI.** React board, "new game" / "resign" controls,
   move history, status. Persist weights via the `localStorage`
   adapter.
6. **Training loop.** Replay buffer fed by completed user-vs-opponent
   games, background gradient steps in a Web Worker so training
   doesn't block the UI, controls for "train for N steps on the
   games you've already played."
7. **Interpretability.** Weight + activation viewer; per-move
   visualization of MCTS priors and value head output.
8. **Stockfish sparring.** Optional external player via `stockfish.js`
   in a Worker so we can watch the opponent improve against a known
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

- Add any chess heuristic to the opponent's evaluation, MCTS prior,
  or reward (including "tiny" ones like material count as a tiebreak).
- Add a self-play loop that is *not* gated by the player-imitator —
  e.g. AlphaZero-style "two copies of the opponent play each other
  forever" is still out of scope. The imitator-mediated self-play is
  the only approved kind because it keeps the data tethered to the user.
- Couple `core/` to `window`, `localStorage`, React, or Astro.
- Skip tests on `core/` modules.
- Inline trained weights into the repo. Weights live in the user's
  browser; the repo ships only the (random-init) architecture.
