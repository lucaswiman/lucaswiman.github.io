# CLAUDE.md

Notes for Claude Code working in this repository.

## What this repo is

A personal blog built with **Astro 5** and deployed to GitHub Pages via
`.github/workflows/` on pushes to `master`. Posts live under
`src/content/blog/`, static assets under `public/`, layouts under
`src/layouts/`, and routes under `src/pages/`.

The build is a plain `npx astro build` producing static HTML in `dist/`
which is uploaded as the Pages artifact.

## Subprojects

### `/nn-chess` — neural-network chess that learns from your gameplay

A browser-based chess agent (MCTS + small value/policy NN, trained from
the win/loss signal of games against the user) that ships at the
`/nn-chess` route on the deployed site. Source lives in the top-level
`nn-chess/` directory.

**Read `nn-chess/README.md` before doing any work on this subproject.**
It describes the goals, non-goals, architecture, library choices,
training-signal constraints, and the explicit list of things not to do
without discussion (e.g. "do not add any chess heuristic beyond
'avoid checkmate'").

Key invariants to preserve when editing nn-chess code:

- `nn-chess/core/` is pure TypeScript with no DOM, no React, no
  `localStorage`, no Astro. It is consumed by the UI, by tests, and
  (eventually) by a Node-based training harness.
- Persistence goes through an abstract `Storage` port. Adapters
  (`localStorage`, IndexedDB, in-memory for tests) live outside
  `core/`.
- The only hand-coded chess heuristic is "checkmate is terminal."
  Everything else is learned.
- Core modules are unit-tested with `vitest`.
- The subproject is structured so it can be lifted into its own
  repository with minimal surgery.
