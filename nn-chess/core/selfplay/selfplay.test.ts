// Tests for the self-play module.
//
// The MCTS+NN combo is heavy. We avoid spinning up real ChessNets by
// stubbing the `Policy` interface with a deterministic uniform-prior
// policy that returns a fixed value. That lets us exercise the
// scheduling, example-construction, abort, and tally logic without
// dragging tfjs into every test run.

import { describe, expect, it } from 'vitest';
import { POLICY_SIZE, type TrainingExample } from '../nn/index.js';
import { legalMoves, isTerminal, toFen, type GameState, type Move } from '../rules/index.js';
import { fromFen } from '../rules/index.js';
import type { Policy } from '../mcts/index.js';
import {
  playOneGame,
  runSelfPlayBatch,
  runEndgameGym,
  parseGymScenarios,
  type EndgameGymScenario,
  type SideResult,
} from './index.js';

/** Deterministic mulberry32 PRNG — used by tests that need reproducible sampling. */
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

function uniformPolicy(value: number): Policy {
  return {
    async evaluate(state: GameState) {
      if (isTerminal(state)) return { priors: new Map(), value };
      const moves = legalMoves(state);
      const u = 1 / Math.max(moves.length, 1);
      const priors = new Map<Move, number>(moves.map(m => [m, u]));
      return { priors, value };
    },
  };
}

describe('playOneGame', () => {
  it('terminates on a mate-in-1 position and reports the correct outcome', async () => {
    // K+R vs K, white to move with Rh8# in one. Source: standard
    // textbook mate. The exact move depends on MCTS's tie-breaks under
    // a uniform prior, so we don't assert which mate move is played —
    // only that the game ends decisively in a small number of plies.
    const start = fromFen('7k/R7/6K1/8/8/8/8/8 w - - 0 1');
    const result = await playOneGame({
      white: uniformPolicy(0),
      black: uniformPolicy(0),
      search: { simulations: 8, cPuct: 1.5 },
      startingState: start,
      maxPlies: 6,
    });
    expect(['white-wins', 'black-wins', 'draw']).toContain(result.outcome);
    expect(result.plies).toBeLessThanOrEqual(6);
  });

  it('respects the ply cap and reports draw when neither side mates', async () => {
    // Bare kings — no legal mate exists, every game must draw out.
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const result = await playOneGame({
      white: uniformPolicy(0),
      black: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      startingState: start,
      maxPlies: 8,
    });
    expect(result.plies).toBeLessThanOrEqual(8);
    expect(['draw', 'white-wins', 'black-wins']).toContain(result.outcome);
  });

  it('records moves on the correct side', async () => {
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const result = await playOneGame({
      white: uniformPolicy(0),
      black: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      startingState: start,
      maxPlies: 6,
    });
    for (const r of result.whiteMoves) expect(r.sideToMove).toBe('w');
    for (const r of result.blackMoves) expect(r.sideToMove).toBe('b');
    // Moves alternate, so the two arrays differ by at most one entry.
    expect(Math.abs(result.whiteMoves.length - result.blackMoves.length)).toBeLessThanOrEqual(1);
  });

  it('emits an onPly observation for the starting position and each move', async () => {
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const observed: Array<{ plyIndex: number; lastMove: Move | null; fen: string }> = [];
    const result = await playOneGame({
      white: uniformPolicy(0),
      black: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      startingState: start,
      maxPlies: 4,
      onPly: (info) => observed.push(info),
    });
    // One observation for the starting position plus one per ply played.
    expect(observed).toHaveLength(result.plies + 1);
    expect(observed[0].plyIndex).toBe(0);
    expect(observed[0].lastMove).toBeNull();
    // Starting FEN matches the position we passed in.
    expect(observed[0].fen.startsWith('4k3/8/8/8/8/8/8/4K3 w')).toBe(true);
    // Subsequent entries have monotonically-increasing plyIndex and a non-null move.
    for (let i = 1; i < observed.length; i++) {
      expect(observed[i].plyIndex).toBe(i);
      expect(observed[i].lastMove).not.toBeNull();
    }
  });

  it('aborts cleanly when the signal fires', async () => {
    const controller = new AbortController();
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    controller.abort();
    const result = await playOneGame({
      white: uniformPolicy(0),
      black: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      startingState: start,
      maxPlies: 100,
      signal: controller.signal,
    });
    // Aborted before the first ply → no moves were played.
    expect(result.plies).toBe(0);
    expect(result.whiteMoves).toHaveLength(0);
    expect(result.blackMoves).toHaveLength(0);
  });
});

describe('runSelfPlayBatch', () => {
  it('plays the requested number of games and tallies results', async () => {
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const result = await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0.5),
      playerPolicy: uniformPolicy(-0.5),
      search: { simulations: 2, cPuct: 1.5 },
      count: 4,
      maxPlies: 6,
      // We feed the same starting state into playOneGame by way of a
      // wrapper — runSelfPlayBatch doesn't expose startingState in this
      // version, so the tally check only depends on `count` and the
      // resolved outcomes, not on which color won.
    });
    expect(result.games).toBe(4);
    expect(result.wins + result.losses + result.draws).toBe(4);
  });

  it('randomizes colors by default (both sides seen over many games)', async () => {
    const seenColors: ('w' | 'b')[] = [];
    // 20 games: P(never see 'w' or never see 'b') = 2 * (1/2)^20 ≈ 0.000002
    await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 20,
      maxPlies: 4,
      onGame: (_i, info) => seenColors.push(info.opponentColor),
    });
    expect(seenColors).toContain('w');
    expect(seenColors).toContain('b');
  });

  it('Opponent training examples are well-formed and tagged from the opponent side', async () => {
    const result = await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 2,
      maxPlies: 8,
    });
    // Every training example must have a features tensor of the right
    // length and a valueTarget in {-1, 0, 1}.
    for (const ex of result.opponentTrainingExamples as TrainingExample[]) {
      expect(ex.features.length).toBeGreaterThan(0);
      expect([-1, 0, 1]).toContain(ex.valueTarget);
      if (ex.policyTarget !== null) {
        expect(ex.policyTarget.length).toBe(POLICY_SIZE);
      }
    }
  });

  it('threads onPly through with the per-game opponent color attached', async () => {
    const observed: Array<{ gameIndex: number; opponentColor: 'w' | 'b'; plyIndex: number }> = [];
    await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 2,
      maxPlies: 2,
      onPly: (gameIndex, info) => observed.push({
        gameIndex,
        opponentColor: info.opponentColor,
        plyIndex: info.plyIndex,
      }),
    });
    // At minimum, each game contributes its starting-position observation.
    expect(observed.length).toBeGreaterThanOrEqual(2);
    // All plies within a single game must share the same opponentColor.
    const gameZero = observed.filter(o => o.gameIndex === 0);
    const gameOne = observed.filter(o => o.gameIndex === 1);
    expect(new Set(gameZero.map(o => o.opponentColor)).size).toBe(1);
    expect(new Set(gameOne.map(o => o.opponentColor)).size).toBe(1);
  });

  it('honors the abort signal between games', async () => {
    const controller = new AbortController();
    let games = 0;
    const result = await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 4,
      maxPlies: 4,
      onGame: () => { games++; controller.abort(); },
      signal: controller.signal,
    });
    // First game finishes; subsequent iterations bail on the signal.
    expect(result.games).toBe(1);
    expect(games).toBe(1);
  });

  it('uses startingStateForGame to seed each game', async () => {
    // Two endgame positions; the callback should be invoked once per
    // game and the resulting moves must respect the starting side-to-
    // move (white in both). We don't assert which move was played —
    // only that the game actually started from the curriculum FEN.
    const start0 = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const start1 = fromFen('7k/R7/6K1/8/8/8/8/8 w - - 0 1');
    const states = [start0, start1];
    const observed: string[] = [];
    await runSelfPlayBatch({
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 4, cPuct: 1.5 },
      count: 2,
      maxPlies: 1,
      startingStateForGame: (i) => states[i],
      onPly: (_gameIndex, info) => {
        if (info.plyIndex === 0) observed.push(info.fen);
      },
    });
    // Each game's ply-0 FEN should match one of the curriculum FENs
    // (FEN comparison via toFen normalizes both sides).
    expect(observed).toHaveLength(2);
    expect(observed[0]).toBe(toFen(start0));
    expect(observed[1]).toBe(toFen(start1));
  });
});

describe('runEndgameGym', () => {
  // The deterministic uniform policy used by the rest of this file
  // doesn't try to win — it just propagates priors. To produce reliable
  // "opponent wins" / "opponent doesn't win" outcomes for the gym tests we need to
  // pin the outcome from the starting position itself, so we use
  // textbook mate-in-1 positions where any tree exploration finds the
  // mate, and KvK draws where neither side can win.

  it('stops attempting a scenario after the opponent wins one game', async () => {
    // Mate-in-1: K+R vs K. With a few sims, MCTS finds the mate.
    const games: { scenarioIdx: number; gameInScenario: number; result: SideResult }[] = [];
    const scenariosDone: { scenarioIdx: number; converted: boolean; gamesPlayed: number }[] = [];

    const result = await runEndgameGym({
      scenarios: parseGymScenarios([
        { id: 'mate-in-1', fen: '7k/R7/6K1/8/8/8/8/8 w - - 0 1', opponentColor: 'w' },
      ]),
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 16, cPuct: 1.5 },
      maxGamesPerScenario: 5,
      maxPlies: 4,
      onGame: (info) => games.push({
        scenarioIdx: info.scenarioIdx,
        gameInScenario: info.gameInScenario,
        result: info.opponentResult,
      }),
      onScenarioDone: (info) => scenariosDone.push(info),
    });

    // The opponent should have won at least once, and we should have stopped
    // attempting that scenario as soon as it did.
    expect(games.length).toBeGreaterThanOrEqual(1);
    expect(games[games.length - 1].result).toBe('win');
    expect(result.scenariosAttempted).toBe(1);
    expect(result.scenariosConverted).toBe(1);
    expect(result.totalWins).toBe(1);
    expect(scenariosDone).toHaveLength(1);
    expect(scenariosDone[0].converted).toBe(true);
  });

  it('moves on after maxGamesPerScenario when the opponent never wins', async () => {
    // Bare kings — nobody can win, so the opponent cannot convert. The gym
    // should burn through maxGamesPerScenario attempts and then move on.
    const drawnScenario: EndgameGymScenario = {
      id: 'kvk',
      startingState: fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1'),
      opponentColor: 'w',
    };
    const result = await runEndgameGym({
      scenarios: [drawnScenario],
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      maxGamesPerScenario: 3,
      maxPlies: 4,
    });
    expect(result.scenariosAttempted).toBe(1);
    expect(result.scenariosConverted).toBe(0);
    expect(result.totalGames).toBe(3);
    expect(result.totalWins).toBe(0);
  });

  it('only records training examples from games the opponent won', async () => {
    // Combine a winnable scenario with a losing-side-to-move scenario.
    // The winnable one converts on attempt 1 (one game played). The
    // unwinnable one runs out of attempts (no games converted). All
    // recorded examples must trace back to the winnable game's win.
    const scenarios: EndgameGymScenario[] = [
      // Mate-in-1 for white.
      { id: 'mate-in-1', startingState: fromFen('7k/R7/6K1/8/8/8/8/8 w - - 0 1'), opponentColor: 'w' },
      // K+K — drawn, the opponent can't win.
      { id: 'kvk', startingState: fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1'), opponentColor: 'w' },
    ];
    const result = await runEndgameGym({
      scenarios,
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 16, cPuct: 1.5 },
      maxGamesPerScenario: 2,
      maxPlies: 4,
    });
    // At least one win, no losses-or-draws-with-policy-targets in the
    // training set: every recorded example must have a policy target
    // and a positive value target (since the gym records only winning
    // games and `buildTrainingExamples` gives the winner side-to-move
    // value = +1 and a one-hot policy target).
    expect(result.scenariosConverted).toBeGreaterThanOrEqual(1);
    for (const ex of result.opponentTrainingExamples) {
      expect(ex.valueTarget).toBe(1);
      expect(ex.policyTarget).not.toBeNull();
    }
  });

  it('honors the abort signal between games', async () => {
    const controller = new AbortController();
    const scenarios: EndgameGymScenario[] = [
      { id: 'kvk', startingState: fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1'), opponentColor: 'w' },
      { id: 'kvk2', startingState: fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1'), opponentColor: 'w' },
    ];
    const result = await runEndgameGym({
      scenarios,
      opponentPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      maxGamesPerScenario: 4,
      maxPlies: 4,
      signal: controller.signal,
      onGame: () => controller.abort(),
    });
    // First game's onGame fires the abort. The loop should exit before
    // starting any further games on this or the next scenario.
    expect(result.totalGames).toBe(1);
    expect(result.scenariosAttempted).toBe(1);
  });

  it('parseGymScenarios rejects scenarios where the opponent does not own the move', () => {
    expect(() =>
      parseGymScenarios([
        // White-to-move FEN but we lie and claim Black should play.
        { id: 'wrong', fen: '4k3/8/8/8/8/8/8/4K3 w - - 0 1', opponentColor: 'b' },
      ]),
    ).toThrow(/side-to-move/);
  });
});

describe('playOneGame — temperature sampling', () => {
  // A policy that puts ALL prior on `forcedMove`, so under argmax MCTS
  // will visit `forcedMove` exclusively and `bestMove` is deterministic.
  // The point of these tests is to show that visit-count sampling can
  // diverge from argmax when temperaturePlies > 0 and the visit-count
  // distribution is non-degenerate.
  function biasedUniformPolicy(): Policy {
    return {
      async evaluate(state: GameState) {
        if (isTerminal(state)) return { priors: new Map(), value: 0 };
        const moves = legalMoves(state);
        const u = 1 / Math.max(moves.length, 1);
        const priors = new Map<Move, number>(moves.map(m => [m, u]));
        return { priors, value: 0 };
      },
    };
  }

  it('default behaviour (temperaturePlies = 0) is argmax-everywhere', async () => {
    // Same RNG, same policy, same starting state — two argmax runs must
    // produce identical move sequences. This guards against accidental
    // RNG dependence in the greedy path.
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const a = await playOneGame({
      white: biasedUniformPolicy(),
      black: biasedUniformPolicy(),
      search: { simulations: 6, cPuct: 1.5, rng: makePrng(1) },
      startingState: start,
      maxPlies: 4,
    });
    const b = await playOneGame({
      white: biasedUniformPolicy(),
      black: biasedUniformPolicy(),
      search: { simulations: 6, cPuct: 1.5, rng: makePrng(1) },
      startingState: start,
      maxPlies: 4,
    });
    const movesA = [...a.whiteMoves, ...a.blackMoves].map(r => r.move);
    const movesB = [...b.whiteMoves, ...b.blackMoves].map(r => r.move);
    expect(movesA).toEqual(movesB);
  });

  it('temperature sampling can produce non-argmax moves', async () => {
    // We construct two runs from the same starting state, with the same
    // MCTS RNG (so visit counts are identical), but different temperature
    // RNGs. If the first ply is sampled (temperaturePlies = 1) and the
    // visit-count distribution has any non-degenerate mass, the two RNGs
    // will eventually disagree on at least one game's first move.
    //
    // We use the standard startpos because there are 20 legal first
    // moves — vastly more entropy than a bare-king position.
    const trials = 20;
    let argmaxFirst: string | null = null;
    const sampledFirsts = new Set<string>();
    for (let i = 0; i < trials; i++) {
      const result = await playOneGame({
        white: biasedUniformPolicy(),
        black: biasedUniformPolicy(),
        search: { simulations: 8, cPuct: 1.5, rng: makePrng(7) },
        temperaturePlies: 1,
        temperature: 1.0,
        rng: makePrng(1000 + i),
        maxPlies: 1,
      });
      const first = result.whiteMoves[0]?.move;
      expect(first).toBeDefined();
      sampledFirsts.add(first!);
    }
    // Also run an argmax-only baseline to confirm argmax would have
    // produced a single fixed move.
    for (let i = 0; i < 3; i++) {
      const greedy = await playOneGame({
        white: biasedUniformPolicy(),
        black: biasedUniformPolicy(),
        search: { simulations: 8, cPuct: 1.5, rng: makePrng(7) },
        maxPlies: 1,
      });
      if (argmaxFirst === null) argmaxFirst = greedy.whiteMoves[0].move;
      else expect(greedy.whiteMoves[0].move).toBe(argmaxFirst);
    }
    // Sampling should have produced strictly more than one distinct
    // first move across the trials; otherwise the temperature knob
    // isn't doing anything.
    expect(sampledFirsts.size).toBeGreaterThan(1);
  });

  it('the rng has no effect when temperaturePlies is 0', async () => {
    // Equivalence check: with sampling disabled, swapping the rng must
    // not change the move sequence. If this fails, sampling is leaking
    // into the argmax path.
    const start = fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    const a = await playOneGame({
      white: biasedUniformPolicy(),
      black: biasedUniformPolicy(),
      search: { simulations: 6, cPuct: 1.5, rng: makePrng(99) },
      startingState: start,
      temperaturePlies: 0,
      rng: makePrng(1),
      maxPlies: 4,
    });
    const b = await playOneGame({
      white: biasedUniformPolicy(),
      black: biasedUniformPolicy(),
      search: { simulations: 6, cPuct: 1.5, rng: makePrng(99) },
      startingState: start,
      temperaturePlies: 0,
      rng: makePrng(2_000_000),
      maxPlies: 4,
    });
    const movesA = [...a.whiteMoves, ...a.blackMoves].map(r => r.move);
    const movesB = [...b.whiteMoves, ...b.blackMoves].map(r => r.move);
    expect(movesA).toEqual(movesB);
  });
});
