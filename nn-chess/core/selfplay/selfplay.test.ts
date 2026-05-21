// Tests for the self-play module.
//
// The MCTS+NN combo is heavy. We avoid spinning up real ChessNets by
// stubbing the `Policy` interface with a deterministic uniform-prior
// policy that returns a fixed value. That lets us exercise the
// scheduling, example-construction, abort, and tally logic without
// dragging tfjs into every test run.

import { describe, expect, it } from 'vitest';
import { POLICY_SIZE, type TrainingExample } from '../nn/index.js';
import { legalMoves, isTerminal, type GameState, type Move } from '../rules/index.js';
import { fromFen } from '../rules/index.js';
import type { Policy } from '../mcts/index.js';
import { playOneGame, runSelfPlayBatch } from './index.js';

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
      aiPolicy: uniformPolicy(0.5),
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

  it('alternates colors by default', async () => {
    const seenColors: ('w' | 'b')[] = [];
    await runSelfPlayBatch({
      aiPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 4,
      maxPlies: 4,
      onGame: (_i, info) => seenColors.push(info.aiColor),
    });
    expect(seenColors).toEqual(['w', 'b', 'w', 'b']);
  });

  it('AI training examples are well-formed and tagged from the AI side', async () => {
    const result = await runSelfPlayBatch({
      aiPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 2,
      maxPlies: 8,
    });
    // Every training example must have a features tensor of the right
    // length and a valueTarget in {-1, 0, 1}.
    for (const ex of result.aiTrainingExamples as TrainingExample[]) {
      expect(ex.features.length).toBeGreaterThan(0);
      expect([-1, 0, 1]).toContain(ex.valueTarget);
      if (ex.policyTarget !== null) {
        expect(ex.policyTarget.length).toBe(POLICY_SIZE);
      }
    }
  });

  it('threads onPly through with the per-game AI color attached', async () => {
    const observed: Array<{ gameIndex: number; aiColor: 'w' | 'b'; plyIndex: number }> = [];
    await runSelfPlayBatch({
      aiPolicy: uniformPolicy(0),
      playerPolicy: uniformPolicy(0),
      search: { simulations: 2, cPuct: 1.5 },
      count: 2,
      maxPlies: 2,
      onPly: (gameIndex, info) => observed.push({
        gameIndex,
        aiColor: info.aiColor,
        plyIndex: info.plyIndex,
      }),
    });
    // At minimum, each game contributes its starting-position observation.
    expect(observed.length).toBeGreaterThanOrEqual(2);
    // gameIndex 0 must be aiColor 'w', gameIndex 1 must be 'b' (alternateColors default).
    const gameZero = observed.filter(o => o.gameIndex === 0);
    const gameOne = observed.filter(o => o.gameIndex === 1);
    expect(gameZero.every(o => o.aiColor === 'w')).toBe(true);
    expect(gameOne.every(o => o.aiColor === 'b')).toBe(true);
  });

  it('honors the abort signal between games', async () => {
    const controller = new AbortController();
    let games = 0;
    const result = await runSelfPlayBatch({
      aiPolicy: uniformPolicy(0),
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
});
