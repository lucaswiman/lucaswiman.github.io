/**
 * Shared message types for the self-play Web Worker protocol.
 *
 * Both `selfplay-worker.ts` (the worker side) and `selfplay-client.ts`
 * (the main-thread wrapper) import from here. No React, no DOM, no UI
 * imports — this file must stay safe to use in a Worker context.
 *
 * A self-play run is three phases inside one worker invocation:
 *   1. Run `gradientSteps` gradient steps on the imitator from its own
 *      replay buffer, so the opponent's self-play stand-in reflects the
 *      user's latest moves before games are generated.
 *   2. Play `count` MCTS games between the opponent net and the
 *      (just-updated) imitator net, collecting AlphaZero-style training
 *      examples for the opponent's side.
 *   3. Run `gradientSteps` gradient steps on the opponent net using the
 *      enlarged replay buffer.
 * Folding all phases into one worker call avoids extra rounds of weight
 * / buffer serialization across the worker boundary.
 */

/** Lightweight mirror of MCTS SearchConfig (only the fields self-play uses). */
export interface SelfPlaySearchConfig {
  simulations: number;
  cPuct: number;
  /**
   * Root-noise alpha for the AlphaZero Dirichlet trick. Enables a noisy
   * exploration boost at the root of the search tree only; leaving it
   * undefined disables noise entirely. Standard AlphaZero used 0.3 for
   * chess.
   */
  dirichletAlpha?: number;
  /**
   * Mixing weight of the Dirichlet noise with the policy priors at the
   * root. 0 = no noise, 1 = noise replaces priors. Standard AlphaZero
   * used 0.25.
   */
  dirichletEpsilon?: number;
}

/**
 * Messages sent FROM the main thread TO the worker.
 *
 * `start` kicks off a self-play batch sandwiched between gradient steps:
 *   - opponentWeights:     serialized opponent ChessNet bytes
 *   - imitatorWeights:     serialized imitator ChessNet bytes
 *   - replay:              serialized opponent ReplayBuffer bytes (may be empty)
 *   - imitatorReplay:      serialized imitator ReplayBuffer bytes (may be empty)
 *   - replayCapacity:      capacity for both replay buffers
 *   - search:              MCTS budget applied to both sides
 *   - count:               number of self-play games to play
 *   - maxPlies:            per-game ply cap
 *   - gradientSteps:       number of gradient steps to run on each net
 *   - gradientBatchSize:   examples per gradient step
 *
 * `stop` requests a graceful early stop. The worker finishes the current
 * game / step, serializes, and posts a 'done' message.
 */
/**
 * One scenario the gym should attempt. The worker parses `fen` to a
 * `GameState` on receipt. `opponentColor` must equal the side to move
 * at `fen` — `parseGymScenarios` enforces this on the way in.
 */
export interface GymScenarioRequest {
  id: string;
  fen: string;
  opponentColor: 'w' | 'b';
}

export type SelfPlayRequest =
  | {
      kind: 'start';
      opponentWeights: Uint8Array;
      imitatorWeights: Uint8Array;
      replay: Uint8Array;
      imitatorReplay: Uint8Array;
      replayCapacity: number;
      search: SelfPlaySearchConfig;
      count: number;
      maxPlies: number;
      gradientSteps: number;
      gradientBatchSize: number;
      /**
       * Plies at the start of each game whose move is sampled from the
       * MCTS visit-count distribution instead of taken greedily. See
       * `PlayGameConfig.temperaturePlies` in core/selfplay.
       */
      temperaturePlies?: number;
      /** Visit-count sampling temperature. See `PlayGameConfig.temperature`. */
      temperature?: number;
      /**
       * Optional pool of starting FENs. When present, each game's
       * starting state is sampled uniformly at random from this pool;
       * when absent, every game starts from the standard initial
       * position. Used to drive the endgame-curriculum self-play mode.
       */
      startingFens?: string[];
    }
  | {
      // Endgame gym: loop through decided positions with the opponent
      // fixed on the winning side. The worker grinds through scenarios
      // in order, stops attempting each one as soon as the opponent
      // converts (or maxGamesPerScenario is hit), and runs a final
      // round of gradient steps on the new examples. Phases 1 / 2 of
      // the regular self-play flow (imitator pre-training and
      // color-alternating games) are skipped — the gym is exclusively
      // about feeding winning-side policy targets to the opponent.
      kind: 'start-gym';
      opponentWeights: Uint8Array;
      imitatorWeights: Uint8Array;
      replay: Uint8Array;
      imitatorReplay: Uint8Array;
      replayCapacity: number;
      search: SelfPlaySearchConfig;
      scenarios: GymScenarioRequest[];
      maxGamesPerScenario: number;
      maxPlies: number;
      gradientSteps: number;
      gradientBatchSize: number;
      temperaturePlies?: number;
      temperature?: number;
    }
  | { kind: 'stop' };

/**
 * Messages sent FROM the worker TO the main thread.
 *
 * `ply`   — emitted once per board state inside a self-play game (one for
 *           the starting position, then one per move). Used by the UI to
 *           render a live board; pure background runs ignore it.
 * `game`  — one self-play game finished.
 * `step`  — periodic gradient-step progress report. `phase` distinguishes
 *           the imitator pre-train pass from the opponent post-train pass.
 * `done`  — the whole batch finished (or stop was requested). Carries
 *           fresh opponent + imitator weights and both replay buffers.
 * `error` — something went wrong; the worker is now idle.
 */
export type SelfPlayResponse =
  | {
      kind: 'ply';
      gameIndex: number;
      opponentColor: 'w' | 'b';
      plyIndex: number;
      fen: string;
      lastMove: string | null;
    }
  | {
      kind: 'game';
      index: number;
      opponentColor: 'w' | 'b';
      opponentResult: 'win' | 'loss' | 'draw';
      plies: number;
      /**
       * When emitted by a gym run, identifies the scenario the game
       * belongs to and the 0-based attempt number within it. Omitted
       * by the regular `start` self-play flow.
       */
      scenarioIdx?: number;
      gameInScenario?: number;
    }
  | {
      kind: 'scenario-done';
      scenarioIdx: number;
      scenarioId: string;
      converted: boolean;
      gamesPlayed: number;
    }
  | {
      kind: 'step';
      phase: 'imitator' | 'opponent';
      stepIndex: number;
      valueLoss: number;
      policyLoss: number;
      /** Illegal-move penalty. 0 if the batch carried no legal-move masks. */
      illegalLoss: number;
      totalLoss: number;
    }
  | {
      kind: 'done';
      opponentWeights: Uint8Array;
      imitatorWeights: Uint8Array;
      replay: Uint8Array;
      imitatorReplay: Uint8Array;
      games: number;
      wins: number;
      losses: number;
      draws: number;
      examplesAdded: number;
      finalStep: number;
      imitatorFinalStep: number;
      /** Gym-only: number of scenarios the opponent converted at least once. */
      scenariosConverted?: number;
      /** Gym-only: scenarios actually attempted before stop/end-of-list. */
      scenariosAttempted?: number;
    }
  | { kind: 'error'; message: string };
