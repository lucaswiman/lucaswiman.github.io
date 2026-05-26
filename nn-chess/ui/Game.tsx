import { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board.js';
import { TrainingPanel } from './TrainingPanel.js';
import { InspectorPanel } from './InspectorPanel.js';
import { SelfPlayPanel } from './SelfPlayPanel.js';
import type { SelfPlayPanelHandle } from './SelfPlayPanel.js';
import { EndgameGymPanel } from './EndgameGymPanel.js';
import { TrainingLog, useTrainingLog } from './TrainingLog.js';
import { TrainingStatsPanel } from './TrainingStats.js';
import { LastMoveMcts } from './LastMoveMcts.js';
import {
  type NetStats,
  emptyNetStats,
  loadNetStats,
  saveNetStats,
  addTrainingPass,
  OPPONENT_TRAIN_STATS_KEY,
  IMITATOR_TRAIN_STATS_KEY,
} from './training-stats.js';
import {
  initialState,
  applyMove,
  fromFen,
  sideToMove,
  outcome,
  toFen,
  legalMoves,
} from '../core/rules/index.js';
import type { GameState, Move, Outcome } from '../core/rules/index.js';
import { encodeState } from '../core/encoding/index.js';
import type { Opponent, GameMoveRecord } from '../core/opponent/index.js';
import { legalMoveIndicesFor, mateInOneIndicesFor, OPPONENT_STORAGE_KEYS } from '../core/opponent/index.js';
import type { PlayerImitator } from '../core/imitator/index.js';
import { IMITATOR_STORAGE_KEYS } from '../core/imitator/index.js';
import { selectMoveFromNet } from '../core/move-selection/index.js';
import MctsToggle from './MctsToggle.js';
import type { SearchResult } from '../core/mcts/index.js';
import type { BlobStorage } from '../core/storage/index.js';
import { getJson, putJson } from '../core/storage/index.js';
import {
  OPPONENT_CONFIG,
  createOpponentStorage,
  loadOpponentFromStorage,
  loadPlayerImitatorFromStorage,
  AUTO_SELFPLAY_PREF_KEY,
  PREVIEW_TRAINING_PREF_KEY,
} from './opponent-config.js';
import { ENDGAME_PRESETS, ENDGAME_PRESETS_BY_ID } from './endgame-presets.js';
import {
  LICHESS_CATEGORIES,
  LICHESS_CATEGORIES_BY_ID,
  randomPuzzle,
  lichessPuzzleUrl,
} from './lichess-endgames.js';
import type { LichessPuzzle } from './lichess-endgames.js';
import {
  loadPersistedSession,
  savePersistedSession,
  type PersistedSession,
  type ColorPreference,
  type OpponentKind,
} from './session-persistence.js';
import { MAIA_ELO_BUCKETS } from '../core/maia/index.js';
import {
  MaiaClient,
  fetchMaiaWeights,
  DEFAULT_MAIA_SOURCE,
  type MaiaLoadProgress,
} from './maia-client.js';
import { StockfishEngine } from './stockfish-protocol.js';

type PlayerColor = 'w' | 'b';
type SetupId = 'standard' | 'custom' | string;

// Engine load state for the heavyweight third-party opponents (Maia,
// Stockfish). The lightweight opponents (the nn-chess opponent net and
// the imitator) are always loaded alongside the page bundle, so they
// don't go through this status machine.
type EngineStatus =
  | { phase: 'idle' }
  | { phase: 'loading'; progress: MaiaLoadProgress | null }
  | { phase: 'ready' }
  | { phase: 'error'; message: string };

interface GameSession {
  state: GameState;
  history: GameMoveRecord[];
  over: boolean;
  finalOutcome: Outcome | null;
  // FEN of the position the game started from. Kept so the persisted
  // session can store `startingFen + moves` instead of every per-move
  // feature tensor.
  startingFen: string;
  // UCI of the most recent move, or null at the start of a game. Used
  // to highlight the from/to squares on the board so the user can see
  // what just moved without scrubbing the history.
  lastMove: Move | null;
}

const OPPONENT_LABELS: Record<OpponentKind, string> = {
  'opponent': 'Opponent NN (the one we\'re training)',
  'imitator': 'Imitator (behaviour-clone of your moves)',
  'stockfish': 'Stockfish (calibrated weak)',
  'maia-2': 'Maia-2 (human-style, by ELO bucket)',
};

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

function sessionFromState(state: GameState): GameSession {
  return {
    state,
    history: [],
    over: false,
    finalOutcome: null,
    startingFen: toFen(state),
    lastMove: null,
  };
}

export default function Game() {
  const [opponent, setOpponent] = useState<Opponent | null>(null);
  const [imitator, setImitator] = useState<PlayerImitator | null>(null);
  const [storage] = useState<BlobStorage>(createOpponentStorage);
  const [playerColor, setPlayerColor] = useState<PlayerColor>('w');
  const [colorPreference, setColorPreference] = useState<ColorPreference>('random');
  const [setupId, setSetupId] = useState<SetupId>('standard');
  const [customFen, setCustomFen] = useState<string>('');
  const [customFenError, setCustomFenError] = useState<string | null>(null);
  const [currentPuzzle, setCurrentPuzzle] = useState<LichessPuzzle | null>(null);
  const [session, setSession] = useState<GameSession>(() => sessionFromState(initialState()));
  const [opponentThinking, setOpponentThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opponentVersion, setOpponentVersion] = useState(0);
  const [lastSearch, setLastSearch] = useState<SearchResult | null>(null);
  // Which engine is playing the other side. Defaults to the nn-chess
  // opponent net — the one the rest of this page exists to train.
  // Switching kinds mid-game just changes who plays the next move; it
  // doesn't reset the position or the history.
  const [opponentKind, setOpponentKind] = useState<OpponentKind>('opponent');
  // Stockfish controls — kept separate from MCTS sims/cPuct because
  // they configure a different engine entirely.
  const [sfSkill, setSfSkill] = useState(5);
  const [sfMovetimeMs, setSfMovetimeMs] = useState(100);
  // Maia controls. ELO bucket 5 ≈ 1500. Maia uses its own MCTS budget
  // because each sim is a full forward pass on WASM (~6-10s), so we
  // cap sims low and default the toggle OFF — Maia's ELO calibration
  // targets its policy head, not search.
  const [maiaEloBucket, setMaiaEloBucket] = useState(5);
  const [maiaUseMcts, setMaiaUseMcts] = useState(false);
  const [maiaSimulations, setMaiaSimulations] = useState(4);
  // Engine status for the heavyweight third-party opponents. Maia is
  // ~40 MiB so we surface download progress; Stockfish is small but
  // still spun up lazily on first use.
  const [maiaStatus, setMaiaStatus] = useState<EngineStatus>({ phase: 'idle' });
  const [sfStatus, setSfStatus] = useState<EngineStatus>({ phase: 'idle' });
  // Top-3 Maia readout for the last move (parallel to LastMoveMcts for
  // the opponent net). Only populated when playing against Maia.
  const [lastMaiaReadout, setLastMaiaReadout] = useState<
    { move: Move; value: number; topMoves: Array<{ move: Move; prob: number }> } | null
  >(null);
  // Default true: the brief asks for auto-batching on by default; the
  // persisted preference (loaded asynchronously below) overrides this
  // once it lands.
  const [autoSelfPlay, setAutoSelfPlay] = useState(true);
  // Default true to match the prior behaviour of always rendering the
  // live training board; the persisted value (loaded below) overrides
  // this once it lands.
  const [previewTraining, setPreviewTraining] = useState(true);
  const [simulations, setSimulations] = useState(OPPONENT_CONFIG.simulations);
  const [cPuct, setCPuct] = useState(OPPONENT_CONFIG.cPuct);
  const [cPuctTipOpen, setCPuctTipOpen] = useState(false);
  // "Use MCTS" toggle. When on (the default, preserving today's
  // behaviour) the opponent runs full MCTS via `opponent.selectMove`.
  // When off, the opponent picks straight from its policy head, which is
  // much faster but skips the visit-count lookahead the inspector panel
  // visualises. The sims + cPuct sliders are still rendered but greyed
  // out so the user understands what the toggle gates.
  const [useMcts, setUseMcts] = useState(true);
  // Shared log fed by every training-side panel. Lets the user see
  // gradient-descent vs MCTS phase transitions and storage saves
  // without staring at the console.
  const trainingLog = useTrainingLog();
  // Cumulative gradient-step / examples-seen counters per net, loaded
  // from storage on mount and bumped after every training pass.
  const [opponentStats, setOpponentStats] = useState<NetStats>(emptyNetStats);
  const [imitatorStats, setImitatorStats] = useState<NetStats>(emptyNetStats);
  // Gates the "persist session on change" effect: we only start writing
  // after the load attempt finishes, so the initial in-memory state
  // can't overwrite a saved session before we've had a chance to read it.
  const [sessionLoaded, setSessionLoaded] = useState(false);

  const opponentRef = useRef<Opponent | null>(null);
  opponentRef.current = opponent;

  const selfPlayPanelRef = useRef<SelfPlayPanelHandle | null>(null);

  // Prevents double-fire of the opponent move under React Strict Mode.
  const opponentMoveInFlight = useRef(false);
  const recordedRef = useRef<GameSession | null>(null);
  // Tracks which finished session has already kicked off auto-self-play.
  // Distinct from `recordedRef` because the auto-batch is gated on
  // both the save having been initiated AND the user's preference.
  const autoBatchedRef = useRef<GameSession | null>(null);

  // Lazy-loaded third-party engines. We hold onto them once spun up so
  // toggling the opponent dropdown back to one we already initialised
  // doesn't pay the load cost twice.
  const maiaClientRef = useRef<MaiaClient | null>(null);
  const sfRef = useRef<StockfishEngine | null>(null);

  // Tear down third-party engines on unmount. Switching opponent kinds
  // mid-session deliberately keeps them around — the user may switch
  // back in the next game.
  useEffect(() => {
    return () => {
      maiaClientRef.current?.dispose();
      maiaClientRef.current = null;
      sfRef.current?.dispose();
      sfRef.current = null;
    };
  }, []);

  const applyStartingState = useCallback((state: GameState) => {
    opponentMoveInFlight.current = false;
    recordedRef.current = null;
    setOpponentThinking(false);
    setSession(sessionFromState(state));
    setLastSearch(null);
    setLastMaiaReadout(null);
    setError(null);
  }, []);

  // Lazily spin up the Maia worker. Cheap to call repeatedly: returns
  // the existing client if one is already ready, only kicks off the
  // 40 MiB blob fetch on the first call.
  const ensureMaia = useCallback(async (): Promise<MaiaClient> => {
    if (maiaClientRef.current) return maiaClientRef.current;
    setMaiaStatus({ phase: 'loading', progress: null });
    try {
      const { blob, manifest } = await fetchMaiaWeights({
        source: DEFAULT_MAIA_SOURCE,
        storage,
        onProgress: p => setMaiaStatus({ phase: 'loading', progress: p }),
      });
      const client = await MaiaClient.create(blob, manifest);
      maiaClientRef.current = client;
      setMaiaStatus({ phase: 'ready' });
      return client;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMaiaStatus({ phase: 'error', message });
      throw err;
    }
  }, [storage]);

  // Lazily spin up Stockfish. The lite-single-threaded build is ~7 MiB
  // so the load is fast but still worth deferring until the user picks
  // this opponent kind. Skill / movetime changes mid-session require
  // tearing down the engine and recreating it with the new config.
  const ensureStockfish = useCallback(async (): Promise<StockfishEngine> => {
    if (sfRef.current) return sfRef.current;
    setSfStatus({ phase: 'loading', progress: null });
    try {
      const engine = await StockfishEngine.create({
        skill: sfSkill,
        movetimeMs: sfMovetimeMs,
      });
      sfRef.current = engine;
      setSfStatus({ phase: 'ready' });
      return engine;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSfStatus({ phase: 'error', message });
      throw err;
    }
  }, [sfSkill, sfMovetimeMs]);

  // When the user nudges Stockfish skill or movetime, drop the existing
  // engine so the next move recreates it with the new config. The
  // Stockfish UCI engine accepts `setoption` mid-life, but the wrapper
  // bakes config in at construction — recreating is the simplest path.
  useEffect(() => {
    if (sfRef.current === null) return;
    sfRef.current.dispose();
    sfRef.current = null;
    setSfStatus({ phase: 'idle' });
  }, [sfSkill, sfMovetimeMs]);

  // Builds the starting state for `id` without sampling — used when the
  // color picker changes mid-game and we want to keep the current puzzle
  // / preset rather than picking a new one.
  const startingStateForCurrent = useCallback((): GameState | null => {
    if (setupId === 'standard') return initialState();
    if (setupId === 'custom') {
      const trimmed = customFen.trim();
      if (trimmed === '') return null;
      try { return fromFen(trimmed); } catch { return null; }
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(setupId);
    if (preset) return fromFen(preset.fen);
    if (currentPuzzle) return fromFen(currentPuzzle.fen);
    return initialState();
  }, [setupId, customFen, currentPuzzle]);

  const pickColor = useCallback((preference: ColorPreference): PlayerColor =>
    preference === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : preference,
  []);

  const startNewGame = useCallback(() => {
    if (setupId === 'standard') {
      setCurrentPuzzle(null);
      setPlayerColor(pickColor(colorPreference));
      applyStartingState(initialState());
      return;
    }
    if (setupId === 'custom') {
      const trimmed = customFen.trim();
      if (trimmed === '') return;
      try {
        applyStartingState(fromFen(trimmed));
      } catch (err) {
        setCustomFenError(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`);
      }
      return;
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(setupId);
    if (preset) {
      setCurrentPuzzle(null);
      applyStartingState(fromFen(preset.fen));
      return;
    }
    const cat = LICHESS_CATEGORIES_BY_ID.get(setupId);
    if (cat) {
      const puzzle = randomPuzzle(cat);
      setCurrentPuzzle(puzzle);
      const state = fromFen(puzzle.fen);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    }
  }, [setupId, customFen, colorPreference, applyStartingState, pickColor]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadOpponentFromStorage(storage),
      loadPlayerImitatorFromStorage(storage),
    ])
      .then(([loadedOpponent, loadedImitator]) => {
        if (cancelled) return;
        setOpponent(loadedOpponent);
        setImitator(loadedImitator);
      })
      .catch(err => {
        if (!cancelled) setError(`Failed to load opponent: ${err instanceof Error ? err.message : String(err)}`);
      });
    return () => { cancelled = true; };
  }, [storage]);

  // Load the auto-self-play preference lazily — if it isn't present
  // yet (first-ever visit) we keep the default of `true`.
  useEffect(() => {
    let cancelled = false;
    getJson<{ enabled: boolean }>(storage, AUTO_SELFPLAY_PREF_KEY)
      .then(saved => {
        if (cancelled || saved === null) return;
        setAutoSelfPlay(saved.enabled);
      })
      .catch(() => { /* preference missing or corrupt — keep default */ });
    return () => { cancelled = true; };
  }, [storage]);

  const handleAutoSelfPlayChange = useCallback((enabled: boolean) => {
    setAutoSelfPlay(enabled);
    void putJson(storage, AUTO_SELFPLAY_PREF_KEY, { enabled });
  }, [storage]);

  useEffect(() => {
    let cancelled = false;
    getJson<{ enabled: boolean }>(storage, PREVIEW_TRAINING_PREF_KEY)
      .then(saved => {
        if (cancelled || saved === null) return;
        setPreviewTraining(saved.enabled);
      })
      .catch(() => { /* preference missing or corrupt — keep default */ });
    return () => { cancelled = true; };
  }, [storage]);

  const handlePreviewTrainingChange = useCallback((enabled: boolean) => {
    setPreviewTraining(enabled);
    void putJson(storage, PREVIEW_TRAINING_PREF_KEY, { enabled });
  }, [storage]);

  // Hydrate cumulative training stats from storage. Failures fall back
  // to zeros — the worst case is a fresh-looking counter, not data
  // corruption, since stats are independent of weights.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadNetStats(storage, OPPONENT_TRAIN_STATS_KEY),
      loadNetStats(storage, IMITATOR_TRAIN_STATS_KEY),
    ])
      .then(([opp, imi]) => {
        if (cancelled) return;
        setOpponentStats(opp);
        setImitatorStats(imi);
      })
      .catch(() => { /* keep defaults */ });
    return () => { cancelled = true; };
  }, [storage]);

  /**
   * Record a finished training pass. Called by each training-side panel
   * after the worker reports `done`. `gradientSteps` is the worker's
   * `finalStep` / `imitatorFinalStep`; `batchSize` is whatever the panel
   * configured the worker to use. Passes with zero steps are no-ops so
   * panels can call this unconditionally on `done`.
   */
  const recordTrainingPass = useCallback((
    net: 'opponent' | 'imitator',
    delta: { gradientSteps: number; batchSize: number },
  ) => {
    if (delta.gradientSteps <= 0) return;
    const bufferSize =
      net === 'opponent'
        ? opponentRef.current?.replayBuffer.size() ?? 0
        : imitator?.replayBuffer.size() ?? 0;
    const setter = net === 'opponent' ? setOpponentStats : setImitatorStats;
    const storageKey = net === 'opponent' ? OPPONENT_TRAIN_STATS_KEY : IMITATOR_TRAIN_STATS_KEY;
    setter(prev => {
      const next = addTrainingPass(prev, delta);
      void saveNetStats(storage, storageKey, next);
      const epochs = bufferSize > 0 ? next.examplesSeen / bufferSize : 0;
      const label = net === 'opponent' ? 'Opponent' : 'Imitator';
      trainingLog.append({
        source: 'training',
        kind: 'info',
        message:
          `${label} lifetime: ${next.gradientSteps.toLocaleString()} step(s), ` +
          `${next.examplesSeen.toLocaleString()} example(s) seen ` +
          `(~${epochs.toFixed(2)} effective epoch(s) over the current ` +
          `${bufferSize.toLocaleString()}-example buffer).`,
      });
      return next;
    });
  }, [storage, imitator, trainingLog]);

  useEffect(() => {
    if (opponent) opponent.setSimulations(simulations);
  }, [opponent, simulations]);

  useEffect(() => {
    if (opponent) opponent.setCPuct(cPuct);
  }, [opponent, cPuct]);

  // Restore the last in-progress (or just-finished) game from storage.
  // Runs once on mount. If the saved session is already terminal we
  // park its reference in `recordedRef` so the game-over effect below
  // doesn't double-record it into the opponent's replay buffer.
  useEffect(() => {
    let cancelled = false;
    loadPersistedSession(storage)
      .then(restored => {
        if (cancelled) return;
        if (restored !== null) {
          const newSession: GameSession = {
            state: restored.state,
            history: restored.history,
            over: restored.over,
            finalOutcome: restored.finalOutcome,
            startingFen: restored.startingFen,
            // The persisted-session blob stores moves but not the
            // last-move highlight directly; recover it from the tail
            // of the history. Empty histories restore as null.
            lastMove: restored.history.length > 0
              ? restored.history[restored.history.length - 1].move
              : null,
          };
          setSession(newSession);
          setPlayerColor(restored.playerColor);
          setColorPreference(restored.colorPreference);
          setSetupId(restored.setupId);
          setCustomFen(restored.customFen);
          setCurrentPuzzle(restored.currentPuzzle);
          setOpponentKind(restored.opponentKind);
          if (restored.over) {
            // The game already ended — either it was recorded in the
            // earlier session, or the tab was closed before the save
            // landed. Either way, don't re-run training on it.
            recordedRef.current = newSession;
          }
        }
        setSessionLoaded(true);
      })
      .catch(() => {
        if (!cancelled) setSessionLoaded(true);
      });
    return () => { cancelled = true; };
  }, [storage]);

  // Mirror the current session + setup back into storage. Gated on
  // `sessionLoaded` so we don't clobber a saved session before reading
  // it. Writes are fire-and-forget; on failure the worst case is that
  // the next reload starts from an earlier checkpoint.
  useEffect(() => {
    if (!sessionLoaded) return;
    const data: PersistedSession = {
      playerColor,
      colorPreference,
      setupId,
      customFen,
      currentPuzzle,
      startingFen: session.startingFen,
      moves: session.history.map(r => r.move),
      opponentKind,
    };
    void savePersistedSession(storage, data);
  }, [session, playerColor, colorPreference, setupId, customFen, currentPuzzle, sessionLoaded, storage, opponentKind]);

  useEffect(() => {
    if (opponent === null) return;
    if (session.over) return;
    if (opponentMoveInFlight.current) return;

    const currentSide = sideToMove(session.state);
    if (currentSide === playerColor) return;

    // The imitator + Maia/Stockfish paths depend on resources that may
    // not be loaded yet. Skip silently rather than erroring — the
    // resource-load effects flip state on completion, which re-runs
    // this effect.
    if (opponentKind === 'imitator' && imitator === null) return;

    opponentMoveInFlight.current = true;
    setOpponentThinking(true);

    const capturedState = session.state;
    const fen = toFen(capturedState);

    // Dispatch to the right backend. Each branch resolves to a UCI
    // move plus optional MCTS/Maia readouts for the inspector panels.
    type MoveOutcome = {
      move: Move;
      mctsResult: SearchResult | null;
      maiaReadout: { value: number; topMoves: Array<{ move: Move; prob: number }> } | null;
    };
    let moveSelection: Promise<MoveOutcome>;
    if (opponentKind === 'opponent') {
      // MCTS branch keeps the rich SearchResult shape so InspectorPanel
      // / LastMoveMcts can read visit counts. Policy-head branch
      // intentionally clears `lastSearch` to a null sentinel.
      moveSelection = useMcts
        ? opponent.selectMove(capturedState).then(r => ({
            move: r.move,
            mctsResult: { ...r, bestMove: r.move, mateInOneMoves: [] as Move[] },
            maiaReadout: null,
          }))
        : selectMoveFromNet(opponent.net, capturedState, { temperature: 0 }).then(r => ({
            move: r.move,
            mctsResult: null,
            maiaReadout: null,
          }));
    } else if (opponentKind === 'imitator') {
      moveSelection = imitator!.selectMove(capturedState, {
        search: useMcts ? { simulations, cPuct } : undefined,
        temperature: 0,
      }).then(r => ({ move: r.move, mctsResult: null, maiaReadout: null }));
    } else if (opponentKind === 'stockfish') {
      moveSelection = ensureStockfish().then(sf => sf.bestMove(fen)).then(raw => {
        // Stockfish may return a queen-promotion as `e7e8q`; that
        // round-trips through our UCI move format. Validate against
        // the rules module — if SF emits something illegal under our
        // ruleset we fall back to the first legal move rather than
        // hard-erroring out of an in-progress game.
        const legal = legalMoves(capturedState);
        const move: Move = legal.includes(raw) ? raw : legal[0];
        return { move, mctsResult: null, maiaReadout: null };
      });
    } else {
      // 'maia-2'
      const elo = MAIA_ELO_BUCKETS[maiaEloBucket].midpoint;
      // Sample from Maia's policy distribution rather than taking the
      // greedy argmax. Maia's top-1 collapses on under-represented
      // opening branches (e.g. after 1.e4 Nf6 2.Nf3 it deterministically
      // retreats Ng8 even at bucket 10 — verified against the PyTorch
      // reference), and with temperature=0 the same wrong move recurs
      // every time the position recurs, producing visible Nf6↔Ng8
      // oscillation. temperature=1 = honest sampling from the policy
      // softmax, which still picks the top-1 most of the time but
      // breaks the deterministic loop on close calls.
      const maiaCall = (client: MaiaClient) => maiaUseMcts
        ? client.searchMove(fen, elo, {
            oppoElo: elo,
            simulations: maiaSimulations,
            cPuct,
          })
        : client.selectMove(fen, elo, { oppoElo: elo, temperature: 1 });
      moveSelection = ensureMaia().then(maiaCall).then(({ move, value, moveProbs }) => {
        const top = Array.from(moveProbs.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([m, p]) => ({ move: m, prob: p }));
        return {
          move,
          mctsResult: null,
          maiaReadout: { value, topMoves: top },
        };
      });
    }

    moveSelection.then(({ move, mctsResult, maiaReadout }) => {
      opponentMoveInFlight.current = false;

      // When MCTS isn't applicable (or the engine doesn't expose visit
      // counts) `mctsResult` is null; LastMoveMcts / InspectorPanel
      // render their "no search yet" placeholder in that case.
      setLastSearch(mctsResult);
      // Maia readout is parallel to the MCTS panel — only set when
      // Maia produced this move, cleared otherwise.
      setLastMaiaReadout(maiaReadout ? { move, ...maiaReadout } : null);

      const record: GameMoveRecord = {
        features: encodeState(capturedState),
        move,
        sideToMove: sideToMove(capturedState),
        legalMoveIndices: legalMoveIndicesFor(capturedState),
        mateMoveIndices: mateInOneIndicesFor(capturedState),
      };

      const nextState = applyMove(capturedState, move);
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';

      setSession(prev => {
        if (prev.state !== capturedState) return prev;
        return {
          state: nextState,
          history: [...prev.history, record],
          over: gameOver,
          finalOutcome: gameOver ? o : null,
          startingFen: prev.startingFen,
          lastMove: move,
        };
      });

      setOpponentThinking(false);
    }).catch(err => {
      opponentMoveInFlight.current = false;
      setOpponentThinking(false);
      setError(`Opponent move failed: ${err instanceof Error ? err.message : String(err)}`);
    });
  }, [
    opponent, imitator, session, playerColor, useMcts, simulations, cPuct,
    opponentKind, maiaEloBucket, maiaUseMcts, maiaSimulations,
    ensureMaia, ensureStockfish,
  ]);

  useEffect(() => {
    if (!session.over) return;
    if (!opponent) return;
    if (!session.finalOutcome || session.finalOutcome === 'ongoing') return;
    if (recordedRef.current === session) return;

    recordedRef.current = session;

    // The opponent NN's replay buffer only receives games where the
    // opponent NN actually played. Games against the imitator,
    // Stockfish, or Maia leave it alone — those engines' moves aren't
    // a reasonable training signal for the opponent's own policy.
    const recordOpponentGame = opponentKind === 'opponent';
    if (recordOpponentGame) {
      opponent.recordGameResult(session.history, session.finalOutcome);
    }

    // The player-imitator trains on the user's moves from EVERY
    // session, regardless of which engine was on the other side. The
    // imitator's job is to model the user, and the user's moves are
    // the user's moves whether they came against the opponent NN, the
    // imitator itself, Stockfish, or Maia.
    if (imitator) {
      const userMoves = session.history.filter(r => r.sideToMove === playerColor);
      imitator.recordUserMoves(userMoves, session.finalOutcome);
    }

    const savePromises: Promise<void>[] = [];
    if (recordOpponentGame) savePromises.push(opponent.saveTo(storage));
    if (imitator) savePromises.push(imitator.saveTo(storage));
    const sessionSnapshot = session;
    Promise.all(savePromises)
      .then(() => {
        // Only kick the auto-batch AFTER the post-game save lands: the
        // worker reads weights+replay we just persisted via
        // opponent.saveTo(...) through the serialized handles passed in,
        // so we want the in-memory `opponent` to reflect every recorded
        // example before we serialize it.
        if (!autoSelfPlay) return;
        if (autoBatchedRef.current === sessionSnapshot) return;
        const handle = selfPlayPanelRef.current;
        if (!handle || handle.isRunning) return;
        autoBatchedRef.current = sessionSnapshot;
        // Fire and forget: the user may start a new game while this
        // runs. onOpponentRefresh re-reads from storage when it finishes,
        // and that doesn't disturb the in-progress game's session state.
        void handle.triggerAutoBatch();
      })
      .catch(err => {
        setError(`Failed to save after game: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [session, opponent, imitator, playerColor, storage, autoSelfPlay, opponentKind]);

  const handleMove = useCallback((nextState: GameState, move: Move) => {
    if (!opponentRef.current) return;
    setSession(prev => {
      if (prev.over) return prev;
      const record: GameMoveRecord = {
        features: encodeState(prev.state),
        move,
        sideToMove: sideToMove(prev.state),
        legalMoveIndices: legalMoveIndicesFor(prev.state),
        mateMoveIndices: mateInOneIndicesFor(prev.state),
      };
      const o = outcome(nextState);
      const gameOver = o !== 'ongoing';
      return {
        state: nextState,
        history: [...prev.history, record],
        over: gameOver,
        finalOutcome: gameOver ? o : null,
        startingFen: prev.startingFen,
        lastMove: move,
      };
    });
  }, []);

  const handleColorChange = useCallback((preference: ColorPreference) => {
    setColorPreference(preference);
    setPlayerColor(pickColor(preference));
    const starting = startingStateForCurrent();
    if (starting) applyStartingState(starting);
  }, [startingStateForCurrent, applyStartingState, pickColor]);

  const handleSetupChange = useCallback((id: SetupId) => {
    setSetupId(id);
    setCustomFenError(null);
    if (id === 'standard') {
      setCurrentPuzzle(null);
      applyStartingState(initialState());
      return;
    }
    if (id === 'custom') {
      setCurrentPuzzle(null);
      return; // wait for Apply
    }
    const preset = ENDGAME_PRESETS_BY_ID.get(id);
    if (preset) {
      setCurrentPuzzle(null);
      setPlayerColor(preset.winningSide);
      applyStartingState(fromFen(preset.fen));
      return;
    }
    const cat = LICHESS_CATEGORIES_BY_ID.get(id);
    if (cat) {
      const puzzle = randomPuzzle(cat);
      setCurrentPuzzle(puzzle);
      const state = fromFen(puzzle.fen);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    }
  }, [applyStartingState]);

  const handleApplyCustomFen = useCallback(() => {
    const trimmed = customFen.trim();
    if (trimmed === '') {
      setCustomFenError('Enter a FEN string.');
      return;
    }
    try {
      const state = fromFen(trimmed);
      setCustomFenError(null);
      setCurrentPuzzle(null);
      setPlayerColor(sideToMove(state));
      applyStartingState(state);
    } catch (err) {
      setCustomFenError(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [customFen, applyStartingState]);

  const handleOpponentRefresh = useCallback(async () => {
    try {
      const [reloadedOpponent, reloadedImitator] = await Promise.all([
        loadOpponentFromStorage(storage),
        loadPlayerImitatorFromStorage(storage),
      ]);
      setOpponent(reloadedOpponent);
      setImitator(reloadedImitator);
      setOpponentVersion(v => v + 1);
    } catch (err) {
      setError(`Failed to reload opponent: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage]);

  // Danger zone: delete one of {weights, replay} for one of {opponent,
  // imitator} and reload. Deleting weights+meta forces `loadOpponent` /
  // `loadPlayerImitator` to fall through their version checks and
  // return a fresh randomly-initialized net. Deleting replay drops the
  // recorded training examples without touching the weights. Both
  // operations are confirmed with a native dialog before they run.
  const handleClearOpponentHistory = useCallback(async () => {
    if (!window.confirm(
      'Clear ALL recorded opponent training examples? This cannot be undone. '
      + 'The opponent\'s weights will be kept.',
    )) return;
    try {
      await storage.delete(OPPONENT_STORAGE_KEYS.REPLAY);
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to clear opponent history: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleClearImitatorHistory = useCallback(async () => {
    if (!window.confirm(
      'Clear ALL recorded user moves (the imitator\'s training examples)? '
      + 'This cannot be undone. The imitator\'s weights will be kept.',
    )) return;
    try {
      await storage.delete(IMITATOR_STORAGE_KEYS.REPLAY);
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to clear imitator history: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleResetOpponentModel = useCallback(async () => {
    if (!window.confirm(
      'Reset the opponent model to a freshly-initialized network? '
      + 'This deletes the trained weights and the recorded training examples. '
      + 'This cannot be undone.',
    )) return;
    try {
      await Promise.all([
        storage.delete(OPPONENT_STORAGE_KEYS.WEIGHTS),
        storage.delete(OPPONENT_STORAGE_KEYS.META),
        storage.delete(OPPONENT_STORAGE_KEYS.REPLAY),
        storage.delete(OPPONENT_TRAIN_STATS_KEY),
      ]);
      setOpponentStats(emptyNetStats());
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to reset opponent model: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  const handleResetImitatorModel = useCallback(async () => {
    if (!window.confirm(
      'Reset the imitator model to a freshly-initialized network? '
      + 'This deletes the trained weights and the recorded user moves. '
      + 'This cannot be undone.',
    )) return;
    try {
      await Promise.all([
        storage.delete(IMITATOR_STORAGE_KEYS.WEIGHTS),
        storage.delete(IMITATOR_STORAGE_KEYS.META),
        storage.delete(IMITATOR_STORAGE_KEYS.REPLAY),
        storage.delete(IMITATOR_TRAIN_STATS_KEY),
      ]);
      setImitatorStats(emptyNetStats());
      await handleOpponentRefresh();
    } catch (err) {
      setError(`Failed to reset imitator model: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [storage, handleOpponentRefresh]);

  if (opponent === null && error === null) {
    return <div style={{ fontFamily: 'sans-serif', padding: '16px' }}>Loading opponent…</div>;
  }

  if (error && opponent === null) {
    return (
      <div style={{ fontFamily: 'sans-serif', padding: '16px', color: '#c00' }}>
        {error}
      </div>
    );
  }

  const humanIsWhite = playerColor === 'w';
  const currentSide = sideToMove(session.state);
  const isHumanTurn = currentSide === playerColor && !session.over;
  const interactive = isHumanTurn && !opponentThinking;

  const o = session.finalOutcome;
  const opponentNoun = opponentKind === 'maia-2' ? 'Maia'
    : opponentKind === 'stockfish' ? 'Stockfish'
    : opponentKind === 'imitator' ? 'Imitator'
    : 'Opponent';
  let outcomeText = '';
  if (o === 'white-wins') outcomeText = playerColor === 'w' ? 'You win!' : `${opponentNoun} wins!`;
  else if (o === 'black-wins') outcomeText = playerColor === 'b' ? 'You win!' : `${opponentNoun} wins!`;
  else if (o === 'draw') outcomeText = 'Draw!';

  const activePreset = ENDGAME_PRESETS_BY_ID.get(setupId);
  const activeLichessCat = LICHESS_CATEGORIES_BY_ID.get(setupId);

  return (
    <div style={{ fontFamily: 'sans-serif' }}>
      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>Setup:</strong>
        <select
          value={setupId}
          onChange={e => handleSetupChange(e.target.value)}
          style={{ padding: '3px 6px' }}
        >
          <option value="standard">Standard game</option>
          <optgroup label="Endgame training">
            {ENDGAME_PRESETS.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </optgroup>
          {LICHESS_CATEGORIES.length > 0 && (
            <optgroup label="Lichess endgames (random)">
              {LICHESS_CATEGORIES.map(c => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </optgroup>
          )}
          <option value="custom">Custom FEN…</option>
        </select>
        {(activeLichessCat || activePreset || setupId === 'standard') && (
          <button
            type="button"
            onClick={startNewGame}
            style={{ padding: '3px 10px', cursor: 'pointer' }}
            title={activeLichessCat ? 'Pick a different random puzzle' : 'Restart this position'}
          >
            {activeLichessCat ? 'Next puzzle' : 'Restart'}
          </button>
        )}
      </div>

      {activePreset && (
        <div
          style={{
            marginBottom: '12px',
            padding: '8px 12px',
            background: '#fffbe6',
            border: '1px solid #e6d97a',
            borderRadius: '6px',
            fontSize: '0.9em',
            color: '#444',
          }}
        >
          <div style={{ marginBottom: '4px' }}>{activePreset.description}</div>
          <div style={{ color: '#666' }}>
            You're set to play{' '}
            <strong>{activePreset.winningSide === 'w' ? 'White' : 'Black'}</strong>{' '}
            (the winning side) — that side's moves teach the opponent the mating pattern.
          </div>
        </div>
      )}

      {activeLichessCat && currentPuzzle && (
        <div
          style={{
            marginBottom: '12px',
            padding: '8px 12px',
            background: '#fffbe6',
            border: '1px solid #e6d97a',
            borderRadius: '6px',
            fontSize: '0.9em',
            color: '#444',
          }}
        >
          <div style={{ marginBottom: '4px' }}>
            <strong>{activeLichessCat.label}</strong>{' '}
            — puzzle{' '}
            <a href={lichessPuzzleUrl(currentPuzzle.id)} target="_blank" rel="noopener noreferrer">
              {currentPuzzle.id}
            </a>{' '}
            (rating {currentPuzzle.rating})
          </div>
          <div style={{ color: '#666', fontSize: '0.85em' }}>
            Themes: {currentPuzzle.themes}. Source: Lichess puzzle database (CC0).
          </div>
        </div>
      )}

      {setupId === 'custom' && (
        <div style={{ marginBottom: '12px', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            value={customFen}
            onChange={e => setCustomFen(e.target.value)}
            placeholder="paste a FEN…"
            style={{ flex: '1 1 360px', padding: '4px 8px', fontFamily: 'monospace', fontSize: '0.9em' }}
          />
          <button type="button" onClick={handleApplyCustomFen} style={{ padding: '4px 12px', cursor: 'pointer' }}>
            Apply
          </button>
          {customFenError && (
            <span style={{ color: '#c00', fontSize: '0.85em' }}>{customFenError}</span>
          )}
        </div>
      )}

      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>Opponent:</strong>
        <select
          value={opponentKind}
          onChange={e => setOpponentKind(e.target.value as OpponentKind)}
          style={{ padding: '3px 6px' }}
        >
          {(Object.keys(OPPONENT_LABELS) as OpponentKind[]).map(k => (
            <option key={k} value={k}>{OPPONENT_LABELS[k]}</option>
          ))}
        </select>
        {opponentKind !== 'opponent' && (
          <span style={{ fontSize: '0.85em', color: '#666' }}>
            Your moves still train the imitator; the opponent NN&apos;s replay buffer is untouched.
          </span>
        )}
      </div>

      <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center' }}>
        <strong>You play as:</strong>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="w"
            checked={colorPreference === 'w'}
            onChange={() => handleColorChange('w')}
          />
          {' '}White
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="b"
            checked={colorPreference === 'b'}
            onChange={() => handleColorChange('b')}
          />
          {' '}Black
        </label>
        <label style={{ cursor: 'pointer' }}>
          <input
            type="radio"
            name="playerColor"
            value="random"
            checked={colorPreference === 'random'}
            onChange={() => handleColorChange('random')}
          />
          {' '}Random
        </label>
      </div>

      {(opponentKind === 'opponent' || opponentKind === 'imitator') && (
        <div style={{ marginBottom: '12px', display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <strong>Strength:</strong>
          <MctsToggle
            useMcts={useMcts}
            onUseMctsChange={setUseMcts}
            simulations={simulations}
            onSimulationsChange={setSimulations}
            minSims={16}
            maxSims={400}
            step={16}
            simsSuffix="sims/move"
          />
          <span style={{ fontSize: '0.9em', color: '#888' }}>|</span>
          <label style={{ fontSize: '0.9em', color: useMcts ? '#555' : '#aaa', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              explore (cPuct):
              <span
                role="button"
                aria-label="Help for cPuct"
                style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: '15px', height: '15px', borderRadius: '50%',
                  background: '#bbb', color: '#fff', fontSize: '0.7em', fontWeight: 'bold',
                  cursor: 'pointer', userSelect: 'none', position: 'relative', flexShrink: 0,
                }}
                onMouseEnter={() => setCPuctTipOpen(true)}
                onMouseLeave={() => setCPuctTipOpen(false)}
                onClick={() => setCPuctTipOpen(o => !o)}
              >
                ?
                {cPuctTipOpen && (
                  <span style={{
                    position: 'absolute', bottom: '120%', left: '50%',
                    transform: 'translateX(-50%)',
                    background: '#333', color: '#fff', fontSize: '0.95em',
                    fontWeight: 'normal', lineHeight: '1.4',
                    padding: '7px 10px', borderRadius: '5px',
                    width: '240px', whiteSpace: 'normal',
                    pointerEvents: 'none', zIndex: 10,
                    boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
                  }}>
                    PUCT exploration constant (AlphaZero default: 1.5).
                    <br /><br />
                    <strong>Lower</strong> → simulations concentrate on the highest-value line, searching deeper but narrower.
                    <br /><br />
                    <strong>Higher</strong> → simulations spread across more candidate moves, shallower but broader.
                  </span>
                )}
              </span>
            </span>
            <input
              type="range"
              min={0.5}
              max={4.0}
              step={0.25}
              value={cPuct}
              onChange={e => setCPuct(Number(e.target.value))}
              disabled={!useMcts}
              style={{ width: '100px', opacity: useMcts ? 1 : 0.4 }}
            />
            <span style={{ fontFamily: 'monospace' }}>{cPuct.toFixed(2)}</span>
          </label>
        </div>
      )}

      {opponentKind === 'stockfish' && (
        <div style={{ marginBottom: '12px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <strong>Stockfish:</strong>
          <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            skill
            <input
              type="range"
              min={0}
              max={20}
              value={sfSkill}
              onChange={e => setSfSkill(Number(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ fontFamily: 'monospace', minWidth: '2em' }}>{sfSkill}</span>
          </label>
          <label style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            movetime (ms)
            <input
              type="number"
              min={10}
              max={10000}
              value={sfMovetimeMs}
              onChange={e => setSfMovetimeMs(Number(e.target.value))}
              style={{ width: '80px' }}
            />
          </label>
          {sfStatus.phase === 'loading' && (
            <span style={{ fontSize: '0.85em', color: '#666' }}>Loading Stockfish…</span>
          )}
          {sfStatus.phase === 'error' && (
            <span style={{ fontSize: '0.85em', color: '#c00' }}>Stockfish error: {sfStatus.message}</span>
          )}
        </div>
      )}

      {opponentKind === 'maia-2' && (
        <div style={{ marginBottom: '12px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
          <strong>Maia strength:</strong>
          <select
            value={maiaEloBucket}
            onChange={e => setMaiaEloBucket(Number(e.target.value))}
            style={{ padding: '3px 6px' }}
          >
            {MAIA_ELO_BUCKETS.map(b => (
              <option key={b.id} value={b.id}>
                {b.label} (≈ {b.midpoint})
              </option>
            ))}
          </select>
          <MctsToggle
            useMcts={maiaUseMcts}
            onUseMctsChange={setMaiaUseMcts}
            simulations={maiaSimulations}
            onSimulationsChange={setMaiaSimulations}
            minSims={1}
            maxSims={16}
          />
          <div style={{ flexBasis: '100%' }} />
          <span style={{ fontSize: '0.8em', color: '#666' }}>
            {maiaUseMcts
              ? 'MCTS on Maia is slow on CPU/WASM — each sim is a full forward pass (~6-10s).'
              : 'MCTS off — Maia samples from its policy distribution (the deterministic argmax loops on under-represented opening branches).'}
          </span>
          {maiaStatus.phase === 'loading' && (
            <div style={{ flexBasis: '100%', fontSize: '0.85em', color: '#666' }}>
              Loading Maia-2 weights…{' '}
              {maiaStatus.progress && (
                <>
                  {fmtBytes(maiaStatus.progress.loaded)}
                  {maiaStatus.progress.total > 0 && (
                    <> / {fmtBytes(maiaStatus.progress.total)}
                      {' '}
                      ({((maiaStatus.progress.loaded / maiaStatus.progress.total) * 100).toFixed(1)}%)
                    </>
                  )}
                </>
              )}
            </div>
          )}
          {maiaStatus.phase === 'error' && (
            <div style={{ flexBasis: '100%', fontSize: '0.85em', color: '#c00' }}>
              Failed to load Maia: {maiaStatus.message}
            </div>
          )}
        </div>
      )}

      <div style={{ marginBottom: '8px', minHeight: '24px' }}>
        {opponentThinking && (
          <span style={{ color: '#666', fontStyle: 'italic' }}>
            {opponentKind === 'maia-2' ? 'Maia thinking…'
              : opponentKind === 'stockfish' ? 'Stockfish thinking…'
              : opponentKind === 'imitator' ? 'Imitator thinking…'
              : 'Opponent thinking…'}
          </span>
        )}
        {!opponentThinking && !session.over && (
          <span>
            {isHumanTurn ? 'Your turn' : `${opponentNoun} to move`}
            {' — '}{currentSide === 'w' ? 'White' : 'Black'} to move
          </span>
        )}
      </div>

      <Board
        state={session.state}
        interactive={interactive}
        onMove={handleMove}
        boardOrientation={humanIsWhite ? 'white' : 'black'}
        lastMove={session.lastMove}
      />

      {opponentKind === 'opponent' && <LastMoveMcts lastSearch={lastSearch} />}

      {opponentKind === 'maia-2' && lastMaiaReadout && (
        <div style={{
          marginTop: '12px',
          padding: '8px 12px',
          background: '#f5f5f5',
          border: '1px solid #ddd',
          borderRadius: '6px',
          fontSize: '0.85em',
          color: '#444',
        }}>
          <strong>Last Maia move:</strong> {lastMaiaReadout.move}{' '}
          <span style={{ color: '#666' }}>
            (value head: {lastMaiaReadout.value.toFixed(3)})
          </span>
          <br />
          <strong>Maia&apos;s top candidates:</strong>{' '}
          {lastMaiaReadout.topMoves
            .map(m => `${m.move} (${(m.prob * 100).toFixed(1)}%)`)
            .join(', ')}
        </div>
      )}

      {session.over && (
        <div
          style={{
            marginTop: '16px',
            padding: '12px 16px',
            background: '#f0f4ff',
            border: '1px solid #aac',
            borderRadius: '6px',
            color: '#1a1a1a',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
          }}
        >
          <strong style={{ fontSize: '1.1em' }}>{outcomeText}</strong>
          <button
            type="button"
            onClick={startNewGame}
            style={{ padding: '6px 16px', cursor: 'pointer' }}
          >
            {activeLichessCat ? 'Next puzzle' : 'New game'}
          </button>
        </div>
      )}

      {error && (
        <div style={{ marginTop: '8px', color: '#c00', fontSize: '0.9em' }}>
          {error}
        </div>
      )}

      {opponent && (
        <TrainingStatsPanel
          opponentStats={opponentStats}
          imitatorStats={imitatorStats}
          opponentBufferSize={opponent.replayBuffer.size()}
          imitatorBufferSize={imitator?.replayBuffer.size() ?? 0}
          opponentBufferCapacity={opponent.replayBuffer.capacity()}
          imitatorBufferCapacity={imitator?.replayBuffer.capacity() ?? opponent.replayBuffer.capacity()}
        />
      )}

      {opponent && (
        <div style={{ marginTop: '24px' }}>
          <TrainingPanel
            opponent={opponent}
            imitator={imitator}
            storage={storage}
            onOpponentRefresh={handleOpponentRefresh}
            onLog={trainingLog.append}
            onTrainingPass={recordTrainingPass}
            key={opponentVersion}
          />
        </div>
      )}

      <SelfPlayPanel
        ref={selfPlayPanelRef}
        opponent={opponent}
        imitator={imitator}
        storage={storage}
        onOpponentRefresh={handleOpponentRefresh}
        autoEnabled={autoSelfPlay}
        onAutoEnabledChange={handleAutoSelfPlayChange}
        previewEnabled={previewTraining}
        onPreviewEnabledChange={handlePreviewTrainingChange}
        simulations={simulations}
        cPuct={cPuct}
        onLog={trainingLog.append}
        onTrainingPass={recordTrainingPass}
      />

      <EndgameGymPanel
        opponent={opponent}
        imitator={imitator}
        storage={storage}
        onOpponentRefresh={handleOpponentRefresh}
        previewEnabled={previewTraining}
        onPreviewEnabledChange={handlePreviewTrainingChange}
        simulations={simulations}
        cPuct={cPuct}
        onLog={trainingLog.append}
        onTrainingPass={recordTrainingPass}
      />

      <TrainingLog entries={trainingLog.entries} onClear={trainingLog.clear} />

      {opponentKind === 'opponent' && (
        <InspectorPanel
          opponent={opponent}
          state={session.state}
          lastSearch={lastSearch}
          opponentVersion={opponentVersion}
        />
      )}

      <details
        style={{
          marginTop: '24px',
          border: '1px solid #d99',
          borderRadius: '6px',
          padding: '10px 14px',
          background: '#fff5f5',
          color: '#1a1a1a',
        }}
      >
        <summary style={{ cursor: 'pointer', fontWeight: 'bold', userSelect: 'none', color: '#a33' }}>
          Danger zone
        </summary>
        <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#444' }}>
          Each action is destructive and applies only to your browser's local
          storage. Trained weights and recorded games cannot be recovered
          after deletion.
        </div>
        <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-start' }}>
          <button
            type="button"
            onClick={handleClearOpponentHistory}
            disabled={opponent === null}
            style={{ padding: '5px 14px', cursor: opponent ? 'pointer' : 'not-allowed' }}
          >
            Clear opponent move history
          </button>
          <button
            type="button"
            onClick={handleClearImitatorHistory}
            disabled={imitator === null}
            style={{ padding: '5px 14px', cursor: imitator ? 'pointer' : 'not-allowed' }}
          >
            Clear user (imitator) move history
          </button>
          <button
            type="button"
            onClick={handleResetOpponentModel}
            disabled={opponent === null}
            style={{ padding: '5px 14px', cursor: opponent ? 'pointer' : 'not-allowed', color: '#a33' }}
          >
            Reset opponent model
          </button>
          <button
            type="button"
            onClick={handleResetImitatorModel}
            disabled={imitator === null}
            style={{ padding: '5px 14px', cursor: imitator ? 'pointer' : 'not-allowed', color: '#a33' }}
          >
            Reset imitator model
          </button>
        </div>
      </details>
    </div>
  );
}
