/**
 * stockfish-protocol.ts — thin async wrapper around the Stockfish UCI engine.
 *
 * Why vendored worker instead of `new Worker(new URL('stockfish/...', import.meta.url))`?
 * The `stockfish` npm package's JS files locate their sibling .wasm file by
 * computing a URL relative to their own script URL. When Vite bundles the JS
 * into a hashed chunk that URL relationship breaks, causing the WASM load to
 * fail at runtime. Vendoring both files in public/nn-chess/stockfish/ and
 * loading them via a plain (non-module) importScripts shim preserves the
 * stable relative URL so the .wasm lookup always succeeds.
 *
 * The lite single-threaded variant is used because:
 *  - No SharedArrayBuffer / COOP/COEP headers required (single-threaded WASM).
 *  - ~7 MB vs ~110 MB for the full engine — acceptable load time.
 *  - Still far stronger than our nn-chess agent at any Skill Level.
 *  - The "Skill Level" UCI option weakens it to give the agent a chance.
 */

export interface StockfishConfig {
  /** Skill Level 0–20; default 5. Lower = weaker, giving the agent a chance. */
  skill?: number;
  /** Move time in milliseconds; default 100. Fast games → volume of data. */
  movetimeMs?: number;
  /** Use fixed depth instead of movetime (mutually exclusive). */
  depth?: number;
}

/**
 * Async wrapper around a Stockfish Web Worker speaking UCI.
 *
 * Usage:
 *   const sf = await StockfishEngine.create({ skill: 5, movetimeMs: 100 });
 *   const move = await sf.bestMove(fen);
 *   sf.dispose();
 */
export class StockfishEngine {
  private worker: Worker;
  private config: Required<Omit<StockfishConfig, 'depth'>> & { depth?: number };

  // Queue of pending bestMove promises. Stockfish can only think about one
  // position at a time; we serialize calls by chaining resolvers.
  private queue: Array<{
    fen: string;
    resolve: (uci: string) => void;
    reject: (err: unknown) => void;
  }> = [];
  private busy = false;

  private constructor(
    worker: Worker,
    config: Required<Omit<StockfishConfig, 'depth'>> & { depth?: number },
  ) {
    this.worker = worker;
    this.config = config;

    this.worker.onmessage = (event: MessageEvent<string>) => {
      this._onLine(event.data);
    };
    this.worker.onerror = (event) => {
      // Reject the front-of-queue item if any.
      const item = this.queue[0];
      if (item) {
        this.queue.shift();
        this.busy = false;
        item.reject(new Error(`Stockfish worker error: ${event.message}`));
        this._drain();
      }
    };
  }

  /**
   * Create and initialize a StockfishEngine.
   *
   * Spawns the plain-JS (non-module) worker shim at nn-chess/ui/stockfish-worker.js,
   * which uses importScripts() to load the vendored lite-single-threaded WASM
   * build from /nn-chess/stockfish/. The stockfishBase query parameter tells
   * the shim where to find the vendored files.
   *
   * Sends `uci` and waits for `uciok`, then applies the Skill Level option.
   */
  static async create(config: StockfishConfig = {}): Promise<StockfishEngine> {
    const skill = config.skill ?? 5;
    const movetimeMs = config.movetimeMs ?? 100;
    const depth = config.depth;

    // The worker shim accepts ?stockfishBase= so it can load the vendored
    // files from the correct origin-relative path at runtime.
    const workerUrl = new URL('./stockfish-worker.js', import.meta.url);
    workerUrl.searchParams.set(
      'stockfishBase',
      `${window.location.origin}/nn-chess/stockfish/`,
    );

    // Plain (non-module) worker — required so importScripts() is available.
    const worker = new Worker(workerUrl.toString());

    const engine = new StockfishEngine(worker, { skill, movetimeMs, depth });

    // Wait for UCI handshake.
    await engine._init(skill);

    return engine;
  }

  /**
   * Return the best UCI move for the given FEN position.
   *
   * Calls are queued — only one `go` command is in flight at a time.
   */
  bestMove(fen: string): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      this.queue.push({ fen, resolve, reject });
      this._drain();
    });
  }

  /**
   * Cancel any in-flight search and terminate the worker.
   * After calling dispose(), all pending bestMove promises will never settle.
   */
  dispose(): void {
    this.queue = [];
    this.busy = false;
    try {
      this.worker.postMessage('stop');
    } catch {
      // Ignore — worker may already be gone.
    }
    this.worker.terminate();
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  /** Pending resolve for the `uciok` handshake. */
  private _uciokResolve: (() => void) | null = null;
  /** Pending resolve for the current `bestmove` response. */
  private _bestmoveResolve: ((uci: string) => void) | null = null;

  private _init(skill: number): Promise<void> {
    return new Promise<void>((resolve) => {
      this._uciokResolve = resolve;
      this.worker.postMessage('uci');
    }).then(() => {
      // Apply skill level before any search.
      this.worker.postMessage(`setoption name Skill Level value ${skill}`);
      // Also apply UCI_LimitStrength to enable elo-based limiting (belt-and-suspenders).
      this.worker.postMessage('setoption name UCI_LimitStrength value true');
      this.worker.postMessage('isready');
      // We don't need to wait for readyok — the engine serializes commands.
    });
  }

  private _onLine(line: string): void {
    // UCI handshake.
    if (line === 'uciok' && this._uciokResolve) {
      const resolve = this._uciokResolve;
      this._uciokResolve = null;
      resolve();
      return;
    }

    // Best move response.
    if (line.startsWith('bestmove ') && this._bestmoveResolve) {
      const parts = line.split(' ');
      const uci = parts[1] ?? '0000';
      const resolve = this._bestmoveResolve;
      this._bestmoveResolve = null;
      this.busy = false;
      resolve(uci);
      this._drain();
    }
  }

  private _drain(): void {
    if (this.busy || this.queue.length === 0) return;

    const item = this.queue.shift()!;
    this.busy = true;
    this._bestmoveResolve = item.resolve;

    this.worker.postMessage(`position fen ${item.fen}`);

    if (this.config.depth !== undefined) {
      this.worker.postMessage(`go depth ${this.config.depth}`);
    } else {
      this.worker.postMessage(`go movetime ${this.config.movetimeMs}`);
    }
  }
}
