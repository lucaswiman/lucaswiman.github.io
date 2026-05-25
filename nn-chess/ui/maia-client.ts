/**
 * Main-thread wrapper around the Maia-2 inference Web Worker.
 *
 * Two responsibilities:
 *
 *   1. Fetch + cache the Maia-2 weights. The blob is ~40 MiB
 *      (fp16) and only changes when we publish a new release, so we
 *      cache it via the existing IndexedDB BlobStorage adapter so
 *      repeat visits start instantly.
 *
 *   2. Send `select-move` requests to the worker and resolve the
 *      returned promises in order.
 *
 * The class is a slim wrapper; the real engine lives in
 * `nn-chess/core/maia/` and runs inside the worker.
 */

import type { BlobStorage } from '../core/storage/index.js';
import { getJson, putJson } from '../core/storage/index.js';
import type {
  MaiaManifest,
} from '../core/maia/index.js';
import type { MaiaRequest, MaiaResponse } from './maia-protocol.js';

// ── Storage keys ────────────────────────────────────────────────────────────

const MAIA_WEIGHTS_KEY = 'maia2/rapid/weights.bin';
const MAIA_MANIFEST_KEY = 'maia2/rapid/manifest.json';

// Default hosting location for the published Maia-2 release. The two
// URLs let us serve directly from the static GH Pages output during
// development (place the files under `public/nn-chess/maia2/`) and
// fall back to a GitHub release asset in production.
export interface MaiaSource {
  /** URL of the manifest JSON. */
  manifestUrl: string;
  /** URL of the weights .bin. */
  weightsUrl: string;
}

export const DEFAULT_MAIA_SOURCE: MaiaSource = {
  manifestUrl: '/nn-chess/maia2/manifest.json',
  weightsUrl: '/nn-chess/maia2/weights.bin',
};

// ── Download progress UI hook ───────────────────────────────────────────────

export interface MaiaLoadProgress {
  /** Bytes received so far. */
  loaded: number;
  /** Total bytes expected (may be 0 if the server didn't send a Content-Length). */
  total: number;
}

export interface MaiaLoadOptions {
  /** Where to fetch the manifest + blob from. */
  source?: MaiaSource;
  /** Optional storage adapter for caching. Skipped if absent. */
  storage?: BlobStorage;
  /** Called periodically during the initial download. */
  onProgress?: (p: MaiaLoadProgress) => void;
  /** Cancel signal — abort the in-flight fetch if requested. */
  signal?: AbortSignal;
}

/**
 * Fetch + cache the Maia-2 weights. Returns {blob, manifest, fromCache}.
 *
 * Cache key: keyed off the manifest's `sha256`. If a cached blob is
 * present with a matching sha, we skip the download entirely.
 */
export async function fetchMaiaWeights(
  options: MaiaLoadOptions = {},
): Promise<{ blob: ArrayBuffer; manifest: MaiaManifest; fromCache: boolean }> {
  const source = options.source ?? DEFAULT_MAIA_SOURCE;
  const storage = options.storage;

  // 1. Fetch the manifest first — small, so always re-fetch to pick up
  //    new release rotations cheaply.
  const manifestResp = await fetch(source.manifestUrl, {
    signal: options.signal,
    cache: 'no-store',
  });
  if (!manifestResp.ok) {
    throw new Error(
      `Maia manifest fetch failed: ${manifestResp.status} ${manifestResp.statusText}`,
    );
  }
  const manifest = (await manifestResp.json()) as MaiaManifest;

  // 2. Try the cache.
  if (storage) {
    const cachedManifest = await getJson<MaiaManifest>(storage, MAIA_MANIFEST_KEY);
    if (cachedManifest && cachedManifest.sha256 === manifest.sha256) {
      const cachedBlob = await storage.get(MAIA_WEIGHTS_KEY);
      if (cachedBlob !== null && cachedBlob.byteLength === manifest.total_bytes) {
        return {
          // Copy into a fresh ArrayBuffer; postMessage to the worker
          // will transfer it, and we don't want to clobber the
          // adapter's underlying buffer.
          blob: copyToArrayBuffer(cachedBlob),
          manifest,
          fromCache: true,
        };
      }
    }
  }

  // 3. Download the blob with progress reporting.
  const blob = await downloadWithProgress(
    source.weightsUrl,
    manifest.total_bytes,
    options.onProgress,
    options.signal,
  );

  // 4. Write back to cache.
  if (storage) {
    try {
      await storage.put(MAIA_WEIGHTS_KEY, new Uint8Array(blob));
      await putJson(storage, MAIA_MANIFEST_KEY, manifest);
    } catch (err) {
      // Caching failure is non-fatal; the engine still works.
      console.warn('Failed to cache Maia weights:', err);
    }
  }

  return { blob, manifest, fromCache: false };
}

function copyToArrayBuffer(view: Uint8Array): ArrayBuffer {
  const copy = new ArrayBuffer(view.byteLength);
  new Uint8Array(copy).set(view);
  return copy;
}

async function downloadWithProgress(
  url: string,
  expectedBytes: number,
  onProgress?: (p: MaiaLoadProgress) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const resp = await fetch(url, { signal });
  if (!resp.ok) {
    throw new Error(
      `Maia weights fetch failed: ${resp.status} ${resp.statusText}`,
    );
  }
  // Prefer streaming so we can show progress. Fall back to .arrayBuffer().
  if (!resp.body) {
    return resp.arrayBuffer();
  }
  const total = Number(resp.headers.get('content-length')) || expectedBytes || 0;
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      onProgress?.({ loaded: received, total });
    }
  }
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    buf.set(c, offset);
    offset += c.byteLength;
  }
  return buf.buffer;
}

// ── MaiaClient ──────────────────────────────────────────────────────────────

/**
 * Promise-based handle for the Maia-2 worker.  Construct once via
 * `MaiaClient.create(blob, manifest)`; call `selectMove` for each
 * Maia move; call `dispose()` when the user navigates away.
 */
export class MaiaClient {
  private worker: Worker;
  private nextRequestId = 1;
  private pending = new Map<number, {
    resolve: (r: {
      move: string;
      value: number;
      moveProbs: Map<string, number>;
    }) => void;
    reject: (err: Error) => void;
  }>();
  private readyResolve: (() => void) | null = null;
  private readyReject: ((err: Error) => void) | null = null;
  private disposed = false;

  private constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (event: MessageEvent<MaiaResponse>) => this.onMessage(event.data);
    this.worker.onerror = (event) => this.onWorkerError(event);
  }

  /**
   * Spin up the worker and hand it the weights. Resolves once the
   * worker has posted `{kind: 'ready'}` (i.e. tensors materialized,
   * tfjs backend ready).
   */
  static async create(
    blob: ArrayBuffer,
    manifest: MaiaManifest,
  ): Promise<MaiaClient> {
    // Plain module worker — bundled by Vite/Astro automatically.
    const worker = new Worker(new URL('./maia-worker.ts', import.meta.url), {
      type: 'module',
    });
    const client = new MaiaClient(worker);
    const ready = new Promise<void>((resolve, reject) => {
      client.readyResolve = resolve;
      client.readyReject = reject;
    });
    // Transfer the ArrayBuffer to avoid a 40MB copy. After this, the
    // local `blob` is detached — callers shouldn't keep a reference.
    worker.postMessage(
      { kind: 'init', blob, manifest } satisfies MaiaRequest,
      [blob],
    );
    await ready;
    return client;
  }

  /**
   * Ask the worker for a move from the given FEN. Resolves with the
   * chosen UCI move, the value-head readout, and a map of move
   * probabilities over legal moves.
   */
  selectMove(
    fen: string,
    elo: number,
    options: { oppoElo?: number; temperature?: number } = {},
  ): Promise<{ move: string; value: number; moveProbs: Map<string, number> }> {
    if (this.disposed) {
      return Promise.reject(new Error('MaiaClient: disposed'));
    }
    const requestId = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      const req: MaiaRequest = {
        kind: 'select-move',
        requestId,
        fen,
        elo,
        oppoElo: options.oppoElo,
        temperature: options.temperature,
      };
      this.worker.postMessage(req);
    });
  }

  /** Terminate the worker. All pending promises will be rejected. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const err = new Error('MaiaClient disposed');
    for (const { reject } of this.pending.values()) reject(err);
    this.pending.clear();
    if (this.readyReject) this.readyReject(err);
    this.readyReject = null;
    this.readyResolve = null;
    this.worker.terminate();
  }

  // ── Internal message handling ─────────────────────────────────────────────

  private onMessage(msg: MaiaResponse): void {
    if (msg.kind === 'ready') {
      this.readyResolve?.();
      this.readyResolve = null;
      this.readyReject = null;
      return;
    }
    if (msg.kind === 'move') {
      const handler = this.pending.get(msg.requestId);
      if (!handler) return;
      this.pending.delete(msg.requestId);
      const probs = new Map<string, number>(Object.entries(msg.moveProbs));
      handler.resolve({ move: msg.move, value: msg.value, moveProbs: probs });
      return;
    }
    if (msg.kind === 'error') {
      const err = new Error(msg.message);
      if (msg.requestId !== undefined) {
        const handler = this.pending.get(msg.requestId);
        if (handler) {
          this.pending.delete(msg.requestId);
          handler.reject(err);
          return;
        }
      }
      // Non-request error: route to readyReject if init is pending,
      // otherwise reject everything outstanding so the UI sees it.
      if (this.readyReject) {
        this.readyReject(err);
        this.readyResolve = null;
        this.readyReject = null;
        return;
      }
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    }
  }

  private onWorkerError(event: ErrorEvent): void {
    const err = new Error(`Maia worker error: ${event.message || 'unknown'}`);
    if (this.readyReject) {
      this.readyReject(err);
      this.readyResolve = null;
      this.readyReject = null;
    }
    for (const { reject } of this.pending.values()) reject(err);
    this.pending.clear();
  }
}
