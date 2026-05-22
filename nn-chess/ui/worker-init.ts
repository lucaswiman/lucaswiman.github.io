// Shared one-time setup for the self-play and training Web Workers.
//
// Two responsibilities:
//
//   1. Install global `error` / `unhandledrejection` handlers that forward
//      problems back to the main thread via `postMessage`. The parent's
//      `worker.onerror` ErrorEvent is often empty/stripped (especially on
//      iOS Safari), so going through the message channel gives us a real,
//      same-origin error message we can render in the UI.
//
//   2. Initialize the WASM backend (with a CPU fallback) without ever
//      rejecting the returned promise. If both backends fail, log to the
//      worker console and let tfjs use whatever default is registered —
//      the message handler should still try to run rather than killing
//      the whole batch over a backend hiccup.

import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

/**
 * Install global error handlers that forward to the main thread via
 * `postMessage`. `errorKind` is the `kind` discriminator of the worker's
 * response protocol (always `'error'` for our two workers).
 */
export function installWorkerErrorHandlers(errorKind: 'error' = 'error'): void {
  self.addEventListener('error', (event) => {
    const err = event.error;
    const detail = err instanceof Error
      ? `${err.name}: ${err.message}`
      : (event.message || 'unknown error');
    const loc = event.filename
      ? ` (${event.filename}:${event.lineno ?? '?'}:${event.colno ?? '?'})`
      : '';
    try {
      self.postMessage({ kind: errorKind, message: `Uncaught error in worker: ${detail}${loc}` });
    } catch {
      // Nothing we can do if postMessage itself fails.
    }
  });

  self.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason instanceof Error
      ? `${event.reason.name}: ${event.reason.message}`
      : String(event.reason);
    try {
      self.postMessage({ kind: errorKind, message: `Unhandled rejection in worker: ${reason}` });
    } catch {
      // Nothing we can do if postMessage itself fails.
    }
  });
}

/**
 * Initialize the tfjs backend (WASM, falling back to CPU). The returned
 * promise NEVER rejects: if both setBackend calls throw, we swallow the
 * errors and let the message handler proceed with whatever default
 * backend tfjs has registered. This avoids killing the entire worker
 * over a backend-init failure when there's usually still a usable CPU
 * path.
 */
export function initBackend(): Promise<void> {
  return (async () => {
    try {
      setWasmPaths(
        'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/',
      );
    } catch (err) {
      console.warn('worker: setWasmPaths failed', err);
    }
    try {
      await tf.setBackend('wasm');
      return;
    } catch (err) {
      console.warn('worker: WASM backend init failed, falling back to CPU', err);
    }
    try {
      await tf.setBackend('cpu');
    } catch (err) {
      console.warn('worker: CPU backend init also failed; using tfjs default', err);
    }
  })();
}
