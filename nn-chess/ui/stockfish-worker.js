/**
 * Stockfish Web Worker shim.
 *
 * This file is loaded as a plain-JS worker (not a module worker) so that
 * importScripts() is available. It delegates to the vendored stockfish
 * lite-single-threaded WASM build in /public/nn-chess/stockfish/.
 *
 * Why vendored instead of importing via the `stockfish` npm package directly?
 * The `stockfish` npm package's JS files load a sibling .wasm file using a
 * URL derived from their own script URL. When Vite bundles the JS into a
 * hashed chunk, the .wasm lookup path breaks. Vendoring both files in
 * public/ preserves their relative URL relationship and avoids any
 * bundler involvement — the files are served as-is at predictable paths.
 *
 * The lite-single-threaded variant (stockfish-18-lite-single.js + .wasm,
 * ~7MB) is chosen because:
 *  1. No SharedArrayBuffer / COOP/COEP headers required (single-threaded).
 *  2. ~7MB vs ~110MB for the full engine — acceptable load time.
 *  3. Still far stronger than the nn-chess agent; the "skill level" option
 *     is used to deliberately weaken it so the agent has a chance.
 *
 * Protocol:
 *   Main thread → worker: postMessage(uciCommandString)
 *   Worker → main thread: postMessage(engineOutputLine)
 *
 * The stockfish-18-lite-single.js file installs its own onmessage handler
 * and posts output lines back, so this shim just needs to importScripts
 * it from the right URL.
 */

// The vendored stockfish script locates its .wasm sibling relative to
// self.location.href (the worker script URL). Because both files live in
// /nn-chess/stockfish/, the relative lookup works automatically.
//
// We derive the base URL from the query parameter ?stockfishBase=<url>
// that the host page passes in. This lets the React component supply the
// correct origin-relative path at runtime without hardcoding it here.

const params = new URLSearchParams(self.location.search);
const stockfishBase = params.get('stockfishBase') || '/nn-chess/stockfish/';
const scriptUrl = stockfishBase + 'stockfish-18-lite-single.js';

importScripts(scriptUrl);
