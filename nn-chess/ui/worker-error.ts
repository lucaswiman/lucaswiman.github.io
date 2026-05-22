// Helpers for formatting Web Worker error reports.
//
// Browsers populate `ErrorEvent` inconsistently. The `message` property
// is often empty (notably on iOS Safari) when the worker script fails to
// evaluate at the top level — typically a load failure, a bad import, or
// an exception in module-level code that runs before our `onmessage`
// handler installs its own try/catch. In those cases we still want to
// surface something more useful than `undefined` to the user.

/**
 * Format a Worker `ErrorEvent` into a human-readable string. Falls back to
 * any available metadata (`event.error`, `event.filename`, line/col) when
 * the primary `message` field is empty.
 */
export function formatWorkerErrorEvent(event: ErrorEvent): string {
  const parts: string[] = [];
  if (typeof event.message === 'string' && event.message.length > 0) {
    parts.push(event.message);
  }
  const err = event.error;
  if (err instanceof Error && err.message && !parts.includes(err.message)) {
    parts.push(err.message);
  } else if (err && typeof err === 'object' && 'message' in err) {
    const m = (err as { message?: unknown }).message;
    if (typeof m === 'string' && m.length > 0 && !parts.includes(m)) parts.push(m);
  }
  if (typeof event.filename === 'string' && event.filename.length > 0) {
    const lineno = typeof event.lineno === 'number' ? event.lineno : '?';
    const colno = typeof event.colno === 'number' ? event.colno : '?';
    parts.push(`(${event.filename}:${lineno}:${colno})`);
  }
  if (parts.length === 0) {
    return '(no details — worker script likely failed to load; check the browser console)';
  }
  return parts.join(' ');
}
