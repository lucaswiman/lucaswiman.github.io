import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  MAIA_MOVES,
  MAIA_POLICY_SIZE,
  maiaPolicyIndex,
} from './action-space.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(__dirname, '__fixtures__', 'all_moves.json');

describe('Maia action space', () => {
  const reference: string[] = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));

  it('has the expected size', () => {
    expect(MAIA_POLICY_SIZE).toBe(1880);
    expect(MAIA_MOVES.length).toBe(1880);
  });

  it('matches the Python reference move list exactly', () => {
    expect(reference.length).toBe(MAIA_MOVES.length);
    // Compare element-by-element so the first mismatch is informative.
    for (let i = 0; i < reference.length; i++) {
      if (reference[i] !== MAIA_MOVES[i]) {
        throw new Error(
          `Mismatch at index ${i}: expected ${reference[i]}, got ${MAIA_MOVES[i]}`,
        );
      }
    }
  });

  it('round-trips through maiaPolicyIndex', () => {
    for (let i = 0; i < MAIA_MOVES.length; i++) {
      expect(maiaPolicyIndex(MAIA_MOVES[i])).toBe(i);
    }
  });

  it('returns -1 for moves not in the action space', () => {
    expect(maiaPolicyIndex('a2a3a')).toBe(-1);  // bogus
    // Black-side promotion (Maia-2's action space is white-side only;
    // black-to-move positions are mirrored before tensorizing).
    expect(maiaPolicyIndex('a2a1q')).toBe(-1);
  });
});
