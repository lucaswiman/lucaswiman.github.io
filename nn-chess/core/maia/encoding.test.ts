import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { encodeForMaia, eloToBucket, mirrorUci } from './encoding.js';
import { fromFen } from '../rules/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ReferenceFixture {
  fen: string;
  elo_self: number;
  elo_oppo: number;
  mirrored: boolean;
  legal_indices: number[];
  encoded_board_flat: number[];
  encoded_board_shape: number[];
}

const REF: ReferenceFixture = JSON.parse(
  readFileSync(join(__dirname, '__fixtures__', 'reference.json'), 'utf-8'),
);

describe('Maia encoding', () => {
  describe('eloToBucket', () => {
    // Reference mapping from maia2.utils.create_elo_dict / map_to_category.
    const cases: Array<[number, number]> = [
      [0, 0], [999, 0], [1099, 0],
      [1100, 1], [1199, 1],
      [1200, 2],
      [1500, 5],
      [1900, 9], [1999, 9],
      [2000, 10], [2500, 10], [3000, 10],
    ];
    for (const [elo, expected] of cases) {
      it(`elo ${elo} → bucket ${expected}`, () => {
        expect(eloToBucket(elo)).toBe(expected);
      });
    }
  });

  describe('mirrorUci', () => {
    it('mirrors common moves', () => {
      expect(mirrorUci('e2e4')).toBe('e7e5');
      expect(mirrorUci('a1h8')).toBe('a8h1');
      expect(mirrorUci('e7e8q')).toBe('e2e1q');
    });
    it('rejects malformed strings', () => {
      expect(() => mirrorUci('foo')).toThrow();
    });
  });

  describe('encodeForMaia', () => {
    it('matches the python reference tensor for the fixture FEN', () => {
      const state = fromFen(REF.fen);
      const { features, mirrored } = encodeForMaia(state);
      expect(mirrored).toBe(REF.mirrored);
      expect(features.length).toBe(REF.encoded_board_flat.length);

      // The maia2 reference tensor is (1, 18, 8, 8). Our encoder is
      // flat [channel, row, col] with the same layout, so direct
      // element-wise comparison is valid.
      let firstMismatch = -1;
      for (let i = 0; i < features.length; i++) {
        if (Math.abs(features[i] - REF.encoded_board_flat[i]) > 1e-6) {
          firstMismatch = i;
          break;
        }
      }
      if (firstMismatch >= 0) {
        const channel = Math.floor(firstMismatch / 64);
        const within = firstMismatch % 64;
        const row = Math.floor(within / 8);
        const col = within % 8;
        throw new Error(
          `encoded board mismatch at flat=${firstMismatch} ` +
          `(channel=${channel}, row=${row}, col=${col}): ` +
          `got ${features[firstMismatch]}, expected ${REF.encoded_board_flat[firstMismatch]}`,
        );
      }
    });

    it('encodes black-to-move via mirroring', () => {
      // Same position with black to move: mirroring should produce a
      // valid 18-plane tensor with side-to-move plane set to 1
      // (because after the mirror, it's white-to-move again).
      const state = fromFen('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2');
      const { features, mirrored } = encodeForMaia(state);
      expect(mirrored).toBe(true);
      // side-to-move plane (idx 12) should be all ones since the mirror
      // makes it white-to-move from the model's perspective.
      const base = 12 * 64;
      let sum = 0;
      for (let i = 0; i < 64; i++) sum += features[base + i];
      expect(sum).toBe(64);
    });
  });
});
