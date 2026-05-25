// Public API for the Maia-2 inference adapter.
//
// Maia-2 is a third-party engine (NeurIPS 2024, MIT-licensed, source at
// https://github.com/CSSLab/maia2). This module re-implements its
// forward pass in TF.js so the nn-chess UI can let the user play
// against it at a chosen skill level. The engine is FROZEN — we do
// NOT train it, do not back-propagate through it, do not feed its
// moves into either of our own networks' training data.  See
// nn-chess/CLAUDE.md / nn-chess/README.md and the parent agent's
// brief for the user-side-only training-data routing convention.
//
// Architecture (from maia2/main.py + config.yaml):
//   - 18×8×8 board encoding (different from `core/encoding/`).
//   - ChessResNet backbone: conv stem + 5 residual blocks (256 ch).
//   - Channel-wise patch embedding → 8 tokens × 1024-d.
//   - 2 skill-aware attention blocks (16 heads, 64 head-dim).
//   - Three heads: policy (1880 moves), side-info (skipped at inference),
//     value (regression in [-1, 1]).
//   - Skill encoded via an 11-bucket embedding table (per-side, concat).
//
// Reference: https://arxiv.org/html/2409.20553v2
// LICENSE NOTE: code in this directory is MIT-licensed, matching the
// upstream Maia-2 LICENSE. Please preserve this notice when copying.

export {
  encodeForMaia,
  eloToBucket,
  mirrorUci,
  MAIA_ELO_BUCKETS,
  MAIA_INPUT_CHANNELS,
  MAIA_FEATURE_COUNT,
} from './encoding.js';

export {
  MAIA_MOVES,
  MAIA_POLICY_SIZE,
  maiaPolicyIndex,
} from './action-space.js';

export {
  loadMaiaWeights,
  MAIA_MANIFEST_VERSION,
} from './weights.js';
export type { MaiaManifest, MaiaManifestTensor, MaiaWeights } from './weights.js';

export {
  maiaForward,
  _MODEL_CONSTANTS,
} from './model.js';
export type {
  MaiaPrediction,
  MaiaForwardOptions,
  MaiaActivationName,
} from './model.js';

export { MaiaEngine } from './engine.js';
export type { MaiaSelectMoveOptions, MaiaSelectMoveResult } from './engine.js';
