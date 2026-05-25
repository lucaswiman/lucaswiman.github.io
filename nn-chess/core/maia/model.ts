// Maia-2 forward pass, re-implemented in TF.js.
//
// The reference is `maia2/main.py:MAIA2Model.forward`. We mirror its
// structure layer-by-layer using raw tf ops inside `tf.tidy` rather
// than `tf.layers`, because:
//   - The custom skill-aware attention block doesn't fit `tf.layers`
//     out of the box.
//   - Loading weights directly from the manifest is simpler when we
//     control where each tensor goes.
//
// Convention: weight names in `MaiaWeights.get(...)` come from the
// manifest and match the (stripped) Python state-dict keys (e.g.
// `chess_cnn.layers.0.conv1.weight`, `transformer.elo_layers.0.0.to_qkv.weight`).
//
// All public functions accept and return plain typed arrays / numbers
// — no `tf.Tensor` leaks out — so the engine on top of this module
// stays framework-agnostic at the boundary.
//
// LICENSE NOTE: derived in structure from
// https://github.com/CSSLab/maia2 (MIT-licensed). See
// `nn-chess/core/maia/index.ts` for the full attribution.

import * as tf from '@tensorflow/tfjs';
import type { MaiaWeights } from './weights.js';
import {
  MAIA_INPUT_CHANNELS,
  BOARD_SIZE,
} from './encoding.js';
import { MAIA_POLICY_SIZE } from './action-space.js';

// ── Architecture constants (must match convert-maia2.py) ─────────────────────

const NUM_BLOCKS_CNN = 5;
const DIM_CNN = 256;
const VIT_LENGTH = 8;
const DIM_VIT = 1024;
const NUM_BLOCKS_VIT = 2;
const NUM_HEADS = 16;
const HEAD_DIM = 64;
const ELO_DIM = 128;  // per-side; concatenated → 256
const BATCH_NORM_EPS = 1e-5;
const LAYER_NORM_EPS = 1e-5;

// ── Public types ─────────────────────────────────────────────────────────────

export interface MaiaPrediction {
  /** Raw logits over the 1880-move action space. */
  policyLogits: Float32Array;
  /** Raw value-head scalar BEFORE the tanh-like rescale.  In the
   * reference, `win_prob = (value/2 + 0.5).clip(0, 1)`. */
  value: number;
}

export interface MaiaForwardOptions {
  /** Skill bucket (0-10) for the side to move. */
  eloSelf: number;
  /** Skill bucket (0-10) for the opponent. */
  eloOppo: number;
  /**
   * Optional intermediate-activation collector. If provided, certain
   * named activations are copied out into the collector as fp32 typed
   * arrays. Used by the parity test.
   */
  collect?: Set<MaiaActivationName>;
  /** If `collect` is set, the captured activations are appended here. */
  out?: Map<MaiaActivationName, Float32Array>;
}

export type MaiaActivationName =
  | 'backbone'
  | 'post_patch'
  | 'elo_emb'
  | 'post_attn'
  | 'pooled'
  | 'policy'
  | 'value';

// ── Forward pass ─────────────────────────────────────────────────────────────

/**
 * Run a single forward pass over one board.
 *
 * `features` must be the 18*8*8 = 1152-length flat tensor produced by
 * `encoding.encodeForMaia`. `eloSelf` / `eloOppo` are the integer
 * bucket IDs from `eloToBucket`.
 *
 * The function does NOT mask illegal moves or apply softmax to the
 * policy — it returns raw logits, mirroring the reference's behavior.
 * The engine layer on top is responsible for masking + softmax.
 */
export function maiaForward(
  weights: MaiaWeights,
  features: Float32Array,
  options: MaiaForwardOptions,
): MaiaPrediction {
  const collect = options.collect ?? null;
  const collectOut = options.out ?? null;

  const result = tf.tidy(() => {
    // 1. Input: (1, 18, 8, 8) — reshape to NHWC for TF.js: (1, 8, 8, 18).
    //    The conv kernels were transposed at conversion time so they
    //    expect HWIO input that's NHWC ordered.
    const ncHw = tf.tensor(features, [1, MAIA_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE], 'float32');
    const nhwc = tf.transpose(ncHw, [0, 2, 3, 1]) as tf.Tensor4D; // (1, 8, 8, 18)

    // 2. CNN stem.
    let x = applyConv2d(nhwc, weights, 'chess_cnn.conv1.weight');
    x = applyBatchNormPreActivation(x, weights, 'chess_cnn.bn1', /*relu=*/true);

    // 3. Residual blocks.
    for (let i = 0; i < NUM_BLOCKS_CNN; i++) {
      const prefix = `chess_cnn.layers.${i}`;
      const identity = x;
      let y = applyConv2d(x, weights, `${prefix}.conv1.weight`);
      y = applyBatchNormPreActivation(y, weights, `${prefix}.bn1`, /*relu=*/true);
      y = applyConv2d(y, weights, `${prefix}.conv2.weight`);
      y = applyBatchNorm(y, weights, `${prefix}.bn2`);
      // residual add then ReLU
      y = tf.relu(tf.add(y, identity));
      x = y;
    }

    // 4. conv_last → bn_last (no activation in reference).
    x = applyConv2d(x, weights, 'chess_cnn.conv_last.weight');
    x = applyBatchNorm(x, weights, 'chess_cnn.bn_last');
    // x: (1, 8, 8, VIT_LENGTH=8). NHWC.

    if (collect?.has('backbone') && collectOut) {
      // Convert back to (1, 8, 8, 8) NCHW for parity-test ease.
      const cb = tf.transpose(x, [0, 3, 1, 2]);
      collectOut.set('backbone', cb.dataSync() as Float32Array);
      cb.dispose();
    }

    // 5. Flatten spatial dims & permute to (1, VIT_LENGTH, 64).
    //    Reference does `embs.view(B, C, 8*8)`, treating the 8 channels
    //    as the token sequence and each spatial 8x8 as a 64-d feature.
    //    We're NHWC, so transpose to NCHW first then reshape.
    let seq = tf.transpose(x, [0, 3, 1, 2]); // (1, 8, 8, 8) in (B, C, H, W)
    seq = tf.reshape(seq, [1, VIT_LENGTH, BOARD_SIZE * BOARD_SIZE]); // (1, 8, 64)

    // 6. Patch embedding: Linear(64 → 1024) + LayerNorm.
    //    Reference uses nn.Sequential[Linear, LayerNorm], no dropout
    //    (dropout is applied AFTER + pos_embedding; at eval it's a no-op).
    let h = applyLinear(seq, weights, 'to_patch_embedding.0.weight', 'to_patch_embedding.0.bias');
    h = applyLayerNorm(h, weights, 'to_patch_embedding.1.weight', 'to_patch_embedding.1.bias');

    // 7. Add positional embedding: shape (1, 8, 1024).
    h = tf.add(h, weights.get('pos_embedding'));

    if (collect?.has('post_patch') && collectOut) {
      collectOut.set('post_patch', h.dataSync() as Float32Array);
    }

    // 8. ELO embedding: lookup self+oppo, concat → (1, 256).
    const eloSelfIdx = tf.tensor1d([options.eloSelf | 0], 'int32');
    const eloOppoIdx = tf.tensor1d([options.eloOppo | 0], 'int32');
    const eloEmbTable = weights.get('elo_embedding.weight'); // (11, 128)
    const eSelf = tf.gather(eloEmbTable, eloSelfIdx); // (1, 128)
    const eOppo = tf.gather(eloEmbTable, eloOppoIdx); // (1, 128)
    const elo = tf.concat([eSelf, eOppo], 1); // (1, 256)

    if (collect?.has('elo_emb') && collectOut) {
      collectOut.set('elo_emb', elo.dataSync() as Float32Array);
    }

    // 9. Transformer blocks (skill-aware attention + MLP).
    for (let i = 0; i < NUM_BLOCKS_VIT; i++) {
      const prefix = `transformer.elo_layers.${i}`;
      const attnOut = eloAwareAttention(h, elo, weights, `${prefix}.0`);
      h = tf.add(h, attnOut);
      const ffOut = feedForward(h, weights, `${prefix}.1.net`);
      h = tf.add(h, ffOut);
    }
    // Final transformer norm.
    h = applyLayerNorm(h, weights, 'transformer.norm.weight', 'transformer.norm.bias');

    if (collect?.has('post_attn') && collectOut) {
      collectOut.set('post_attn', h.dataSync() as Float32Array);
    }

    // 10. Pool across patches (mean), then last_ln.
    let pooled = tf.mean(h, 1); // (1, 1024)
    pooled = applyLayerNorm(pooled, weights, 'last_ln.weight', 'last_ln.bias');

    if (collect?.has('pooled') && collectOut) {
      collectOut.set('pooled', pooled.dataSync() as Float32Array);
    }

    // 11. Policy + value heads. fc_2 (side-info) is not run.
    const policy = applyLinear(pooled, weights, 'fc_1.weight', 'fc_1.bias'); // (1, 1880)
    if (collect?.has('policy') && collectOut) {
      collectOut.set('policy', policy.dataSync() as Float32Array);
    }

    const hValue = tf.relu(applyLinear(pooled, weights, 'fc_3_1.weight', 'fc_3_1.bias'));
    const value = applyLinear(hValue, weights, 'fc_3.weight', 'fc_3.bias'); // (1, 1)
    if (collect?.has('value') && collectOut) {
      collectOut.set('value', value.dataSync() as Float32Array);
    }

    return {
      policyData: policy.dataSync() as Float32Array,
      valueData: value.dataSync() as Float32Array,
    };
  });

  return {
    policyLogits: result.policyData.slice() as Float32Array,
    value: result.valueData[0],
  };
}

// ── Layer helpers ────────────────────────────────────────────────────────────

/**
 * Apply a Conv2d with stride=1, padding=1, no bias. Kernel weight is
 * expected in HWIO order (set at conversion time).
 */
function applyConv2d(x: tf.Tensor, w: MaiaWeights, weightName: string): tf.Tensor4D {
  const kernel = w.get(weightName) as tf.Tensor4D;
  return tf.conv2d(
    x as tf.Tensor4D,
    kernel,
    /*strides=*/ 1,
    /*pad=*/ 'same',
    'NHWC',
  );
}

/**
 * Apply a 2D BatchNorm in inference mode using stored running stats.
 * Matches PyTorch's eval-mode behaviour exactly:
 *   y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
 */
function applyBatchNorm(x: tf.Tensor, w: MaiaWeights, prefix: string): tf.Tensor4D {
  const mean = w.get(`${prefix}.running_mean`); // (C,)
  const variance = w.get(`${prefix}.running_var`); // (C,)
  const scale = w.get(`${prefix}.weight`); // (C,)
  const offset = w.get(`${prefix}.bias`); // (C,)
  // tf.batchNorm broadcasts (C,) over the last NHWC axis.
  return tf.batchNorm(x as tf.Tensor4D, mean, variance, offset, scale, BATCH_NORM_EPS);
}

/**
 * Apply BatchNorm followed by an optional ReLU activation.  Used for
 * `conv1 → bn1 → relu` in the residual block stems.  Kept separate so
 * the linear-residual-add path (`conv2 → bn2 → +x → relu`) stays
 * straightforward.
 */
function applyBatchNormPreActivation(
  x: tf.Tensor,
  w: MaiaWeights,
  prefix: string,
  relu: boolean,
): tf.Tensor4D {
  const out = applyBatchNorm(x, w, prefix);
  return (relu ? tf.relu(out) : out) as tf.Tensor4D;
}

/**
 * Apply a Linear (Dense) layer: `y = x @ W^T + b`.
 *
 * PyTorch stores Linear weights as `(out_features, in_features)`, which
 * is what `weight.shape` already gives us. We therefore transpose
 * before the matmul.
 */
function applyLinear(
  x: tf.Tensor,
  w: MaiaWeights,
  weightName: string,
  biasName: string | null,
): tf.Tensor {
  const W = w.get(weightName); // (out, in)
  const Wt = tf.transpose(W); // (in, out)
  let y: tf.Tensor;
  if (x.rank === 2) {
    y = tf.matMul(x, Wt);
  } else {
    // Generic (..., in) @ (in, out) → (..., out). tf.matMul handles
    // ranks up to 4 with broadcast for the batch dims.
    y = tf.matMul(x, Wt);
  }
  if (biasName !== null) {
    y = tf.add(y, w.get(biasName));
  }
  return y;
}

/**
 * LayerNorm over the LAST axis (matching PyTorch `nn.LayerNorm(dim)`).
 * Reference behaviour:
 *   mean = x.mean(-1); var = x.var(-1, unbiased=False);
 *   y = (x - mean) / sqrt(var + eps) * weight + bias
 *
 * `tf.layers.layerNormalization` does the right thing here but we go
 * via `tf.moments` so we can be explicit about the axis and the eps
 * matches.
 */
function applyLayerNorm(
  x: tf.Tensor,
  w: MaiaWeights,
  weightName: string,
  biasName: string,
): tf.Tensor {
  const axis = x.rank - 1;
  const { mean, variance } = tf.moments(x, axis, /*keepDims=*/ true);
  const invStd = tf.rsqrt(tf.add(variance, LAYER_NORM_EPS));
  let y = tf.mul(tf.sub(x, mean), invStd);
  y = tf.mul(y, w.get(weightName));
  y = tf.add(y, w.get(biasName));
  return y;
}

/**
 * Skill-aware attention block. Mirrors maia2's `EloAwareAttention`:
 *   x_norm = LN(x)
 *   q, k, v = (Linear(x_norm) → split). Reshape to (B, heads, N, d)
 *   q += reshape(Linear(elo))  ← the skill-aware twist
 *   attn = softmax(q @ k^T / sqrt(d))
 *   out = (attn @ v) → reshape → Linear(out_dim)
 */
function eloAwareAttention(
  x: tf.Tensor,
  elo: tf.Tensor,
  w: MaiaWeights,
  prefix: string,
): tf.Tensor {
  // x: (B, N=8, 1024). elo: (B, 256).
  const innerDim = NUM_HEADS * HEAD_DIM; // 1024
  const scale = 1 / Math.sqrt(HEAD_DIM);

  // 1. LayerNorm.
  const xn = applyLayerNorm(x, w, `${prefix}.norm.weight`, `${prefix}.norm.bias`);

  // 2. QKV projection: (B, N, 3*inner).
  // Reference: `nn.Linear(dim, inner*3, bias=False)`.
  const qkvW = w.get(`${prefix}.to_qkv.weight`); // (3*inner, in_dim)
  const qkv = tf.matMul(xn, tf.transpose(qkvW)); // (B, N, 3*inner)
  // Split into three (B, N, inner). tf.split takes the axis arg as the
  // second positional argument; we want to slice on the last axis.
  const [qFlat, kFlat, vFlat] = tf.split(qkv, 3, -1) as [tf.Tensor, tf.Tensor, tf.Tensor];
  // Reshape to (B, heads, N, head_dim). PyTorch's rearrange
  // 'b n (h d) -> b h n d' is equivalent to reshape→transpose.
  const B = qFlat.shape[0]!;
  const N = qFlat.shape[1]!;
  const reshape4 = (t: tf.Tensor) =>
    tf.transpose(tf.reshape(t, [B, N, NUM_HEADS, HEAD_DIM]), [0, 2, 1, 3]);
  let q = reshape4(qFlat);
  const k = reshape4(kFlat);
  const v = reshape4(vFlat);

  // 3. ELO query: Linear(elo) → (B, inner), reshape to (B, heads, 1, head_dim).
  const eloQ = tf.matMul(elo, tf.transpose(w.get(`${prefix}.elo_query.weight`)));
  const eloReshaped = tf.reshape(eloQ, [B, NUM_HEADS, 1, HEAD_DIM]); // broadcasts over N
  q = tf.add(q, eloReshaped);

  // 4. Attention dots: (B, h, N, N).
  const kT = tf.transpose(k, [0, 1, 3, 2]); // (B, h, head_dim, N)
  let dots = tf.matMul(q, kT);
  dots = tf.mul(dots, tf.scalar(scale));
  const attn = tf.softmax(dots, -1);

  // 5. Attend.
  const ctx = tf.matMul(attn, v); // (B, h, N, head_dim)
  // Rearrange 'b h n d -> b n (h d)'.
  const out = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [B, N, innerDim]);

  // 6. Output projection with bias.
  return applyLinear(out, w, `${prefix}.to_out.0.weight`, `${prefix}.to_out.0.bias`);
}

/**
 * FeedForward block (Maia-2 / standard ViT). Order:
 *   LayerNorm, Linear, GELU, [Dropout], Linear, [Dropout]
 *
 * Dropout is a no-op at inference time so we skip it. `prefix.net` is
 * the parent of indices 0 (LN), 1 (Linear-in), 4 (Linear-out).
 */
function feedForward(x: tf.Tensor, w: MaiaWeights, netPrefix: string): tf.Tensor {
  let y = applyLayerNorm(x, w, `${netPrefix}.0.weight`, `${netPrefix}.0.bias`);
  y = applyLinear(y, w, `${netPrefix}.1.weight`, `${netPrefix}.1.bias`);
  // Reference uses nn.GELU() which defaults to the exact (erf-based)
  // formulation, not the tanh approximation. TF.js tf.gelu is the same.
  // We use the explicit erf formulation for portability across tfjs
  // versions that don't ship `tf.gelu`.
  y = exactGelu(y);
  y = applyLinear(y, w, `${netPrefix}.4.weight`, `${netPrefix}.4.bias`);
  return y;
}

/**
 * Exact GELU activation: y = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * TF.js sometimes exposes `tf.erf`; if it doesn't, we approximate erf
 * using a Chebyshev series accurate to ~1e-7 in the range we care about.
 * Use the tf op when present so the GPU/WASM backend can stay on graph.
 */
function exactGelu(x: tf.Tensor): tf.Tensor {
  // tf.erf is present in @tensorflow/tfjs >= 3.x.
  const erfFn = (tf as unknown as { erf?: (t: tf.Tensor) => tf.Tensor }).erf;
  if (typeof erfFn === 'function') {
    const z = tf.div(x, tf.scalar(Math.SQRT2));
    const erf = erfFn(z);
    return tf.mul(tf.mul(tf.scalar(0.5), x), tf.add(tf.scalar(1), erf));
  }
  // Fallback: tanh-based GELU. Not bit-equivalent but the tolerance
  // we test against (≈ fp16 → fp32 noise) covers the discrepancy.
  const c = Math.sqrt(2 / Math.PI);
  const inner = tf.mul(tf.scalar(c), tf.add(x, tf.mul(tf.scalar(0.044715), tf.pow(x, 3))));
  return tf.mul(tf.mul(tf.scalar(0.5), x), tf.add(tf.scalar(1), tf.tanh(inner)));
}

// Constants exported for tests.
export const _MODEL_CONSTANTS = {
  NUM_BLOCKS_CNN,
  DIM_CNN,
  VIT_LENGTH,
  DIM_VIT,
  NUM_BLOCKS_VIT,
  NUM_HEADS,
  HEAD_DIM,
  ELO_DIM,
  POLICY_SIZE: MAIA_POLICY_SIZE,
} as const;
