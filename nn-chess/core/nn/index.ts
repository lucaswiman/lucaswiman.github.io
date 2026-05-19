// Value + policy neural network for the chess agent.
//
// Design choices documented here:
//
// Architecture: Dense (MLP) vs Convolutional
//   We use a flat dense MLP (input 1152 → dense(128) → dense(128) → two heads)
//   rather than a conv net because:
//   1. Pedagogical clarity: the forward pass is a few matrix multiplies, easy
//      to trace in a debugger or interpret.
//   2. This is a tiny model trained on tiny data; conv filters are more useful
//      when the spatial structure really matters (e.g. AlphaZero trains for
//      millions of games). With one human's worth of games, the dense network
//      will converge first.
//   3. We can upgrade to conv later (NN_VERSION bump) without changing the
//      surrounding API.
//
// Policy representation: raw logits over 64×64 = 4096 from-square × to-square
//   - Cheap, AlphaZero-style move encoding.
//   - The agent module is responsible for masking illegal moves before
//     computing softmax — we return raw logits here.
//   - TODO: extend to 64×64×3 = 12288 to cover underpromotion piece choices
//     (queen/rook/bishop). Currently all 5-char UCI promotions map to the
//     same 4-char prefix index, which means underpromotion is folded into
//     the queen promotion slot.
//
// Serialization wire format:
//   A single Uint8Array whose first 4 bytes are a little-endian uint32
//   holding the byte-length of the topology JSON, followed by the topology
//   JSON bytes (UTF-8), followed by the weight data (raw ArrayBuffer from
//   tfjs weightData). On load, we split at the boundary and feed both halves
//   back to tf.loadLayersModel via tf.io.fromMemory(). The weightSpecs array
//   is embedded in the topology JSON (under the 'weightsManifest' key that
//   tfjs itself injects during save).
//
// Policy output: raw logits (not softmax probabilities).
//   Callers should apply softmax + illegal-move masking before sampling.

import * as tf from '@tensorflow/tfjs';
import { FEATURE_COUNT } from '../encoding/index.js';

// ── Public constants ────────────────────────────────────────────────────────

/** Number of squares × squares in the flat from→to move encoding. */
export const POLICY_SIZE = 64 * 64; // 4096

/**
 * Bump this whenever the architecture or move encoding changes so that
 * saved weights from an older version can be detected and discarded.
 */
export const NN_VERSION = 1;

// ── Move-index helpers ──────────────────────────────────────────────────────

/** Maps 'a'–'h' → 0–7, '1'–'8' → 0–7. */
function squareToIndex(sq: string): number {
  const file = sq.charCodeAt(0) - 97; // 'a' = 0
  const rank = sq.charCodeAt(1) - 49; // '1' = 0
  return rank * 8 + file;
}

function indexToSquare(idx: number): string {
  const file = idx % 8;
  const rank = Math.floor(idx / 8);
  return String.fromCharCode(97 + file) + String.fromCharCode(49 + rank);
}

/**
 * Convert a UCI move string (e.g. "e2e4", "a1h8", promotion "e7e8q")
 * to a flat policy index in [0, POLICY_SIZE).
 *
 * Promotion suffix is ignored — the 4-char from/to prefix determines
 * the index. See TODO in file header about underpromotion extension.
 */
export function policyIndexFromUci(uci: string): number {
  const from = squareToIndex(uci.slice(0, 2));
  const to = squareToIndex(uci.slice(2, 4));
  return from * 64 + to;
}

/**
 * Convert a flat policy index back to a 4-char UCI string (without
 * promotion piece). Inverse of `policyIndexFromUci`.
 */
export function uciFromPolicyIndex(idx: number): string {
  const from = Math.floor(idx / 64);
  const to = idx % 64;
  return indexToSquare(from) + indexToSquare(to);
}

// ── Types ───────────────────────────────────────────────────────────────────

export interface TrainingExample {
  /** Encoded board, length FEATURE_COUNT (= 1152). */
  features: Float32Array;
  /** Terminal value target for the side to move: -1 | 0 | 1. */
  valueTarget: number;
  /**
   * One-hot over POLICY_SIZE for the move played, or null to skip the
   * policy loss for this example (draws, or positions where the loser moved).
   */
  policyTarget: Float32Array | null;
}

export interface Prediction {
  /** Scalar value ∈ [-1, 1] (tanh), from side-to-move's perspective. */
  value: number;
  /**
   * Raw logits over POLICY_SIZE (one entry per from→to square pair).
   * Callers must apply softmax + illegal-move masking before use.
   */
  policy: Float32Array;
}

export interface TrainResult {
  valueLoss: number;
  policyLoss: number;
  totalLoss: number;
}

export interface ChessNetOptions {
  /**
   * If provided, every layer's kernel initializer is seeded with this
   * value, making weight initialization deterministic across instances.
   * Useful for reproducible unit tests.
   */
  seed?: number;
}

// ── ChessNet ────────────────────────────────────────────────────────────────

/**
 * ChessNet — a small value+policy network implemented as a tfjs
 * functional model with a shared trunk and two output heads.
 *
 * We expose a class (not a factory function) so that callers can hold a
 * typed reference and call methods without indirection. The constructor is
 * private; use `ChessNet.create(opts?)` or `deserialize(bytes)` to obtain
 * an instance.
 *
 * Layer names are stable strings so that `getActivations` and `layerNames`
 * have a reliable contract even across serialize/deserialize round-trips.
 */
export class ChessNet {
  private readonly _model: tf.LayersModel;
  private readonly _optimizer: tf.Optimizer;
  private readonly _activationModels = new Map<string, tf.LayersModel>();

  // Layer names for the activation inspector (interpretability hook).
  // These match the `name` field passed to each tf.layers.* call below.
  private static readonly INSPECTABLE_LAYERS = [
    'hidden1',
    'hidden2',
    'value_head',
    'policy_head',
  ] as const;

  private constructor(model: tf.LayersModel) {
    this._model = model;
    // Adam with a small learning rate. L2 regularization is applied inside
    // the model definition (kernelRegularizer on each dense layer).
    this._optimizer = tf.train.adam(1e-3);
  }

  // ── Factory ──────────────────────────────────────────────────────────────

  /** Build a fresh network with (optionally) a fixed random seed. */
  static create(opts: ChessNetOptions = {}): ChessNet {
    const seed = opts.seed;

    const initOpts = seed !== undefined
      ? { seed }
      : {};

    // Shared trunk: flatten 1152-dim input → two ReLU dense layers.
    const input = tf.input({ shape: [FEATURE_COUNT], name: 'board_input' });

    const h1 = tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: tf.initializers.glorotUniform(initOpts),
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      name: 'hidden1',
    }).apply(input) as tf.SymbolicTensor;

    const h2 = tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: tf.initializers.glorotUniform(
        seed !== undefined ? { seed: seed + 1 } : {},
      ),
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      name: 'hidden2',
    }).apply(h1) as tf.SymbolicTensor;

    // Value head: scalar output through tanh, giving a value in [-1, 1].
    const valueHead = tf.layers.dense({
      units: 1,
      activation: 'tanh',
      kernelInitializer: tf.initializers.glorotUniform(
        seed !== undefined ? { seed: seed + 2 } : {},
      ),
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      name: 'value_head',
    }).apply(h2) as tf.SymbolicTensor;

    // Policy head: raw logits over POLICY_SIZE.
    // No activation — callers apply softmax after masking illegal moves.
    const policyHead = tf.layers.dense({
      units: POLICY_SIZE,
      activation: 'linear',
      kernelInitializer: tf.initializers.glorotUniform(
        seed !== undefined ? { seed: seed + 3 } : {},
      ),
      biasInitializer: 'zeros',
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      name: 'policy_head',
    }).apply(h2) as tf.SymbolicTensor;

    const model = tf.model({
      inputs: input,
      outputs: [valueHead, policyHead],
      name: 'chess_net',
    });

    return new ChessNet(model);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Single-board forward pass.
   * Returns `value` ∈ [-1, 1] and raw `policy` logits of length POLICY_SIZE.
   */
  predict(features: Float32Array): Prediction {
    return tf.tidy(() => {
      const input = tf.tensor2d(features, [1, FEATURE_COUNT]);
      const [valueTensor, policyTensor] = this._model.predict(input) as [
        tf.Tensor,
        tf.Tensor,
      ];
      const value = (valueTensor.dataSync() as Float32Array)[0];
      const policy = policyTensor.dataSync() as Float32Array;
      return { value, policy: policy.slice() as Float32Array };
    });
  }

  /**
   * Batched forward pass — a single tfjs op, not a loop.
   * Returns one `Prediction` per input feature vector.
   */
  predictBatch(featuresBatch: Float32Array[]): Prediction[] {
    if (featuresBatch.length === 0) return [];

    return tf.tidy(() => {
      const batchSize = featuresBatch.length;
      // Stack feature vectors into a single [batchSize, FEATURE_COUNT] tensor.
      const combined = new Float32Array(batchSize * FEATURE_COUNT);
      for (let i = 0; i < batchSize; i++) {
        combined.set(featuresBatch[i], i * FEATURE_COUNT);
      }
      const input = tf.tensor2d(combined, [batchSize, FEATURE_COUNT]);
      const [valueTensor, policyTensor] = this._model.predict(input) as [
        tf.Tensor,
        tf.Tensor,
      ];

      const values = valueTensor.dataSync() as Float32Array;
      const policies = policyTensor.dataSync() as Float32Array;

      const results: Prediction[] = [];
      for (let i = 0; i < batchSize; i++) {
        results.push({
          value: values[i],
          policy: policies.slice(i * POLICY_SIZE, (i + 1) * POLICY_SIZE) as Float32Array,
        });
      }
      return results;
    });
  }

  /**
   * One gradient step over a batch of training examples.
   *
   * Loss breakdown:
   *   - Value loss: MSE between predicted tanh value and valueTarget.
   *   - Policy loss: softmax cross-entropy, masked per-example.
   *     Examples with `policyTarget === null` contribute 0 to the policy
   *     loss. We implement this by zeroing their contribution before summing.
   *   - L2 regularization is applied inside the layer definitions
   *     (kernelRegularizer) and is included in the total loss automatically
   *     by tfjs when we call model.trainOnBatch — but here we run a manual
   *     gradient step so we add the regularization terms explicitly.
   *
   * Returns the scalar losses (for logging/monitoring).
   */
  async trainBatch(examples: TrainingExample[]): Promise<TrainResult> {
    if (examples.length === 0) {
      return { valueLoss: 0, policyLoss: 0, totalLoss: 0 };
    }

    const n = examples.length;

    // Build tensors outside tidy so we can track them for disposal.
    const featureData = new Float32Array(n * FEATURE_COUNT);
    const valueTargetData = new Float32Array(n);
    // Policy targets: n × POLICY_SIZE, zero-filled; mask tracks which rows
    // have a real target.
    const policyTargetData = new Float32Array(n * POLICY_SIZE);
    const maskData = new Float32Array(n); // 1 where policyTarget != null, else 0

    for (let i = 0; i < n; i++) {
      featureData.set(examples[i].features, i * FEATURE_COUNT);
      valueTargetData[i] = examples[i].valueTarget;
      if (examples[i].policyTarget !== null) {
        policyTargetData.set(examples[i].policyTarget!, i * POLICY_SIZE);
        maskData[i] = 1;
      }
    }

    let valueLoss = 0;
    let policyLoss = 0;

    const { value: totalLossValue } = this._optimizer.minimize(
      () => tf.tidy(() => {
        const features = tf.tensor2d(featureData, [n, FEATURE_COUNT]);
        const valueTargets = tf.tensor2d(valueTargetData, [n, 1]);
        const policyTargets = tf.tensor2d(policyTargetData, [n, POLICY_SIZE]);
        const mask = tf.tensor1d(maskData); // [n]

        const [valuePred, policyLogits] = this._model.apply(features, {
          training: true,
        }) as [tf.Tensor, tf.Tensor];

        // MSE value loss.
        const vLoss = tf.losses.meanSquaredError(valueTargets, valuePred) as tf.Scalar;

        // Masked softmax cross-entropy policy loss.
        // crossEntropy shape: [n] (one scalar per example).
        const logSoftmax = tf.logSoftmax(policyLogits); // [n, POLICY_SIZE]
        // Sum of (target * log_softmax) per example → [n].
        const perExampleCE = tf.neg(
          tf.sum(tf.mul(policyTargets, logSoftmax), 1),
        ); // [n]
        // Multiply by mask and take the mean over the batch.
        const maskedCE = tf.mul(perExampleCE, mask); // [n]
        const numMasked = tf.maximum(tf.sum(mask), tf.scalar(1)); // avoid /0
        const pLoss = tf.div(tf.sum(maskedCE), numMasked) as tf.Scalar;

        // L2 regularization: sum kernel norms manually so the optimizer sees
        // them in the gradient path (tfjs model regularization is only
        // applied through model.fit/trainOnBatch).
        let l2Sum = tf.scalar(0);
        for (const w of this._model.trainableWeights) {
          if (w.name.includes('kernel')) {
            l2Sum = tf.add(l2Sum, tf.mul(tf.scalar(1e-4), tf.sum(tf.square(w.read())))) as tf.Scalar;
          }
        }

        const totalLoss = tf.add(tf.add(vLoss, pLoss), l2Sum) as tf.Scalar;

        // Capture scalar values for return (inside tidy, so we read before
        // tidy disposes).
        valueLoss = (vLoss.dataSync() as Float32Array)[0];
        policyLoss = (pLoss.dataSync() as Float32Array)[0];

        return totalLoss;
      }),
      true, // returnCost
    );

    const totalLoss = valueLoss + policyLoss;

    // Dispose the returned cost tensor.
    if (totalLossValue) totalLossValue.dispose();

    return { valueLoss, policyLoss, totalLoss };
  }

  // ── Serialization ─────────────────────────────────────────────────────────

  /**
   * Serialize the model to a self-contained Uint8Array suitable for storage
   * via BlobStorage.put().
   *
   * Wire format:
   *   [0..3]   little-endian uint32: byte length of the header JSON
   *   [4..4+L) UTF-8 header JSON: { modelTopology, weightSpecs }
   *   [4+L..)  raw weight data (ArrayBuffer from tfjs)
   *
   * Both `modelTopology` and `weightSpecs` are top-level fields in the tfjs
   * ModelArtifacts object. We bundle them together in a single JSON header so
   * the deserializer can reconstruct the full ModelArtifacts without any
   * external manifest. The weight binary follows immediately after the header.
   *
   * This format is intentionally simple: one contiguous blob, no zip, no
   * external manifest file. The tfjs model.save() IOHandler is used
   * internally to extract the topology + weight bytes; tf.loadLayersModel +
   * tf.io.fromMemory() restores them.
   */
  async serialize(): Promise<Uint8Array> {
    let artifacts: tf.io.ModelArtifacts | null = null;

    const saveHandler = tf.io.withSaveHandler(async (a) => {
      artifacts = a;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON' as const,
        },
      };
    });

    await this._model.save(saveHandler);

    if (artifacts === null) throw new Error('ChessNet.serialize: save handler not called');

    const a = artifacts as tf.io.ModelArtifacts;

    // Bundle modelTopology and weightSpecs into one JSON header so the
    // deserializer has everything it needs in one place.
    const header = {
      modelTopology: a.modelTopology,
      weightSpecs: a.weightSpecs ?? [],
    };
    const headerJson = JSON.stringify(header);
    const headerBytes = new TextEncoder().encode(headerJson);

    const weightBytes = new Uint8Array(a.weightData as ArrayBuffer);

    // Pack: 4-byte length header + header JSON + weights.
    const out = new Uint8Array(4 + headerBytes.length + weightBytes.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, headerBytes.length, /* littleEndian= */ true);
    out.set(headerBytes, 4);
    out.set(weightBytes, 4 + headerBytes.length);

    return out;
  }

  // ── Interpretability ──────────────────────────────────────────────────────

  /**
   * Returns the stable names of layers whose activations can be retrieved
   * via `getActivations`. These names survive serialize/deserialize.
   */
  layerNames(): string[] {
    return [...ChessNet.INSPECTABLE_LAYERS];
  }

  /**
   * Return the post-activation output tensor values for the named layers,
   * given a single board encoding. Layers not found are silently omitted.
   *
   * This is the hook for the future interpretability / visualization viewer.
   * It creates a sub-model on the fly (not cached, so not suitable for
   * hot loops — call `predict` for inference).
   */
  getActivations(
    features: Float32Array,
    layerNames: string[],
  ): Record<string, Float32Array> {
    return tf.tidy(() => {
      const input = tf.tensor2d(features, [1, FEATURE_COUNT]);
      const result: Record<string, Float32Array> = {};

      for (const name of layerNames) {
        let subModel = this._activationModels.get(name);
        if (!subModel) {
          let layer: tf.layers.Layer | null = null;
          try {
            layer = this._model.getLayer(name);
          } catch {
            continue;
          }
          if (!layer) continue;
          // Cached because tf.model() registers in tfjs's internal model
          // registry; building one per call leaks. Disposing the sub-model
          // is not safe — it shares layer instances with _model, so its
          // dispose() would corrupt the parent.
          subModel = tf.model({
            inputs: this._model.inputs,
            outputs: layer.output as tf.SymbolicTensor,
          });
          this._activationModels.set(name, subModel);
        }
        const out = subModel.predict(input) as tf.Tensor;
        result[name] = out.dataSync().slice() as Float32Array;
      }

      return result;
    });
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /** Release tfjs tensors. Call when this instance is no longer needed. */
  dispose(): void {
    this._activationModels.clear();
    this._model.dispose();
    this._optimizer.dispose();
  }
}

// ── Top-level deserialize ───────────────────────────────────────────────────

/**
 * Reconstruct a `ChessNet` from bytes previously produced by
 * `instance.serialize()`. Compatible with BlobStorage.get() output.
 */
export async function deserialize(bytes: Uint8Array): Promise<ChessNet> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(0, /* littleEndian= */ true);

  const headerBytes = bytes.slice(4, 4 + headerLen);
  const weightBytes = bytes.slice(4 + headerLen);

  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson) as {
    modelTopology: tf.io.ModelJSON;
    weightSpecs: tf.io.WeightsManifestEntry[];
  };

  // tf.io.fromMemory with a full ModelArtifacts object (single-argument form).
  const artifacts: tf.io.ModelArtifacts = {
    modelTopology: header.modelTopology,
    weightSpecs: header.weightSpecs,
    weightData: weightBytes.buffer.slice(
      weightBytes.byteOffset,
      weightBytes.byteOffset + weightBytes.byteLength,
    ),
  };

  const loadHandler = tf.io.fromMemory(artifacts);
  const model = await tf.loadLayersModel(loadHandler);

  return createFromModel(model);
}

/**
 * Internal helper: wrap a loaded LayersModel in a ChessNet. Exported only
 * so `deserialize` can call the private constructor without a public
 * escape hatch.
 */
function createFromModel(model: tf.LayersModel): ChessNet {
  // Access private constructor via Object.create + property assignment.
  // This avoids exposing a public constructor while still allowing
  // `deserialize` (in the same module) to build instances.
  return new (ChessNet as any)(model) as ChessNet;
}
