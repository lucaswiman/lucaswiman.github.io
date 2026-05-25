#!/usr/bin/env python3
"""
convert-maia2.py — convert a Maia-2 PyTorch checkpoint into a layout the
TypeScript engine can stream from a single fetch.

Pipeline:
  1. Load the checkpoint (default: the `rapid` model from
     `maia2.model.from_pretrained`).
  2. Read every tensor out of `model_state_dict`.
  3. Convert Conv2d kernels from PyTorch NCHW (out, in, kH, kW) layout to
     the HWIO layout TF.js expects (kH, kW, in, out). Linear / norm /
     embedding / bias tensors are left as-is.
  4. Concatenate every tensor (in a stable order) into one flat binary
     blob. Default dtype is fp16; pass `--dtype fp32` for full precision.
  5. Emit `manifest.json` describing each tensor's offset, shape, and
     dtype, plus a sha256 of the blob.

The output files (`weights.bin`, `manifest.json`) are written to
`--out-dir`. They are NOT committed to the repo — the blog ships them
either as a GitHub release asset fetched at runtime, or (later) as
static assets under `public/nn-chess/maia2/`. Either way the
production-time download is fired from the browser and cached in
IndexedDB via the existing storage adapter; see
`nn-chess/ui/maia-worker.ts`.

In addition, when `--ref-fen` is supplied, the script also dumps a JSON
fixture under `--fixtures-dir` containing the post-backbone,
post-attention, policy-logits and value-output activations for that
position. The TS parity test loads the same fixture to verify the
ported forward pass matches the PyTorch reference to within float-16
tolerance.

Licence: this conversion script depends on the Maia-2 model
(MIT-licensed, https://github.com/CSSLab/maia2). It is part of the
nn-chess subproject; see nn-chess/core/maia/index.ts for the MIT
attribution on the TS side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# ── Architecture constants (mirrors maia2/main.py + config.yaml) ─────────────

# Number of residual blocks in the ChessResNet backbone.
NUM_BLOCKS_CNN = 5
# Width of conv backbone (input/output channels of every residual layer).
DIM_CNN = 256
# Number of "channel patches" the attention block sees as its sequence.
VIT_LENGTH = 8
# Per-patch hidden dim into the attention block.
DIM_VIT = 1024
# Number of attention blocks (EloAwareAttention + FeedForward).
NUM_BLOCKS_VIT = 2
# Per-side ELO embedding dim. Self+oppo concat → 256.
ELO_DIM = 128
# Size of the action space; see maia2.utils.get_all_possible_moves().
POLICY_SIZE = 1880
# Side-info head output size: 1880 (policy duplicate) + 6 + 6 + 1 + 64 + 64.
SIDE_INFO_SIZE = 2021
# Number of ELO buckets in the embedding table.
NUM_ELO_BUCKETS = 11


# ── Tensor naming + conv-kernel transpose ────────────────────────────────────

# Conv2d weight tensors live in PyTorch NCHW kernel layout: (out, in, kH, kW).
# We re-pack to (kH, kW, in, out) so the TS side can drop straight into
# tf.layers.conv2d's expected `kernelInitializer` of HWIO.
CONV_WEIGHT_KEYS = {
    "module.chess_cnn.conv1.weight",
    "module.chess_cnn.conv_last.weight",
}
for i in range(NUM_BLOCKS_CNN):
    CONV_WEIGHT_KEYS.add(f"module.chess_cnn.layers.{i}.conv1.weight")
    CONV_WEIGHT_KEYS.add(f"module.chess_cnn.layers.{i}.conv2.weight")


def maybe_transpose(name: str, t: torch.Tensor) -> np.ndarray:
    """Strip the leading 'module.' (DataParallel artifact) is the caller's
    job; this function only does the conv-kernel transpose."""
    arr = t.detach().cpu().numpy()
    if name in CONV_WEIGHT_KEYS:
        # PyTorch: (out, in, kH, kW) → TF.js HWIO: (kH, kW, in, out).
        arr = np.transpose(arr, (2, 3, 1, 0)).copy()
    return arr


# ── Stable tensor ordering ────────────────────────────────────────────────────

# Hard-coded order so manifest layout is reproducible across runs and so the
# TS side can sanity-check it. Buffer integers / num_batches_tracked are
# intentionally omitted — inference doesn't need them, but BatchNorm running
# stats (running_mean / running_var / weight / bias) ARE included since they
# fold into the affine transform at inference time.

def ordered_tensor_names() -> List[str]:
    out: List[str] = []
    # Positional embedding.
    out.append("pos_embedding")

    # CNN stem.
    out.append("chess_cnn.conv1.weight")
    out.append("chess_cnn.bn1.weight")
    out.append("chess_cnn.bn1.bias")
    out.append("chess_cnn.bn1.running_mean")
    out.append("chess_cnn.bn1.running_var")

    # Residual blocks.
    for i in range(NUM_BLOCKS_CNN):
        prefix = f"chess_cnn.layers.{i}"
        out.append(f"{prefix}.conv1.weight")
        out.append(f"{prefix}.bn1.weight")
        out.append(f"{prefix}.bn1.bias")
        out.append(f"{prefix}.bn1.running_mean")
        out.append(f"{prefix}.bn1.running_var")
        out.append(f"{prefix}.conv2.weight")
        out.append(f"{prefix}.bn2.weight")
        out.append(f"{prefix}.bn2.bias")
        out.append(f"{prefix}.bn2.running_mean")
        out.append(f"{prefix}.bn2.running_var")

    # Final conv + bn → channel patches.
    out.append("chess_cnn.conv_last.weight")
    out.append("chess_cnn.bn_last.weight")
    out.append("chess_cnn.bn_last.bias")
    out.append("chess_cnn.bn_last.running_mean")
    out.append("chess_cnn.bn_last.running_var")

    # Patch embedding (Linear + LayerNorm).
    out.append("to_patch_embedding.0.weight")
    out.append("to_patch_embedding.0.bias")
    out.append("to_patch_embedding.1.weight")
    out.append("to_patch_embedding.1.bias")

    # Transformer (per-block).
    for i in range(NUM_BLOCKS_VIT):
        prefix = f"transformer.elo_layers.{i}"
        # Attention.
        out.append(f"{prefix}.0.norm.weight")
        out.append(f"{prefix}.0.norm.bias")
        out.append(f"{prefix}.0.to_qkv.weight")
        out.append(f"{prefix}.0.elo_query.weight")
        out.append(f"{prefix}.0.to_out.0.weight")
        out.append(f"{prefix}.0.to_out.0.bias")
        # FeedForward (LayerNorm, Linear, GELU, Dropout, Linear, Dropout).
        # nn.Sequential names: 0 = LN, 1 = Linear(dim,hidden), 4 = Linear(hidden,dim).
        out.append(f"{prefix}.1.net.0.weight")
        out.append(f"{prefix}.1.net.0.bias")
        out.append(f"{prefix}.1.net.1.weight")
        out.append(f"{prefix}.1.net.1.bias")
        out.append(f"{prefix}.1.net.4.weight")
        out.append(f"{prefix}.1.net.4.bias")

    out.append("transformer.norm.weight")
    out.append("transformer.norm.bias")

    out.append("last_ln.weight")
    out.append("last_ln.bias")

    # Heads. fc_1 is policy. fc_2 is side-info (not used at inference).
    # fc_3_1 + fc_3 form the value head.
    out.append("fc_1.weight")
    out.append("fc_1.bias")
    out.append("fc_3_1.weight")
    out.append("fc_3_1.bias")
    out.append("fc_3.weight")
    out.append("fc_3.bias")

    out.append("elo_embedding.weight")

    return out


# ── Pack to a single blob ────────────────────────────────────────────────────


def to_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp16":
        return arr.astype(np.float16)
    if dtype == "fp32":
        return arr.astype(np.float32)
    raise ValueError(f"unknown dtype {dtype}")


def pack(
    state_dict: Dict[str, torch.Tensor],
    dtype: str,
) -> Tuple[bytes, List[Dict[str, Any]]]:
    """Return (blob_bytes, manifest_entries) ordered by `ordered_tensor_names`."""
    blob_chunks: List[bytes] = []
    manifest: List[Dict[str, Any]] = []
    offset = 0
    bytes_per_el = 2 if dtype == "fp16" else 4

    for name in ordered_tensor_names():
        sd_key = f"module.{name}"
        if sd_key not in state_dict:
            raise KeyError(
                f"checkpoint missing expected tensor: {sd_key}.  "
                f"Available: {sorted(state_dict.keys())[:3]}..."
            )
        np_arr = maybe_transpose(sd_key, state_dict[sd_key])
        np_arr = to_dtype(np_arr, dtype)
        # Force C-contiguous so tobytes() matches the declared shape.
        np_arr = np.ascontiguousarray(np_arr)
        raw = np_arr.tobytes()
        size_bytes = len(raw)
        manifest.append({
            "name": name,
            "shape": list(np_arr.shape),
            "offset": offset,
            "size_bytes": size_bytes,
            "dtype": dtype,
        })
        blob_chunks.append(raw)
        offset += size_bytes
        assert size_bytes == np_arr.size * bytes_per_el, (
            f"size mismatch for {name}"
        )

    return b"".join(blob_chunks), manifest


# ── Reference-activation dump ────────────────────────────────────────────────


def _import_maia2_for_ref() -> Any:
    """Import the Maia-2 package (used for the reference activation dump).

    We do this lazily so the conversion-only path doesn't require the
    `maia2` pip package — only the `--ref-fen` path does.
    """
    try:
        from maia2 import model as maia2_model_mod, inference as maia2_inference  # noqa: F401
        from maia2.main import MAIA2Model  # noqa: F401
        from maia2.utils import (
            board_to_tensor,
            create_elo_dict,
            get_all_possible_moves,
            map_to_category,
        )
        return {
            "MAIA2Model": MAIA2Model,
            "board_to_tensor": board_to_tensor,
            "create_elo_dict": create_elo_dict,
            "get_all_possible_moves": get_all_possible_moves,
            "map_to_category": map_to_category,
        }
    except ImportError as e:
        sys.stderr.write(
            f"--ref-fen requires the maia2 package: pip install maia2.  "
            f"Original error: {e}\n"
        )
        sys.exit(2)


def dump_reference(
    state_dict: Dict[str, torch.Tensor],
    fen: str,
    elo_self: int,
    elo_oppo: int,
    out_path: Path,
) -> None:
    """Run the actual maia2 PyTorch model on (fen, elo_self, elo_oppo) and
    dump the activations the TS parity test needs."""
    deps = _import_maia2_for_ref()
    import chess  # noqa: F401  – guaranteed by maia2

    # Recreate the Config object the way maia2.model.from_pretrained does,
    # without re-downloading anything.
    class _Cfg:
        input_channels = 18
        dim_cnn = DIM_CNN
        num_blocks_cnn = NUM_BLOCKS_CNN
        dim_vit = DIM_VIT
        num_blocks_vit = NUM_BLOCKS_VIT
        vit_length = VIT_LENGTH
        elo_dim = ELO_DIM
        side_info = True
        value = True

    elo_dict = deps["create_elo_dict"]()
    all_moves = deps["get_all_possible_moves"]()

    model = deps["MAIA2Model"](len(all_moves), elo_dict, _Cfg())
    # Strip the "module." prefix added by DataParallel.
    stripped = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    model.load_state_dict(stripped)
    model.eval()

    # Replicate inference.preprocessing exactly so the fixture matches what
    # the high-level inference_each() call would have produced.
    import chess as chess_mod
    if fen.split(" ")[1] == "w":
        board = chess_mod.Board(fen)
        mirrored = False
    elif fen.split(" ")[1] == "b":
        board = chess_mod.Board(fen).mirror()
        mirrored = True
    else:
        raise ValueError(f"invalid FEN: {fen}")
    board_input = deps["board_to_tensor"](board).unsqueeze(0)
    # Capture the raw encoded board so the TS-side encoding parity test
    # can compare without reimplementing maia2.utils.board_to_tensor.
    encoded_board = board_input.cpu().numpy().astype(np.float32)

    es = torch.tensor([deps["map_to_category"](elo_self, elo_dict)])
    eo = torch.tensor([deps["map_to_category"](elo_oppo, elo_dict)])

    legal_idx_in_canonical = sorted(
        all_moves.index(m.uci()) for m in board.legal_moves
    )

    # We want intermediate activations: the post-backbone (post bn_last,
    # post conv_last) tensor, the post-attention tensor, and the policy /
    # value head outputs.
    activations: Dict[str, np.ndarray] = {}

    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        # Backbone.
        embs = model.chess_cnn(board_input)
        activations["backbone"] = to_np(embs)  # (1, 8, 8, 8) but actually (1, vit_length, 8, 8)
        flat = embs.view(1, embs.size(1), 8 * 8)
        patch = model.to_patch_embedding(flat)
        x = patch + model.pos_embedding
        activations["post_patch"] = to_np(x)

        es_emb = model.elo_embedding(es)
        eo_emb = model.elo_embedding(eo)
        elo_emb = torch.cat([es_emb, eo_emb], dim=1)
        activations["elo_emb"] = to_np(elo_emb)

        # Attention. The model.transformer applies dropout (eval=0) plus
        # the two attn + ff blocks plus the post-norm. We re-run those
        # ops to keep the intermediate.
        # The transformer drops out x BEFORE the first block (.dropout
        # member of MAIA2Model wraps the patch+pos sum). At eval time it's
        # a no-op so we just use `x`.
        for attn, ff in model.transformer.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x
        x = model.transformer.norm(x)
        activations["post_attn"] = to_np(x)

        pooled = x.mean(dim=1)
        pooled = model.last_ln(pooled)
        activations["pooled"] = to_np(pooled)

        logits_policy = model.fc_1(pooled)
        logits_value = model.fc_3(
            torch.relu(model.fc_3_1(pooled))
        ).squeeze(-1)
        activations["policy"] = to_np(logits_policy)
        activations["value"] = to_np(logits_value)

    # Top-5 policy moves by logits (after legal masking) for the parity test.
    # The TS test only checks rank order over the legal subset so we score
    # only the legal subset here.
    legal_set = set(legal_idx_in_canonical)
    legal_with_logits = sorted(
        ((i, float(activations["policy"][0, i])) for i in legal_set),
        key=lambda p: -p[1],
    )
    top5_legal = [
        {"index": int(i), "uci": all_moves[i], "logit": float(l)}
        for i, l in legal_with_logits[:5]
    ]

    fixture = {
        "fen": fen,
        "elo_self": elo_self,
        "elo_oppo": elo_oppo,
        "mirrored": mirrored,
        "legal_indices": legal_idx_in_canonical,
        "encoded_board_flat": [float(x) for x in encoded_board.reshape(-1).tolist()],
        "encoded_board_shape": list(encoded_board.shape),
        "shapes": {k: list(v.shape) for k, v in activations.items()},
        # Full activations would be huge; only keep the smaller ones in
        # the JSON. For the larger ones (post_patch / post_attn) we store
        # a compact summary (sum / max / first 8 + last 8).
        "value": float(activations["value"].reshape(-1)[0]),
        "policy_top5_legal": top5_legal,
        "pooled_summary": _summarize(activations["pooled"]),
        "post_patch_summary": _summarize(activations["post_patch"]),
        "post_attn_summary": _summarize(activations["post_attn"]),
        "backbone_summary": _summarize(activations["backbone"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2))


def _summarize(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.astype(np.float32).reshape(-1)
    return {
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "max": float(flat.max()),
        "min": float(flat.min()),
        "first8": [float(x) for x in flat[:8].tolist()],
        "last8": [float(x) for x in flat[-8:].tolist()],
        "n": int(flat.size),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the Maia-2 .pt checkpoint (the file from gdown)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write weights.bin + manifest.json",
    )
    ap.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Output dtype (default fp16, ~47MB)",
    )
    ap.add_argument(
        "--ref-fen",
        type=str,
        default=None,
        help="If supplied, also dump reference activations for this FEN",
    )
    ap.add_argument(
        "--ref-elo-self",
        type=int,
        default=1500,
        help="ELO for the side to move (default 1500)",
    )
    ap.add_argument(
        "--ref-elo-oppo",
        type=int,
        default=1500,
        help="ELO for the opponent (default 1500)",
    )
    ap.add_argument(
        "--fixtures-dir",
        type=Path,
        default=None,
        help="Where to write the reference-activation JSON. "
             "Required if --ref-fen is set.",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = {str(k): v for k, v in sd.items()}

    blob, manifest_entries = pack(sd, args.dtype)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = args.out_dir / "weights.bin"
    manifest_path = args.out_dir / "manifest.json"
    blob_path.write_bytes(blob)

    sha = hashlib.sha256(blob).hexdigest()
    manifest = {
        "version": 1,
        "dtype": args.dtype,
        "model": "maia2",
        "model_type": "rapid",
        "total_bytes": len(blob),
        "sha256": sha,
        "num_elo_buckets": NUM_ELO_BUCKETS,
        "policy_size": POLICY_SIZE,
        "tensors": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote {blob_path} ({len(blob) / (1 << 20):.2f} MiB, dtype={args.dtype})")
    print(f"wrote {manifest_path} ({len(manifest_entries)} tensors)")
    print(f"sha256: {sha}")

    if args.ref_fen is not None:
        if args.fixtures_dir is None:
            ap.error("--fixtures-dir is required when --ref-fen is supplied")
        ref_path = args.fixtures_dir / "reference.json"
        dump_reference(
            sd,
            args.ref_fen,
            args.ref_elo_self,
            args.ref_elo_oppo,
            ref_path,
        )
        print(f"wrote {ref_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
