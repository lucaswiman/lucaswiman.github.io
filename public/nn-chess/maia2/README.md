# Maia-2 weights — third-party redistribution

`weights.bin` and `manifest.json` in this directory are a port of the
**Maia-2 "rapid" pretrained checkpoint** from CSSLab/Toronto, served as
a static asset so the browser play page doesn't have to round-trip to a
release server (the original `/nn-chess/maia2/manifest.json` 404'd on
the deployed site).

## Source

- Upstream project: https://github.com/CSSLab/maia2
- Paper: McIlroy-Young, R., et al. *Maia-2: A Unified Model for Human-AI
  Alignment in Chess*. NeurIPS 2024.
- Pretrained checkpoint: the `rapid` variant fetched by
  `maia2.model.from_pretrained("rapid", ...)`
  (`rapid_model.pt`, ~267 MiB PyTorch state dict).

The conversion that produced these files runs every tensor through the
TS-engine-friendly layout (Conv2d kernels NCHW → HWIO, everything packed
into a single flat blob with a JSON offset/shape/dtype manifest). The
script lives at `nn-chess/scripts/convert-maia2.py` in this repo.

The two files in this directory:

- `weights.bin` — packed fp16 tensors, 40.5 MiB,
  `sha256 d475c4a6a94d1a1e56b634abdd3f61eb23892319154b4c2857b8387054a8ba99`
- `manifest.json` — version, dtype, total bytes, sha, and per-tensor
  `{name, offset, shape, dtype}` entries.

## Licence

Maia-2 is released under the **MIT Licence**. The upstream `LICENSE`
file (reproduced verbatim as `LICENSE` next to this README) leaves the
copyright line as `Copyright (c) [year] [fullname]` — that is what the
upstream repo publishes, and it is preserved here unchanged so this
copy matches the source exactly. Practically, attribution belongs to
the **Maia-2 authors at the Computational Social Science Lab,
University of Toronto** (see the paper above).

We claim no copyright on `weights.bin` or `manifest.json`; they are a
mechanical transform of the upstream checkpoint and are redistributed
under the same MIT licence. If anything in this directory is ever
removed, the play page falls back to "Maia weights not available."

## How to regenerate

```bash
pip install maia2 numpy torch gdown
python3 -c "import maia2.model as m; m.from_pretrained('rapid', 'cpu', save_root='/tmp/maia2_dl')"
python3 nn-chess/scripts/convert-maia2.py \
  --ckpt /tmp/maia2_dl/rapid_model.pt \
  --out-dir /tmp/maia2_out \
  --dtype fp16
cp /tmp/maia2_out/weights.bin /tmp/maia2_out/manifest.json public/nn-chess/maia2/
```
