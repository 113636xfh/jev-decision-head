# Shipped weights

| File | sha256 | Size | What it is |
|---|---|---|---|
| `best.pt` | `8d3f60094cc79bdbee2b66837963c8f9f094819a97fc6d774d24fbcbbb674d2b` | 52,787,698 B | The decision head: 36 tensors, **13,194,368 parameters** (13.19 M), FP32 |
| `norm.npz` | `b00ddad2c2b0ddc0ae0c9dfc50fcc1ec16de1f9f21683a13698aadca2e9ab2b2` | 819,694 B | Per (pool, layer, dimension) z-score statistics, fitted on **train split only** |

Verified against the training run on 2026-10-20. The loader re-hashes on load and refuses a
mismatch.

## What these weights are bound to

These weights are **not** a standalone model. They are a readout over the intermediate
representations of one specific backbone build, and they assume the exact feature protocol
shipped in `jev/`:

- backbone: `Qwen3.8-Flash-Next-GSQ-RCO`, used as released — no fine-tuning, no LoRA, no
  gradient into it;
- readout layers **23, 29, 35, 41, last** (the archive protocol stores 15 as well; this head
  reads five of the six), pooling `last` + `span mean`, 4 residual streams (4 × 2560 per tap);
- spans: question stem, every candidate, final decision position;
- `compute_backend`: the batched-prefill GPU path (features from a different backend are not
  interchangeable — see `REPRODUCE.md`).

Re-running the frozen head against a different build of the same model moved accuracy
78.06% → 77.08% over 7,718 items and made calibration worse (README §2.3). The loader therefore
refuses to run without a compatibility certificate naming the exact build.

## Licence

Weights: MIT (same as the code). The backbone weights, the feature archives and the third-party
datasets behind the training data are **not** re-distributed — see `README.md` §8.
