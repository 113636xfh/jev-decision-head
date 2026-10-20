# System identities: which build answered

A runtime identity says **what ran**: engine / server / config / vision file hashes (read
through `/proc`), model name and pack paths, tokenizer file hashes, engine capacity keys,
head file hashes and the readout layer taps. `python -m jev.identity` records it on the
serving host; every harness re-checks it before and during a run and stops when the server
changes mid-run. The environment we ran on is pinned in `REPRODUCE.md` §0a — a rebuild on
your machine will not be byte-identical and does not need to be; the client contract is the
readout API the patch adds, not a binary hash.

`manifests/system-identities.json` is the machine-readable, path-free index; full original
and shipped artifact-byte hashes are in `manifests/artifact-digests.json`.

## 1. Three identities that must not be merged

Prefixes below identify full 64-digit hashes in the index; they are navigation aids, not
truncated matching rules.

| Role | Engine | Server | Config | What it supports |
|---|---|---|---|---|
| **Training / extraction archive** | `44f99383…` | `32368d78…` | `5e7a3152…` | Original feature archive, normalization and trained head; archived-feature held-out baselines |
| **Later adaptive runtime (2026-10-15/16)** | `ce7559e0…` | `3ebd690b…` | `b6b687b8…` | Bounded empirical decision/routing acceptance for the old head, not byte-identical archive extraction |
| **Abliterated measurement (2026-10-18/19)** | `c1d3070c…` | `d85622eb…` | `5440f05c…` | Explicit frozen-head remeasurement on the abliterated backbone |

The training/extraction assets identify the IQ3_S `Qwen3.8-Flash-Next-GSQ-RCO` model;
the archive manifest's artifact names refer to the older runtime tree. It is incorrect to
rename the later Strata 0.1.44.1 adaptive serving pair as the system that originally generated
all training features. This is a distinction between recorded systems, not a deployment
instruction.

The abliterated run reused the same head, normalization, archived token IDs, spans and
feature protocol for held-out comparisons; its **live feature values were different**.
Weights, engine, server and config identities differ, so this is not a clean weights-only
causal ablation. The two `identity/runtime-identity.*20261018.json` files both describe
measurement-stage systems, not the training archive.

## 2. Results mapped to the system that produced them

| Evaluation | System / input | Counts or scope |
|---|---|---|
| Archived dev/test/OOD/cal | Original archived features, training identity | 1,488/1,996; 1,528/1,999; 1,668/1,991; 1,341/1,732 (74.55 / 76.44 / 83.78 / 77.42%) |
| Adaptive runtime dev impact | Historical comparison on the adaptive runtime | Old 1,488 → new 1,486 correct; 3 prediction flips, 2 routing changes; not a new archive baseline |
| JevBench public 231, adaptive runtime | Live tokenizer/readout, frozen head, no decode fallback | 198/231 (85.71%); both renderings had that aggregate, not identical per-item answers |
| Abliterated held-out measurement | Same archived input IDs/spans, new live readouts | 1,473/1,996; 1,498/1,999; 1,661/1,991; 1,317/1,732 |
| Abliterated JevBench public 231 | Live tokenizer/readout | Criteria 198/231; labels 194/231; not an official full/sealed benchmark score |

Archived-feature correctness is not evidence of byte equality with a later live service.
The aggregate 7,718-item comparison is 6,025 correct (78.06%) versus 5,949 (77.08%), with
690 prediction flips. Its item-level nominal significance does not account for correlated
parent groups.

The published 20-item smoke compares **`pred_old`** with historical archived predictions:
20 compared / 0 mismatch. The same smoke records 3 live/archive prediction flips and zero
items with all vectors identical. It does not prove a new live pair or the shipped patch
chain is compatible with your stack.

## 3. Feature protocol: exact spelling matters

The archive's protocol value is **`batched-prefill-one-shot-r2`**. The phrase
`gpu_batched_prefill` in earlier narrative documentation is only a descriptive shorthand,
not a byte-compatible alternate protocol identifier that a validator may silently accept.
CPU verify features are not interchangeable with this GPU batched-prefill archive.

- Prompt version: `lossless-state-final-assistant-token-v2`.
- Archive layers: `[15, 23, 29, 35, 41, -1]`; selected head: `[23, 29, 35, 41, -1]`.
- Archive pools: `[span_last, span_mean]`; live request spellings `last,mean` describe the
  corresponding API choices, not permission to change pooling definitions.
- Four streams × 2,560 dimensions = 10,240 values per tap, stored FP16 in the archive.
- Stem, every candidate and final assistant decision-token spans; original state JSON is
  lossless. `-1` aliases final layer 47 in the 48-layer backbone, not an extra layer.
- Each item's tensor varies with candidate count; sharing weights/protocol does not mean
  one fixed-size vector, the same feature values, or reuse of a prior chat's hidden states.
- Head and normalization hashes are fixed in the identity index. Normalization is fitted
  on train only; no backbone gradients, LoRA or task weighting were used for this head.

## 4. Apply/build success is a different gate

Keep at least these gates distinct:

| Gate | What a pass establishes | What it does not establish |
|---|---|---|
| Patch apply on a pinned clean upstream | The particular patch series applies to that source tree | Completeness of unshipped prerequisites or successful compilation |
| Build / startup / interface and safety tests | That compiled system implements the tested interface/behavior | Exact archived features, head calibration or unchanged scheduling numerics |
| Explicit accuracy/drift measurement | Observed outcomes on named inputs and identity | Compatibility with a different backbone, engine, config or policy |

A new build is not the historical binary just because its version string is 0.1.44.1 or
because the same patch filenames were used. Numbers measured on one row of the table in §1
belong to that row; comparing across rows requires re-measurement on each.
