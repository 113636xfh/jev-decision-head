# Jev Decision Head

A lightweight Jev-style decision head, plus the Strata patches that let one resident Strata
process serve **both** LLM generation and Jev-style typed decisions from a single set of
weights. This repository ships the trained decision-head weights, the Strata interface
patches, and the training / evaluation code used to get the numbers below.

**The model the head was trained on**: `Qwen3.8-Flash-Next-GSQ-RCO`, served by Strata 0.1.44.1.
The head is a bolt-on: the base model is used as released, **no fine-tuning, no LoRA, no
gradient into it**. Every number in this README was produced by that base model unless a
paragraph says otherwise.

**A Jev-style decision**: the caller supplies a *state* (text or JSON) and a question whose
**answer space is declared up front** — a set of options, a yes/no criterion, or an ordered
rating scale. The model returns a probability distribution over exactly that space in a
single forward pass; no text is generated, so there is nothing to parse. Because the answer
space is known in advance, all candidates are scored in parallel inside one read of the
input instead of being decoded token by token. (Terminology follows the upstream typed-decision
interface and the JevBench methodology document; links in §8.)

On the 231 publicly released JevBench items this head answers **198/231 (85.71%)** while
running inside the same resident model process that serves chat — no second model, no second
copy of the hardware ([§2](#2-accuracy)).

![Figure 1 — one backbone, two interfaces](docs/fig/architecture.png)
*Figure 1. The prompt is read once by the resident backbone. Generation continues as usual;
in parallel, readouts from five backbone layers feed a 13.19 M-parameter head that returns a
probability for each declared option.*

## 1. Why this design

A serving stack that already keeps a large model resident has already paid to load it and to
understand the context. Adding a separate decision model means choosing between two bad
options: a model small enough to be cheap does not carry enough world knowledge and
generalises poorly to unseen tasks, while a model large enough to be good needs another
copy of the hardware.

We keep the typed-decision interface exactly as it is defined upstream, and move the readout
from the language-model head (or the probability of each option's text) to a small head over
the **intermediate representations of several backbone layers**, sharing the weights that are
already serving generation. The head is 13.19 M parameters (~50 MiB in FP32) and reads five
backbone layers; the backbone is never updated and no gradient flows into it.

The head returns a distribution. What you do with a low-confidence one — fall back to the
generation path of the same backbone, ask a human, refuse — is a caller-side policy; we ship
a router as a worked example, not as the subject of this work.

## 2. Accuracy

Every number is bound to the exact system that produced it: engine binary, server, service
config, model/tokenizer, head and normalisation parameters, and the compute backend. Unless a
paragraph says otherwise, all of it was produced by the model named above, with the head and
feature protocol shipped in this repository. `REPRODUCE.md` lists the hashes and the command
that produced each table.

The split names used in the tables mean:
**dev** (1,996) — checkpoint and threshold selection only; **test** (1,999) — final
evaluation, never used for selection; **OOD** (1,991) — task families and sources that never
appeared in training, i.e. a generalisation probe; **cal** (1,732) — a split held out for
temperature scaling / calibration. No item appears in two splits, and splitting is done by
parent group and semantics, never by id arithmetic.

### 2.1 Public benchmark — JevBench, 231 released items

The comparison below is not model-vs-model: our entry keeps a released base model **unchanged**
and adds a small head, while every other row is a complete model on its own.

| System | Base | Head | Correct / 231 | p50 latency |
|---|---|---|---|---|
| **Jev Decision Head (this repo)** | `Qwen3.8-Flash-Next-GSQ-RCO` | **13.19 M** (base untouched) | **198 (85.71%)** — identical for both option renderings | 1.82 s (see §3) |
| Jev 1.13.0 | closed (hosted API) | — | 200 (86.58%) | 291 ms |
| Open-Jev 27B v1.1 | Qwen3.5-27B | — | 197 (85.28%) | not printed |
| Open-Jev 9B | Qwen3.5-9B | — | 179 (77.49%) | 189 ms |
| Open-Jev 2B | Qwen3.5-2B | — | 150 (64.94%) | 138 ms |
| Kev 0.8B | Kev 0.8B | — | 138 (59.74%) | — |
| Laya typed-decisions | ModernBERT 421M | — | 124 (53.68%) | — |
| GPT-5.6 Luna | closed (hosted API) | — | 206 (89.18%) | 954 ms |
| GPT-6 Astra | closed (hosted API) | — | 231 (100%) | 2,206 ms |

Our run is a single pass with a live tokenizer, **231/231 strict-valid** probability vectors
(0 renormalisations), scored by the benchmark's own `scoring.py`; the two option renderings
(“label” vs “criteria”) both give 198/231, with only 3 items changing answer. A leakage check
against all 23,718 of our own items found 0 exact state matches, and we never trained, fitted
a temperature or chose a threshold on these questions. The public subset is 231 of the
benchmark's 534 items; the remaining 303 are not available to us, so this is **not** an
official Intelligence / Calibration / Capability score and not a leaderboard rank. Latency is
not comparable to a hosted decision API (§3). Third-party rows are quoted from their authors'
published runs ([Open-Jev benchmarks](https://zefan-cai.github.io/open-jev/benchmarks/),
[Jev News public-subset table](https://jevainews.com/compare/),
[OpenDecisions / Inner Jev, Table 3](https://arxiv.org/html/2610.17205)); we did not rerun
them.

### 2.2 Held-out splits (7,718 items)

These splits are derived from public decision datasets —
[`pngwn/typed-decisions-v2`](https://huggingface.co/datasets/pngwn/typed-decisions-v2) and
[`pngwn/typed-decisions-v2-system-one`](https://huggingface.co/datasets/pngwn/typed-decisions-v2-system-one),
whose own upstream sources include [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
and [MMLU-Pro](https://huggingface.co/datasets/MrTornado/MMLU-Pro) — plus internal control
tasks for the OOD split. We do not re-distribute them; the repository ships the split manifests
(item ids, group ids and hashes) so the same four splits can be rebuilt from the upstream data.

| Split | accuracy | ECE |
|---|---|---|
| dev (1,996) | 74.55% | 0.0266 |
| test (1,999) | 76.44% | 0.0280 |
| OOD (1,991) | 83.78% | 0.0396 |
| cal (1,732) | 77.42% | 0.0276 |

These are held-out numbers from the base model named at the top; the checkpoint and every
threshold were chosen on dev alone. Note that the benchmark's 85.71% and our test 76.44% are
different distributions and splits — they are not comparable and not a progress curve.

![Figure 2 — reliability](docs/fig/reliability.png)
*Figure 2. Reliability diagram (dev, per item) and ECE by split for the same frozen head on the
trained build and on a second build of the same model.*

### 2.3 Portability: the head belongs to one model build

We re-ran the **frozen** head, with the archived features and identical input tokens, against
a second build of the same model (different weights, `…-abliterated-IQ3_S`). Nothing in the
head or the feature protocol changed. Accuracy moved 78.06% → 77.08% over 7,718 items
(+253/−329, z = −3.15, p ≈ 0.0016); on test alone −1.50 pt, McNemar p = 0.027; 8.94% of items
changed their answer, and ECE rose (dev 0.0266 → 0.0366, test 0.0280 → 0.0412, OOD 0.0396 →
0.0433, cal 0.0276 → 0.0401). Applying a temperature fitted on the first build made
calibration worse rather than better. Per-family effects do not extrapolate: on test the loss
concentrates in choice items (−2.68 pt) while score items gain (+3.09 pt), and on OOD choice
items gain (+1.06 pt).

That is the reason for the identity machinery in §4: a decision head is not portable across
model builds for free, and any number you publish has to name the build it came from.

![Figure 3 — backbone swap](docs/fig/backbone-swap.png)
*Figure 3. Same head, same features, same tokens; only the backbone build differs.*

## 3. Speed

Measured on this machine; please quote it together with the numbers.

| Component | Configuration |
|---|---|
| CPU | Intel Xeon E5-2673 v4 @ 2.30 GHz (40 threads) |
| GPU | 3× Tesla V100-SXM2 16GB in use (a fourth is installed and unused) |
| RAM | 94 GB |
| NVLink | none |

- The head itself: **4.8 ms** per decision once features are available.
- The whole decision channel (read the prompt + read out + score): **p50 1.82 s / p95 7.10 s**
  over the 231 public items (149,519 prompt tokens, 645 s wall clock). Per tier: original
  1.52 s, easy 1.56 s, hard 3.04 s — the long policy/multi-hop prompts dominate p95.
  Essentially all of that is reading the prompt once; a request with multi-layer readout does
  not reuse the prefix cache by design (at equal length it costs ~0.1 s more than a plain
  chat request).

This is a 2016-era platform's prefill bandwidth, **not the cost of the method**. On a modern
GPU (more memory, higher bandwidth, prefix caching) that part drops substantially; the head
cost stays at the millisecond level.

![Figure 4 — where the time goes](docs/fig/latency-breakdown.png)
*Figure 4. Head cost versus channel cost, and channel p50 by prompt tier.*

## 4. How it works

![Figure 5 — feature layout](docs/fig/readout-layout.png)
*Figure 5. Spans, layers and pooling that define one feature vector.*

- **Input layout.** Full prompt + all candidates + the final decision position. Readouts are
  taken over several text spans: the question stem, each candidate, and the decision position.
- **Readout.** The archive stores 6 backbone layers (15/23/29/35/41/last) × 2 pooling modes
  (last position, span mean) × 4 residual streams. **This head reads 5 of them**
  (23/29/35/41/last); the per-record feature vector is stored as FP16 with a per-record hash.
  The layer set and span set are part of the feature protocol: changing them changes the
  meaning of the weights, so they are recorded in the archive manifest
  (`prompt_version: lossless-state-final-assistant-token-v2`).
- **Head.** Shared projection → candidate-set encoder (candidates see each other) → one score
  per candidate; 13.19 M parameters. Trained against soft targets; the checkpoint is selected
  on dev NLL only, with early stopping on no improvement.
- **Normalisation.** Layer scales differ by a factor of ~33, so a per (pool, layer,
  dimension) z-score is required; it is fitted on train data only and shipped with the head.
- **Router (example policy).** Confidence / margin / normalised entropy decide whether to
  accept the head's answer. Thresholds are chosen on an independent dev split only; if no
  operating point is feasible, everything falls back. The shipped
  `policies/dev-thresholds.json` was produced by `jev/tune_policy.py` from the dev predictions
  in `baselines/` alone: `min_confidence 0.80`, `min_margin 0.05`,
  `max_normalised_entropy 0.40` → 42.2% coverage at 95.5% accuracy on dev. On dev alone,
  `conf ≥ 0.85` accepts 44.6% of items at 94.8% accuracy; `conf ≥ 0.95` accepts 31.8% at 98.3%.

![Figure 6 — risk and coverage](docs/fig/risk-coverage.png)
*Figure 6. Accuracy on the accepted subset as a function of coverage, head answering directly.*

- **Depth does not buy accuracy.** With everything else fixed, 2 / 4 / 8 transformer blocks in
  the head give dev NLL 0.6523 / 0.6680 / 0.6770 and test accuracy 75.99 / 76.34 / 75.89% —
  inside seed noise, and 2 blocks has the best dev NLL. A wider head (d = 1024) was also worse
  on dev NLL (0.6739). We report this as a negative result rather than picking the best test
  number.

![Figure 7 — head depth](docs/fig/head-depth-ablation.png)
*Figure 7. Head depth ablation: one variable per run, architecture never selected on test.*

- **Identity binding.** Every evaluation records the running system's binary/config/weight
  hashes and re-verifies them during the run; if the service changes mid-run, the run refuses
  to continue.

## 5. Strata patches

Five patches against Strata 0.1.44.1 are needed on the serving side; `patches/README.md` gives
the per-patch technical report (files and functions touched, blast radius, switch, acceptance
numbers). They are intended as upstream PRs, and until they are merged you apply them from
`patches/` before starting Strata.

| Patch | What it changes | Why it is needed |
|---|---|---|
| `0001-session-kv-retention` | a slot's KV stays resident across pipeline stages instead of being released (`serve/server.py`, `src/program/generate.cpp`) | a decision request and the chat request that produced its context share a slot; without retention every decision re-reads the whole conversation |
| `0003-latent-readout` | adds the readout request path and the capability bit the client checks (`src/program/generate.cpp`) | the head scores hidden states; generating text and parsing it is slower and loses the distribution |
| `0004-latent-taps` | one forward pass can be read at several layers, with schema validation (`generate.cpp`, `src/core/verify.cpp/.hpp`, `serve/server.py`) | the protocol reads six layers; six passes would multiply the prefill cost |
| `0005-latent-spans-and-pooling` | span selection (stem / each option / decision position) and pooling (last, span mean) on the engine side | if engine and training extractor disagree about spans, the weights mean something else |
| `0006-decision-region-and-tap-read-path` | defines the decision region and fixes the tap read path to it | a readout taken from the wrong position silently produces a different feature |

Measured effect of the readout chain: with taps on the batched-prefill path, 402/512/950-token
requests run 1.62× / 2.32× / 2.87× faster than the serial small-window path (107 tok/s) used
before. Readouts are verified byte-for-byte against the archived features for identical tokens
(`certs/live-readout-compatibility-r2-bitwise.json`), and a request that asks for readouts on a
server without the capability is rejected rather than silently degraded.

**Caveats we publish rather than hide.** Multi-layer readout does not reuse the prefix cache by
design; feature values depend on expert placement, so a CPU-verified feature set and a GPU
batched-prefill feature set are **not** interchangeable (`compute_backend` is part of the
feature protocol); the capability bit is reported only under `batch_groups ≤ 2`.

## 6. Quick start

```bash
export JEV_DATA=$HOME/jev-data          # archives, split subsets, evaluation records
export STRATA_ROOT=$HOME/strata         # patched Strata tree
export CUDA_VISIBLE_DEVICES=3           # the harness GPU; never the serving GPUs

# 1. build Strata with the patches, then start it with the shipped config
git apply patches/*.patch && ./start.sh

# 2. record the runtime identity (required, not optional)
python -m jev.identity --base http://127.0.0.1:8100 --out runtime-identity.json

# 3. score one item with the shipped head
python -m jev.predict --identity runtime-identity.json \
  --state '{"ticket": "..."}' --question "route" \
  --options '["billing","account","other"]'

# 4. or use the router (thresholds from dev only; the shipped file is policies/dev-thresholds.json)
python -m jev.cascade --identity runtime-identity.json --policy policies/dev-thresholds.json
```

Without a compatibility certificate matching your pair, the loader refuses to run rather than
silently scoring with the wrong features.

**This path is tested, not aspirational.** Replaying 20 dev items through the code in this
repository reproduced the accepted 2026-10-15 baseline exactly — 20 compared, 0 gold
mismatches, 0 prediction mismatches (`evidence/smoke-20-dev-20261020/smoke20-check.json`, with
the runtime identity that produced it). Reproduce with:

```bash
python -m jev.eval_replay --split dev --limit 20 --out rows-smoke20 \
        --identity runtime-identity.json --confirm-window
python -m jev.check_baseline --rows rows-smoke20.jsonl \
        --accepted baselines/middle5-s1.dev-predictions.jsonl
```

## 7. Training your own

```bash
python -m jev.extract  --endpoint ... --items data/train.jsonl --out archive/
python -m jev.train    --archive archive/ --config jev/experiments.json --run middle5-s1
```

`jev/experiments.json` is the pre-registered experiment matrix: one variable per run, fixed
splits / seeds / layer sets / budget. `jev/check_baseline.py` re-checks a run against a known
prediction file so a framework bug is never mistaken for a model change.

Labelled-data volume was the variable that kept paying: with everything else fixed, 2k / 4k /
8k / 16k train items give dev accuracy 66.48 / 67.54 / 69.94 / 71.94% and test accuracy
68.28 / 70.24 / 72.09 / 75.99%.

![Figure 8 — data scaling](docs/fig/data-scaling.png)
*Figure 8. Training-scale learning curve, same head and protocol.*

## 8. Data and weights

| Item | Where it comes from | Redistributed here |
|---|---|---|
| Public benchmark | [JevBench](https://jevbench.dev/) · [methodology](https://benchmarkheaven.com/jev-models/methodology) | no (231 items fetched by the eval script) |
| Decision datasets behind our splits | [`pngwn/typed-decisions-v2`](https://huggingface.co/datasets/pngwn/typed-decisions-v2) and [`-v2-system-one`](https://huggingface.co/datasets/pngwn/typed-decisions-v2-system-one), whose own upstream pins include [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) and [MMLU-Pro](https://huggingface.co/datasets/MrTornado/MMLU-Pro) | no |
| Our splits (dev 1,996 / test 1,999 / OOD, cal) | derived from the above, isolated by parent group and semantics — never by id arithmetic | no; manifest + hashes |
| Feature archives | produced by `jev.extract` | no; hashes in the archive manifest |
| Backbone weights | `Qwen3.8-Flash-Next-GSQ-RCO` (the model this head was trained on) | no (re-distribution unconfirmed) |
| **Head weights + normalisation** | this repository (`weights/best.pt`, `weights/norm.npz`) | **yes** |

## 9. Licence

Code, patches and weights: MIT (`LICENSE`). Data, feature archives, backbone weights and
third-party datasets are not re-distributed — see `NOTICE` and §8.
