# Reproduce

Every number in `README.md` is listed here next to the system that produced it. A number
without its identity is not a result.

## 0. The two backbone builds

| Name | model | engine binary | server | service config |
|---|---|---|---|---|
| **trained build** | `Qwen3.8-Flash-Next-GSQ-RCO` (IQ3_S GGUF) | `ce7559e09c26a53313d283e1da8b3773c28f1a51e93d700124ee7633965522d2` | `3ebd690b5a977939447f8b8f5c5d619edb9a9ba69a7860293ff6f1805803bbbf` | `b6b687b8ce1e68b19d5bc8e6d4ca11099a3211559e3769a783a5d761bacc1464` |
| **second build** | `Qwen3.8-Flash-Next-GSQ-RCO-abliterated` (IQ3_S GGUF) | `c1d3070c7c289fcb09875cb909ce9a5fd3270a5f5fc1b548a75df4bc773a3b7b` | `d85622eb988638e089c07d6692b09f09f516a17c74d20c6540f6b874c6f9b65d` | `5440f05cbf03c283c530b63202c4bea93948a5154aae9fcb22990b5f6b732238` |

Both run Strata 0.1.44.1 with the patches in `patches/`. Vision module (both):
`914d5da3650755c3711265a9c437e57d6058958afa145e40d4a70a66c4ddc63b`.

Shared by both: tokenizer `chat_template.jinja` `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`,
`tokenizer.json` `c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410…`;
head `weights/best.pt` `8d3f60094cc79bdbee2b66837963c8f9f094819a97fc6d774d24fbcbbb674d2b`;
normalisation `weights/norm.npz` `b00ddad2c2b0ddc0ae0c9dfc50fcc1ec16de1f9f21683a13698aadca2e9ab2b2`.

Feature protocol (identical in both): `prompt_version`
`lossless-state-final-assistant-token-v2`, archived layers `[15, 23, 29, 35, 41, -1]`, this head
reads `[23, 29, 35, 41, -1]`, pools `[span_last, span_mean]`, 4 residual streams × 2560,
`compute_backend = gpu_batched_prefill`. **A feature set from another backend is not
interchangeable with this one.**

### 0a. Build the server we ran on (pinned environment)

One build command, in a new directory (no model is loaded or service started):

```sh
GGML_DIR=/path/to/pinned-llama.cpp \
  bash patches/reproduce-build.sh /tmp/strata-repro-NEW
```

The script clones `Niko1221/Strata`, pins commit `296b59d2f73c8f04a805df29c2fa97655adace78`
(tag `v0.1.44.1`, tree `063c7282837ecfe9fccad829ff7d7873400f6777`), applies the single
`0001-strata-v0.1.44.1-jev-readout.patch`, verifies all patched files against the hashes in
`patches/manifest.json`, and builds the recipe recorded there. Default tools: CUDA 12.9,
sm70, one build job; overrides and external GGML requirements are documented in
`patches/README.md`. Clean apply and compilation are verified in
`patches/evidence/validation-{local,linux}.json`.

A rebuild on your machine will not be byte-identical to the binaries we ran - build paths
and toolchain leak into the image - and it does not need to be. What the released client
requires is the readout API the patch adds, checked at startup by `python -m jev.identity`.

## 1. Held-out splits — trained build

`compute_backend = gpu_batched_prefill`; head + normalisation as above; identity as in the
"trained build" row.

| Split | n | accuracy | ECE |
|---|---|---|---|
| dev | 1,996 | 74.55% | 0.0266 |
| test | 1,999 | 76.44% | 0.0280 |
| OOD | 1,991 | 83.78% | 0.0396 |
| cal | 1,732 | 77.42% | 0.0276 |

Produced by:

```bash
python -m jev.train --shards "$JEV_ARCHIVE_ROOT/jev-v4-gpu/cache" \
        --out runs/middle5-s1 --layers 23,29,35,41,-1 --seed 1 --epochs 8   # -> results/middle5-s1.results.json
JEV_ARCHIVE_ROOT=/path/to/archives python -m jev.eval_replay --split dev --limit 20 \
        --identity identity/runtime-identity.pre-run.json --confirm-window --out rows-smoke
python -m jev.check_baseline --rows rows-smoke.jsonl --expected-count 20 \
        --accepted baselines/middle5-s1.dev-predictions.jsonl
```

Historical full archived-feature replay reproduced 1488/1996 (dev), 1528/1999 (test),
1668/1991 (OOD), 1341/1732 (cal), matching the training run. The current command above
requests only a 20-item replay/live-comparison smoke, not those full historical counts.
Private archives are required and are not shipped. `jev/check_baseline.py` compares per item against
`baselines/middle5-s1.dev-predictions.jsonl`.

## 2. Held-out splits — second build, same frozen head

Same frozen head and input tokens; live readouts are compared against the same archived
reference features. The live feature values themselves are not the same; the backbone build differs.
The readout backend is named differently in three artifacts — the training manifest records
`batched-prefill-one-shot-r2`, the live server answers `batched-prefill` under contract
`jev-readout-v2`, and older notes write `gpu_batched_prefill`. They describe the same fast path,
but each name belongs to a specific artifact; do not treat them as interchangeable labels.
The table below is a **measurement on the abliterated backbone**.

| Split | n | accuracy | Δ vs trained build | McNemar p | ECE |
|---|---|---|---|---|---|
| dev | 1,996 | 73.80% | +68 / −83 | 0.2545 | 0.0366 |
| test | 1,999 | 74.94% | +71 / −101 | 0.0267 | 0.0412 |
| OOD | 1,991 | 83.43% | +55 / −62 | 0.5793 | 0.0433 |
| cal | 1,732 | 76.04% | +59 / −83 | 0.0532 | 0.0401 |
| **all** | 7,718 | **77.08%** | 78.06% → 77.08%, +253 / −329 | z = −3.15, p ≈ 0.0016 | — |

Produced by:

```bash
python -m jev.identity --base http://127.0.0.1:8100 --out identity/runtime-identity.run-20261018.json
JEV_ARCHIVE_ROOT=/path/to/archives python -m jev.run_all --confirm-window --out-prefix rows
python -m jev.analyze_flips --rows rows-dev.jsonl --split dev --out-flips flips-dev.jsonl
```

Supporting evidence: 8.94% of items changed answer; FP16 vectors byte-identical in only
0.086–0.089% of positions; no item had all of its vectors identical. A three-timepoint probe
(`jev/probe_drift.py`, same 24 items at start / middle / end with a real chat request in
between) gave identical results, which rules out scheduler state or chat-history accumulation
as the cause.

## 3. External benchmark — JevBench public 231

| Run | rendering | correct / 231 | strict-valid | ECE | channel p50 |
|---|---|---|---|---|---|
| trained build, live tokenizer | criteria | **198 (85.71%)** | 231/231, 0 renorm | 0.0349 | 1.82 s |
| second build, live tokenizer | criteria | 198 (85.71%) | 231/231, 0 renorm | 0.0253 | 1.97 s |
| second build, live tokenizer | label | 194 (83.98%) | 231/231, 0 renorm | — | 1.86 s |

```bash
python -m jev.eval_benchmark --identity identity/runtime-identity.run-20261018.json \
        --mode criterion_as_option --tokenizer-dir /path/to/served/tokenizer \
        --confirm-window --out bench-criteria
python -m jev.eval_benchmark --identity identity/runtime-identity.run-20261018.json \
        --mode label_as_option --tokenizer-dir /path/to/served/tokenizer \
        --confirm-window --out bench-labels
```

Scoring uses the benchmark's own `scoring.py` (strict Σ = 1 within 1e-3, renormalisation
within 2e-2). The trained-build run: 149,519 prompt tokens, 645 s wall clock, head 4.8 ms,
channel p50 1.82 s / p95 7.10 s, tier p50 original 1.52 s / easy 1.56 s / hard 3.04 s.
Temperature scaling fitted on our `cal` split makes JevBench calibration worse
(0.0253 → 0.0589 on the second build), so the shipped policy applies no temperature.

The third-party rows in README §2.1 are quoted from their authors' published runs and were not
rerun here.

## 3a. Release closeout acceptance

| Gate | Actual result |
|---|---|
| Patch application, source hashes, compilation | PASS on pinned sources; external GGML/toolchain needed |
| Genuine isolated CPU dependency install | PASS, Windows CPython 3.12 / torch 2.6.0+cpu profile |
| Unit tests + offline selfcheck in that venv | PASS (66 tests); one earlier selfcheck run hit an unexplained native crash that later passes did not reproduce |
| Tokenizer/template | PASS — 20/20 prompts' complete token IDs and all span positions matched pinned serving code |
| Online interface measurement | PASS — single head call, 20 head distributions, forced decode, both natural cascade routes |
| Baseline comparison | PASS — 1,996-row self-check; separately derived 20 archived rows match 20/20 |
| Original smoke row recovery | NOT RECOVERED — derived archived subset is not original live evidence |

The historical tables above describe the systems named in their own headers; they are not
claims about any particular current deployment.

The vendored tokenizer reads Strata's descriptor/vocab/merges/token-types and matches its Jinja
whitespace behavior. It does not treat the descriptor as a Hugging Face tokenizer serialization.
Head inference rounds normalized inputs through FP16, restores FP32, supplies the batch axis,
and places model/normalization/input on the requested device. CPU head use with remote GPU
prefill was exercised; this does not validate every CUDA/Python/platform combination.

Full new results: `evidence/final-validation-20261020/results.json`,
`evidence/live-closeout-20261020/`, and `evidence/dependency-install-20261020/`.
The natural four-item router test was selected from
the smoke's confidence extremes to exercise both routes, not to estimate benchmark accuracy.

## 4. Figures

`docs/fig/make_figs.py` generates all eight SVGs without running a model or contacting a
service. Training metrics, reliability bins and coverage curves come from the adjacent
`_run_metrics.json`, `_splits_replay.json` and `_dev_thresholds.json` aggregates (no item text
or ids). Latency and backbone-swap summaries retain the published values encoded in the
generator. These aggregates are plot inputs, not a replacement for per-item source evidence.

```bash
npm ci --prefix docs/fig               # locked renderer: @resvg/resvg-js 2.6.2
python docs/fig/make_figs.py
node docs/fig/render.cjs               # all eight PNGs, 1600 px wide
```

The renderer checks PNG dimensions and prints content hashes. Rendering is deterministic on
the same renderer/font environment; the figures use Arial with Helvetica/sans-serif fallbacks.
Charts distinguish archived features from second-build live readouts; shared feature protocol
does not mean identical feature values. Depth results are single-seed, and the scaling curve
uses the three-input-layer baseline rather than the released five-input-layer head.

## 5. Known hash changes inside this repository

| File | before | after | why |
|---|---|---|---|
| `jev/experiments.json` | `d6a9ba964d9f…` | `42b611cd3438…` | subset paths made relative to `$JEV_DATA` (no server-absolute paths in a public repository) |
| provenance JSON files | see `PROVENANCE.md` | see `PROVENANCE.md` | server-absolute-path redaction (`/home/<user>` → `<JEV_ROOT>`); original digests recorded there |
