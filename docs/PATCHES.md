# Complete Strata v0.1.44.1 patch package

## Fixed baseline and replacement rule

- Upstream: <https://github.com/Niko1221/Strata>
- Tag: `v0.1.44.1`
- **Commit:** `296b59d2f73c8f04a805df29c2fa97655adace78`
- **Git tree:** `063c7282837ecfe9fccad829ff7d7873400f6777`
- Upstream ggml/llama.cpp pin: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Apply **one** file: `patches/0001-strata-v0.1.44.1-jev-readout.patch`.
It is the complete clean-upstream-to-custom diff, not a patch against a privately customized base.
The former `0001-session-kv-retention` and `0003`–`0006` chain has been removed. Do **not** apply it first,
combine old/new patches, or select patches by `*.patch` globs. Its retired hashes remain in `patches/manifest.json`.

The consolidated patch changes exactly six upstream files:

```text
include/strata/core/verify.hpp
include/strata/prefill/prefill.hpp
serve/server.py
src/core/verify.cpp
src/prefill/prefill.cpp
src/program/generate.cpp
```

`patches/manifest.json` contains the patch SHA-256, all six base/target SHA-256 values, base Git blob IDs,
validation-tool hashes and implementation inventory. These are integrity/provenance records.

## What is included

| Component | Actual implementation | Boundary |
|---|---|---|
| Base `/v1/latent`, multi-layer taps and pools | Route/parser + STRLAT1/2/3 writers; `Verifier::enable_taps/read_taps`; explicit token IDs, spans, last/mean/raw rows | Variable-size span/layer tensors, not a single fixed-size vector |
| Fast batched-prefill readout | `Prefill::on_layer` observer callback, `LatPrefillHooks`, `lat_fast_active` path and cleanup | A real prompt forward; not free reuse of a previous chat's hidden states |
| Shared lease and isolation | `latent_lease`, `cooperative_readout`, guarded `generate_batched`; native GEN admission guard, pipeline drains and request-local MTP disable | Latent calls remain serial with one another; eligible fast calls interleave with chat; legacy/unsupported cases are exclusive; latent never becomes BGEN |
| Official-adapt capability | `INFO jev_slot_readout` uses `batch > 0 && batch_groups <= 2` | Does not disable official expert adaptation; capability is not bitwise or head compatibility |
| Unified API-key authentication | The same `_post` → `_authorized` gate for chat and latent | Bearer / `x-api-key`; no separate diagnostic token. **No configured key means both routes are open**—configure normal Strata authentication before exposing a service |
| Input/output and allocation guards | Reject command whitespace/control characters, caller output paths, invalid layer/final-layer aliases, spans/pools and token-ID shapes; bounded per-read staging, 1 GiB raw tap-row budget, per-layer coverage, finite/truncation checks | Targeted safeguards, not a complete security audit |
| Session KV retention | Longest-common-token-prefix slot affinity; `PGroup.first` trims leading idle rows; corrected local/global slot mapping | Keeps eligible paused slots out of padding writes; interior holes for groups larger than two remain a limitation |

The `decision` cache region isolates parked latent sessions from chat sessions; it does **not** define the final decision-token position.
The caller still supplies the correct full-chat-suffix token IDs and span positions.

## Clean apply and offline tests

Requirements: Git and Python 3.9+. These steps do not install Python/CUDA packages, load weights or send any model requests.
Run from this release repository; use a new empty validation directory.

```sh
# Obtain the fixed upstream, not a floating latest/main branch.
git init /tmp/strata-upstream
git -C /tmp/strata-upstream config core.longpaths true
git -C /tmp/strata-upstream fetch --depth=1 https://github.com/Niko1221/Strata.git \
  296b59d2f73c8f04a805df29c2fa97655adace78
git -C /tmp/strata-upstream checkout --detach FETCH_HEAD

python3 patches/verify.py --source /tmp/strata-upstream \
  --work-dir /tmp/strata-patch-validation
```

The verifier:

1. Requires the exact original commit **and** tree and a clean source checkout. It never resets/cleans the supplied tree.
2. Verifies the patch and validation tools against the manifest.
3. Creates a new local shared-object clone/sandbox and checks it is clean.
4. Checks all six base-file bytes, executes **`git apply --check` and actual `git apply`**, then checks all six target-file bytes and the exact changed-path set.
5. Executes `git diff --check`, Python AST parsing and the offline contract tests.
6. Saves `report.json`, `commands.log` and `last-command.log` in the work directory. A failure exits nonzero with `status=FAIL`; previous nonempty work directories are never overwritten.

On Windows the verifier explicitly sets `core.longpaths=true` and `core.autocrlf=false` for sandbox Git commands.
It does not relax target-byte comparisons or modify global Git settings. Keep the upstream source/object store while using the shared-object sandbox.

The 17 offline tests execute the actual server methods extracted with Python AST, with mocked engine I/O:
shared/exclusive lease behavior, serial latent calls, busy-slot GEN/BGEN routing, authentication (20 cases),
invalid inputs, command/path guards and KV slot affinity. They do **not** execute CUDA callbacks or demonstrate real GPU scheduling/numerical correctness.

### Building the patched engine

Requires CMake, a C++ toolchain, CUDA and the pinned llama.cpp source containing `ggml/`.
Prefer an existing verified cache; `GGML_DIR` is an explicit source path, not an instruction to
download or install anything.

```sh
GGML_DIR=/path/to/pinned-llama.cpp \
  bash patches/reproduce-build.sh /tmp/strata-repro-NEW
```

This is the full build entry point. A rebuilt binary will not be byte-identical to a
historical one (build paths and toolchain leak into the image); that is expected and does
not matter — the client contract is the readout API. There is no separate reduced `verify.py` build recipe: the shell delegates to it,
and both consume `manifest.json:build_recipe.cmake_definitions`.

The builder clones/checks out `296b59d2…` (or accepts `SOURCE=/path/to/clean-pinned-checkout`),
checks commit/tree/cleanliness and tool integrity, applies the patch in a **new** sandbox, verifies
all six target byte hashes and the exact changed-path set, runs the offline contracts, then builds
`strata`. The full recipe is Release, CUDA sm70, native experts ON, `STRATA_MMQ_KQUANTS=OFF`,
`STRATA_EXPERIMENTAL_SM60=ON`, tests OFF, `GGML_NATIVE=ON`, and GGML AVX/AVX2/FMA/F16C/BMI2/SSE42,
AVX512 (including BF16/VBMI/VNNI), AVX-VNNI and AMX extensions explicitly OFF. The input `ggml/`
tree must match the audited **content digest** in the manifest, including extra-file detection;
this works for both archives and Git checkouts and is not a substitute for a signed dependency provenance.
The configured cache is checked against the complete recipe before compiling. Toolchain/build paths
and host architecture can still affect the output; no path-derived difference is automatically excused.

Default parallelism is **one** (`JOBS` overrides it). CUDA devices are hidden from verifier subprocesses.
The output is hashed; it is **never run, installed or used to restart a service**.
On shared machines, independently enforce background CPU/memory/time limits and preserve the PID/log.

### What the build verification establishes

The pinned commit + patch reproduces our source tree exactly: the five changed C++/header
files and the patched server file match the bytes we ran. Binary byte-identity additionally
depends on build paths and toolchain, so no byte-identity claim is made for the executable —
and none is needed. Per-section audits of our own internal rebuilds are internal audit
material, kept outside this package. None of this is required of third parties: the client
contract is the readout API, verified at startup by `python -m jev.identity`.

## Validation recorded in this package

Evidence is under `patches/evidence/`; each record states its scope. No runtime result is inferred from source markers.

- Upstream identity/provenance: cached archive checked against 3,177 Git blobs, original Git tree/commit hash matched; an independent pinned Git fetch was also obtained (`upstream-provenance.json`). Ten upstream BAT working-tree CRLF conversions were restored to Git LF only for archive-object verification, not for C++/CUDA/Python bytes.
- Clean Windows/Linux apply, exact six-file target hashes and offline tests: see `validation-local.json`, `validation-linux.json` and their command/test logs; an independent auto-clone run is in `validation-auto-clone-linux-20261020.json`.
- Fail-closed build-tooling regression tests on Windows and Linux: `build-tools-tests-{local,linux}-v2-20261020.log`.
- Expected source negatives (wrong upstream, dirty upstream, corrupt patch): see `negative-tests.json`. Initial tooling failures and fixes are retained in `validation-history.json`.
- Cached-recipe consistency checks: `cached-recipe-check-v2-20261020.json`.

## Not accepted by these checks

This is a source/build package, **not a deployment or head-compatibility acceptance**.
No files were replaced, no services restarted, no GPU inference performed and no large
evaluations requested in this audit.

A newly built binary/config/tokenizer/model tuple must run its own measurements before you
trust head numbers on it. The Strata source patch supplies raw readout; the release client
checks the readout API at startup with `python -m jev.identity`.

Not tested in this patch audit: real latent/chat interleaving and cancellation, native per-layer numerical agreement,
head decision/routing compatibility, 4×256K capacity/load, live KV retention after tool pauses, GPU performance, HIP/SYCL/Metal,
other CUDA architectures, vision, or a fresh dependency install. Numbers measured on one
backbone/engine/config pair do not transfer to another; re-measure on yours.
