# Strata patch package

Use **only** `0001-strata-v0.1.44.1-jev-readout.patch`, on clean upstream Strata
`296b59d2f73c8f04a805df29c2fa97655adace78` (tag `v0.1.44.1`).

This is a consolidated replacement, **not** an addition to the former five-patch custom/0.1.41 chain.
The retired files' hashes and the six complete before/after source hashes are in `manifest.json`.
Do not apply patches by glob or apply this to an already-customized tree.

See [`../docs/PATCHES.md`](../docs/PATCHES.md) for scope, exact commands, validation evidence and limitations.

```sh
python3 patches/verify.py --source /path/to/clean-strata-upstream \
  --work-dir /tmp/strata-patch-validation
```

The verifier leaves the input checkout untouched and applies into a new sandbox. Its default does **not** build.
`--build cuda` explicitly requests a compile; it never runs a model or installs/restarts anything.
## Build: one complete recipe

```sh
GGML_DIR=/path/to/audited-pinned-llama.cpp bash patches/reproduce-build.sh /tmp/strata-repro-NEW
```

This is the full build recipe we ran with. A rebuilt binary will not be byte-identical to a
historical one (build paths and toolchain leak into the image); that is expected and does not
matter — the client contract is the readout API, checked at startup by `python -m jev.identity`.
It clones/pins upstream (or accepts `SOURCE=/path/to/clean-pinned-checkout`), verifies/applies the patch,
runs offline contracts, and builds with the **single complete recipe** in `manifest.json`:
Release/CUDA sm70, native experts ON, K-quants OFF, `GGML_NATIVE=ON`, GGML x86 SIMD extensions OFF.
`verify.py --build cuda` uses this exact same implementation and recipe, not a second reduced recipe.
The external `ggml/` source bytes must match the audited tree digest in the manifest (a fresh Git checkout
is acceptable if those bytes match). Default compile parallelism is one; `JOBS` is explicit.

An apply/test/build PASS proves the pinned sources compile; it does not transfer accuracy,
calibration or latency claims to your build — re-measure on yours.
