# Provenance files (protocol, certificates, identity)

These are the files that make the numbers in `README.md` checkable. They are small on purpose.

**Redaction rule.** Every absolute server path in these JSON files was replaced
(server-absolute paths: `/home/<user>` → `<JEV_ROOT>`); nothing else was changed. The sha256 of the
**original** file is given so that anyone with access to the original can confirm it; the
sha256 of the redacted copy shipped here is given so the copy in this repository is itself
pinned.

| File | What it is | original sha256 | shipped (redacted) sha256 |
|---|---|---|---|
| `manifests/middle5-s1.manifest.json` | Run manifest of the shipped head: feature protocol (layers, spans, pooling), split definitions, seeds, and the hashes of the backbone/engine/server/config/tokenizer it was trained against | `5d23d7db39855584…` | `0140b9e5ff27d6d0…` |
| `results/middle5-s1.results.json` | Per-split metrics of that run (accuracy, NLL, Brier, ECE, risk–coverage) | `ad05b76274f79e13…` | `3b544770b7b1900b…` |
| `baselines/middle5-s1.dev-predictions.jsonl` | Per-item dev predictions of that run — the reference used by `jev/check_baseline.py` to prove a framework change did not alter results | not redacted (no paths) | — |
| `identity/runtime-identity.run-20261018.json` | Runtime identity captured before the frozen-head re-measurement: engine exe sha256, runtime file shas, running config, tokenizer sha, engine metrics, GPU snapshot | `d042caaf738e5b26…` | `1b46772d6a5f0418…` |
| `identity/runtime-identity.pre-run.json` | Identity captured before the earlier stage of the same measurement | `7847cde4fffffb66…` | `b2119b27cfe30a48…` |
| `certs/live-readout-compatibility-r2-bitwise.json` | Compatibility certificate: proves the live multi-layer readout reproduces the archived features byte-for-byte on the pair it names | `97e7a6986fb4522d…` | `2c8828de2407c898…` |
| `certs/old-pair-adaptive-head-v1.compatibility.json` | The earlier certificate for the previous backbone build. Kept unchanged as history; **it does not cover the current build** | `37236f092d5e0b01…` | `703daf8f29776180…` |

## Why this exists

A decision head is a readout over one specific backbone build. Re-running the frozen head
against a different build of the same model moved accuracy 78.06% → 77.08% over 7,718 items
and made calibration worse (README §2.3). So every number must name: engine binary, server,
service config, model/tokenizer, head + normalisation hashes, and `compute_backend`. The
loader in `jev/head_loader.py` refuses to run when the running system does not match a
certificate in this directory.
