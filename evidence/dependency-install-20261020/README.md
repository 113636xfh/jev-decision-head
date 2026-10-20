# Clean CPU dependency installation — 2026-10-20

## Final status: quality gate **FAIL** (intermittent native selfcheck crash)

The dependency installation itself **passed** in a new venv with no inherited
system/user packages. `pip check`, **15/15 CLI `--help`** invocations and **28/28
contract/provenance tests** passed. However, the first canonical offline selfcheck
exited with Windows native status **3221225477 (`0xc0000005`, access violation)**.
Two later canonical reruns and one unbuffered/faulthandler diagnostic run passed
without changing packages, runtime, weights or selfcheck code. The original failure
is unexplained and remains a release-quality **FAIL**; successful retries do not
erase it. There is no failing-stage traceback because the first run crashed before
its buffered output was flushed. The diagnostic run did not reproduce the fault.

## Actual isolation and tested version boundary

- Host: Windows 11 x86-64 (`Windows-11-10.0.26200-SP0`).
- Interpreter: CPython **3.12.10**, `D:/Python312/python.exe` bootstrap.
- New venv: `D:/tmp/release-install-20261020`.
- `include-system-site-packages = false`; user site disabled; initial `pip list`
  contained **only pip 25.0.1**. See `pyvenv.cfg.txt`, `02-clean-inventory.log` and
  `v2-03-platform.log` (prefix/base-prefix, complete sys.path, torch module path).
- Installed direct dependencies: **torch 2.6.0+cpu**, **numpy 1.26.4**,
  **tokenizers 0.23.3**, **jinja2 3.1.6**.
- `torch.version.cuda` is **null**; installation report contains no `nvidia-*` or
  `cuda-*` distributions. All resolved package versions are in `v2-02-freeze.log`.
- Broader requirement ranges, Python 3.9/3.14, other OSes, GPU builds, external
  tokenizer/model packs and online numerical compatibility are **not certified**.
  Extraction remains a Strata-host boundary (on this Windows host it encounters
  missing `fcntl`); it is not one of the 15 standalone CLI gates.

## Installation and official CPU wheel provenance

The real installation command was run asynchronously, not replaced with a
system-site-packages or preinstalled-package smoke test:

```text
D:/tmp/release-install-20261020/Scripts/python.exe -m pip install --no-cache-dir --report evidence/dependency-install-20261020/pip-install-report.json -r requirements-cpu.txt
```

`requirements-cpu.txt` includes the release `requirements.txt` and pins the `+cpu`
build using PyTorch's official CPU index. Ordinary packages resolved through this
host's preconfigured Tsinghua PyPI mirror; every actual URL and package archive hash
is recorded in `pip-install-report.json`. Requirements file hashes are recorded in
the gate. The complete pip install returned **0** (1,195.75 seconds).

The first conservative provenance check accepted only `download.pytorch.org` and
rejected the official index's `download-r2.pytorch.org` CDN. That original **FAIL**
is preserved, unchanged, in `install-gate.json`; the successful installation was
not rerun or replaced. `verify_install_v2.py` subsequently fetched the official
HTTPS CPU index, saved its raw response, and verified the **exact reported wheel
URL plus SHA-256** against the index link before running the offline checks.
This was not merely a relaxed host allowlist.

- Official index: `https://download.pytorch.org/whl/cpu/torch/`
- Actual wheel:
  `https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-win_amd64.whl`
- Wheel SHA-256:
  `4027d982eb2781c93825ab9527f17fbbb12dbabf422298e4b954be60016f87d8`
- Index/link binding: `verification-gate-v2.json` and
  `official-torch-cpu-index-v2.html`.

## Audit artifacts

- `verification-gate-v2.json`: primary gate; installation **PASS**, quality **FAIL**;
  complete command return codes and timings, official index/wheel binding.
- `v2-01-pip-check.log`: no broken requirements.
- `v2-04-selfcheck.log`: original native-crash attempt, return code in primary gate.
- `selfcheck-native-probe-v3.{log,json}`: diagnostic invocation passed, including
  13,194,368-parameter weight load, forward distribution, policy/certificate checks
  and 231-task benchmark snapshot.
- `selfcheck-canonical-rechecks-v4.json` and two matching logs: canonical reruns
  both passed; aggregate quality gate intentionally remains **FAIL**.
- `v2-05-help-*` through `v2-19-help-*`: all 15 help commands returned zero:
  predict, cascade, identity, fingerprint_model, domain, eval_replay, eval_benchmark,
  probe_drift, check_baseline, train, train_distilled, tune_policy, analyze_flips,
  finalize_rows, run_all. Training commands ran **only `--help`**.
- `v2-20-unittest.log`: 28 tests passed in this same fresh venv, including actual
  1,996-row baseline comparison and source-archive/derived-row equivalence.
- `sealed-summary-v1.json`: final machine-readable summary with artifact hashes
  and a post-check source snapshot, not a claim that external concurrent source
  changes were prevented.

No e5 environment was created. No GPU, training, production inference request,
runtime/selfcheck edit, commit or push was used. Read-only server access was solely
for locating and hashing already-recorded source evidence.
