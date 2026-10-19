# Smoke test: the published code reproduces the accepted baseline

Run on 2026-10-20 against a live patched Strata 0.1.44.1 service, using **only the files in this
repository** (the `release` tree unpacked into a scratch directory).

```
JEV_DATA=<JEV_ROOT>/data CUDA_VISIBLE_DEVICES=3 python jev/eval_replay.py \
    --split dev --limit 20 --out rows-smoke20 --identity identity-fresh.json --confirm-window
python jev/check_baseline.py --rows rows-smoke20.jsonl \
    --accepted baselines/middle5-s1.dev-predictions.jsonl --out smoke20-check.json
```

Result (`smoke20-check.json`):

| field | value |
|---|---|
| accepted rows available | 1,996 |
| rows replayed successfully | 20 |
| compared | 20 |
| gold mismatches | 0 |
| prediction mismatches (archived features vs accepted run) | 0 |
| passed | **true** |

`identity-fresh.json` is the runtime identity captured immediately before the run (server-absolute
paths redacted). `rows-smoke20-summary.json` is the aggregate the replay itself reported.

What this proves: the extractor/assembly/normalisation/head code shipped here produces the same
predictions as the run whose numbers are quoted in `README.md` §2.2. What it does not prove: that
your hardware gives the same latency, or that a different backbone build gives the same accuracy —
see README §2.3.
