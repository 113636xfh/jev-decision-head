# 20-row archive fallback (not the missing original live smoke)

## Result

The original `rows-smoke20.jsonl` was not located in the main workspace's
`work/jev-abliterated-eval-20261018` files/archive or in a recursive, read-only search
under the server's `<BUILD_ROOT>`. The archive does contain earlier `smoke/` and
`smoke2/` runs, but those are not the missing 20-item run.

`rows-smoke20-derived-archive-v2.jsonl` is therefore **derived evidence**, not an
original-smoke recovery. It copies only the existing archived-decision fields of
the first 20 records, in file order, from the full 1,996-item dev replay. No model
was run, no features were extracted, no production request was sent, and no gold,
prediction, confidence or status was invented. The transformation script refuses
divergent duplicate archive members.

- Compared: **20/20**; gold mismatches **0**, archived prediction mismatches **0**.
  Latest digest-pinned reports: `smoke20-derived-check-v4.json` and
  `baseline-self-check-v4.json` (earlier reports were preserved).
- Archived correct decisions: **13/20** (copied source values; not a new evaluation).
- Full 1,996-row published baseline self-comparison: **1,996/1,996**, no mismatches.
- Tests also compare the full original dev replay to the baseline when the external
  source archive is present; that source is not required for portable unit tests.

## Provenance

See `rows-smoke20-derived-provenance-v2.json` for source path, archive SHA-256,
member SHA-256, duplicate-member checks, copied fields, ordered UIDs and output hash.

- Source archive SHA-256:
  `bbb661829ddfb93483f9259022587aa30e550b8e45b88ff87c12c3c1025d5809`
- Full `rows-dev.jsonl` SHA-256:
  `7e9b6fc13e66abe29d806688f73acd0ecbe0adda518b617e84d6f5000ec81aea`
- Derived 20-row SHA-256:
  `d5ccd8957755eda3de3752c89f4cade1c911a2d1f2398d691ed73ed8f210ff4e`

## Recheck

From the release root, with a fresh output filename if saving the JSON result:

```sh
python -m jev.check_baseline --rows evidence/smoke-20-dev-20261020/rows-smoke20-derived-archive-v2.jsonl --expected-count 20
python -m jev.check_baseline --rows baselines/middle5-s1.dev-predictions.jsonl --expected-count 1996 --require-complete
python -m unittest discover -s tests -v
```

The checker defaults to full baseline coverage. An explicit `--expected-count 20`
permits the smoke subset but still requires exactly 20 unique valid known rows;
error rows (including contradictory error metadata), unknown UIDs, missing required
fields and incompatible/mixed schemas are failures, not skipped records.
`--require-complete` adds full UID coverage even when a count is given. The published
reference's SHA-256 is independently pinned in the checker, with CRLF normalized to
LF so Windows checkouts remain valid; its expected size is 1,996 rows. Explicit custom
references are permitted for fixtures but reported as `custom-unverified`, not as the
authenticated publication. Output evidence files are created exclusively, not replaced.

Supported inputs are the real baseline's indexed predictions and probabilities,
archived replay's `status/gold/pred_old/n_options`, and release head-route labeled
predictions. Head-route passes are **conditional on independently trusted candidate
identity and ordering**: the baseline stores indices, not option labels, so changing
the labels while preserving positions cannot be detected by this checker. It does
not prove prompt identity or live numerical compatibility; decode-route rows are rejected.

## Boundary

The four pre-existing files (`README.md`, `identity-fresh.json`,
`rows-smoke20-summary.json`, `smoke20-check.json`) were not overwritten. Their
original live-smoke claims remain separate. **Do not attach this fallback to that
old summary or identity**: it cannot reproduce the old timing, live feature
accuracy, byte-equality, logit deltas or runtime identity. The missing original
20-item live smoke remains unrecovered. This fallback closes only the offline
stored archived-decision comparison requested for release packaging.
