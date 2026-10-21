# Data identity and reconstruction boundaries

This document and `manifests/` describe the **actual selected data**, not a claim that all
experiments can be reproduced from this repository alone. See [IDENTITIES.md](IDENTITIES.md)
for the separate archive, compatibility and measurement systems. Older descriptions of OOD
as entirely unseen sources/families, or of these splits as a direct download from two
`pngwn` repositories, are not accurate for the files indexed here.

## 1. Direct source, not an inferred upstream mixture

The direct input was [ZefanCai/Open-Jev-v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1),
configuration `community-hard-mix-v2-redistributable`: **326,619** published records across
train / validation / test / OOD / calibration. The exporter identifies its parent mixture as
`community-hard-mix-v2-final` (328,672 records). Its public projection omits 2,053
Wikispeedia records; **we did not restore or use those omitted records in these splits**.
This is not a claim to reproduce the complete mixture used to train Open-Jev 27B.

The original raw JSONL gzip artifacts, not a newly flattened Arrow representation, were
used. The public Parquet `record_json` field is another route to the original record, but
its exact round trip is an upstream claim, not a new Parquet test performed for this release.
`manifests/data-lineage.json` records the cached export-manifest, raw gzip/uncompressed
hashes, row counts, preparation-code hashes, selection parameters and prepared-file hashes.

- Upstream export manifest SHA-256: recorded in `data-lineage.json`.
- Upstream code checkout at export: `bf818f86a74869f65bc6b93966bf4d5a6f604c03`.
  This is **not** a Hugging Face dataset revision.
- The original download's dataset commit was not recorded in the inspected cache. We pin
  the exact artifact bytes by SHA-256 instead; a URL on `main` is not an immutable pin.
- The `pngwn/typed-decisions-v2` and `pngwn/typed-decisions-v2-system-one` repositories are
  related typed-decision lineage references, **not the direct inputs for regenerating these
  selected files**. HotpotQA/MMLU-Pro names in earlier summaries do not substitute for
  row-level provenance. The selected natural-language-inference rows are WANLI-derived.

### Public control tasks versus private/internal tasks

The five member sets contain **23,718/23,718 exact matches to the cached published
Open-Jev projection**, checked against each complete upstream record's canonical hash,
original id/group/source/split, complete metadata, and content. See `*-origin.json`.
No additional project-private control-task rows were found in these selected files.
Names such as `customer-control-v1` and `workflow-controls-v1/...` refer here to controls
already present in that public projection, not proof that a task is private.

No task body is redistributed here: no state, question, option wording, conversations or
full metadata. Membership identifiers, categories, target vectors and hash commitments
are published. These identifiers are linkable to the public upstream; this is **not an
anonymized dataset**. Any separate private/internal tasks outside these five files are
out of scope: no payload, reconstructibility or membership claim is made for them.

## 2. Conversion and selection chain

1. Parse the original JSON record without flattening `state`. Lists, dictionaries, strings,
   nulls and `state.candidates` retain their JSON types; no prompt truncation is involved.
2. Connect original `group_id`, optional `metadata.semantic_context_sha256`, and exact
   canonical `[state, question]` anchors. Components spanning official splits are
   quarantined rather than reassigned. The historical full preparation manifest reports
   **zero quarantined components**. This is an exact-key grouping policy, not a semantic
   similarity detector.
3. Preserve original id (`uid` in prepared rows), group, source, source split, full metadata,
   state, original options and original targets. Shuffle options deterministically with
   seed `20261012`, reordering target entries with the same permutation. Retain the whole
   state serialization, question, kind, component and raw-record hash.
4. Choose hash-ordered prefixes of **whole components**, stopping before exceeding the
   requested size. The fast cohort excludes the components represented by the older
   CPU-verify archive (1,932 legacy archive records; component hashes are in
   `data-lineage.json`). No legacy UID is re-extracted. The resulting fast train sizes are
   1,994 / 3,998 / 7,994 / **16,000**, not the older preparation's smaller-subset counts.
5. The four selected evaluation files are byte-identical to their earlier prepared subsets:
   dev 1,996 / test 1,999 / OOD 1,991 / calibration 1,732. They were not supplemented to
   force round counts. The archive/backend selection changed; the evaluation labels did not.

The **stored hard gold is frozen**, not recomputed from a convenient argmax. In tied soft
targets, 74 train / 8 dev / 3 test / 6 OOD / 7 cal records have a stored hard label different
from the *first* maximum in original option order; all remain within the original target's
maximum set. Members preserve `gold_original_index` and its wording hash. Regenerating a
first-argmax label would fail exact reconstruction. No gold labels or soft targets were
changed for this release.

## 3. Actual released members and hashes

`manifests/splits/<split>/members-*.jsonl` contains the selected members in original prepared
file order. Shards are lexical-order consecutive segments, at most 512 rows and 1,500,000
bytes each. Every shard has its own byte hash in `manifests/split-audit.json`.

| Logical split | Members | Parent groups / components | Shards | SHA-256 of concatenated member-shard bytes |
|---|---:|---:|---:|---|
| train | 16,000 | 6,230 | 32 | `03c80c24176aa2b49d7f2357e638958bfa360b7356884a8aa46e7920189f1122` |
| dev | 1,996 | 393 | 4 | `fc87de4381456b378665fc0ef2f0d1f325b86308c4cead287b689bbf30107048` |
| test | 1,999 | 412 | 4 | `536b0207a542ee4ccc643733b9405e27134894144035db135d298e16925fc5fa` |
| OOD | 1,991 | 160 | 4 | `0efc5412aebc45237aee790cb218537c51f059163ed823377e9027b63d8c37dd` |
| cal | 1,732 | 265 | 4 | `85a0124b31db5c6b253b226ff4fb46df4063f44374d5102f6286db0f1dbc29ad` |
| **Total** | **23,718** | | **48** | |

A member preserves literal `id`/`uid`, `group_id`, `source`, `split`, `source_split`,
`component`, `kind`, original/prepared target vectors and option permutation. `id` is an
explicit alias of prepared `uid`, which is the original upstream `id`, not a new generated
identifier. The `cal` **directory** is a convenience alias: row `split` remains
`calibration`; original validation remains `source_split=validation`, prepared `split=dev`.
We do not silently normalize those fields. Selected categorical metadata is copied as-is;
complete metadata is committed by hash rather than leaked as free text.

## 4. Hash protocol (version 1)

There are two distinct hash types. All are SHA-256, written as 64 lowercase hex digits.

**File hashes** cover exact bytes, including encoding, JSON whitespace, line order and line
endings. Newly generated member shards are UTF-8, no BOM, one canonical object plus LF per
line. `members_file_sha256` in the audit/origin receipts means the virtual file obtained by
concatenating that split's shards in lexical filename order. It is not a hash of sorted ids,
a directory name, or a list of shard hashes. A gzip hash covers compressed bytes, separately
from its uncompressed JSONL byte hash.

**Canonical value hashes** use this exact Python JSON definition:

```python
json.dumps(value, ensure_ascii=False, sort_keys=True,
           separators=(',', ':'), allow_nan=False).encode('utf-8')
```

No trailing LF, Unicode normalization, case folding, float-to-string coercion, list sorting,
field truncation or removal of nulls is applied. This is **not RFC 8785/JCS**. Array order,
JSON types, numeric serialization (including `-0.0`), empty values and all dictionary fields
matter. Duplicate JSON keys are outside the accepted record contract.

| Field | Exact value hashed |
|---|---|
| `raw_sha256` | Complete decoded **original upstream record**, including every metadata field; inherited commitment independently verified against raw gzip |
| `prepared_record_sha256` | Complete prepared record, including original fields, full metadata, state, text, reordered targets, hard gold, component and protocol fields |
| `content_sha256` | Object with keys `state`, `question`, `kind`, `original_options`, `original_target`, using original unpermuted values |
| `state_question_sha256` | Two-element list `[state, question]` |
| `state_sha256` | Complete original `state` value |
| `metadata_sha256` | Complete original `metadata` object |
| `original_options_sha256` / `original_target_sha256` | Original complete option / target list, respectively |
| `option_permutation_sha256` | Complete option-index permutation list |
| `gold_sha256` | Stored hard-gold option string; `gold_original_index` identifies it without publishing wording |
| `source_instance_id_sha256` | Original metadata value, only when the field exists |
| `semantic_context_sha256` | Copied upstream metadata commitment, **not rehashed** or asserted present on every record |

Missing metadata keys remain missing; null remains null. Missing/null/empty values are
excluded from intersection sets and counted separately. Hashing commits to withheld content;
it does not make that content recoverable or confer redistribution rights.

## 5. What OOD actually isolates

OOD means the selected upstream **OOD assignment with source-specific holdouts**, not an
entirely novel-source or entirely novel coarse-family test. We preserve upstream policy
strings in `provenance_split_policy` so that claims can be traced without exposing payloads.

Examples of different granularities in these 1,991 OOD rows:

- Authored `community-diversity-v2`: held-out **policy/scenario families within domains**
  (136 selected rows); the domains themselves can be shared with train.
- WANLI: held-out **connected seed/premise components within the same English NLI source**
  (124 selected rows), not a new language or a new external source.
- Other controls: parent document/contract/task-graph families, reserved layouts/vocabulary/
  wording, separate episode seeds/control regimes, or game grid/geometry regimes. The
  precise per-row declaration and counts are in the audit, not inferred from source names.

The local preparation never repartitions by UID arithmetic. Some **upstream** policy
strings mention index modulo or id hashes within their own authored family rules; retaining
those strings is not a claim that the upstream used one universal semantic splitting rule.

### Recomputed train–OOD intersections

| Exact key | Shared distinct values | Train rows with shared value | OOD rows with shared value |
|---|---:|---:|---:|
| id / group_id / connected component | 0 each | 0 | 0 |
| canonical state / `[state, question]` / content hash | 0 each | 0 | 0 |
| present semantic-context / source-instance hashes | 0 each | 0 | 0 |
| `source` | **19** | **12,055** | **1,855** |
| coarse `metadata.family` | **6** | **16,000** | **1,991** |
| present `metadata.domain` | **13** | **2,643** | **256** |
| present `scenario_family` / `template_id` | 0 each | 0 | 0 |

Coverage matters: OOD lacks `domain` on 1,735 rows, `scenario_family` and
`semantic_context_sha256` on 1,855, and `template_id` on 154. Zero overlap in an optional
field is not full-corpus family-disjointness. All six coarse families recur: policy,
evidence, routing, rubric, natural-language inference and doom_basic.

The published audit covers **all ten pairs of the five selected splits**, with exact
intersection values/counts and missing-field counts. Every pair has zero shared id,
parent group, connected component, present semantic context, `[state, question]`, content
hash and complete raw-record hash. Within-split correlated/duplicate views remain; the
number of rows is not the number of independent scenarios. These checks do not detect
paraphrases, all semantic leakage, or backbone pretraining exposure.

## 6. What can actually be reconstructed

**Verified here:** all 23,718 member commitments match cached public raw records. The
reconstruction helper also regenerated each complete prepared row virtually, preserving
stored hard-label indices, and obtained **exactly the five frozen prepared-file hashes**
in `data-lineage.json`. No task-body files were written remotely during verification.

**Conditional data-only recipe:** independently obtain the five raw gzip artifacts from
[the upstream raw directory](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1/tree/main/raw/community-hard-mix-v2-redistributable)
under their terms, then verify the exact compressed and uncompressed hashes in
`data-lineage.json`. Obtain the same bytes, not merely the same number of rows.

```bash
# No GPU, model download, service or activation archive is needed for these checks.
python manifests/audit_members.py > local-split-audit.json
python manifests/verify_all_origins.py --raw-root /path/to/upstream/raw

# Optional data-only reconstruction to a NEW file (existing files are refused):
python manifests/verify_upstream.py --raw /path/to/upstream/raw/train.jsonl.gz \
  --members manifests/splits/train --prepared-out /path/to/new/train-16000.jsonl \
  > local-train-origin.json
```

Use validation raw + dev members, test + test, OOD + OOD, and calibration + cal analogously.
A failed run may leave a partial optional output; never accept it without the final receipt
and expected full-file SHA. `verify_all_origins.py` rewrites only its local `*-origin.json`
receipts after each successful check; redirecting the read-only single-file verifier to a
separate output is preferable when retaining the shipped receipts unchanged.

This is **not** an end-to-end training/serving reproduction. Activation archives,
exact original model/runtime assets and full historical acceptance dependencies are not
shipped. Public raw-file availability and rights are external dependencies; if the upstream
no longer serves the pinned bytes, the hashes do not reconstruct them. Unpublished internal
tasks, omitted Wiki records, the complete Open-Jev model-training mixture, and JevBench
sealed questions are not reconstructible from this release. A new extraction/build is a new
system until independently checked; it cannot borrow a historical compatibility assertion.

## 7. Terms and attribution

See [NOTICE](../NOTICE). Open-Jev generated content is source-dependent CC0; WANLI-derived
text remains CC BY 4.0. Customer-control short question descriptions retain an upstream
TypeSafe documentation licensing qualification. This project's licenses (Apache-2.0 code,
CC BY 4.0 head weights) do not
relicense those inputs. Only indices, categories, numeric labels and hashes are shipped,
not any of the original question/state/option bodies.
