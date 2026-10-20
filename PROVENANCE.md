# Provenance and evidence boundaries

This release now provides actual split membership, not just the head's run manifest.
Start with [docs/DATA.md](docs/DATA.md) (direct source, conversion, hashes, OOD audit and
conditional data-only reconstruction) and [docs/IDENTITIES.md](docs/IDENTITIES.md) (three
different system identities and what each run recorded).
[WEIGHTS.md](WEIGHTS.md) pins the shipped head and normalization.

These clarifications take precedence over earlier shorthand that called the later
compatibility runtime the “trained build,” described OOD as wholly novel sources/families,
or promised to rebuild every split from the two `pngwn` dataset names. They are not evidence
that unrelated application/patch release gates have passed.

## Data artifacts

| Artifact | Meaning |
|---|---|
| `manifests/splits/{train,dev,test,ood,cal}/members-*.jsonl` | **23,718 actual members**, 48 small ordered shards. Original identifiers, parent groups, sources, split names, target vectors, permutations, frozen hard-label indices, categorical metadata and content commitments; no task-body text |
| `manifests/split-audit.json` | Exact per-split counts, per-shard byte hashes, virtual concatenated-stream hashes, missing fields and **all ten pairwise intersection audits** |
| `manifests/data-lineage.json` | Direct `ZefanCai/Open-Jev-v1.1` input and export hashes, source-dependent terms, transformation/selection identities, excluded legacy components and five actual prepared-file hashes |
| `manifests/{train,dev,test,ood,cal}-origin.json` | Full selected-member match against cached public raw records, plus exact virtual reconstruction of prepared rows/file hashes; **not** transferable compatibility evidence |
| `manifests/generate_members.py`, `shard_members.py`, `audit_members.py` | Stdlib-only member export, small-file sharding and locally reproducible intersection audit |
| `manifests/verify_upstream.py`, `verify_all_origins.py` | Streaming raw-source verification and optional local data-only reconstruction; no activation arrays or model calls |

Every selected member matched the published Open-Jev projection; none requires an additional
project-private task source. Some of that public data consists of authored controls. It is
incorrect to equate “control task” with “private input.” Separate internal tasks outside
these files are not described or reconstructible here. Only membership/labels/hashes and
categories are released, never question, state or option bodies.

The exact input artifacts are pinned by compressed and uncompressed SHA-256, not by an
invented dataset Git revision. The recorded exporter code commit is not the dataset commit.
If the upstream stops serving those pinned bytes, a hash is not a replacement for the data.
The complete parent mixture, excluded Wiki rows, hidden-state archives and sealed benchmark
payloads are not shipped and cannot be recreated from this repository alone.

## System and historical artifact identities

`manifests/system-identities.json` separates:

- training/archive: engine **44f99383…**, server **32368d78…**, config **5e7a3152…**;
- later historical adaptive compatibility: **ce7559e0… / 3ebd690b… / b6b687b8…**;
- explicit abliterated measurement: **c1d3070c… / d85622eb… / 5440f05c…**.

Full hashes and tokenizer values are in that index. It contains no service location.

`manifests/artifact-digests.json` records full **original archival byte SHA-256** and
**shipped working-file byte SHA-256** values. Prefixes below are navigation aids only:

| Shipped file | Original SHA prefix | Shipped SHA prefix | Scope |
|---|---|---|---|
| `manifests/middle5-s1.manifest.json` | `5d23d7db39855584` | `0140b9e5ff27d6d0` | Training run/feature source, **not** a split-member list |
| `results/middle5-s1.results.json` | `ad05b76274f79e13` | `3b544770b7b1900b` | Archived-feature held-out metrics |
| `baselines/middle5-s1.dev-predictions.jsonl` | `7e76aa9a6a3df075` | `3e0fc712aa2c6aa3` | Dev-only per-item baseline, not all-split inputs |
| `identity/runtime-identity.run-20261018.json` | `d042caaf738e5b26` | `1b46772d6a5f0418` | Historical abliterated measurement identity |
| `identity/runtime-identity.pre-run.json` | `7847cde4fffffb66` | `b2119b27cfe30a48` | Earlier capture for the same measurement stage |

### Redaction, serialization and immutability

The shipped historical JSON files were already path-redacted before this clarification:
the archival user-home prefix became `<JEV_ROOT>`. The present working copies also use
CRLF in places where the originals use LF; formatting can change without changing parsed
JSON. Thus **“only paths changed” is not a correct file-byte claim**. The baseline's 1,996
extra bytes are its CRLF-vs-LF difference, not additional prediction rows.

We independently compared every listed original with its shipped parsed content after the
legacy prefix redaction. All seven match at that content level; the full comparison digests
are in `manifests/legacy-redaction-digests.json`. This establishes which transformations were
observed; it does not make the redacted bytes equal to their original SHA. The original byte
digests themselves were re-read from retained archival files and recorded in
`manifests/original-artifact-digests.json`.

Do not replace original
hash references inside a historical artifact with hashes of its redacted copy, relabel a
measurement as accepted, relax old assertions, or reissue the old artifact. New indices
point at both forms explicitly. Git line-ending settings on another checkout can produce a
different working-file SHA; compare exact bytes with the corresponding recorded form, or
use a declared parsed-content comparison for content identity—not as a substitute for a
byte-strict numerical assertion.

## Evidence of implementation versus measured claims

Applying/building patches on clean pinned upstream, passing interface/safety tests, and
demonstrating numerical behavior for one exact pair are **different gates**. The 20-item
archived-side smoke and the online interface measurement exercise the release client; they
do not transfer accuracy, calibration or latency numbers to a different backbone, engine,
config or policy. No new apply/build run, model training, activation extraction or service
change was performed for this provenance update.
