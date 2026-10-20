# Released head weights and their identity

This is the release-level weight/identity clarification. For exact system roles and
identity limits see [docs/IDENTITIES.md](docs/IDENTITIES.md); for data membership and
provenance see [docs/DATA.md](docs/DATA.md). Earlier wording that a later compatible runtime
was the training system is superseded
by those distinctions.

| File | Exact SHA-256 | Bytes | Contents |
|---|---|---:|---|
| `weights/best.pt` | `8d3f60094cc79bdbee2b66837963c8f9f094819a97fc6d774d24fbcbbb674d2b` | 52,787,698 | Frozen decision head; 36 FP32 tensors, 13,194,368 parameters |
| `weights/norm.npz` | `b00ddad2c2b0ddc0ae0c9dfc50fcc1ec16de1f9f21683a13698aadca2e9ab2b2` | 819,694 | Per-pool/layer/dimension z-score statistics fitted on train only |

These byte hashes were independently verified. Neither file was retrained, converted,
recalibrated or modified for this documentation/data-manifest update. A file hash provides
integrity, not a guarantee that an arbitrary live feature source is compatible. Load only
trusted PyTorch artifacts; use a weights-only deserialization path where supported.

## Training source and feature contract

The head is not a standalone language model. It was trained on archived hidden-state
readouts from `Qwen3.8-Flash-Next-GSQ-RCO` IQ3_S, with no backbone update, LoRA or backbone
gradient. The **archive identity** is engine `44f99383…`, server `32368d78…`, config
`5e7a3152…`, not the later `ce7559e0… / 3ebd690b… / b6b687b8…` compatibility runtime.
Full hashes are in `manifests/system-identities.json` and the frozen run manifest.

- Exact backend value: `batched-prefill-one-shot-r2`; `gpu_batched_prefill` is prose
  shorthand only. CPU verify and GPU batched-prefill features are not interchangeable.
- Prompt version: `lossless-state-final-assistant-token-v2`.
- Head layers `[23, 29, 35, 41, -1]`; the archive additionally stores layer 15.
- Pools `[span_last, span_mean]`, stem/all candidates/final-decision-token spans,
  4 residual streams × 2,560 dimensions per tap.
- Shared projection and two candidate-set transformer blocks, width 512. Train-only
  normalization; checkpoint selected on dev NLL. No task weighting was used.
- Actual training members: **16,000**. Separate dev/test/OOD/cal membership is now
  published in `manifests/splits/`, not inferred from the run manifest alone.

## Compatibility is not portability

The frozen head was trained on one backbone (engine `44f99383…` / server `32368d78…` /
config `5e7a3152…`). Later systems measured with the same weight bytes gave different
numbers: the archived-feature baseline is 6,025/7,718 (78.06%); the abliterated live-readout
measurement is 5,949/7,718 (77.08%), with feature values and serving-stack identities both
different. Reusing these weight bytes is not enough to transfer compatibility, calibration
or latency claims. A new model/engine/server/config/backend/policy needs its own
measurements; a successful patch apply/build alone is insufficient.

`python -m jev.identity` records which build answered each run, and the harnesses stop when
it changes mid-run. That is bookkeeping for honest attribution, not a compatibility
guarantee.

## License and reproduction boundary

The project's original head weights and normalization are offered under MIT. That does not
relicense the backbone, third-party data or original upstream wording; see [NOTICE](NOTICE).
The source-dependent Open-Jev/WANLI inputs and their caveats remain relevant even though
this repository redistributes only indices, numeric labels and commitments, not task bodies.

Backbone assets and hidden-state archives are not included. Data-only reconstruction from
the exact separately obtained public raw artifacts has been verified at the prepared-file
hash level; that is not an end-to-end training or online-serving reproduction.
Do not treat these weights as a second resident backbone, nor assume that a decision request
can reuse a previous chat's hidden states without another forward pass.
