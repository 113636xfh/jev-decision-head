# Strata patches

Five patches, in apply order, against **Strata 0.1.44.1**. They are the serving-side half of
this work: without them a resident Strata process can serve generation *or* a naive
single-token readout, but not a multi-layer, multi-span decision readout that shares a slot
with live chat. We intend to submit them upstream; until then apply them before starting the
server.

```
git apply patches/0001-session-kv-retention.patch \
          patches/0003-latent-readout.patch \
          patches/0004-latent-taps.patch \
          patches/0005-latent-spans-and-pooling.patch \
          patches/0006-decision-region-and-tap-read-path.patch
```

`0002` is intentionally absent: the web-UI session-tabs patch in our tree is unrelated to
decisions and is not part of this work.

---

## 0001 — session KV retention (`serve/server.py`, `src/program/generate.cpp`)

**Change.** A slot's KV was released when the pipeline advanced past the request that produced
it; the patch keeps it resident per slot and re-attaches it to the next request on the same
slot, inside the existing conversation-cache budget (8 GiB, 4 slots, evict-oldest). The
`generate.cpp` hunks carry the slot-lifetime bookkeeping; the `server.py` hunks carry the
scheduler side.

**Why.** A decision request and the chat request that produced its context land on the same
slot. Without retention every decision re-reads the whole conversation, which is exactly the
cost that dominates the channel latency (README §3).

**Blast radius.** Scheduler and KV-cache lifetime only; no change to model math.

**Enable.** None — on once applied. Budget stays configurable through the existing
conversation-cache settings.

**Acceptance.** Four slots at full 256K context concurrently: 8/8 ok (prompt 259,325 tokens,
259,249 cached), 20/20 ok at `batch_groups=1` (259,656–260,318), 48/48 ok in a dedicated 258K
audit; chat behaviour unchanged; 0 OOM kills of the production process. On a shared slot a
decision call went 6.385 s → 2.555 s while the interleaved chat call went 4.39 s → 7.03 s —
the gain is throughput interleaving, not free speed, and the chat side pays part of it.

## 0003 — latent readout (`src/program/generate.cpp`)

**Change.** Adds the readout request path: after prefill, the engine can return the hidden
state at the requested position(s) instead of sampling a token, and reports the capability bit
the client checks before asking.

**Why.** The head scores candidates from hidden states; asking the model to *write* its answer
and parsing it is slower and loses the distribution.

**Acceptance.** Readouts verified byte-for-byte against the archived features for identical
tokens (`certs/live-readout-compatibility-r2-bitwise.json`).

## 0004 — latent taps (`src/program/generate.cpp`, `src/core/verify.cpp/.hpp`, `serve/server.py`)

**Change.** Turns the single readout position into a **set of taps**: several transformer
layers can be read from one forward pass, with the request schema validated
(`verify.*`) and the API surface exposed in `server.py`.

**Why.** The feature protocol archives six backbone layers and this head reads five of them
(23/29/35/41/last). Reading them in separate passes would multiply the prefill cost; taps make
it one pass.

**Acceptance.** Batched-prefill readout at 402/512/950 tokens runs 1.62× / 2.32× / 2.87×
faster than the serial small-window path (107 tok/s) used before.

## 0005 — latent spans and pooling (`src/program/generate.cpp`, `serve/server.py`)

**Change.** Adds span selection (question stem, each candidate, decision position) and pooling
(last position, span mean) on the engine side, so the client asks for semantic regions instead
of raw indices.

**Why.** Those spans are what define the feature vector; if the engine and the training
extractor disagree about them, the weights mean something else.

**Acceptance.** Span/pooling parameters are echoed in the response and recorded in the archive
manifest; changing them changes the feature protocol and invalidates the weights.

## 0006 — decision region and tap read path (`serve/server.py`, `src/program/generate.cpp`)

**Change.** Defines the decision region (the position after the final candidate, where the
choice is made) and fixes the tap read path so taps are taken from that region rather than
from the last generated token.

**Why.** A one-token readout taken from the wrong position silently produces a different
feature. This was a real failure mode in an earlier revision: a 1-token latent could fall into
the generation path and change the residual.

**Acceptance.** Requests that ask for readouts on a server without the capability are rejected
rather than silently degraded; the same 20 items re-read through the live path match the
archived features byte-for-byte.

---

## Caveats we publish rather than hide

- Multi-layer readout does **not** reuse the prefix cache by design; at equal prompt length it
  costs ~0.1 s more than a plain chat request.
- Feature values depend on expert placement, so a CPU-verified feature set and a GPU
  batched-prefill feature set are **not** interchangeable: `compute_backend` is part of the
  feature protocol.
- The capability bit is reported only under `batch_groups ≤ 2`; with more groups the client
  refuses to run instead of falling back.
