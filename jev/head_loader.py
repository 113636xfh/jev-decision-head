"""Live head: the frozen head driven against a running patched backbone.

It binds to an identity recorded by `jev.identity` (which build is running) and
refuses to keep going when that build changes mid-run.  The environment the head
was trained and tested on is pinned in `REPRODUCE.md`.

Archived feature shards are not part of the release (they are large).  Set
`JEV_ARCHIVE_ROOT` to a directory holding `taps-dev2k/`, `taps-test2k/`, `taps-ood2k/`,
`taps-cal2k/` if you have them.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch

from .gpu_data import LAYERS                    # noqa: E402
from .runtime import WEIGHTS, Client, HeadRuntime, load_json  # noqa: E402

RUN = WEIGHTS
V3 = Path(os.environ.get('JEV_ARCHIVE_ROOT', str(Path(WEIGHTS).parent / 'archives')))


class MeasurementHead(HeadRuntime):
    """Same frozen inference against a live backbone."""

    def __init__(self, base='http://127.0.0.1:8100', identity=None, parallel=True,
                 weights_dir=WEIGHTS, device='cuda', tokenizer_dir=None):
        ident = load_json(identity) if isinstance(identity, (str, Path)) else (identity or {})
        super().__init__(ident, base=base, weights_dir=weights_dir, device=device,
                         tokenizer_dir=tokenizer_dir, parallel=parallel)
        self.parallel = parallel
        self._requests = 0

    def infer_request(self, request, k):
        """Live readout for a built request, chunked over spans."""
        self.check_runtime()
        started = time.perf_counter()
        values, api_ms = self.readout(request, request['latent_spans'])
        raw = self.stack(values, k)
        out = self.normalized_from_raw(raw, k)
        out.update(feature_api_ms=api_ms,
                   head_path_ms=(time.perf_counter() - started) * 1000)
        self._requests += 1
        return out

    def raw_request(self, request, spans, k):
        """Readout for an explicit span list (archive replay sends the archived spans/ids)."""
        self.check_runtime()
        return self.readout(request, spans)

    def normalized_from_raw(self, raw, k):
        """FP16 readout -> frozen normalization -> JSON-safe lists, as in HeadRuntime.

        Array arithmetic belongs at the consumer boundary (np.asarray); do not
        change these lists to tensors/ndarrays just to support an old caller.
        Materializing both lists synchronizes CUDA before measuring head_ms.
        """
        started = time.perf_counter()
        logits = self.logits(np.ascontiguousarray(raw), k)
        temperature = self.temperature or 1.0
        probabilities = torch.softmax(logits / temperature, -1).tolist()
        values = logits.tolist()
        return {'logits': values, 'probabilities': probabilities,
                'argmax': int(np.argmax(probabilities)),
                'head_ms': (time.perf_counter() - started) * 1000}
