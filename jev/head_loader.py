import os as _os
JEV_DATA = _os.environ.get('JEV_DATA', _os.path.expanduser('~/jev-data'))
STRATA = _os.environ.get('STRATA_ROOT', _os.path.expanduser('~/strata'))
JEV_PY = _os.environ.get('JEV_PY', 'python3')
"""GPU3 head that answers from the CURRENT live pair, without the old certificate.

`adaptive-head-v1/adapter.py:LiveHead` refuses to open unless the runtime matches its
certificate (engine ce7559e0 / server 3ebd690b / config b6b687b8, the non-abliterated
backbone).  That refusal is by design and the old adapter is left untouched.  This subclass
keeps every numeric behaviour of that adapter - same frozen norm, same best.pt, same
readout.build prompt protocol, same chunking, same temperature - but binds to the identity
recorded by `identity.py` instead of the stale certificate.

Nothing here retrains or renormalises anything: normalization is the frozen train-fit
`runs/middle5-s0/norm.npz` and the head is `runs/middle5-s1/best.pt`.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(JEV_DATA + '/strata-upgrade-20261016')
AD = ROOT / 'adaptive-head-v1'
V3 = Path(JEV_DATA + '/jev-v3-fast')
V4 = Path(JEV_DATA + '/jev-v4-gpu')
RUN = V4 / 'runs/middle5-s1'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(V4), str(V3), str(AD)]

import accept_v01441 as A                                            # noqa: E402
import identity as IDENT                                             # noqa: E402
from adapter import Client, LiveHead, RuntimeMismatch                # noqa: E402
from head import Head                                                # noqa: E402
from gpu_data import normalize_gpu, LAYERS                           # noqa: E402
from readout import build                                            # noqa: E402

BASE = 'http://127.0.0.1:8100'
FULL_CHECK_EVERY = 200       # full SHA re-verification cadence (the plan's ~200-item recheck)


class UncertifiedLiveHead(LiveHead):
    """Same inference path as the accepted adapter, bound to a recorded live identity."""

    def __init__(self, base: str = BASE, identity: str = None, parallel: bool = True):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
            raise ValueError('CUDA_VISIBLE_DEVICES must be 3')
        self.identity = json.loads(Path(identity or HERE / 'runtime-identity.json').read_text())
        if self.identity.get('schema') != 'jev-abliterated-live-identity-v1':
            raise ValueError('unsupported identity file')
        self.base = base
        self.parallel = parallel
        self.client = Client(base.rstrip('/'))
        state = IDENT.verify(self.identity, base)                    # full check before any request
        self.engine_pids = state['engine_pids']
        self.stats = IDENT._stats(self.identity['watched_files'])
        with np.load(RUN / 'norm.npz', allow_pickle=False) as z:
            self.mu = torch.tensor(z['mu'], device='cuda')
            self.sd = torch.tensor(z['sd'], device='cuda')
        self.head = Head(5, fusion='concat').cuda().eval()
        self.head.load_state_dict(torch.load(RUN / 'best.pt', map_location='cuda', weights_only=True))
        self.head.requires_grad_(False)
        self.temperature = self.identity['policy']['temperature']
        self.threshold = self.identity['policy']['threshold']
        self._tokenizer = None
        self._requests = 0

    # ---- identity gates -------------------------------------------------------------
    def check_runtime(self):
        """Cheap per-item gate: files untouched, engine alive, capacity keys unchanged."""
        if IDENT._stats(self.identity['watched_files']) != self.stats:
            raise RuntimeMismatch('runtime or head files changed since the identity was recorded')
        for pid in self.engine_pids:
            if not (Path('/proc') / str(pid)).exists():
                raise RuntimeMismatch('engine restarted; reopen the head')
        metrics = self.client.get('/metrics')
        live = {k: metrics['engine'].get(k) for k in IDENT.CAPACITY_KEYS}
        if live != self.identity['engine_metrics']:
            raise RuntimeMismatch('capacity/placement changed: %s -> %s'
                                  % (self.identity['engine_metrics'], live))

    def check_identity(self):
        """Full gate (re-hashes engine/server/config/tokenizer); run every FULL_CHECK_EVERY items."""
        state = IDENT.verify(self.identity, self.base)
        self.engine_pids = state['engine_pids']
        self.stats = IDENT._stats(self.identity['watched_files'])

    # ---- request construction -------------------------------------------------------
    def request_for(self, item):
        if self._tokenizer is None:
            from extract import tokenizer
            tok, tpl, hashes = tokenizer()
            if hashes != self.identity['tokenizer_sha256']:
                raise RuntimeMismatch('feature tokenizer differs from the recorded identity')
            self._tokenizer = (tok, tpl)
        # Frozen prompt/suffix and six-layer readout, exactly as the accepted adapter does.
        return build(item, *self._tokenizer, layers='15,23,29,35,41,-1', max_tokens=4096)

    # ---- inference -----------------------------------------------------------------
    def infer_request(self, request, k):
        """Parent's path with one change: the scheduling mode is ours (shared-slots by default)."""
        self.check_runtime()
        start = time.monotonic()
        values, api_ms = {}, 0.0
        if len(request['latent_spans']) != k + 2:
            raise ValueError('candidate/span mismatch')
        for offset in range(0, k + 2, 16):
            chunk = dict(request, latent_spans=request['latent_spans'][offset:offset + 16])
            got, timing = self.client.latent(chunk, parallel=self.parallel, timeout=600)
            for name, value in got.items():
                span, suffix = name.split('.', 1)
                key = 'span%d.%s' % (int(span[4:]) + offset, suffix)
                if key in values:
                    raise ValueError('duplicate readout')
                values[key] = value
            api_ms += timing['api_ms']
        raw = np.stack([np.stack([np.stack([values[f'span{s}.L{l}.{p}'] for l in LAYERS])
                                  for p in ('span_last', 'span_mean')]) for s in range(k + 2)]).astype(np.float16)
        if not np.isfinite(raw).all():
            raise ValueError('FP16 readout overflow')
        head_start = time.monotonic()
        with torch.inference_mode():
            normalized = normalize_gpu(torch.from_numpy(raw).cuda(), self.mu, self.sd)
            result = self.infer_normalized(normalized, k)
        result.update(feature_api_ms=api_ms, head_ms=1000 * (time.monotonic() - head_start),
                      head_path_ms=1000 * (time.monotonic() - start))
        self._requests += 1
        if self._requests % FULL_CHECK_EVERY == 0:
            self.check_identity()
        else:
            self.check_runtime()
        return result

    def raw_request(self, request, spans, k):
        """Readout for an explicit span list (archive replay sends the archived spans/ids)."""
        self.check_runtime()
        values, api_ms = {}, 0.0
        for offset in range(0, len(spans), 16):
            chunk = dict(request, latent_spans=spans[offset:offset + 16])
            got, timing = self.client.latent(chunk, parallel=self.parallel, timeout=600)
            for name, value in got.items():
                span, suffix = name.split('.', 1)
                key = 'span%d.%s' % (int(span[4:]) + offset, suffix)
                if key in values:
                    raise ValueError('duplicate readout')
                values[key] = value
            api_ms += timing['api_ms']
        return values, api_ms

    def normalized_from_raw(self, raw, k):
        """FP16 readout -> frozen normalization -> head logits (shared by both harnesses)."""
        with torch.inference_mode():
            normalized = normalize_gpu(torch.from_numpy(raw).cuda(), self.mu, self.sd)
            return self.infer_normalized(normalized, k)
