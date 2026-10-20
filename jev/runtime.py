"""Self-contained runtime: HTTP client, tokenizer, and the frozen head.

This module is the release's only path into a running patched Strata server.  It
needs no files from the original training workspace: weights come from `weights/`
and the model tokenizer from the serving host.  Record which server you run
against with `python -m jev.identity`; the environment this head was trained and
tested on is pinned in `REPRODUCE.md`.
"""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .common import digest                       # noqa: F401  (re-exported for callers)
from .gpu_data import LAYERS, normalize_gpu      # noqa: E402
from .head import Head                           # noqa: E402
from .readout import build, validate_response    # noqa: E402

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
WEIGHTS = Path(os.environ.get('JEV_WEIGHTS', RELEASE / 'weights'))
MANIFEST = Path(os.environ.get('JEV_HEAD_MANIFEST', RELEASE / 'manifests/middle5-s1.manifest.json'))
DEFAULT_BASE = os.environ.get('JEV_BASE', 'http://127.0.0.1:8100')
CONTRACT = 'jev-readout-v2'
BACKEND = 'batched-prefill'


class RuntimeMismatch(Exception):
    """The live server differs from the recorded identity."""


# ----------------------------------------------------------------------------- io ---
def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


# -------------------------------------------------------------------------- http ---
class Client:
    """Minimal HTTP client for the patched Strata server (stdlib only)."""

    def __init__(self, base=DEFAULT_BASE, api_key=None, timeout=180):
        self.base = base.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key or os.environ.get('JEV_API_KEY')

    def _headers(self):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        return headers

    def _post(self, path, payload, timeout=None):
        request = urllib.request.Request(self.base + path, data=json.dumps(payload).encode('utf-8'),
                                         headers=self._headers())
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.load(response)

    def post(self, path, payload, timeout=None):
        return self._post(path, payload, timeout)

    def get(self, path, timeout=15):
        request = urllib.request.Request(self.base + path, headers=self._headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def metrics(self):
        return self.get('/metrics')

    def latent(self, chunk, timeout=None):
        blob = self._post('/v1/latent', chunk, timeout)
        if blob.get('contract_version') != CONTRACT:
            raise RuntimeMismatch('server readout contract %r is not %r'
                                  % (blob.get('contract_version'), CONTRACT))
        if blob.get('readout_backend') != BACKEND:
            raise RuntimeMismatch('readout backend %r is not %r; the fast path is required'
                                  % (blob.get('readout_backend'), BACKEND))
        return blob

    def chat(self, model, prompt, max_tokens=2048, timeout=None):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'temperature': 0}
        return self._post('/v1/chat/completions', payload, timeout)


# --------------------------------------------------------------------- tokenizer ---
def load_tokenizer(tokenizer_dir):
    """Load a Strata GGUF tokenizer pack with the reference byte-BPE semantics.

    The pack's tokenizer.json is a description, not a Hugging Face serialization.
    Special-token handling and Jinja whitespace must match the serving extractor.
    """
    from .strata_tokenizer import Tokenizer, QWEN35_PATTERN
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    directory = Path(tokenizer_dir)
    config = load_json(directory / 'tokenizer.json')
    if config.get('model') != 'gpt2' or config.get('pre') != 'qwen35':
        raise ValueError('unsupported tokenizer pack; expected gpt2/qwen35')
    if config.get('pre_pattern') != QWEN35_PATTERN or config.get('add_bos_token') is not False:
        raise ValueError('tokenizer pre-pattern/BOS contract changed')
    vocab = load_json(directory / 'vocab.json')
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('vocabulary indices must be unique and contiguous')
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (directory / 'merges.txt').read_text(encoding='utf-8').split('\n')
    types = load_json(directory / 'token_type.json')
    if (len(tokens) != config['vocab_size'] or len(merges) != config['n_merges']
            or len(types) != len(tokens)):
        raise ValueError('tokenizer pack sizes differ from its descriptor')
    tok = Tokenizer(tokens, merges, types, pre=config['pre'], special_ids=config.get('special_ids'))

    class Template:
        def __init__(self, text):
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=['jinja2.ext.loopcontrols'])
            def raise_exception(message):
                raise ValueError(message)
            env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=kw.pop('ensure_ascii', False), **kw)
            env.globals['raise_exception'] = raise_exception
            self._tpl = env.from_string(text)

        def render(self, messages, add_generation_prompt=True, enable_thinking=False):
            return self._tpl.render(messages=messages, tools=None, add_generation_prompt=add_generation_prompt,
                                    enable_thinking=enable_thinking)

    template = Template((directory / 'chat_template.jinja').read_text(encoding='utf-8'))
    hashes = {p.name: sha256_bytes(p.read_bytes())
              for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in
              ('.json', '.jinja', '.txt')}
    return tok, template, hashes


# --------------------------------------------------------------------------- head ---
def head_geometry(manifest_path=MANIFEST):
    manifest = load_json(manifest_path)
    return {'layers': manifest['layers'], 'blocks': manifest['blocks'], 'd': manifest['d'],
            'dropout': manifest['dropout'], 'fusion': manifest['args']['fusion'],
            'hc': manifest['extraction']['hc'], 'n_embd': manifest['extraction']['n_embd'],
            'compute_backend': manifest['extraction']['compute_backend'],
            'trained_on': {k: manifest['extraction'][k]
                           for k in ('engine_sha256', 'server_sha256', 'model_config_sha256')}}


def load_head(weights_dir=WEIGHTS, device='cuda', manifest_path=MANIFEST):
    geometry = head_geometry(manifest_path)
    head = Head(len(geometry['layers']), hc=geometry['hc'], n_embd=geometry['n_embd'],
                d=geometry['d'], blocks=geometry['blocks'], dropout=geometry['dropout'],
                fusion=geometry['fusion'])
    state = torch.load(Path(weights_dir) / 'best.pt', map_location=device, weights_only=True)
    head.load_state_dict(state)
    head.to(device)
    head.requires_grad_(False)
    head.eval()
    with np.load(Path(weights_dir) / 'norm.npz', allow_pickle=False) as archive:
        mu = torch.tensor(archive['mu'], device=device)
        sd = torch.tensor(archive['sd'], device=device)
    return head, mu, sd, geometry


class HeadRuntime:
    """Frozen head + frozen normalization + the patched server's fast readout."""

    def __init__(self, identity, base=DEFAULT_BASE, weights_dir=WEIGHTS, device='cuda',
                 tokenizer_dir=None, parallel=True, timeout=600):
        self.identity = identity
        self.base = base
        self.client = Client(base)
        self.parallel = parallel
        self.timeout = timeout
        required = ('runtime_files', 'running_config', 'head_files', 'tokenizer_sha256', 'engine_metrics', 'taps')
        if any(not identity.get(key) for key in required):
            raise RuntimeMismatch('identity is incomplete; record it with jev.identity on the serving host')
        self.policy = identity.get('policy', {})
        self.temperature = self.policy.get('temperature')
        self.threshold = self.policy.get('threshold')
        self.taps = identity.get('taps', '15,23,29,35,41,-1')
        self._tokenizer = None
        if tokenizer_dir:
            self.set_tokenizer(tokenizer_dir)
        self.weights_dir = Path(weights_dir)
        self.head, self.mu, self.sd, self.geometry = load_head(weights_dir, device)
        self._check_head_files()

    # ---- identity gates ---------------------------------------------------------
    def _check_head_files(self):
        recorded = self.identity.get('head_files') or {}
        for name, want in recorded.items():
            if name == 'manifest.json':
                continue
            target = self.weights_dir / name
            if not target.is_file() or digest(target) != want:
                raise RuntimeMismatch('%s differs from the recorded identity' % name)

    def check_runtime(self):
        """Cheap gate: the server binaries/config are the recorded ones and capacity is unchanged."""
        metrics = self.client.get('/metrics')
        live = {k: metrics['engine'].get(k) for k in self.identity.get('engine_metrics', {})}
        want = self.identity.get('engine_metrics', {})
        if live != want:
            raise RuntimeMismatch('engine capacity/placement changed: %s -> %s'
                                  % (json.dumps(want, sort_keys=True), json.dumps(live, sort_keys=True)))

    def set_tokenizer(self, tokenizer_dir):
        tok, template, hashes = load_tokenizer(tokenizer_dir)
        recorded = self.identity.get('tokenizer_sha256') or {}
        mismatched = {k: (v, hashes.get(k)) for k, v in recorded.items() if hashes.get(k) != v}
        if mismatched:
            raise RuntimeMismatch('tokenizer files differ from the recorded identity: %s' % mismatched)
        self._tokenizer = (tok, template)

    # ---- request construction ---------------------------------------------------
    def request_for(self, item):
        if self._tokenizer is None:
            raise ValueError('a tokenizer is required for live prompts: pass --tokenizer-dir '
                             'pointing at the served model\'s tokenizer directory')
        return build(item, self._tokenizer[0], self._tokenizer[1], layers=self.taps, max_tokens=4096)

    # ---- readout ----------------------------------------------------------------
    def readout(self, request, spans):
        values, api_ms = {}, 0.0
        for offset in range(0, len(spans), 16):
            chunk = dict(request, latent_spans=spans[offset:offset + 16], latent_parallel=self.parallel)
            blob = self.client.latent(chunk, timeout=self.timeout)
            timing = blob.get('timing', {})
            api_ms += timing.get('api_ms', 0)
            for (span, layer, pool), data in validate_response(blob, chunk,
                                                               hc=self.geometry['hc'],
                                                               n_embd=self.geometry['n_embd']).items():
                key = 'span%d.L%d.%s' % (span + offset, layer, pool)
                if key in values:
                    raise ValueError('duplicate readout vector %s' % key)
                values[key] = np.frombuffer(data, dtype='<f4').astype(np.float16)
                if not np.isfinite(values[key]).all():
                    raise ValueError('FP16 readout overflow')
        return values, api_ms

    def stack(self, values, k):
        return np.stack([np.stack([np.stack([values['span%d.L%d.%s' % (s, l, p)] for l in LAYERS])
                                   for p in ('span_last', 'span_mean')]) for s in range(k + 2)])

    def features_for(self, item):
        request = self.request_for(item)
        k = len(item['options'])
        values, api_ms = self.readout(request, request['latent_spans'])
        return self.stack(values, k), api_ms

    # ---- inference --------------------------------------------------------------
    def normalized(self, raw):
        """FP16 readout -> frozen z-score with FP16 rounding -> FP32 for the head.

        The FP16 round is part of the feature contract (train fit and live inference
        must round identically); the head itself runs in FP32.
        """
        with torch.inference_mode():
            return normalize_gpu(torch.from_numpy(np.ascontiguousarray(raw)).to(self.mu.device),
                                 self.mu, self.sd).float()

    def logits(self, raw, k):
        with torch.inference_mode():
            return self.head(self.normalized(raw).unsqueeze(0), torch.tensor([k], device=self.mu.device))[0]

    def probabilities(self, raw, k):
        logits = self.logits(raw, k)
        temperature = self.temperature or 1.0
        return torch.softmax(logits / temperature, -1).tolist()

    def predict(self, item):
        self.check_runtime()
        raw, api_ms = self.features_for(item)
        k = len(item['options'])
        probs = self.probabilities(raw, k)
        return {'probabilities': probs, 'argmax': int(np.argmax(probs)), 'feature_api_ms': api_ms}


def load_runtime(identity_path, base=DEFAULT_BASE, **kwargs):
    """Build a HeadRuntime from an identity file recorded by jev.identity."""
    return HeadRuntime(load_json(identity_path), base=base, **kwargs)
