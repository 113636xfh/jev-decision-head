"""Independent, offline acceptance of the real ``python -m jev.probe_drift``.

Only weight loading and urllib's transport are replaced in disposable child
processes. The Client, tokenizer/hash checks, readout validation, FP16/CPU
normalization, MeasurementHead list contract, CLI and report writing are real.
No listener, external service, GPU, persistent evidence or other test helper.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from jev.gpu_data import LAYERS
from jev.strata_tokenizer import QWEN35_PATTERN, bytes_to_unicode

RELEASE = Path(__file__).resolve().parents[1]
CHILD_HOOK = r'''
import base64
import io
import json
import os
import socket
import urllib.error
import urllib.request
import numpy as np
import torch
from jev import runtime

torch.set_num_threads(2)
def forbidden(*args, **kwargs):
    raise AssertionError('probe test forbids real network/GPU access')
socket.create_connection = forbidden
torch.cuda._lazy_init = forbidden

def trace(event):
    with open(os.environ['PROBE_TEST_TRACE'], 'a', encoding='utf-8') as sink:
        sink.write(json.dumps(event) + '\n')

class MockHead:
    def __call__(self, x, counts):
        assert x.device.type == 'cpu' and x.dtype == torch.float32
        assert tuple(x.shape) == (1, 4, 2, 5, 10240)
        assert counts.tolist() == [2]
        value = x[0, 0, 0, 0, 0]
        result = torch.stack([value, 1 - value]).unsqueeze(0)
        failure = os.environ.get('PROBE_TEST_FAIL')
        if failure == 'nonfinite':
            result[0, 0] = float('nan')
        if failure == 'short-logits':
            result = result[:, :1]
        return result

def load_head(weights_dir, device):
    trace({'event': 'load_head', 'actual_device': 'cpu'})
    mu = torch.zeros((2, 5, 10240))
    return MockHead(), mu, torch.ones_like(mu), {'hc': 4, 'n_embd': 2560}
runtime.load_head = load_head

def urlopen(request, timeout=None):
    assert isinstance(request, urllib.request.Request)
    assert request.full_url.startswith('http://mock.invalid/')
    path = request.full_url[len('http://mock.invalid'):]
    payload = json.loads(request.data) if request.data is not None else None
    trace({'event': path, 'payload': payload, 'timeout': timeout})
    failure = os.environ.get('PROBE_TEST_FAIL')
    if path == '/metrics':
        body = {'engine': {'model': 'wrong' if failure == 'runtime' else 'offline-mock'}}
    elif path == '/v1/models':
        body = {'data': [{'id': 'offline-mock'}]}
    elif path == '/v1/chat/completions':
        if failure == 'chat-http':
            raise urllib.error.HTTPError(request.full_url, 503, 'mock chat unavailable', {}, None)
        # Exactly two keys also catches accidental tuple-unpacking of a dict.
        body = {'choices': [{'message': {'content': 'ok'}}], 'id': 'mock-chat'}
        if failure == 'chat-shape':
            body = {'choices': []}
    elif path == '/v1/latent':
        if failure == 'latent-http':
            raise urllib.error.HTTPError(request.full_url, 502, 'mock readout unavailable', {}, None)
        data = base64.b64encode(np.full(10240, 2, dtype='<f4').tobytes()).decode('ascii')
        segments = []
        for span, (a, b) in enumerate(payload['latent_spans']):
            for layer in map(int, payload['latent_taps'].split(',')):
                for kind in ('span_last', 'span_mean'):
                    segments.append({'span': span, 'layer': layer, 'kind': kind,
                                     'pos0': b - 1 if kind == 'span_last' else a,
                                     'rows': b - a, 'data_b64': data})
        if failure == 'readout':
            segments.pop()
        body = {'contract_version': runtime.CONTRACT, 'readout_backend': runtime.BACKEND,
                'hc': 4, 'n_embd': 2560, 'prompt_tokens': len(payload['latent_ids']),
                'segments': segments, 'timing': {'api_ms': 7.25}}
    else:
        raise AssertionError('unexpected mock endpoint: ' + path)
    return io.BytesIO(json.dumps(body).encode('utf-8'))
urllib.request.urlopen = urlopen

if os.environ.get('PROBE_TEST_LEGACY'):
    from jev.head_loader import MeasurementHead
    original = MeasurementHead.normalized_from_raw
    def legacy(self, raw, k):
        result = original(self, raw, k)
        result['logits'] = np.asarray(result['logits'])
        return result
    MeasurementHead.normalized_from_raw = legacy
'''


class ProbeDriftEntry(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jev probe entry ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hooks = self.root / 'hooks'
        self.hooks.mkdir()
        (self.hooks / 'sitecustomize.py').write_text(CHILD_HOOK, encoding='utf-8')
        self.trace = self.root / 'transport.jsonl'
        self.weights = self.root / 'weights'
        self.weights.mkdir()
        (self.weights / 'best.pt').write_bytes(b'offline-weight-boundary')
        self.tokenizer = self.root / 'tokenizer pack'
        self.tokenizer.mkdir()
        chars = list(bytes_to_unicode().values()) + ['ab']
        self.write_json(self.tokenizer / 'tokenizer.json', {
            'model': 'gpt2', 'pre': 'qwen35', 'pre_pattern': QWEN35_PATTERN,
            'add_bos_token': False, 'vocab_size': 257, 'n_merges': 1,
        })
        self.write_json(self.tokenizer / 'vocab.json', dict(zip(chars, range(257))))
        self.write_json(self.tokenizer / 'token_type.json', [1] * 257)
        (self.tokenizer / 'merges.txt').write_text('a b', encoding='utf-8')
        (self.tokenizer / 'chat_template.jinja').write_text(
            '{{ messages[0]["content"] }}\nassistant:\n', encoding='utf-8')
        self.identity = self.root / 'identity.json'
        self.ident = {
            'runtime_files': {'serve/server.py': 'fixture', 'engine-cuda12/strata': 'fixture'},
            'running_config': {'model_name': 'offline-mock', 'tokenizer_dir': str(self.tokenizer)},
            'engine_metrics': {'model': 'offline-mock'},
            'tokenizer_sha256': {p.name: self.digest(p) for p in self.tokenizer.iterdir()},
            'head_files': {'best.pt': self.digest(self.weights / 'best.pt')},
            'taps': '15,23,29,35,41,-1', 'policy': {'temperature': None},
        }
        self.write_json(self.identity, self.ident)
        self.archive = self.root / 'archives'
        self.records = self.archive / 'taps-dev2k'
        self.records.mkdir(parents=True)
        self.spans = [[0, 1], [1, 2], [2, 3], [3, 4]]
        vectors = {'latent_ids': np.array([1, 2, 3, 4]), 'latent_spans': np.array(self.spans)}
        for span in range(4):
            for layer in LAYERS:
                for pool in ('span_last', 'span_mean'):
                    vectors['span%d.L%d.%s' % (span, layer, pool)] = np.zeros(10240, np.float16)
        self.npz = self.records / 'one.npz'
        np.savez_compressed(self.npz, **vectors)
        self.record = {'item': {'uid': 'one', 'options': ['a', 'b'], 'gold': 'b'},
                       'spans': self.spans, 'npz_sha256': self.digest(self.npz)}
        self.write_json(self.records / 'one.json', self.record)
        self.write_json(self.records / 'manifest.json', {'not_an_item': True})
        self.out = self.root / 'probe output'
        self.env = dict(os.environ, JEV_WEIGHTS=str(self.weights), JEV_ARCHIVE_ROOT=str(self.archive),
                        PROBE_TEST_TRACE=str(self.trace), CUDA_VISIBLE_DEVICES='',
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                        PYTHONPATH=os.pathsep.join([str(self.hooks), str(RELEASE)]))
        for name in ('PROBE_TEST_FAIL', 'PROBE_TEST_LEGACY'):
            self.env.pop(name, None)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value) + '\n', encoding='utf-8')

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def events(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text(encoding='utf-8').splitlines()]

    def cli(self, extra=(), *, failure=None, legacy=False, code=0):
        env = dict(self.env)
        if failure:
            env['PROBE_TEST_FAIL'] = failure
        if legacy:
            env['PROBE_TEST_LEGACY'] = '1'
        result = subprocess.run(
            [sys.executable, '-m', 'jev.probe_drift', '--phase', 'start',
             '--identity', str(self.identity), '--base', 'http://mock.invalid',
             '--out-prefix', str(self.out)] + list(extra),
            cwd=str(self.root), env=env, capture_output=True, text=True, timeout=60)
        self.assertNotIn('Error in sitecustomize', result.stderr)
        self.assertEqual(result.returncode, code, result.stdout + '\n' + result.stderr)
        if code:
            self.assertFalse((self.out / 'repeat-start.json').exists(), result.stdout)
        return result

    def test_real_one_item_list_contract_chat_and_current_tokenizer(self):
        result = self.cli(['--chat', '--exclusive'])
        summary = self.read_json(self.out / 'repeat-start.json')
        self.assertEqual(summary['identity_sha256'], self.digest(self.identity))
        self.assertEqual((summary['n'], summary['decisions_stable_vs_archived']), (1, 0))
        self.assertEqual(summary['max_logit_delta_max'], 2)
        self.assertEqual(summary['chat']['content'], 'ok')
        self.assertEqual(summary['chat']['status'], 200)
        self.assertGreaterEqual(summary['chat']['seconds'], 0)
        self.assertEqual(json.loads(result.stdout)['n'], 1)
        row, = [json.loads(line) for line in (self.out / 'repeat-start.jsonl').read_text(
            encoding='utf-8').splitlines()]
        self.assertEqual((row['uid'], row['gold'], row['pred'], row['pred_archived']), ('one', 1, 0, 1))
        self.assertEqual(row['max_abs_diff_vs_archived'], 2)
        self.assertEqual(row['max_logit_delta'], 2)
        np.testing.assert_allclose(row['probs'], np.exp([2., -1.]) / np.exp([2., -1.]).sum())
        self.assertEqual(row['api_ms'], 7.25)
        events = self.events()
        self.assertEqual([r['event'] for r in events], [
            'load_head', '/metrics', '/v1/models', '/v1/chat/completions', '/metrics', '/v1/latent'])
        chat = events[3]
        self.assertEqual(chat['timeout'], 300)
        self.assertEqual(chat['payload'], {'model': 'offline-mock',
            'messages': [{'role': 'user', 'content': 'Reply with exactly: ok'}],
            'max_tokens': 16, 'temperature': 0})
        payload = events[-1]['payload']
        self.assertEqual(payload['latent_ids'], [1, 2, 3, 4])
        self.assertEqual(payload['latent_spans'], self.spans)
        self.assertEqual(payload['latent_taps'], self.ident['taps'])
        self.assertEqual(payload['latent_pool'], 'last,mean')
        self.assertEqual(payload['cache_region'], 'decision')
        self.assertFalse(payload['latent_parallel'])

    def test_legacy_arrays_and_tokenizer_override(self):
        self.ident['running_config']['tokenizer_dir'] = str(self.root / 'missing recorded pack')
        self.write_json(self.identity, self.ident)
        self.cli(['--tokenizer-dir', str(self.tokenizer)], legacy=True)
        summary = self.read_json(self.out / 'repeat-start.json')
        self.assertEqual(summary['n'], 1)
        self.assertIsNone(summary['chat'])
        self.assertEqual(summary['max_logit_delta_max'], 2)
        self.assertTrue(self.events()[-1]['payload']['latent_parallel'])

    def test_archived_ids_do_not_require_tokenizer_when_no_path_recorded(self):
        del self.ident['running_config']['tokenizer_dir']
        self.write_json(self.identity, self.ident)
        self.cli()
        self.assertEqual(self.events()[-1]['payload']['latent_ids'], [1, 2, 3, 4])

    def test_current_tokenizer_hash_failure_before_chat_or_readout(self):
        self.ident['tokenizer_sha256']['vocab.json'] = '0' * 64
        self.write_json(self.identity, self.ident)
        result = self.cli(['--chat'], code=1)
        self.assertIn('tokenizer files differ from the recorded identity', result.stderr)
        self.assertEqual(self.events(), [])

    def test_pack_loading_errors_propagate(self):
        self.write_json(self.tokenizer / 'tokenizer.json', {'model': 'unsupported'})
        result = self.cli(code=1)
        self.assertIn('unsupported tokenizer pack', result.stderr)
        self.assertEqual(self.events(), [])

    def test_empty_archive_stops_before_head_chat_or_output(self):
        (self.records / 'one.json').unlink()
        result = self.cli(['--chat'], code=1)
        self.assertIn('no archived items selected', result.stderr)
        self.assertEqual(self.events(), [])
        self.assertFalse(self.out.exists())

    def test_runtime_mismatch_stops_before_chat(self):
        result = self.cli(['--chat'], failure='runtime', code=1)
        self.assertIn('engine capacity/placement changed', result.stderr)
        self.assertEqual([r['event'] for r in self.events()], ['load_head', '/metrics'])

    def test_chat_http_error_propagates_without_readout(self):
        result = self.cli(['--chat'], failure='chat-http', code=1)
        self.assertIn('HTTP Error 503: mock chat unavailable', result.stderr)
        self.assertNotIn('/v1/latent', [r['event'] for r in self.events()])

    def test_malformed_chat_is_not_reported_as_success(self):
        self.cli(['--chat'], failure='chat-shape', code=1)
        self.assertNotIn('/v1/latent', [r['event'] for r in self.events()])

    def test_latent_http_error_propagates(self):
        result = self.cli(failure='latent-http', code=1)
        self.assertIn('HTTP Error 502: mock readout unavailable', result.stderr)

    def test_invalid_readout_propagates(self):
        result = self.cli(failure='readout', code=1)
        self.assertIn('missing per-span/layer/pool', result.stderr)

    def test_nonfinite_and_short_logits_are_rejected(self):
        for failure in ('nonfinite', 'short-logits'):
            with self.subTest(failure=failure):
                result = self.cli(failure=failure, code=1)
                self.assertIn('invalid head logits', result.stderr)

    def test_archive_span_count_mismatch_is_rejected_without_readout(self):
        with np.load(self.npz, allow_pickle=False) as archive:
            vectors = {name: archive[name] for name in archive.files}
        vectors['latent_spans'] = np.array(self.spans[:3])
        np.savez_compressed(self.npz, **vectors)
        result = self.cli(code=1)
        self.assertIn('span/candidate mismatch', result.stderr)
        self.assertNotIn('/v1/latent', [r['event'] for r in self.events()])


if __name__ == '__main__':
    unittest.main()
