"""Offline one-item acceptance for the actual module/subprocess entry points.

No HTTP listener, external endpoint, weights/GPU, or service mutation is
needed. Only the head loader and HTTP transport boundary are replaced in child
processes; tokenizer loading, request building, response validation, normalization,
MeasurementHead list/timing contracts, scoring, evidence, orchestration and exit
codes are real. Set JEV_WORKFLOW_EVIDENCE to retain fixtures, commands and outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import torch

from jev.eval_benchmark import score_one, subset_metrics
from jev.gpu_data import LAYERS
from jev.head_loader import MeasurementHead
from jev.strata_tokenizer import QWEN35_PATTERN, bytes_to_unicode

RELEASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

# This hook is written ONLY to the disposable test/evidence directory. All child
# processes (including run_all's children) inherit it through PYTHONPATH.
CHILD_HOOK = r'''
import base64
import json
import os
from pathlib import Path
import urllib.request
import numpy as np
import torch
from jev import runtime
torch.set_num_threads(2)

def forbidden(*args, **kwargs):
    raise AssertionError('offline workflow test forbids network/GPU access')
urllib.request.urlopen = forbidden
torch.cuda._lazy_init = forbidden

def trace(event):
    with open(os.environ['JEV_WORKFLOW_TRACE'], 'a', encoding='utf-8') as sink:
        sink.write(json.dumps(event) + '\n')

class MockHead:
    def __call__(self, x, counts):
        assert x.device.type == 'cpu' and x.dtype == torch.float32
        assert tuple(x.shape[:4]) == (1, int(counts[0]) + 2, 2, 5)
        k = int(counts[0])
        logits = torch.arange(k, dtype=torch.float32).unsqueeze(0)
        if os.environ.get('JEV_WORKFLOW_FAIL') == 'nonfinite':
            logits[0, 0] = float('nan')
        if os.environ.get('JEV_WORKFLOW_FAIL') == 'drift':
            # Start and mid are equal; only the end probe changes its decision.
            if 'repeat-end' in ' '.join(__import__('sys').argv):
                logits = logits.flip(-1)
        return logits

def load_head(weights_dir, device):
    trace({'event': 'load_head', 'actual_device': 'cpu'})
    mu = torch.zeros((2, 5, 10240))
    return MockHead(), mu, torch.ones_like(mu), {'hc': 4, 'n_embd': 2560}
runtime.load_head = load_head

def get(self, path, timeout=15):
    assert self.base == 'http://mock.invalid' and path == '/metrics'
    return {'engine': {'model': 'offline-mock'}, 'live': {'running': 0}}

def post(self, path, payload, timeout=None):
    assert self.base == 'http://mock.invalid' and path == '/v1/latent'
    trace({'event': 'latent', 'tokens': len(payload['latent_ids']),
           'spans': len(payload['latent_spans']), 'parallel': payload['latent_parallel']})
    data = base64.b64encode(np.zeros(10240, dtype='<f4').tobytes()).decode('ascii')
    segments = []
    for span, (a, b) in enumerate(payload['latent_spans']):
        for layer in map(int, payload['latent_taps'].split(',')):
            for kind in ('span_last', 'span_mean'):
                segments.append({'span': span, 'layer': layer, 'kind': kind,
                                 'pos0': b - 1 if kind == 'span_last' else a,
                                 'rows': b - a, 'data_b64': data})
    if os.environ.get('JEV_WORKFLOW_FAIL') == 'readout':
        segments.pop()
    return {'contract_version': runtime.CONTRACT, 'readout_backend': runtime.BACKEND,
            'hc': 4, 'n_embd': 2560, 'prompt_tokens': len(payload['latent_ids']),
            'segments': segments, 'timing': {'api_ms': 7.25}}
runtime.Client.get = get
runtime.Client._post = post

# Compatibility check: callers must accept old array-valued loaders as well as
# MeasurementHead's real list contract. Never change the live runtime.
if os.environ.get('JEV_WORKFLOW_LEGACY_ARRAY'):
    from jev.head_loader import MeasurementHead
    original = MeasurementHead.normalized_from_raw
    def legacy(self, raw, k):
        result = original(self, raw, k)
        result['logits'] = np.asarray(result['logits'])
        return result
    MeasurementHead.normalized_from_raw = legacy
'''


class WorkflowEntries(unittest.TestCase):
    def setUp(self):
        evidence = os.environ.get('JEV_WORKFLOW_EVIDENCE')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + '-', dir=evidence))
        else:
            temporary = tempfile.TemporaryDirectory(prefix='jev workflow ')
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
        self.hooks = self.root / 'hooks'
        self.hooks.mkdir()
        # Keep the orchestrator stdlib-only, just like the shipped client; do not import
        # an extra copy of torch while it waits for its evaluation subprocess.
        hook = ("import sys\nif 'jev.run_all' not in getattr(sys, 'orig_argv', sys.argv):\n"
                + textwrap.indent(textwrap.dedent(CHILD_HOOK), '    '))
        (self.hooks / 'sitecustomize.py').write_text(hook, encoding='utf-8')
        self.trace = self.root / 'transport.jsonl'
        self.weights = self.root / 'weights'
        self.weights.mkdir()
        (self.weights / 'best.pt').write_bytes(b'offline-head-boundary-fixture')
        np.savez(self.weights / 'norm.npz', mu=np.zeros((2, 5, 10240), np.float32),
                 sd=np.ones((2, 5, 10240), np.float32))
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
            'schema': 'jev-abliterated-live-identity-v1',
            'running_config': {'model_name': 'offline-mock', 'tokenizer_dir': str(self.tokenizer),
                               'sha256': '3' * 64},
            'engine_metrics': {'model': 'offline-mock'},
            'runtime_files': {'serve/server.py': '0' * 64,
                              'engine-cuda12/strata': '1' * 64,
                              'engine-cuda12/strata-vision': '2' * 64},
            'taps': '15,23,29,35,41,-1',
            'tokenizer_sha256': {p.name: self.digest(p) for p in self.tokenizer.iterdir()},
            'head_files': {p.name: self.digest(p) for p in self.weights.iterdir()},
            'policy': {'temperature': None},
        }
        self.write_json(self.identity, self.ident)
        self.archive = self.root / 'archives'
        for split in ['dev', 'test']:
            directory = self.archive / ('taps-' + split + '2k')
            directory.mkdir(parents=True)
            # The same one item is reused in each stage and each repeat sample.
            path = directory / 'one.npz'
            spans = [[0, 1], [1, 2], [2, 3], [3, 4]]
            vectors = {'latent_ids': np.array([1, 2, 3, 4]), 'latent_spans': np.array(spans)}
            for s in range(4):
                for layer in LAYERS:
                    for pool in ('span_last', 'span_mean'):
                        vectors['span%d.L%d.%s' % (s, layer, pool)] = np.zeros(10240, np.float16)
            np.savez_compressed(path, **vectors)
            self.write_json(path.with_suffix('.json'), {
                'item': {'uid': 'one', 'options': ['a', 'b'], 'gold': 'b'},
                'spans': spans, 'npz_sha256': self.digest(path),
            })
            # Must not be mistaken for an archive item.
            self.write_json(directory / 'manifest.json', {'fixture': True})
        self.env = dict(os.environ, JEV_WEIGHTS=str(self.weights), JEV_ARCHIVE_ROOT=str(self.archive),
                        JEV_WORKFLOW_TRACE=str(self.trace), JEV_PY=sys.executable,
                        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                        PYTHONPATH=os.pathsep.join([str(self.hooks), str(RELEASE)]))
        self.env.pop('JEV_WORKFLOW_FAIL', None)
        self.env.pop('JEV_WORKFLOW_LEGACY_ARRAY', None)
        self.count = 0

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    def read_rows(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def cli(self, module, extra=(), *, fail=None, legacy=False, code=0):
        self.count += 1
        command = [sys.executable, '-m', module, '--identity', str(self.identity),
                   '--base', 'http://mock.invalid', '--confirm-window', '--limit', '1'] + list(extra)
        env = dict(self.env)
        if fail:
            env['JEV_WORKFLOW_FAIL'] = fail
        if legacy:
            env['JEV_WORKFLOW_LEGACY_ARRAY'] = '1'
        result = subprocess.run(command, cwd=str(self.root), env=env, capture_output=True,
                                text=True, timeout=120)
        stem = self.root / ('command-%02d' % self.count)
        stem.with_suffix('.stdout.txt').write_text(result.stdout, encoding='utf-8')
        stem.with_suffix('.stderr.txt').write_text(result.stderr, encoding='utf-8')
        self.write_json(stem.with_suffix('.json'), {
            'command': command, 'returncode': result.returncode, 'expected_returncode': code,
            'offline': True, 'mock_head_and_transport_only': True,
        })
        self.assertEqual(result.returncode, code, result.stdout + '\n' + result.stderr)
        self.assertNotIn('Error in sitecustomize', result.stderr)
        return result

    def test_measurement_lists_argmax_and_timings(self):
        from jev.head_loader import MeasurementHead
        head = object.__new__(MeasurementHead)
        head.temperature = None
        head._requests = 0
        head.check_runtime = lambda: None
        head.readout = lambda request, spans: ({}, 7.25)
        head.stack = lambda values, k: np.zeros((4, 2, 5, 10240), np.float16)
        head.logits = lambda raw, k: torch.tensor([0., 1.])
        result = head.infer_request({'latent_spans': [[0, 1]]}, 2)
        self.assertIsInstance(result['logits'], list)
        self.assertIsInstance(result['probabilities'], list)
        self.assertEqual(result['argmax'], 1)
        self.assertEqual(result['feature_api_ms'], 7.25)
        self.assertGreaterEqual(result['head_ms'], 0)
        self.assertGreaterEqual(result['head_path_ms'], result['head_ms'])
        json.dumps(result)
        self.assertEqual(head._requests, 1)

    def test_replay_real_module_one_item_and_legacy_array(self):
        for legacy in (False, True):
            out = self.root / ('replay-' + str(legacy))
            self.cli('jev.eval_replay', ['--split', 'dev', '--out', str(out)], legacy=legacy)
            row, = self.read_rows(Path(str(out) + '.jsonl'))
            self.assertEqual((row['status'], row['pred_new'], row['pred_old']), ('ok', 1, 1))
            self.assertEqual(row['max_logit_delta'], 0)
            self.assertEqual(row['vectors'], row['byte_equal'])
            summary = self.read_json(Path(str(out) + '-summary.json'))
            self.assertTrue(summary['passed'])
            self.assertEqual(summary['metrics']['valid'], 1)
            self.assertEqual(summary['identity_sha256'], self.digest(self.identity))

    def test_benchmark_real_module_current_identity_and_tokenizer_override(self):
        # Override the recorded path but keep the file identities: tests the CLI
        # path as well as the runtime's real pack loading/hash verification.
        self.ident['running_config']['tokenizer_dir'] = 'not-a-real-tokenizer-path'
        self.write_json(self.identity, self.ident)
        out = self.root / 'benchmark'
        self.cli('jev.eval_benchmark', ['--mode', 'criterion_as_option', '--out', str(out),
                                     '--tokenizer-dir', str(self.tokenizer)])
        row, = self.read_rows(out / 'predictions.jsonl')
        self.assertEqual(row['status'], 'ok', row)
        self.assertEqual(row['predicted'], row['predicted_raw'])
        self.assertEqual(row['probabilities_raw'], row['probabilities_calibrated'])
        self.assertEqual(row['feature_api_ms'], 7.25)
        self.assertGreaterEqual(row['head_path_ms'], row['head_ms'])
        self.assertIsInstance(row['logits'], list)
        summary = self.read_json(out / 'summary.json')
        self.assertTrue(summary['passed'])
        self.assertEqual((summary['all']['planned'], summary['all']['valid']), (1, 1))
        self.assertIsNotNone(summary['all']['top_label_ece'])
        manifest = self.read_json(out / 'manifest.json')
        self.assertEqual(manifest['tokenizer_dir'], str(self.tokenizer))

    def test_benchmark_identity_tokenizer_fallback_and_legacy_array(self):
        out = self.root / 'benchmark'
        self.cli('jev.eval_benchmark', ['--mode', 'label_as_option', '--out', str(out)], legacy=True)
        self.assertTrue(self.read_json(out / 'summary.json')['passed'])
        # Completed resume still writes a valid summary and adds no item requests.
        (out / 'summary.json').unlink()
        self.cli('jev.eval_benchmark', ['--mode', 'label_as_option', '--out', str(out)])
        self.assertTrue(self.read_json(out / 'summary.json')['passed'])
        self.assertEqual(len(self.read_rows(out / 'predictions.jsonl')), 1)
        self.assertEqual(sum(r['event'] == 'latent' for r in self.read_rows(self.trace)), 1)

    def test_benchmark_tokenizer_identity_mismatch_fails_before_requests(self):
        self.ident['tokenizer_sha256']['vocab.json'] = '0' * 64
        self.write_json(self.identity, self.ident)
        result = self.cli('jev.eval_benchmark', ['--mode', 'label_as_option',
                          '--out', str(self.root / 'bad')], code=1)
        self.assertIn('tokenizer files differ', result.stderr)
        self.assertFalse(self.trace.exists())

    def test_missing_tokenizer_has_actionable_error(self):
        del self.ident['running_config']['tokenizer_dir']
        self.write_json(self.identity, self.ident)
        result = self.cli('jev.eval_benchmark', ['--mode', 'label_as_option',
                          '--out', str(self.root / 'missing tokenizer')], code=2)
        self.assertIn('pass --tokenizer-dir', result.stderr)
        self.assertFalse(self.trace.exists())

    def test_head_identity_mismatch_fails_before_requests(self):
        self.ident['head_files']['best.pt'] = '0' * 64
        self.write_json(self.identity, self.ident)
        result = self.cli('jev.eval_replay', ['--split', 'dev',
                          '--out', str(self.root / 'bad head')], code=1)
        self.assertIn('best.pt differs from the recorded identity', result.stderr)
        self.assertFalse(any(r['event'] == 'latent' for r in self.read_rows(self.trace)))

    def test_archive_digest_failure_is_recorded_without_request(self):
        path = self.archive / 'taps-dev2k' / 'one.json'
        record = self.read_json(path)
        record['npz_sha256'] = '0' * 64
        self.write_json(path, record)
        out = self.root / 'tampered archive'
        self.cli('jev.eval_replay', ['--split', 'dev', '--out', str(out)], code=2)
        row, = self.read_rows(Path(str(out) + '.jsonl'))
        self.assertIn('archived npz digest changed', row['error'])
        self.assertFalse(any(r['event'] == 'latent' for r in self.read_rows(self.trace)))

    def test_benchmark_single_error_is_recorded_and_nonzero(self):
        out = self.root / 'benchmark'
        self.cli('jev.eval_benchmark', ['--mode', 'criterion_as_option', '--out', str(out)],
                 fail='readout', code=2)
        row, = self.read_rows(out / 'predictions.jsonl')
        self.assertEqual(row['status'], 'error')
        self.assertIn('missing per-span/layer/pool', row['error'])
        summary = self.read_json(out / 'summary.json')
        self.assertTrue(summary['finished'])
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['all']['errors'], 1)
        self.assertEqual(summary['all']['accuracy_all_planned'], 0)
        # Resume must not turn a recorded failure into exit 0.
        self.cli('jev.eval_benchmark', ['--mode', 'criterion_as_option', '--out', str(out)], code=2)

    def test_replay_single_error_is_recorded_and_nonzero(self):
        out = self.root / 'replay'
        self.cli('jev.eval_replay', ['--split', 'dev', '--out', str(out)], fail='readout', code=2)
        row, = self.read_rows(Path(str(out) + '.jsonl'))
        self.assertEqual(row['status'], 'error')
        self.assertFalse(self.read_json(Path(str(out) + '-summary.json'))['passed'])
        self.cli('jev.eval_replay', ['--split', 'dev', '--out', str(out)], code=2)

    def test_nonfinite_logits_fail_closed(self):
        for module, argv in [('jev.eval_replay', ['--split', 'dev']),
                             ('jev.eval_benchmark', ['--mode', 'label_as_option'])]:
            self.cli(module, argv + ['--out', str(self.root / module)], fail='nonfinite', code=2)

    def test_empty_archive_fails_instead_of_false_success(self):
        for path in (self.archive / 'taps-dev2k').glob('one.*'):
            path.unlink()
        result = self.cli('jev.eval_replay', ['--split', 'dev', '--out', str(self.root / 'empty')], code=1)
        self.assertIn('no archived items selected', result.stderr)
        self.assertFalse(self.trace.exists())

    def test_score_fields_use_the_distribution_being_reported(self):
        row = {'id': 'one', 'status': 'ok', 'labels': ['0', '1'], 'gold_label': '1',
               'type': 'score', 'head_ms': 1., 'head_path_ms': 2.,
               'probabilities_raw': {'0': .8, '1': .2},
               'probabilities_calibrated': {'0': .1, '1': .9}}
        raw = subset_metrics([row], {'one', 'missing'}, 'probabilities_raw')
        cal = subset_metrics([row], {'one', 'missing'}, 'probabilities_calibrated')
        self.assertEqual((raw['planned'], raw['valid'], raw['errors'], raw['missing']), (2, 1, 1, 1))
        self.assertEqual(raw['accuracy_all_planned'], 0)
        self.assertEqual(cal['accuracy_all_planned'], .5)
        self.assertAlmostEqual(raw['top_label_ece'], .8)
        self.assertAlmostEqual(cal['top_label_ece'], .1)
        self.assertAlmostEqual(score_one(row['probabilities_calibrated'], row['labels'], '1', 'score')['ordinal_ev'], .9)

    def test_run_all_real_children_and_real_repeated_probes(self):
        out = self.root / 'sweep outputs'
        self.cli('jev.run_all', ['--out-prefix', str(out), '--exclusive',
                               '--tokenizer-dir', str(self.tokenizer)])
        status = self.read_json(out / 'run-status.json')
        self.assertTrue(status['stages'][-1]['finished'])
        stages = [r['stage'] for r in status['stages']]
        for name in ['instrument-start', 'instrument-mid', 'instrument-end',
                     'jevbench-criterion', 'jevbench-label', 'dev', 'test']:
            self.assertIn(name, stages)
        comparisons = [r for r in status['stages'] if r['stage'].endswith('-comparison')]
        self.assertEqual(len(comparisons), 3)
        self.assertTrue(all(r['passed'] and r['n'] == 1 for r in comparisons))
        commands = [r['command'] for r in status['stages'] if 'command' in r]
        self.assertEqual(len(commands), 7)
        self.assertTrue(all(command[1] == '-m' for command in commands))
        self.assertTrue(all('--exclusive' in command for command in commands))
        latent = [r for r in self.read_rows(self.trace) if r['event'] == 'latent']
        self.assertEqual(len(latent), 7)
        self.assertTrue(all(not r['parallel'] for r in latent))
        for mode in ['criterion_as_option', 'label_as_option']:
            row, = self.read_rows(out / 'jevbench' / mode / 'predictions.jsonl')
            self.assertEqual(row['status'], 'ok')
        for split in ['dev', 'test']:
            row, = self.read_rows(out / ('rows-' + split + '.jsonl'))
            self.assertEqual(row['status'], 'ok')

    def test_run_all_probe_failure_stops_and_propagates(self):
        out = self.root / 'failed sweep'
        self.cli('jev.run_all', ['--out-prefix', str(out)], fail='readout', code=2)
        status = self.read_json(out / 'run-status.json')
        self.assertEqual([r['stage'] for r in status['stages']], ['instrument-start', 'all'])
        self.assertFalse(status['stages'][-1]['finished'])
        self.assertEqual(status['stages'][0]['returncode'], 2)

    def test_run_all_stage_failure_stops_and_propagates(self):
        out = self.root / 'failed stage'
        self.cli('jev.run_all', ['--out-prefix', str(out), '--skip-instrument'], fail='readout', code=2)
        status = self.read_json(out / 'run-status.json')
        self.assertEqual([r['stage'] for r in status['stages']], ['jevbench-criterion', 'all'])
        self.assertFalse(status['stages'][-1]['finished'])

    def test_run_all_launch_failure_is_recorded_and_nonzero(self):
        self.env['JEV_PY'] = str(self.root / 'missing-python.exe')
        out = self.root / 'missing child interpreter'
        self.cli('jev.run_all', ['--out-prefix', str(out)], code=2)
        status = self.read_json(out / 'run-status.json')
        self.assertIn('FileNotFoundError', status['stages'][0]['error'])
        self.assertFalse(status['stages'][-1]['finished'])
        self.assertFalse(self.trace.exists())

    def test_run_all_repeat_drift_is_an_acceptance_failure(self):
        out = self.root / 'drift'
        self.cli('jev.run_all', ['--out-prefix', str(out)], fail='drift', code=2)
        status = self.read_json(out / 'run-status.json')
        comparison = status['stages'][-2]
        self.assertEqual(comparison['stage'], 'instrument-end-comparison')
        self.assertEqual(comparison['decision_changes_vs_start'], 1)
        self.assertFalse(comparison['passed'])
        self.assertFalse(status['stages'][-1]['finished'])

    def test_deadline_is_incomplete_not_success(self):
        out = self.root / 'deadline'
        self.cli('jev.run_all', ['--out-prefix', str(out), '--deadline', '2000-01-01 00:00:00'], code=2)
        status = self.read_json(out / 'run-status.json')
        self.assertFalse(status['stages'][-1]['finished'])
        self.assertFalse(self.trace.exists())
        for module, extra in [('jev.eval_replay', ['--split', 'dev']),
                              ('jev.eval_benchmark', ['--mode', 'label_as_option'])]:
            self.cli(module, extra + ['--out', str(self.root / module),
                                     '--deadline', '2000-01-01 00:00:00'], code=2)


if __name__ == '__main__':
    unittest.main()
