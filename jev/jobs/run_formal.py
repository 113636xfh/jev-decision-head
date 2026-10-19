"""Frozen, serial extraction -> first-seed ablations. Run in a bounded unit.

Workflow code lives outside the extractor's frozen root Python set. No silent
data skips; only transient extraction transport failures get bounded resume.
"""
import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))
from common import atomic_json, digest, source_hashes, volunteer_for_oom
from fingerprint_model import verify_manifest

SOURCES = [('train16k', 'train-16000.jsonl', 16000), ('dev2k', 'dev-2000.jsonl', 1996), ('test2k', 'test-2000.jsonl', 1999), ('ood2k', 'ood-2000.jsonl', 1991), ('cal2k', 'calibration-2000.jsonl', 1732)]
TAGS = ['baseline-s0', 'data2k-s0', 'data4k-s0', 'data8k-s0', 'depth4-s0', 'depth8-s0',
        'middle5-s0', 'weighted3-s0', 'width1024-s0']
BACKBONE = {'server_sha256': '/home/xfh/strata/v0.1.41/serve/server.py',
            'engine_sha256': '/home/xfh/strata/v0.1.41/engine-cuda12/strata',
            'model_config_sha256': '/home/xfh/strata/v0.1.41/strata-iq3_s.json'}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def preflight():
    accepted = json.loads((ROOT / 'evidence/online-acceptance.json').read_text())
    if not accepted.get('passed'): raise ValueError('online acceptance failed')
    for key, path in BACKBONE.items():
        if digest(path) != accepted['installed_hashes'][key]: raise ValueError('installed pair changed: ' + key)
    verify_manifest(ROOT / 'evidence/model-identity.fixed-cache.json', BACKBONE['model_config_sha256'])
    plan = json.loads((ROOT / 'evidence/extraction-plan.json').read_text())
    for name, sha in plan['inputs_sha256'].items():
        if digest(ROOT / 'data-fast-20261014' / name) != sha: raise ValueError('frozen input changed')
    matrix = json.loads((ROOT / 'experiments.json').read_text())
    runs = {r['tag']: r for r in matrix['runs']}
    for tag in TAGS:
        if not Path(runs[tag]['train_subset']).is_absolute() or not Path(runs[tag]['train_subset']).is_file():
            raise ValueError('invalid remote train subset path: ' + tag)
    return accepted, plan

class Workflow:
    def __init__(self, work, contract):
        self.work, self.contract = work, contract
        self.state = {'status': 'running', 'started_utc': now(), 'phase': 'preflight',
                      'completed_extractions': [], 'completed_runs': [], 'child_pid': None}
        if (work / 'status.json').exists():
            old = json.loads((work / 'status.json').read_text())
            for key in ('started_utc', 'completed_extractions', 'completed_runs'):
                self.state[key] = old[key]
        self.child = None

    def save(self, **changes):
        self.state.update(changes); self.state['updated_utc'] = now()
        atomic_json(self.work / 'status.json', self.state)

    def guard(self):
        if source_hashes() != self.contract['code']: raise ValueError('extractor/trainer source changed during workflow')
        if digest(ROOT / 'experiments.json') != self.contract['matrix_sha256']: raise ValueError('experiment matrix changed')
        if digest(ROOT / 'evidence/online-acceptance.json') != self.contract['acceptance_sha256']: raise ValueError('acceptance certificate changed')
        for key, path in BACKBONE.items():
            if digest(path) != self.contract['installed_hashes'][key]: raise ValueError('production pair changed: ' + key)
        if shutil.disk_usage(ROOT).free < 5 * 2**30: raise ValueError('disk reserve below 5GiB')

    def command(self, argv, log_path, environment=None):
        self.guard()
        with log_path.open('ab') as log:
            self.child = subprocess.Popen(argv, cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
            self.save(child_pid=self.child.pid, log=str(log_path), argv=argv)
            try:
                while self.child.poll() is None:
                    time.sleep(5)
                    if int(time.monotonic()) - getattr(self, 'checked', 0) >= 60:
                        self.guard(); self.checked = int(time.monotonic())
                        self.save()
                return self.child.returncode
            finally:
                if self.child.poll() is None:
                    self.child.terminate()
                    try: self.child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.child.kill(); self.child.wait()
                self.child = None; self.save(child_pid=None)

    def run(self, args):
        directories = []
        for label, filename, expected in SOURCES:
            target = ROOT / ('taps-' + label); directories.append(str(target))
            if label in self.state['completed_extractions']:
                receipt_path = self.work / (label + '.complete.json')
                if not receipt_path.is_file(): raise ValueError('completed phase lost completion receipt')
                receipt = json.loads(receipt_path.read_text())
                if receipt['manifest_sha256'] != digest(target / 'manifest.json'):
                    raise ValueError('completed extraction manifest changed')
                continue
            self.save(phase='extract:' + label, expected_rows=expected)
            argv = [sys.executable, str(ROOT / 'extract.py'), '--items', str(ROOT / 'data-fast-20261014' / filename),
                    '--out', str(target), '--layers', '15,23,29,35,41,-1',
                    '--model-identity', str(ROOT / 'evidence/model-identity.fixed-cache.json')]
            log = self.work / (label + '.log')
            for attempt in range(1, 4):
                self.save(attempt=attempt)
                code = self.command(argv, log)
                if code == 0: break
                tail = log.read_bytes()[-6000:].decode('utf-8', 'replace')
                transient = any(x in tail for x in ('TimeoutError:', 'URLError:', 'ConnectionResetError:',
                                                   'RemoteDisconnected:', 'HTTP Error 502', 'HTTP Error 503', 'HTTP Error 504'))
                if not transient or attempt == 3: raise ValueError(f'{label} failed (exit {code}); see {log}')
                self.save(phase='retry:' + label, retry_reason='transient transport only')
                time.sleep(30)
            # Success must mean the complete source made it to verified shards.
            records = [p for p in target.glob('*.json') if p.name != 'manifest.json']
            if len(records) != expected or len(list(target.glob('*.npz'))) != expected:
                raise ValueError('extraction count mismatch: ' + label)
            atomic_json(self.work / (label + '.complete.json'), {'rows': expected, 'input_sha256': digest(ROOT / 'data-fast-20261014' / filename),
                                                  'manifest_sha256': digest(target / 'manifest.json'), 'completed_utc': now()})
            self.state['completed_extractions'].append(label); self.save()
        self.save(phase='train', expected_rows=None)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '3', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        for tag in TAGS:
            output = ROOT / 'runs' / tag
            if tag in self.state['completed_runs']:
                if not (output / 'results.json').is_file(): raise ValueError('completed training result missing')
                continue
            if output.exists(): raise ValueError('partial training output exists; preserve it and choose a reviewed recovery strategy: ' + tag)
            self.save(phase='train:' + tag)
            argv = [sys.executable, str(ROOT / 'launch_training.py'), '--matrix', str(ROOT / 'experiments.json'),
                    '--tag', tag, '--shards', *directories, '--out-root', str(ROOT / 'runs')]
            if self.command(argv, self.work / (tag + '.log'), environment) != 0:
                raise ValueError('training failed: ' + tag)
            if not (output / 'results.json').is_file(): raise ValueError('trainer produced no final result')
            self.state['completed_runs'].append(tag); self.save()
        summary = {tag: json.loads((ROOT / 'runs' / tag / 'results.json').read_text()) for tag in TAGS}
        chosen = min(TAGS, key=lambda tag: summary[tag]['final']['dev']['nll'])
        atomic_json(self.work / 'comparison.json', {'selection': 'dev NLL only', 'selected_tag': chosen,
                                                   'runs': {t: {'parameters': r['parameters'], 'metrics': r['final']} for t, r in summary.items()}})
        self.save(status='complete', phase='complete', finished_utc=now())

def main():
    volunteer_for_oom()
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    accepted, plan = preflight()
    work = Path(args.work)
    if not work.is_absolute(): raise ValueError('workflow directory must be absolute')
    remaining = int(plan['estimated_with_overhead_gib'] * 2**30)
    if not work.exists() and shutil.disk_usage(ROOT).free < remaining + 8 * 2**30:
        raise ValueError('disk too small for extraction plus 5GiB reserve and training outputs')
    contract = {'installed_hashes': accepted['installed_hashes'], 'code': source_hashes(),
                'matrix_sha256': digest(ROOT / 'experiments.json'), 'acceptance_sha256': digest(ROOT / 'evidence/online-acceptance.json'),
                'input_hashes': plan['inputs_sha256'], 'workflow_sha256': digest(__file__),
                'layers': '15,23,29,35,41,-1', 'sources': SOURCES, 'tags': TAGS}
    if args.check_only:
        print(json.dumps({'preflight': 'PASS', 'rows': plan['totals']['items'], 'tags': TAGS,
                          'free_gib': shutil.disk_usage(ROOT).free / 2**30})); return
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'workflow.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = work / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(contract)):
            raise ValueError('workflow provenance changed; preserve existing run')
    else: atomic_json(manifest, contract)
    workflow = Workflow(work, contract)
    def stop(signum, frame): raise InterruptedError('workflow stopped by signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    try: workflow.run(args)
    except BaseException as exc:
        workflow.save(status='failed', error=type(exc).__name__ + ': ' + str(exc)); raise

if __name__ == '__main__': main()
