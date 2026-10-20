"""Seal final install evidence without overwriting an older summary."""
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    primary_name = 'verification-gate-v2.json' if (OUT / 'verification-gate-v2.json').exists() else 'install-gate.json'
    gate = json.loads((OUT / primary_name).read_text())
    initial = json.loads((OUT / 'install-gate.json').read_text())
    if 'finished_utc' not in gate:
        raise SystemExit('installation still running; not a completed gate')
    helps = [s for s in gate['steps'] if '-help-' in s['name']]
    summary = {'sealed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'primary_gate': primary_name, 'initial_quality_gate': initial['quality_gate'],
               'initial_error': initial.get('error'),
               'official_cpu_provenance': gate.get('official_cpu_provenance'),
               'quality_gate': gate['quality_gate'],
               'installation_gate': gate.get('installation_gate', 'FAIL'),
               'help_successes': sum(s['returncode'] == 0 and 'usage:' in (OUT / s['log']).read_text().lower()
                                     for s in helps),
               'help_attempts': len(helps),
               'failed_steps': [s for s in gate['steps'] if s['returncode'] != 0],
               'source_snapshot_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                                           for p in sorted((ROOT / 'jev').glob('*.py'))},
               'requirements_sha256': {p: digest(ROOT / p) for p in ['requirements.txt', 'requirements-cpu.txt']},
               'protected_smoke_evidence_sha256': {p: digest(ROOT / 'evidence/smoke-20-dev-20261020' / p)
                                                   for p in ['README.md', 'identity-fresh.json',
                                                             'rows-smoke20-summary.json', 'smoke20-check.json']},
               'boundary': 'Only this Windows CPython 3.12 CPU install is tested; broad requirement ranges and other OS/Python/GPU configurations are not certified. No live inference, training or production request.'}
    for name, key in [('selfcheck-native-probe-v3.json', 'selfcheck_diagnostic'),
                      ('selfcheck-canonical-rechecks-v4.json', 'selfcheck_canonical_rechecks')]:
        if (OUT / name).exists():
            summary[key] = json.loads((OUT / name).read_text())
    report = OUT / 'pip-install-report.json'
    if report.is_file():
        entries = json.loads(report.read_text())['install']
        summary['installed_versions'] = {i['metadata']['name']: i['metadata']['version'] for i in entries}
        torch = [i for i in entries if i['metadata']['name'].lower() == 'torch']
        summary['torch_download'] = [i['download_info'] for i in torch]
        summary['cuda_dependencies'] = [i['metadata']['name'] for i in entries
                                         if i['metadata']['name'].lower().startswith(('nvidia-', 'cuda-'))]
    summary['artifact_sha256'] = {p.name: digest(p) for p in sorted(OUT.iterdir())
                                   if p.is_file() and p.name not in ('runner.log',)}
    with (OUT / 'sealed-summary-v1.json').open('x', encoding='utf-8') as sink:
        sink.write(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
