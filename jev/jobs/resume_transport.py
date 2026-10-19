import os as _os
JEV_DATA = _os.environ.get('JEV_DATA', _os.path.expanduser('~/jev-data'))
STRATA = _os.environ.get('STRATA_ROOT', _os.path.expanduser('~/strata'))
JEV_PY = _os.environ.get('JEV_PY', 'python3')
"""Resume the exact frozen workflow with a bounded HTTP transport override.

Only extractor subprocess arguments gain --timeout; source, data, feature and
workflow manifests remain unchanged. All original guards/retries are retained.
"""
import argparse, datetime, importlib.util, json, subprocess, sys
from pathlib import Path

ROOT=Path(JEV_DATA + '/jev-v3-fast')
UNIT='jev-v3-fast-recovery-s0-20261015.service'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--timeout',type=float,default=1200)
    ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    if not 180 < args.timeout <= 1200:raise ValueError('transport timeout must be >180 and <=1200 seconds')
    original=ROOT/'jobs/run_formal.py'
    work=ROOT/'workflows/20261014-fast-formal-s0'
    sys.path.insert(0,str(ROOT))
    from common import atomic_json,digest,source_hashes
    frozen=json.loads((work/'manifest.json').read_text())
    if digest(original)!=frozen['workflow_sha256'] or source_hashes()!=frozen['code']:
        raise ValueError('original workflow/source changed; do not resume')
    spec=importlib.util.spec_from_file_location('jev_frozen_workflow',original)
    wf=importlib.util.module_from_spec(spec);spec.loader.exec_module(wf)
    original_command=wf.Workflow.command
    def command(self,argv,log_path,environment=None):
        if len(argv)>1 and Path(argv[1])==ROOT/'extract.py':
            if '--timeout' in argv:raise ValueError('unexpected competing timeout override')
            argv=[*argv,'--timeout',str(args.timeout)]
        # Original command still checks all frozen hashes, bounded retries and
        # disk reserve, records actual argv and cleans up its child on failure.
        return original_command(self,argv,log_path,environment)
    wf.Workflow.command=command
    if not args.check_only:
        props=subprocess.check_output(['systemctl','--user','show',
             'jev-v3-fast-formal-s0-20261014.service','-p','MainPID'],text=True).strip()
        if props!='MainPID=0':raise ValueError('original workflow still active')
        failed=json.loads((work/'status.json').read_text())
        if failed.get('status')!='failed' or failed.get('completed_runs'):
            raise ValueError('recovery requires the inspected extraction-only failure')
        if set(failed.get('completed_extractions',[]))!={'train16k','dev2k','test2k','ood2k'}:
            raise ValueError('unexpected recovery phase')
        record=work/'transport-recovery-20261015.json'
        receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'unit':UNIT,'timeout_before_seconds':180,'timeout_after_seconds':args.timeout,
                 'wrapper_sha256':digest(__file__),'frozen_workflow_sha256':digest(original),
                 'workflow_manifest_sha256':digest(work/'manifest.json'),
                 'reason':'three HTTP response timeouts; no feature/data/protocol change',
                 'failure_snapshot':failed}
        if record.exists():raise ValueError('recovery receipt exists; inspect current unit before launching again')
        atomic_json(record,receipt)
    sys.argv=[str(original),'--work',str(work)]+(['--check-only'] if args.check_only else [])
    wf.main()

if __name__=='__main__':main()
