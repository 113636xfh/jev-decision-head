"""Train or warm-start a head with teacher sidecars; v2 files/shards remain untouched."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from .domain import digest,identity,validate_distribution,atomic_json
from . import train as trainer

def overlay(records,teachers,alpha):
    result=[];used=set()
    for rec in records:
        original=rec['item'];item=dict(original);new={**rec,'item':item}
        if item['split']=='train':
            t=teachers.get(item['uid'])
            if t is None or t.get('status')!='ok':raise ValueError('every training item needs a valid teacher target')
            if t['item_sha256']!=identity(original) or t['options']!=item['options']:raise ValueError('teacher/input candidate mismatch')
            target=t['teacher_target'];validate_distribution(target,len(item['options']))
            supplied=item.get('target')
            if supplied is not None:
                validate_distribution(supplied,len(target));item['target']=[(1-alpha)*a+alpha*b for a,b in zip(supplied,target)]
            else:item['target']=target
            if item.get('gold') is None:item['gold']=item['options'][max(range(len(target)),key=lambda i:target[i])]
            item['metadata']={**item.get('metadata',{}),'training_teacher_overlay':True};used.add(item['uid'])
        else:
            if item.get('target') is None or item.get('gold') not in item['options'] or item.get('metadata',{}).get('supervision_origin')!='provided':
                raise ValueError('dev/test/ood/calibration need independently provided labels, never teacher-generated holdouts')
        result.append(new)
    return result,used

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher-targets',required=True);ap.add_argument('--teacher-weight',type=float,default=.5)
    ap.add_argument('--init-run',help='warm start best.pt AND its train-fitted norm; matching architecture required')
    args,remaining=ap.parse_known_args()
    if not 0<=args.teacher_weight<=1:raise ValueError('teacher weight must be in [0,1]')
    if '--train-subset' in remaining:raise ValueError('use domain subset archives explicitly; do not hide unneeded unlabeled items')
    teachers={}
    with open(args.teacher_targets,encoding='utf-8') as f:
        for line in f:
            t=json.loads(line)
            if t['uid'] in teachers:raise ValueError('duplicate teacher uid')
            teachers[t['uid']]=t
    original_collect=trainer.collect;used=set()
    def collect(paths,layers):
        records,contract=original_collect(paths,layers)
        changed,ids=overlay(records,teachers,args.teacher_weight);used.update(ids)
        if any(teachers[u]['teacher_signature'].get('model_artifacts')!=contract['model_artifacts'] for u in ids):
            raise ValueError('teacher and archived frozen backbone fingerprints differ')
        return changed,contract
    trainer.collect=collect
    if args.init_run:
        prior=Path(args.init_run);manifest=json.loads((prior/'manifest.json').read_text());old_head=trainer.Head
        def init_head(*a,**kw):
            h=old_head(*a,**kw);h.load_state_dict(torch.load(prior/'best.pt',map_location='cpu',weights_only=True));return h
        def init_norm(records,layers):
            if ','.join(map(str,layers))!=manifest['args']['layers']:raise ValueError('warm-start layer order must match')
            with np.load(prior/'norm.npz',allow_pickle=False) as z:mu,sd=z['mu'].copy(),z['sd'].copy()
            if mu.shape!=sd.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<1e-3).any():raise ValueError('invalid prior norm')
            return mu,sd
        trainer.Head=init_head;trainer.fit_norm=init_norm
    sys.argv=[sys.argv[0],*remaining]
    # Same trainer: dev checkpoint selection and final independent test/OOD.
    out=Path(remaining[remaining.index('--out')+1])
    if out.exists():raise ValueError('output exists; preserve previous run')
    trainer.main()
    atomic_json(out/'distillation.json',{'teacher_sha256':digest(args.teacher_targets),'teacher_weight_with_provided_labels':args.teacher_weight,
                'unlabeled_train_weight':1,'used_train_uids':sorted(used),'holdout_policy':'provided labels only; no teacher targets',
                'init_run':args.init_run,'framework_sha256':digest(__file__),'teacher_target_origin':'same-backbone decode, not independent gold'})

if __name__=='__main__':main()
