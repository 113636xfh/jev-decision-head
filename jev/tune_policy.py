"""Dev-only operating point selection. No use of test or teacher labels."""
import argparse,itertools,json
from pathlib import Path
from .cascade import Policy
from .domain import atomic_json,digest

def tune(rows,target_accuracy=.95,min_accepted=20):
    if not rows or any(r.get('split')!='dev' or r.get('independent_gold') not in r.get('options',[]) for r in rows):
        raise ValueError('only independently labeled dev predictions are allowed')
    candidates=[]
    for conf,margin,entropy in itertools.product((.5,.6,.7,.8,.9,.95),(.05,.1,.2,.3),(.25,.4,.55,.7,1.0)):
        policy=Policy(conf,margin,entropy);accepted=correct=0
        for row in rows:
            probs=row.get('head_probabilities')
            if probs is None:continue
            if policy.assess(probs)['accepted']:
                accepted+=1;correct+=int(row['options'][max(range(len(probs)),key=lambda i:probs[i])]==row['independent_gold'])
        if accepted>=min_accepted and correct/accepted>=target_accuracy:
            candidates.append((accepted,correct/accepted,policy))
    if not candidates:return {'policy':vars(Policy(force_decode=True)),'dev_head_coverage':0,'dev_selective_accuracy':None,'reason':'no feasible head operating point'}
    accepted,acc,policy=max(candidates,key=lambda x:(x[0],x[1]))
    return {'policy':vars(policy),'dev_head_coverage':accepted/len(rows),'dev_selective_accuracy':acc,'accepted':accepted}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dev-predictions',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--target-accuracy',type=float,default=.95);ap.add_argument('--min-accepted',type=int,default=20);a=ap.parse_args()
    if not 0<=a.target_accuracy<=1 or a.min_accepted<1:raise ValueError('invalid operating point target')
    if Path(a.out).exists():raise ValueError('policy output exists')
    with open(a.dev_predictions,encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    result=tune(rows,a.target_accuracy,a.min_accepted);result.update(dev_predictions_sha256=digest(a.dev_predictions),selection='dev only; empirical operating point, not a correctness guarantee')
    atomic_json(a.out,result)
