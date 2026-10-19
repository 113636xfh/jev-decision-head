"""Validate saved predictions against frozen gold, without model inference."""
import collections, json, math, sys
from pathlib import Path
ROOT=Path('/home/xfh/data/jev-v3-fast');sys.path.insert(0,str(ROOT))
from common import atomic_json,digest
TAGS=('depth8-s0',)
ref={}
for split in ('dev','test','ood','calibration'):
    for line in (ROOT/'data-fast-20261014'/f'{split}-2000.jsonl').open():
        item=json.loads(line);assert item['uid'] not in ref
        ref[item['uid']]={'split':split,'gold':item['options'].index(item['gold']),
                          'options':len(item['options']),'target':item['target'],
                          'family':str(item['metadata'].get('family','?'))}
summary={}
for tag in TAGS:
    run=ROOT/'runs'/tag
    result=json.loads((run/'results.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['extraction']['compute_backend']=='batched-prefill-one-shot-r2'
    assert manifest['args']['seed']==0
    for name,sha in result['files'].items():assert digest(run/name)==sha
    seen=set();counts=collections.defaultdict(lambda:[0,0]);fam=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
    brier=collections.defaultdict(float)
    for line in (run/'predictions.jsonl').open():
        pred=json.loads(line);uid=pred['uid'];assert uid in ref and uid not in seen;seen.add(uid)
        truth=ref[uid];split=truth['split'];p=pred['probabilities'];K=truth['options']
        assert pred['split']==split and pred['gold']==truth['gold']
        assert len(p)==K and all(math.isfinite(v) and 0<=v<=1 for v in p)
        assert abs(sum(p)-1)<1e-5 and 0<=pred['pred']<K
        assert p[pred['pred']]>=max(p)-1e-7
        correct=int(pred['pred']==truth['gold'])
        counts[split][0]+=correct;counts[split][1]+=1
        fam[split][truth['family']][0]+=correct;fam[split][truth['family']][1]+=1
        brier[split]+=sum((v-t)**2 for v,t in zip(p,truth['target']))
    assert seen==set(ref)
    for split,(correct,n) in counts.items():
        metrics=result['final'][split]
        assert (correct,n)==(metrics['correct'],metrics['n'])
        assert abs(correct/n-metrics['accuracy'])<1e-12
        assert dict(fam[split])==metrics['families']
        assert abs(brier[split]/n-metrics['brier'])<1e-5
    selected=min(result['history'],key=lambda e:e['dev']['nll'])
    assert selected['dev']==result['final']['dev']
    summary[tag]={'passed':True,'evaluation_rows':len(seen),'train_rows':manifest['split_counts']['train'],
                  'selected_epoch':selected['epoch'],'epochs_run':len(result['history']),
                  'checkpoint_selection':'dev NLL only','accuracy':{k:v['accuracy'] for k,v in result['final'].items()},
                  'gold_scope':'exact frozen evaluation gold; soft-target tasks use stored hard gold for accuracy',
                  'results_sha256':digest(run/'results.json')}
receipt={'passed':True,'runs':summary,
         'scope':'All 7718 saved evaluation predictions per completed run checked; no model inference'}
atomic_json(ROOT/'evidence/depth8-results-audit-20261015-0748.json',receipt)
print(json.dumps(receipt))

