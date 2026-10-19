"""End-to-end live or cached feature routing; threshold defaults are examples, not calibrated values."""
import argparse,json,sys,urllib.request,os
from pathlib import Path
import numpy as np
import torch
from domain import DecodeClient,canonical,identity,atomic_json,digest
from cascade import Policy,decide,summarize
from head import Head
from readout import build,validate_response
from fingerprint_model import verify_manifest
from readout_compatibility import verify_live_readout
import train as trainer

class FeatureClient:
    def __init__(self,base,tok,template,layers):self.base=base.rstrip('/');self.tok=tok;self.template=template;self.layers=layers
    def __call__(self,item):
        req=build(item,self.tok,self.template,','.join(map(str,self.layers)),4096);vectors={}
        for offset in range(0,len(req['latent_spans']),16):
            chunk={**req,'latent_spans':req['latent_spans'][offset:offset+16],'latent_backend':'fast'}
            headers={'Content-Type':'application/json'}
            if os.environ.get('JEV_API_KEY'):headers['Authorization']='Bearer '+os.environ['JEV_API_KEY']
            request=urllib.request.Request(self.base+'/v1/latent',data=json.dumps(chunk).encode(),headers=headers)
            with urllib.request.urlopen(request,timeout=180) as response:blob=json.load(response)
            if blob.get('contract_version')!='jev-readout-v2':raise ValueError('accepted readout-v2 endpoint required')
            if blob.get('readout_backend')!='batched-prefill':raise ValueError('fast readout backend required')
            for (span,layer,pool),data in validate_response(blob,chunk).items():
                v=np.frombuffer(data,dtype='<f4').astype(np.float16)
                if not np.isfinite(v).all():raise ValueError('FP16 feature overflow')
                vectors[span+offset,layer,pool]=v
        return np.stack([np.stack([np.stack([vectors[s,l,p] for l in self.layers]) for p in trainer.POOLS]) for s in range(len(item['options'])+2)])

class HeadPredictor:
    def __init__(self,run,feature_provider):
        run=Path(run);m=json.loads((run/'manifest.json').read_text());a=m['args'];self.layers=list(map(int,a['layers'].split(',')))
        geometry=m['extraction']
        self.head=Head(len(self.layers),hc=geometry['hc'],n_embd=geometry['n_embd'],d=a['d'],blocks=a['blocks'],fusion=a['fusion']);self.head.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True));self.head.eval()
        with np.load(run/'norm.npz',allow_pickle=False) as z:self.norm=(z['mu'].copy(),z['sd'].copy())
        self.provider=feature_provider
    def __call__(self,item):
        x=trainer.preprocess(self.provider(item),self.norm)
        with torch.inference_mode():p=self.head(torch.from_numpy(x)[None],torch.tensor([len(item['options'])])).softmax(-1)[0].tolist()
        return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--items',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--base',default='http://127.0.0.1:8100');ap.add_argument('--model',required=True);ap.add_argument('--cached-shards',nargs='+')
    ap.add_argument('--min-confidence',type=float,default=.8);ap.add_argument('--min-margin',type=float,default=.2);ap.add_argument('--max-entropy',type=float,default=.55)
    ap.add_argument('--max-tokens',type=int,default=2048);ap.add_argument('--policy',help='dev-selected policy JSON')
    ap.add_argument('--model-identity',required=True);ap.add_argument('--model-config',default='/home/xfh/strata/v0.1.41/strata-iq3_s.json')
    ap.add_argument('--readout-compatibility',help='explicit evidence-bound old/new pair certificate; original head manifest stays unchanged');args=ap.parse_args()
    manifest=json.loads((Path(args.run)/'manifest.json').read_text());layers=list(map(int,manifest['args']['layers'].split(',')))
    artifacts=verify_manifest(args.model_identity,args.model_config)
    if artifacts!=manifest['extraction']['model_artifacts']:raise ValueError('current backbone differs from trained head')
    readout_identity={'mode':'cached archive'}
    if args.cached_shards:
        records,_=trainer.collect(args.cached_shards,layers);index={r['item']['uid']:r for r in records}
        def provider(item):
            rec=index[item['uid']]
            if identity(item)!=rec['item_sha256']:raise ValueError('cached item changed; never relabel a feature prompt')
            return trainer.load(rec,layers)
    else:
        # Identical weights do not imply identical features across CPU verify and
        # GPU prefill kernels. Refuse live inference with a legacy-normalized head.
        if manifest['extraction'].get('compute_backend')!='batched-prefill-one-shot-r2':
            raise ValueError('legacy verify head cannot use fast live features; cached offline evaluation remains supported')
        cfg_path=Path(args.model_config);cfg=json.loads(cfg_path.read_text());exe=Path(cfg['exe'])
        if not exe.is_absolute():exe=cfg_path.parent/exe
        # Current project Strata adapter; other backbones must supply their own tokenizer/template/provider.
        from extract import tokenizer
        tok,tpl,token_hashes=tokenizer()
        installed={'engine_sha256':digest(exe),'server_sha256':digest(cfg_path.parent/'serve/server.py'),
                   'model_config_sha256':digest(cfg_path)}
        readout_identity=verify_live_readout(manifest['extraction'],installed,token_hashes,args.readout_compatibility,args.base)
        provider=FeatureClient(args.base,tok,tpl,layers)
    head=HeadPredictor(args.run,provider);decode=DecodeClient(args.base,args.model,args.max_tokens)
    policy=Policy(**json.loads(Path(args.policy).read_text())['policy']) if args.policy else Policy(args.min_confidence,args.min_margin,args.max_entropy)
    out=Path(args.out)
    if out.exists():raise ValueError('prediction output exists')
    out.parent.mkdir(parents=True,exist_ok=True);rows=[]
    with open(args.items,encoding='utf-8') as f,out.open('w',encoding='utf-8') as w:
        for line in f:
            item=json.loads(line);result=decide(item,head,decode,policy)
            result.update(uid=item['uid'],split=item['split'],group_id=item['group_id'],options=item['options'],
                          independent_gold=item.get('gold') if item.get('metadata',{}).get('supervision_origin')=='provided' else None)
            rows.append(result);w.write(canonical(result)+'\n');w.flush()
    atomic_json(out.with_suffix('.summary.json'),{'policy':vars(policy),'thresholds_status':'user supplied; select on dev only','metrics':summarize(rows),'readout_identity':readout_identity})

if __name__=='__main__':main()
