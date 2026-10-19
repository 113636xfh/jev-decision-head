"""Adapters around the frozen v2 pipeline; independent gold is never overwritten."""
import argparse, json, math, os, sys, urllib.request
from pathlib import Path
ROOT=Path(os.path.abspath(__file__)).parent
sys.path.insert(0,str(ROOT.parent/'jev-v2'))
from common import SCHEMA, atomic_json, canonical, digest, identity
from fingerprint_model import verify_manifest

def prepare(raw):
    if not isinstance(raw.get('question'),str) or not raw['question'].strip():raise ValueError('nonempty question required')
    split=raw['split']
    if split not in ('train','dev','test','ood','calibration'): raise ValueError('explicit original split required')
    criteria=raw['criteria']
    if not isinstance(criteria,dict) or len(criteria)<2 or not all(isinstance(k,str) and isinstance(v,str) and k and v for k,v in criteria.items()):
        raise ValueError('criteria must be at least two named text candidates')
    options=list(criteria)
    target=raw.get('target')
    if target is not None: validate_distribution(target,len(options))
    if split!='train' and target is None: raise ValueError('evaluation needs independently provided targets')
    uid=str(raw['id']);group=str(raw['group_id'])
    return {'schema':SCHEMA,'uid':'domain:'+uid,'id':raw['id'],'group_id':raw['group_id'],
            'source':raw.get('source','user_domain'),'split':split,'state':raw['state'],
            'text':canonical(raw['state']),'question':raw['question'],'options':options,'crit':criteria,
            'target':target,'gold':options[max(range(len(options)),key=lambda i:target[i])] if target is not None else None,
            'component':'domain-group:'+group,'metadata':{**raw.get('metadata',{}),
            'supervision_origin':'provided' if target is not None else 'unlabeled'}}

def validate_distribution(values,k):
    if not isinstance(values,list) or len(values)!=k or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('invalid target probability vector')
    if abs(sum(values)-1)>1e-5: raise ValueError('target must sum to one')

def teacher_prompt(item):
    # The teacher instruction is separate from the frozen feature prompt.
    # It changes supervision, never the archived input or its provenance.
    return ('Read the complete state, question and candidate criteria. Solve the decision carefully. '
            'The final answer must be ONLY a JSON object {"choice":"candidate_id"}. '
            'Use {"choice":null} if you cannot answer. Do not invent candidate IDs.\n'
            'STATE:\n'+item['text']+'\nQUESTION:\n'+item['question']+'\nCANDIDATES:\n'+canonical(item['crit']))

def parse_final(content,options):
    if not isinstance(content,str): raise ValueError('no final assistant content')
    final=content.strip()
    if final.startswith('```json\n') and final.endswith('\n```'): final=final[8:-4].strip()
    data=json.loads(final)  # Never scan reasoning or search embedded text for an answer.
    if not isinstance(data,dict) or data.get('choice') not in options: raise ValueError('unanswered or invalid final choice')
    choice=data['choice'];target=[float(o==choice) for o in options]
    return choice,target

class DecodeClient:
    def __init__(self,base,model,max_tokens=2048,timeout=180):
        self.base=base.rstrip('/');self.model=model;self.max_tokens=max_tokens;self.timeout=timeout
    def __call__(self,item):
        payload={'model':self.model,'messages':[{'role':'user','content':teacher_prompt(item)}],
                 'max_tokens':self.max_tokens,'temperature':0}
        headers={'Content-Type':'application/json'}
        key=os.environ.get('JEV_API_KEY')
        if key:headers['Authorization']='Bearer '+key
        req=urllib.request.Request(self.base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=self.timeout) as response:data=json.load(response)
        if data['choices'][0].get('finish_reason')=='length':raise ValueError('decode truncated by token budget')
        message=data['choices'][0]['message']
        content=message.get('content')
        choice,target=parse_final(content,item['options'])
        return {'choice':choice,'teacher_target':target,'final_content':content,
                'finish_reason':data['choices'][0].get('finish_reason'),'usage':data.get('usage',{})}

def prepare_file(source,out):
    # First pass stores grouping keys only; second pass emits complete records.
    parent={};splits={};seen=set();keys=[]
    def find(k):
        parent.setdefault(k,k)
        if parent[k]!=k:parent[k]=find(parent[k])
        return parent[k]
    def union(a,b):parent[find(b)]=find(a)
    with open(source,encoding='utf-8') as f:
        for line in f:
            raw=json.loads(line);item=prepare(raw)
            if item['uid'] in seen:raise ValueError('duplicate id')
            seen.add(item['uid'])
            group=item['component'];same='same:'+identity([item['state'],item['question']]);union(group,same)
            context=raw.get('metadata',{}).get('context_id')
            if context is not None:union(group,'context:'+identity(context))
            keys.append((group,item['split']))
    for group,split in keys:
        component=find(group)
        if component in splits and splits[component]!=split:raise ValueError('group/context crosses splits; fix original split')
        splits[component]=split
    dest=Path(out)
    if dest.exists():raise ValueError('prepared output exists; preserve it')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp')
    with open(source,encoding='utf-8') as f,tmp.open('w',encoding='utf-8') as w:
        for line in f:
            item=prepare(json.loads(line));item['component']=find(item['component']);w.write(canonical(item)+'\n')
    os.replace(tmp,dest)
    atomic_json(dest.with_suffix('.manifest.json'),{'schema':SCHEMA,'source_sha256':digest(source),'rows':len(keys),'output_sha256':digest(dest),'split_policy':'original group + same state/question + explicit context'})

def collect_teacher(items,out,client,signature):
    dest=Path(out);old={}
    if dest.exists():
        with dest.open(encoding='utf-8') as f:
            for line in f:
                rec=json.loads(line)
                if rec['uid'] in old:raise ValueError('duplicate teacher record')
                old[rec['uid']]=rec
    dest.parent.mkdir(parents=True,exist_ok=True)
    with open(items,encoding='utf-8') as f,dest.open('a',encoding='utf-8') as w:
        for line in f:
            item=json.loads(line)
            if item['split']!='train':continue
            item_sha=identity(item)
            if item['uid'] in old:
                rec=old[item['uid']]
                if rec['item_sha256']!=item_sha or rec['teacher_signature']!=signature:raise ValueError('teacher resume provenance changed')
                if rec['status']!='ok':raise ValueError('previous teacher failure; review it before resuming')
                continue
            rec={'uid':item['uid'],'item_sha256':item_sha,'options':item['options'],
                 'teacher_signature':signature,'label_source':'same_backbone_decode','independent_gold':False}
            try:rec.update(client(item));rec['status']='ok'
            except Exception as exc:
                rec.update(status='failed',error=type(exc).__name__+': '+str(exc));w.write(canonical(rec)+'\n');w.flush();os.fsync(w.fileno());raise
            w.write(canonical(rec)+'\n');w.flush();os.fsync(w.fileno())
            print(canonical({'uid':item['uid'],'status':'ok'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--items',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('teacher');p.add_argument('--items',required=True);p.add_argument('--out',required=True)
    p.add_argument('--base',default='http://127.0.0.1:8100');p.add_argument('--model',required=True)
    p.add_argument('--max-tokens',type=int,default=2048);p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--model-identity',required=True);p.add_argument('--model-config',default='/home/xfh/strata/v0.1.41/strata-iq3_s.json')
    args=ap.parse_args()
    if args.mode=='prepare':prepare_file(args.items,args.out)
    else:
        artifacts=verify_manifest(args.model_identity,args.model_config)
        signature={'base':args.base,'model':args.model,'max_tokens':args.max_tokens,'temperature':0,'prompt_version':'domain-teacher-json-v1','model_artifacts':artifacts}
        collect_teacher(args.items,args.out,DecodeClient(args.base,args.model,args.max_tokens,args.timeout),signature)
