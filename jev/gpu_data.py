"""Bounded mmap input, pinned prefetch, GPU padding/casting and GPU metrics."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

LAYERS = [23, 29, 35, 41, -1]
SHAPE = (2, 5, 10240)

def normalize_gpu(raw, mu, sd):
    # Match numpy's separate FP32 subtract, FP32 divide and FP16 rounding.
    # FP64 division then FP32 round avoids approximate CUDA FP32 division at half ties.
    delta = raw.float() - mu
    divided = (delta.double() / sd.double()).float()
    return divided.half()

class Dataset:
    def __init__(self, root, allow_pilot=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema'] != 'jev-normalized-fp16-cache-v1': raise ValueError('wrong cache')
        if self.manifest['pilot'] and not allow_pilot: raise ValueError('pilot cache cannot train formal models')
        if not self.manifest.get('all_normalized_bytes_match_cpu'): raise ValueError('unverified rounding')
        h=hashlib.sha256()
        with (self.root/'index.jsonl').open('rb') as f:
            for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
        if h.hexdigest()!=self.manifest['index_sha256']:raise ValueError('cache index changed')
        self.rows = [json.loads(line) for line in (self.root/'index.jsonl').open()]
        if len(self.rows) != self.manifest['rows']: raise ValueError('cache row count mismatch')
        self.maps = {}
        for entry in self.manifest['chunks']:
            path = Path(entry['path'])
            st=path.stat()
            if st.st_size != entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']: raise ValueError('cache chunk changed')
            self.maps[str(path)] = np.memmap(path, mode='r', dtype=np.float16).reshape((-1,)+SHAPE)
        self.splits = {s:[i for i,r in enumerate(self.rows) if r['item']['split']==s] for s in ['train','dev']}

    def host_batch(self, ids):
        rows = [self.rows[i] for i in ids]
        nspans = sum(r['spans'] for r in rows)
        if nspans * np.prod(SHAPE) * 2 > 512*2**20: raise ValueError('bounded batch size exceeded')
        host = torch.empty((nspans,)+SHAPE, dtype=torch.float16, pin_memory=True)
        dest = host.numpy(); offset=0; counts=[]; targets=[]; gold=[]; starts=[]
        max_k = max(len(r['item']['options']) for r in rows)
        for r in rows:
            n = r['spans']; item=r['item']; starts.append(offset)
            np.copyto(dest[offset:offset+n], self.maps[r['chunk']][r['start']:r['start']+n])
            counts.append(len(item['options']))
            targets.append(item['target']+[0.]*(max_k-len(item['options'])))
            gold.append(item['options'].index(item['gold'])); offset+=n
        return host, counts, targets, gold, starts, rows

class Prefetch:
    """One worker, two batches; the host buffer is kept until DMA completes."""
    def __init__(self, data, ids, batch_size=16):
        self.data=data; self.batches=[ids[i:i+batch_size] for i in range(0,len(ids),batch_size)]
        self.stream=torch.cuda.Stream(); self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.previous=None

    def stage(self, ids):
        host, counts, targets, gold, starts, rows=self.data.host_batch(ids)
        with torch.cuda.device(0), torch.cuda.stream(self.stream):
            flat=host.to('cuda:0',non_blocking=True)
            c=torch.tensor(counts,device='cuda:0')
            ss=torch.arange(max(counts)+2,device='cuda:0')[None,:]
            valid=ss<c[:,None]+2
            indices=torch.tensor(starts,device='cuda:0')[:,None]+ss
            indices=indices.masked_fill(~valid,0)
            x=flat[indices].float().masked_fill(~valid[:,:,None,None,None],0)
            target=torch.tensor(targets,device='cuda:0',dtype=torch.float32)
            g=torch.tensor(gold,device='cuda:0'); event=torch.cuda.Event(); event.record(self.stream)
        return (x,c,target,g),rows,host,event

    def __iter__(self):
        if not self.batches: self.executor.shutdown(); return
        future=self.executor.submit(self.stage,self.batches[0])
        try:
            for i in range(len(self.batches)):
                packet=future.result()
                if i+1<len(self.batches): future=self.executor.submit(self.stage,self.batches[i+1])
                values,rows,host,event=packet
                torch.cuda.current_stream().wait_event(event)
                for value in values: value.record_stream(torch.cuda.current_stream())
                if self.previous is not None: self.previous[1].synchronize()
                self.previous=(host,event)
                yield values,rows
        finally:
            self.executor.shutdown(wait=True)
            if self.previous is not None: self.previous[1].synchronize()

def loss_parts(logits,target):
    valid=torch.isfinite(logits)
    logp=logits.log_softmax(-1).masked_fill(~valid,0)
    return -(target*logp).sum(-1),(logits.softmax(-1)-target).square().sum(-1)

def evaluate(model,data,ids,batch_size=16,predictions=False):
    model.eval(); totals=torch.zeros(3,device='cuda:0',dtype=torch.float64)
    family_ids=sorted({str(data.rows[i]['item']['metadata'].get('family','?')) for i in ids})
    family_map={s:i for i,s in enumerate(family_ids)}
    families=torch.zeros((len(family_ids),2),device='cuda:0',dtype=torch.int64)
    saved=[]
    with torch.inference_mode():
        for (x,c,target,gold),rows in Prefetch(data,ids,batch_size):
            logits=model(x,c); nll,brier=loss_parts(logits,target)
            pred=logits.argmax(-1); correct=pred.eq(gold)
            # Match the original per-batch FP32 sum, then accumulate without a host sync.
            totals+=torch.stack([correct.sum(),nll.sum(),brier.sum()]).double()
            fi=torch.tensor([family_map[str(r['item']['metadata'].get('family','?'))] for r in rows],device='cuda:0')
            families[:,0].scatter_add_(0,fi,correct.long())
            families[:,1].scatter_add_(0,fi,torch.ones_like(fi))
            if predictions:
                probs=logits.softmax(-1).cpu().tolist(); pp=pred.cpu().tolist(); gg=gold.cpu().tolist()
                for r,prob,p,g in zip(rows,probs,pp,gg):
                    item=r['item']; k=len(item['options'])
                    saved.append({'uid':item['uid'],'split':item['split'],'gold':g,'pred':p,'probabilities':prob[:k]})
    a,n,b=totals.cpu().tolist(); fam=families.cpu().tolist(); count=len(ids)
    return {'n':count,'correct':int(a),'accuracy':a/count,'nll':n/count,'brier':b/count,
            'families':dict(zip(family_ids,fam))},saved
