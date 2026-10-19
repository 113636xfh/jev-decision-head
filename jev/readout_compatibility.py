"""Explicit, evidence-bound pair compatibility; never edits training manifests."""
import hashlib, json
from pathlib import Path
from urllib.parse import urlsplit

HASH_KEYS=('server_sha256','engine_sha256','model_config_sha256')
CONTRACT_KEYS=('prompt_version','layers','pools','hc','n_embd')
BACKEND='batched-prefill-one-shot-r2'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_live_readout(extraction,installed,tokenizer,certificate=None,base='http://127.0.0.1:8100'):
    if extraction.get('compute_backend')!=BACKEND:raise ValueError('legacy backend cannot use fast live features')
    trained={k:extraction[k] for k in HASH_KEYS}
    if set(installed)!=set(HASH_KEYS):raise ValueError('incomplete installed identity')
    if tokenizer!=extraction['tokenizer']:raise ValueError('tokenizer/template changed')
    if installed==trained and certificate is None:return {'mode':'exact trained pair','installed_hashes':installed}
    if certificate is None:raise ValueError('live extractor differs from trained head; explicit compatibility certificate required')
    endpoint=urlsplit(base)
    if endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','localhost','::1') or endpoint.port!=8100 or endpoint.path not in ('','/'):
        raise ValueError('local pair certificate only covers the local port 8100 endpoint')
    path=Path(certificate).resolve();cert=json.loads(path.read_text())
    if cert.get('schema')!='jev-readout-pair-compat-v1':raise ValueError('unknown compatibility schema')
    if cert.get('from_hashes')!=trained or cert.get('to_hashes')!=installed:raise ValueError('certificate pair mismatch')
    if cert.get('compute_backend')!=BACKEND:raise ValueError('certificate backend mismatch')
    if cert.get('feature_contract')!={k:extraction[k] for k in CONTRACT_KEYS}:raise ValueError('feature contract mismatch')
    if cert.get('tokenizer')!=tokenizer or cert.get('model_artifacts')!=extraction['model_artifacts']:
        raise ValueError('certificate tokenizer/weights mismatch')
    # Both halves of this rollout retain exactly the same tokenizer and config.
    if trained['model_config_sha256']!=installed['model_config_sha256']:raise ValueError('config change is outside this certificate protocol')
    evidence=cert.get('evidence',{})
    if set(evidence)!={'cooperative','online','chat_latency'}:raise ValueError('incomplete acceptance evidence')
    reports={}
    for name,entry in evidence.items():
        target=(path.parent/entry['path']).resolve()
        try:target.relative_to(path.parent)
        except ValueError:raise ValueError('evidence path escapes certificate directory')
        if digest(target)!=entry['sha256']:raise ValueError('acceptance evidence changed: '+name)
        report=json.loads(target.read_text())
        if report.get('passed') is not True:raise ValueError('acceptance did not pass: '+name)
        reports[name]=report
    r=reports['cooperative'];con=r.get('concurrent',{})
    if r.get('from_hashes')!=trained or r.get('installed_hashes')!=installed or r.get('compute_backend')!=BACKEND:
        raise ValueError('cooperative acceptance covers another pair/backend')
    if len(r.get('archived_fp16_checks',[]))<20 or not all(con.get(k) is True for k in
            ('two_readouts_before_chat_end','archived_fp16_equal','long_raw_fp32_equal')) or con.get('chat_content_during_latent',0)<1:
        raise ValueError('incomplete numerical/interleaving proof')
    r=reports['online']
    required={'short_real_prompt_six_layers','invalid_requests_rejected_and_engine_survived',
              'long_5000_token_full_span','wide_18_spans_serial_chunks','concurrent_chat_and_two_latents'}
    if r.get('installed_hashes')!=installed or not required.issubset({c.get('name') for c in r.get('checks',[])}):
        raise ValueError('online coverage or pair mismatch')
    mixed={c.get('case'):c for c in reports['chat_latency'].get('checks',[]) if c.get('case') in ('exclusive-with-chat','shared-with-chat')}
    if set(mixed)!={'exclusive-with-chat','shared-with-chat'}:raise ValueError('missing full chat comparison')
    for row in mixed.values():
        chat=row.get('chat',{})
        if row.get('complete_chat_matches_baseline') is not True or chat.get('finish_reason')!='length' or chat.get('usage',{}).get('completion_tokens')!=256:
            raise ValueError('complete chat evidence missing')
    if mixed['shared-with-chat'].get('latent_before_chat_done') is not True or mixed['shared-with-chat'].get('chat_content_events_during_latent',0)<1:
        raise ValueError('mixed chat did not actually advance')
    return {'mode':'explicit bounded compatibility certificate','certificate_sha256':digest(path),
            'installed_hashes':installed,'scope':cert['scope']}
