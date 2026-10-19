"""Strict prompt/readout contract shared by extraction and offline validation."""
import base64
import math
import struct
from common import PROMPT_VERSION

def layers_of(text, n_layers=48):
    if not isinstance(text, str) or not text:
        raise ValueError('empty layer list')
    parts = text.split(',')
    if any(p != '-1' and not p.isdigit() for p in parts):
        raise ValueError('invalid layer list')
    layers = [int(p) for p in parts]
    resolved = [n_layers - 1 if v == -1 else v for v in layers]
    if any(v < 0 or v >= n_layers for v in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError('layer out of range or duplicate final-layer alias')
    return layers

def build(item, tok, tpl, layers='15,29,-1', max_tokens=262144):
    layers_of(layers)
    sentinel = '\0JEV_V2_SENTINEL\0'
    rendered = tpl.render([{'role': 'user', 'content': sentinel}], add_generation_prompt=True, enable_thinking=False)
    if rendered.count(sentinel) != 1:
        raise ValueError('template must contain sentinel once')
    prefix, suffix = rendered.split(sentinel)
    ids, spans = [], []
    def add(text, span=False):
        encoded = tok.encode(text, parse_special=True)
        if span:
            if not encoded:
                raise ValueError('empty span')
            spans.append([len(ids), len(ids) + len(encoded)])
        ids.extend(encoded)
    add(prefix)
    add('### State:\n' + item['text'] + '\n\n### Questions:\n')
    add(item['question'] + '\n', True)
    for opt in item['options']:
        add('   - ' + opt + ': ' + item['crit'][opt] + '\n', True)
    before = len(ids)
    add(suffix)
    if len(ids) <= before:
        raise ValueError('missing assistant generation suffix')
    spans.append([len(ids) - 1, len(ids)])
    if len(ids) + 1 > max_tokens:
        raise ValueError('prompt exceeds context; truncation is forbidden')
    return {'latent_ids': ids, 'latent_spans': spans, 'latent_taps': layers, 'latent_pool': 'last,mean',
            'cache_region': 'decision'}

def validate_response(response, request, hc=4, n_embd=2560):
    layers = layers_of(request['latent_taps'])
    spans = request['latent_spans']
    if response.get('hc') != hc or response.get('n_embd') != n_embd:
        raise ValueError('backbone geometry mismatch')
    if response.get('short'):
        raise ValueError('server reported short spans')
    if response.get('prompt_tokens') != len(request['latent_ids']):
        raise ValueError('prompt token count mismatch')
    expected = {(s, l, p) for s in range(len(spans)) for l in layers for p in ('span_last', 'span_mean')}
    result = {}
    for segment in response.get('segments', []):
        key = (segment.get('span'), segment.get('layer'), segment.get('kind'))
        if key not in expected or key in result:
            raise ValueError('unexpected/duplicate segment')
        a, b = spans[key[0]]
        # The engine labels BOTH pooled vectors with the number of covered
        # positions; span_last still contains only one vector.
        pos, count = (b - 1, b - a) if key[2] == 'span_last' else (a, b - a)
        if segment.get('pos0') != pos or segment.get('rows') != count:
            raise ValueError('wrong position or incomplete pool coverage')
        blob = base64.b64decode(segment['data_b64'], validate=True)
        if len(blob) != hc * n_embd * 4:
            raise ValueError('invalid vector byte length')
        if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', blob)):
            raise ValueError('nonfinite latent vector')
        result[key] = blob
    if set(result) != expected:
        raise ValueError('missing per-span/layer/pool vectors')
    return result
