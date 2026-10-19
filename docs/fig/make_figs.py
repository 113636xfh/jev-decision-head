#!/usr/bin/env python3
"""Generate the eight README figures as SVG. Data comes from files already in the
repository (docs/fig/_run_metrics.json, docs/fig/_splits_replay.json,
baselines/middle5-s1.dev-predictions.jsonl) or from numbers verified in the result
documents. No experiment is re-run."""
import json, math, os

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
F = 'Arial, Helvetica, sans-serif'
INK, MUT, GRID, A1, A2, ACC = '#111111', '#5a6472', '#d8dee6', '#1f6feb', '#c2410c', '#0f766e'


class S:
    def __init__(s, w, h, title=None):
        s.w, s.h = w, h
        s.p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
               f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#ffffff"/>']
        if title:
            s.txt(28, 34, title, 21, 'start', '700')

    def txt(s, x, y, t, size=14, anchor='start', weight='400', fill=INK, spacing=None):
        sp = f' letter-spacing="{spacing}"' if spacing else ''
        s.p.append(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" '
                   f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{sp}>{t}</text>')

    def rect(s, x, y, w, h, fill='#ffffff', stroke=INK, sw=1.4, dash=None, rx=8):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(s, x1, y1, x2, y2, stroke=INK, sw=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                   f'stroke-width="{sw}"{d}/>')

    def arrow(s, x1, y1, x2, y2, stroke=INK, sw=1.8, dash=None, head=8):
        s.line(x1, y1, x2, y2, stroke, sw, dash)
        ang = math.atan2(y2 - y1, x2 - x1)
        for da in (math.radians(152), -math.radians(152)):
            s.p.append(f'<line x1="{x2}" y1="{y2}" x2="{x2+head*math.cos(ang+da):.1f}" '
                       f'y2="{y2+head*math.sin(ang+da):.1f}" stroke="{stroke}" stroke-width="{sw}"/>')

    def box(s, x, y, w, h, title, lines=(), fill='#f6f8fa', stroke=INK, dash=None, tcol=INK):
        s.rect(x, y, w, h, fill, stroke, 1.4, dash)
        cx, n = x + w / 2, len(lines)
        s.txt(cx, y + (24 if n else 30), title, 15, 'middle', '700', tcol)
        for i, ln in enumerate(lines):
            s.txt(cx, y + 44 + i * 19, ln, 12.5, 'middle', '400', MUT)

    def save(s, name):
        s.p.append('</svg>')
        open(os.path.join(OUT, name), 'w', encoding='utf-8').write('\n'.join(s.p))
        print('wrote', name)


def axes(s, x0, y0, w, h, xlab, ylab, xt, yt, xmax=1.0, ymax=1.0, grid=True):
    """x0,y0 = bottom-left of plot area."""
    for i, v in enumerate(yt):
        yy = y0 - h * (v / ymax)
        if grid:
            s.line(x0, yy, x0 + w, yy, GRID, 1)
        s.txt(x0 - 10, yy + 4, f'{v:g}', 12, 'end', '400', MUT)
    for i, v in enumerate(xt):
        xx = x0 + w * (v / xmax)
        s.txt(xx, y0 + 22, f'{v:g}', 12, 'middle', '400', MUT)
    s.line(x0, y0, x0 + w, y0, INK, 1.4)
    s.line(x0, y0, x0, y0 - h, INK, 1.4)
    s.txt(x0 + w / 2, y0 + 44, xlab, 13, 'middle', '600', INK)
    s.p.append(f'<text x="{x0-52}" y="{y0-h/2}" font-family="{F}" font-size="13" font-weight="600" '
               f'fill="{INK}" text-anchor="middle" transform="rotate(-90 {x0-52} {y0-h/2})">{ylab}</text>')


def poly(s, x0, y0, w, h, pts, col, xmax=1.0, ymax=1.0, sw=2.4, dash=None):
    d = ' '.join(('M' if i == 0 else 'L') + f'{x0+w*(p[0]/xmax):.1f},{y0-h*(p[1]/ymax):.1f}'
                 for i, p in enumerate(pts))
    dd = f' stroke-dasharray="{dash}"' if dash else ''
    s.p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"{dd}/>')


def legend(s, x, y, items, bg=None):
    if bg:
        s.rect(x - 8, y - 16, bg[0], bg[1], '#ffffff', '#ffffff', 0, rx=4)
    for i, (col, lab, dash) in enumerate(items):
        yy = y + i * 22
        s.line(x, yy, x + 26, yy, col, 2.6, dash)
        s.txt(x + 34, yy + 5, lab, 12.5, 'start', '400', INK)


# ---------------------------------------------------------------- figure 1
def fig_architecture():
    s = S(1240, 640, 'One resident backbone, two interfaces')
    s.box(40, 96, 210, 96, 'Caller request', ['state (text / JSON)', 'declared answer space L'])
    s.box(300, 76, 300, 136, 'Strata backbone (one process)',
          ['prompt read once', 'generation continues as usual', 'IQ3_S GGUF, 48 layers'])
    s.box(660, 60, 210, 168, 'Multi-layer readout',
          ['taps: 15 / 23 / 29 / 35', '/ 41 / last', 'spans: stem, each option,', 'decision position',
           'pool: last + span mean'])
    s.box(910, 76, 290, 136, 'Decision head (13.19 M)',
          ['shared projection', 'candidate-set encoder', 'one score per option'])
    s.box(910, 268, 290, 84, 'p(. | s, q) over L', ['distribution, not text', 'calibration is a first-class output'])
    s.box(560, 268, 280, 84, 'Generation interface', ['tokens, sampling, chat', 'same weights, same slot'])
    s.rect(40, 404, 1160, 190, '#ffffff', MUT, 1.4, '6 5')
    s.txt(60, 432, 'Caller-side policy (not part of the head)', 15, 'start', '700', MUT)
    s.box(70, 452, 250, 108, 'accept', ['threshold / margin /', 'normalised entropy on dev'])
    s.box(370, 452, 250, 108, 'fall back', ['ask the same backbone', 'to generate an answer'])
    s.box(670, 452, 250, 108, 'ask a human', ['or refuse, when the', 'operating point is infeasible'])
    s.box(970, 452, 200, 108, 'log', ['keep failures in the', 'denominator'])
    s.arrow(250, 144, 300, 144)
    s.arrow(600, 144, 660, 144)
    s.arrow(870, 144, 910, 144)
    s.arrow(1055, 212, 1055, 268)
    s.arrow(600, 212, 600, 268)
    s.arrow(1055, 352, 1055, 404)
    s.arrow(195, 192, 195, 404)
    s.arrow(195, 404, 195, 452)
    s.txt(40, 610, 'The dashed box is a policy we ship as a worked example; the head itself only returns a distribution.',
          13, 'start', '400', MUT)
    s.save('architecture.svg')


# ---------------------------------------------------------------- figure 2
def fig_readout_layout():
    s = S(1240, 620, 'What one feature vector is made of')
    s.box(40, 80, 300, 150, 'Prompt layout', ['question stem', 'option 1 … option K', 'final decision position'])
    s.box(40, 270, 300, 150, 'Spans (3 kinds)', ['stem', 'each option', 'decision position'])
    s.box(40, 460, 300, 110, 'Pooling (2 modes)', ['last position', 'span mean'])
    s.box(420, 80, 300, 300, 'Backbone taps (6)', ['layer 15', 'layer 23', 'layer 29', 'layer 35',
                                                   'layer 41', 'last layer', '', '4 residual streams × 2560'])
    s.box(800, 120, 380, 220, 'One item = one fixed-size vector',
          ['6 layers × 2 poolings × 4 streams', 'per span, concatenated', 'stored as FP16',
           'per-record sha256 in the manifest'])
    s.box(800, 380, 380, 110, 'Normalisation', ['z-score per (pool, layer, dim)', 'fitted on train split only'])
    s.arrow(340, 155, 420, 155)
    s.arrow(340, 345, 420, 300)
    s.arrow(340, 515, 420, 340)
    s.arrow(720, 230, 800, 230)
    s.arrow(990, 340, 990, 380)
    s.txt(40, 600, 'Changing the layer set, the span set or the pooling changes the meaning of the weights: '
                   'they are recorded in the archive manifest and checked by the loader.', 13, 'start', '400', MUT)
    s.save('readout-layout.svg')


# ---------------------------------------------------------------- figure 3
def fig_reliability():
    d = json.load(open(os.path.join(OUT, '_splits_replay.json')))
    s = S(1240, 470, 'Reliability: the head is close to calibrated, and a backbone swap makes it worse')
    x0, y0, w, h = 90, 380, 330, 300
    axes(s, x0, y0, w, h, 'predicted probability', 'observed accuracy', [0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0])
    s.line(x0, y0, x0 + w, y0 - h, MUT, 1.2, '5 4')
    for key, col, dash, lab in (('rel_old', A1, None, 'trained build'), ('rel_new', A2, '7 4', 'second build')):
        pts = []
        for cnt, cor, mean in d['dev'][key]:
            if cnt:
                pts.append((mean, cor / cnt))
        poly(s, x0, y0, w, h, pts, col, 1.0, 1.0, 2.6, dash)
    legend(s, x0 + 16, y0 - h + 34,
           [(A1, 'trained build (ECE 0.0266)', None),
            (A2, 'second build (ECE 0.0366)', '7 4')], bg=(232, 62))
    s.txt(x0, 70, 'dev split, 1,996 items, per item', 14, 'start', '700')
    # right: ECE bars
    bx0, by0, bw, bh = 620, 380, 560, 300
    axes(s, bx0, by0, bw, bh, 'split', 'expected calibration error', [], [0, 0.01, 0.02, 0.03, 0.04, 0.05], 4.0, 0.05)
    splits = ['dev', 'test', 'ood', 'cal']
    for i, sp in enumerate(splits):
        for j, (key, col) in enumerate((('ece_old', A1), ('ece_new', A2))):
            v = d[sp][key]
            xx = bx0 + (i + 0.16 + j * 0.36) * (bw / 4.0)
            hh = bh * (v / 0.05)
            s.rect(xx, by0 - hh, bw / 4.0 * 0.32, hh, col, col, 0, rx=2)
            s.txt(xx + bw / 4.0 * 0.16, by0 - hh - 8, f'{v:.4f}', 11.5, 'middle', '400', INK)
        s.txt(bx0 + (i + 0.5) * (bw / 4.0), by0 + 22, sp, 12.5, 'middle', '600', INK)
    legend(s, bx0 + 18, by0 - bh + 34, [(A1, 'trained build', None), (A2, 'second build', None)], bg=(190, 62))
    s.txt(bx0, 70, 'ECE by split (aggregate, same head and features)', 14, 'start', '700')
    s.txt(40, 452, 'Left: reliability diagram computed from per-item predictions of the same frozen head on both builds. '
                   'Right: ECE from the same replay. Both are held-out splits.', 13, 'start', '400', MUT)
    s.save('reliability.svg')


# ---------------------------------------------------------------- figure 4
def fig_risk_coverage():
    d = json.load(open(os.path.join(OUT, '_splits_replay.json')))
    s = S(1160, 470, 'Risk - coverage: what you give up when you only accept confident answers')
    x0, y0, w, h = 100, 380, 900, 300
    axes(s, x0, y0, w, h, 'coverage', 'accuracy on the accepted subset',
         [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1.0, 1.0)
    for key, col, dash, lab in (('rc_old', A1, None, 'trained build'), ('rc_new', A2, '7 4', 'second build')):
        poly(s, x0, y0, w, h, [(c, a) for c, a in d['dev'][key]], col, 1.0, 1.0, 2.6, dash)
    thr = json.load(open(os.path.join(OUT, '_dev_thresholds.json')))
    items = sorted(((float(k), v) for k, v in thr.items()))
    for t, (cov, acc) in items:
        xx = x0 + w * cov
        s.line(xx, y0, xx, y0 - h, '#9aa4b0', 1.1, '4 4')
        s.txt(xx, y0 - 12, f'{t:.2f}', 12, 'middle', '600', MUT)
    s.rect(x0 + 24, y0 - 176, 330, 96, '#ffffff', '#d8dee6', 1, rx=6)
    s.txt(x0 + 40, y0 - 152, 'operating points chosen on dev:', 12.5, 'start', '700', INK)
    for k, (t, (cov, acc)) in enumerate(items):
        s.txt(x0 + 40, y0 - 128 + k * 22, f'conf >= {t:.2f}   coverage {cov*100:4.1f}%   accuracy {acc*100:.1f}%',
              12, 'start', '400', MUT)
    legend(s, x0 + 20, y0 - h + 40, [(A1, 'trained build (dev)', None), (A2, 'second build (dev)', '7 4')], bg=(215, 62))
    s.txt(40, 452, 'Curves: dev split, 1,996 items, head answering directly (no fallback). Dashed lines: the operating '
                   'points chosen on dev. Thresholds are never tuned on test.', 13, 'start', '400', MUT)
    s.save('risk-coverage.svg')


# ---------------------------------------------------------------- figure 5
def fig_latency():
    s = S(1160, 470, 'Where the time goes: the head is milliseconds, the prompt is seconds')
    x0, y0, w, h = 120, 360, 560, 280
    items = [('head only', 4.8, A1), ('channel p50', 1820.0, A2), ('channel p95', 7100.0, ACC)]
    lo, hi = math.log10(1.0), math.log10(10000.0)
    for i, (lab, v, col) in enumerate(items):
        xx = x0 + i * (w / 3) + 30
        lw = w / 3 - 70
        hh = h * (math.log10(v) - lo) / (hi - lo)
        s.rect(xx, y0 - hh, lw, hh, col, col, 0, rx=3)
        s.txt(xx + lw / 2, y0 - hh - 10, f'{v:.1f} ms' if v < 100 else f'{v/1000:.2f} s', 13, 'middle', '700', INK)
        s.txt(xx + lw / 2, y0 + 22, lab, 12.5, 'middle', '600', INK)
    s.line(x0, y0, x0 + w, y0, INK, 1.4)
    s.txt(x0, 92, 'per decision, bar height on a log scale (1 ms - 10 s)', 14, 'start', '700')
    s.rect(740, 110, 380, 250, '#f6f8fa', MUT, 1.2)
    s.txt(760, 140, 'Channel latency by tier (p50)', 15, 'start', '700')
    for i, (lab, v) in enumerate([('original', 1.52), ('easy', 1.56), ('hard', 3.04)]):
        yy = 175 + i * 46
        s.txt(760, yy + 16, lab, 13, 'start', '400', INK)
        s.rect(880, yy, int(v / 3.2 * 165), 22, A2, A2, 0, rx=3)
        s.txt(880 + int(v / 3.2 * 165) + 8, yy + 16, f'{v:.2f} s', 12.5, 'start', '400', INK)
    s.txt(760, 330, '231 public items, 149,519 prompt tokens,', 12.5, 'start', '400', MUT)
    s.txt(760, 350, '645 s wall clock, one resident process', 12.5, 'start', '400', MUT)
    s.txt(40, 440, 'Essentially all of the channel cost is reading the prompt once. A readout request does not reuse the '
                   'prefix cache by design (~0.1 s more than a plain chat request at equal length).', 13, 'start', '400', MUT)
    s.save('latency-breakdown.svg')


# ---------------------------------------------------------------- figure 6
def fig_backbone_swap():
    s = S(1160, 470, 'The head belongs to one backbone build')
    x0, y0, w, h = 110, 370, 700, 290
    axes(s, x0, y0, w, h, 'split', 'accuracy', [], [0.70, 0.75, 0.80, 0.85], 5.0, 0.15)
    for v in (0.70, 0.75, 0.80, 0.85):
        s.txt(x0 - 10, y0 - h * ((v - 0.70) / 0.15) + 4, f'{v:.2f}', 12, 'end', '400', MUT)
    data = [('dev', .7455, .7380), ('test', .7644, .7494), ('ood', .8378, .8343),
            ('cal', .7742, .7604), ('all 7,718', .7806, .7708)]
    for i, (lab, a, b) in enumerate(data):
        for j, (v, col) in enumerate(((a, A1), (b, A2))):
            xx = x0 + (i + 0.16 + j * 0.36) * (w / 5.0)
            hh = h * ((v - 0.70) / 0.15)
            s.rect(xx, y0 - hh, w / 5.0 * 0.32, hh, col, col, 0, rx=2)
            s.txt(xx + w / 5.0 * 0.16, y0 - hh - 7, f'{v*100:.2f}', 11, 'middle', '400', MUT)
        s.txt(x0 + (i + 0.5) * (w / 5.0), y0 + 22, lab, 12.5, 'middle', '600', INK)
    legend(s, x0 + 18, y0 - h + 34, [(A1, 'trained build', None), (A2, 'second build of the same model', None)], bg=(265, 62))
    s.rect(850, 110, 270, 250, '#f6f8fa', MUT, 1.2)
    s.txt(870, 140, 'What changed', 15, 'start', '700')
    for i, ln in enumerate(['head: identical', 'features: identical', 'input tokens: identical',
                            'backbone weights: different', '', '8.94% of items changed', 'their answer',
                            'merged -0.98 pt, p ~ 0.0016']):
        s.txt(870, 172 + i * 23, ln, 12.5, 'start', '400', INK if ln else MUT)
    s.txt(40, 452, 'Same frozen head, same archived features, same tokens; only the backbone build differs. '
                   'Per-family effects do not extrapolate across splits.', 13, 'start', '400', MUT)
    s.save('backbone-swap.svg')


# ---------------------------------------------------------------- figure 7
def fig_depth():
    m = json.load(open(os.path.join(OUT, '_run_metrics.json')))
    s = S(1160, 470, 'A bigger head did not help: 2 / 4 / 8 blocks')
    x0, y0, w, h = 120, 360, 620, 280
    axes(s, x0, y0, w, h, 'head depth (blocks)', 'dev NLL  (lower is better)', [2, 4, 8], [], 8, 0.06)
    for v in (0.64, 0.66, 0.68, 0.70):
        yy = y0 - h * ((v - 0.64) / 0.06)
        s.line(x0, yy, x0 + w, yy, GRID, 1)
        s.txt(x0 - 10, yy + 4, f'{v:.2f}', 12, 'end', '400', MUT)
    pts = []
    for tag, xb in (('baseline-s0', 2), ('depth4-s0', 4), ('depth8-s0', 8)):
        v = m[tag]['dev_nll']
        pts.append((xb, v))
        xx = x0 + w * (xb / 8) - 45
        hh = h * ((v - 0.64) / 0.06)
        s.rect(xx, y0 - hh, 90, hh, A1, A1, 0, rx=3)
        s.txt(xx + 45, y0 - hh - 8, f'{v:.4f}', 12, 'middle', '700', INK)
    s.txt(x0 + 30, y0 - 150, 'width 1024 (2 blocks): dev NLL 0.6739, test 75.09%', 12, 'start', '400', MUT)
    s.rect(790, 110, 330, 250, '#f6f8fa', MUT, 1.2)
    s.txt(810, 140, 'test accuracy (1,999)', 15, 'start', '700')
    for i, (lab, tag) in enumerate([('2 blocks', 'baseline-s0'), ('4 blocks', 'depth4-s0'), ('8 blocks', 'depth8-s0')]):
        yy = 175 + i * 40
        s.txt(810, yy + 15, lab, 13, 'start', '400', INK)
        s.rect(930, yy, int(m[tag]['test_acc'] * 200), 20, A2, A2, 0, rx=3)
        s.txt(930 + int(m[tag]['test_acc'] * 200) + 8, yy + 15, f"{m[tag]['test_acc']*100:.2f}%", 12.5, 'start', '400', INK)
    s.txt(810, 320, 'Differences are inside seed noise;', 12.5, 'start', '400', MUT)
    s.txt(810, 340, '2 blocks has the best dev NLL.', 12.5, 'start', '400', MUT)
    s.txt(40, 440, 'Same split, seed, layer set and budget; one variable changed per run. Architecture was never selected '
                   'on test.', 13, 'start', '400', MUT)
    s.save('head-depth-ablation.svg')


# ---------------------------------------------------------------- figure 8
def fig_scaling():
    m = json.load(open(os.path.join(OUT, '_run_metrics.json')))
    s = S(1160, 470, 'More labelled items kept paying: 2k / 4k / 8k / 16k train')
    x0, y0, w, h = 120, 360, 700, 280
    axes(s, x0, y0, w, h, 'train items', 'accuracy', [2000, 4000, 8000, 16000], [], 16000, 0.10)
    for v in (0.65, 0.70, 0.75):
        yy = y0 - h * ((v - 0.65) / 0.10)
        s.line(x0, yy, x0 + w, yy, GRID, 1)
        s.txt(x0 - 10, yy + 4, f'{v:.2f}', 12, 'end', '400', MUT)
    for key, col, lab in (('dev_acc', A1, 'dev (1,996)'), ('test_acc', A2, 'test (1,999)')):
        pts = [(n, m[t][key]) for n, t in ((2000, 'data2k-s0'), (4000, 'data4k-s0'),
                                           (8000, 'data8k-s0'), (16000, 'baseline-s0'))]
        dpath = ' '.join(('M' if i == 0 else 'L') + f'{x0+w*(n/16000):.1f},{y0-h*((v-0.65)/0.10):.1f}'
                         for i, (n, v) in enumerate(pts))
        s.p.append(f'<path d="{dpath}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for n, v in pts:
            s.p.append(f'<circle cx="{x0+w*(n/16000):.1f}" cy="{y0-h*((v-0.65)/0.10):.1f}" r="4" fill="{col}"/>')
    legend(s, x0 + 20, y0 - h + 34, [(A1, 'dev (1,996)', None), (A2, 'test (1,999)', None)], bg=(150, 62))
    s.txt(40, 440, 'Same head, same protocol; only the number of labelled train items changes. dev is used for '
                   'checkpoint selection, test only for reporting.', 13, 'start', '400', MUT)
    s.save('data-scaling.svg')


if __name__ == '__main__':
    fig_architecture(); fig_readout_layout(); fig_reliability(); fig_risk_coverage()
    fig_latency(); fig_backbone_swap(); fig_depth(); fig_scaling()
