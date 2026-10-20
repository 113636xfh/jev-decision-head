#!/usr/bin/env python3
"""Generate the eight README SVGs; render with `node docs/fig/render.cjs`.

Only existing aggregates are plotted. No model, service or experiment is run.
Run metrics, reliability bins and coverage curves are read from the adjacent JSONs.
The latency and backbone-swap summaries below retain the previously published values.
"""
from pathlib import Path
from html import escape
import json
import math

OUT = Path(__file__).resolve().parent
FONT = 'Arial, Helvetica, sans-serif'
INK, MUT, BORDER, GRID = '#1a202c', '#5a6a80', '#dce3eb', '#e8edf3'
ORANGE, ORANGE_BG, ORANGE_INK = '#e8850c', '#fdeacc', '#9a4d00'
BLUE, BLUE_BG = '#6480a0', '#eef3fb'
GRAY_BG, GREEN_BG, GREEN = '#f7f9fb', '#eefaf3', '#1e7a4c'
METRICS = json.loads((OUT / '_run_metrics.json').read_text())
SPLITS = json.loads((OUT / '_splits_replay.json').read_text())
THRESHOLDS = json.loads((OUT / '_dev_thresholds.json').read_text())


class Figure:
    def __init__(self, number, title, subtitle, height=680):
        self.w, self.h = 1280, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" '
                      f'viewBox="0 0 1280 {height}" role="img">',
                      f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>']
        self.rect(0, 0, self.w, self.h, '#ffffff', 'none', radius=0)
        self.rect(48, 34, 40, 30, ORANGE_BG, 'none', radius=7)
        self.text(68, 55, f'{number:02}', 16, 'middle', '700', ORANGE_INK)
        self.text(104, 57, 'JEV DECISION HEAD', 15, weight='700', fill=MUT)
        self.text(48, 108, title, 28, weight='700')
        self.text(48, 140, subtitle, 17, fill=MUT)

    def text(self, x, y, value, size=17, anchor='start', weight='400', fill=INK):
        # Reject obvious canvas overflow before rendering. Nodes have a tighter width gate.
        extent = len(str(value)) * size * .55
        left = x - (extent if anchor == 'end' else extent / 2 if anchor == 'middle' else 0)
        assert left >= -2 and left + extent <= self.w + 2, (value, left, extent)
        assert 0 <= y - size and y + size * .3 < self.h, (value, y)
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
                          f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill=GRAY_BG, stroke=BORDER, radius=12, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"{extra}/>')

    def line(self, x1, y1, x2, y2, color=BORDER, width=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{color}" stroke-width="{width}"{extra}/>')

    def circle(self, x, y, r=5, fill=ORANGE, stroke='none', width=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{width}"/>')

    def arrow(self, x1, y1, x2, y2):
        # Explicit polygon heads, not rotated SVG markers (resvg 2.x).
        angle = math.atan2(y2-y1, x2-x1)
        ux, uy = math.cos(angle), math.sin(angle)
        bx, by = x2-12*ux, y2-12*uy
        self.line(x1, y1, bx, by, '#718096', 2.2)
        self.parts.append(f'<polygon points="{x2},{y2} {bx-5*uy},{by+5*ux} '
                          f'{bx+5*uy},{by-5*ux}" fill="#718096"/>')

    def node(self, x, y, w, title, support, accent=False, green=False, h=104):
        for value, size in [(title, 20), (support, 16)]:
            assert len(value)*size*.55 <= w-28, (title, value, w)
        self.rect(x, y, w, h, ORANGE_BG if accent else GREEN_BG if green else GRAY_BG,
                  ORANGE if accent else BORDER)
        self.text(x+w/2, y+h/2-5, title, 20, 'middle', '700', ORANGE_INK if accent else INK)
        self.text(x+w/2, y+h/2+23, support, 16, 'middle', fill=MUT)

    def footer(self, first, second=None):
        y = self.h - (74 if second else 58)
        self.line(48, y-23, 1232, y-23)
        self.text(48, y+4, first, 16, fill=MUT)
        if second:
            self.text(48, y+30, second, 16, fill=MUT)

    def save(self, name):
        (OUT / (name+'.svg')).write_text('\n'.join(self.parts+['</svg>'])+'\n', encoding='utf-8')
        print('SVG', name)


def legend(f, x, y, labels):
    for i, (label, color, dashed) in enumerate(labels):
        yy = y+i*29
        f.line(x, yy-5, x+28, yy-5, color, 3, '6 4' if dashed else None)
        f.text(x+38, yy, label, 16, fill=MUT)


def chart(f, x, y, w, h, xticks, yticks, xmin=0, xmax=1, ymin=0, ymax=1):
    """Plot origin is top-left; all series use this same explicit transform."""
    def xy(a, b):
        return x+w*(a-xmin)/(xmax-xmin), y+h-h*(b-ymin)/(ymax-ymin)
    for value, label in yticks:
        _, yy = xy(xmin, value)
        f.line(x, yy, x+w, yy, GRID, 1)
        f.text(x-14, yy+6, label, 16, 'end', fill=MUT)
    for value, label in xticks:
        xx, _ = xy(value, ymin)
        f.text(xx, y+h+28, label, 16, 'middle', fill=MUT)
    f.line(x, y+h, x+w, y+h, '#94a3b8')
    return xy


def series(f, xy, points, color, dashed=False, dots=False):
    points = list(points)
    path = ' '.join(('M' if i == 0 else 'L')+f'{xy(a,b)[0]:.2f},{xy(a,b)[1]:.2f}'
                    for i, (a,b) in enumerate(points))
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    f.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" '
                   f'stroke-linejoin="round"{dash}/>')
    if dots:
        for a,b in points:
            f.circle(*xy(a,b), r=5, fill=color)


def architecture():
    f = Figure(1, 'One backbone. Two interfaces.',
               'A frozen resident model supplies features to a small, separately trained decision head.', 660)
    xs = [48, 356, 664, 972]
    for x, title, support, accent in [
        (xs[0], 'Decision request', 'State + declared options', False),
        (xs[1], 'Resident backbone', 'One set of frozen weights', False),
        (xs[2], 'Multi-layer readout', 'Five taps; last + mean', False),
        (xs[3], '13.19 M head', 'A probability per option', True)]:
        f.node(x, 224, 260, title, support, accent)
    for a,b in zip(xs, xs[1:]):
        f.arrow(a+260, 276, b, 276)
    f.node(xs[1], 436, 260, 'Generation API', 'Normal text generation')
    f.arrow(xs[1]+130, 328, xs[1]+130, 436)
    f.node(xs[3], 436, 260, 'Caller-side policy', 'Accept / generate / review', green=True)
    f.arrow(xs[3]+130, 328, xs[3]+130, 436)
    f.text(48, 197, 'DECISION PATH', 15, weight='700', fill=MUT)
    f.footer('Weights are shared; a decision still requires its own readout forward pass.',
             'The head returns a distribution. Fallback is an optional caller policy, not part of the head.')
    f.save('architecture')


def readout_layout():
    f = Figure(5, 'From prompt spans to head inputs',
               'The protocol is fixed; the number of spans depends on the number of candidates K.', 660)
    f.node(48, 228, 350, 'Prompt spans', 'Stem + K options + decision')
    f.node(465, 228, 350, 'FP16 archive', '(K+2) x 2 pools x 6 x 10240')
    f.node(882, 228, 350, 'Selected head inputs', '(K+2) x 2 pools x 5 x 10240', accent=True)
    f.arrow(398, 280, 465, 280)
    f.arrow(815, 280, 882, 280)
    f.node(882, 426, 350, 'Train-fit z-score', 'Same frozen mean and scale', green=True)
    f.arrow(1057, 332, 1057, 426)
    f.rect(48, 405, 767, 132, BLUE_BG, 'none')
    f.text(72, 441, 'HEAD LAYERS', 15, weight='700', fill=MUT)
    f.text(72, 475, '23 / 29 / 35 / 41 / last', 22, weight='700')
    f.text(72, 509, 'Archive also retains layer 15. Each tap has 4 streams x 2560.', 17, fill=MUT)
    f.footer('The final decision position follows the complete assistant suffix, not the last option.',
             'The feature tensor is variable in K; changing spans, pooling or layers changes its meaning.')
    f.save('readout-layout')


def reliability():
    f = Figure(2, 'Calibration under a backbone change',
               'Same frozen head and feature protocol; archived features versus a second-build live readout.', 740)
    f.text(100, 200, 'DEV RELIABILITY', 16, weight='700', fill=MUT)
    xy = chart(f, 100, 248, 440, 330,
               [(i/4, f'{i*25}%') for i in range(5)], [(i/4, f'{i*25}%') for i in range(5)])
    series(f, xy, [(0,0),(1,1)], '#b7c3cf', True)
    for key, color, dashed in [('rel_old', BLUE, False), ('rel_new', ORANGE, True)]:
        series(f, xy, [(mean, correct/n) for n,correct,mean in SPLITS['dev'][key] if n], color, dashed, True)
    f.text(320, 639, 'Mean confidence (10 equal-width bins)', 16, 'middle', fill=MUT)
    f.text(100, 229, 'Observed accuracy', 16, fill=MUT)
    f.rect(620, 182, 612, 464, GRAY_BG, 'none')
    f.text(650, 221, 'EXPECTED CALIBRATION ERROR', 16, weight='700', fill=MUT)
    f.text(650, 253, 'Lower is better', 16, fill=MUT)
    f.text(925, 293, 'Archived', 16, 'end', weight='700', fill=BLUE)
    f.text(1180, 293, 'Second build', 16, 'end', weight='700', fill=ORANGE_INK)
    for i, sp in enumerate(['dev','test','ood','cal']):
        yy = 343+i*69
        f.text(650, yy, sp.upper(), 17, weight='700')
        f.text(925, yy, f"{SPLITS[sp]['ece_old']:.4f}", 22, 'end', '700', BLUE)
        f.text(1180, yy, f"{SPLITS[sp]['ece_new']:.4f}", 22, 'end', '700', ORANGE_INK)
        if i<3:
            f.line(650, yy+24, 1200, yy+24)
    legend(f, 107, 286, [('Archived features', BLUE, False), ('Second-build readout', ORANGE, True)])
    f.footer('Dev: 1,996 items. Low-confidence bins have few items; these are descriptive calibration estimates.')
    f.save('reliability')


def risk_coverage():
    f = Figure(6, 'Confidence trades coverage for accuracy',
               'Dev-only operating points; the head answers directly, without generation fallback.', 740)
    xy = chart(f, 104, 223, 700, 336,
               [(i/5, f'{i*20}%') for i in range(6)], [(i/10, f'{i*10}%') for i in range(7,11)],
               ymin=.7, ymax=1.01)
    for key, color, dashed in [('rc_old', BLUE, False), ('rc_new', ORANGE, True)]:
        series(f, xy, [(c,a) for c,a in SPLITS['dev'][key] if c>0], color, dashed)
    f.text(104, 196, 'Accuracy on the accepted subset', 16, fill=MUT)
    f.text(454, 619, 'Coverage: share of items answered by the head', 16, 'middle', fill=MUT)
    legend(f, 126, 455, [('Archived features', BLUE, False), ('Second-build readout', ORANGE, True)])
    f.rect(850, 190, 382, 402, GRAY_BG, 'none')
    f.text(878, 228, 'EXAMPLE THRESHOLDS', 16, weight='700', fill=MUT)
    for i, (threshold, values) in enumerate(sorted(THRESHOLDS.items())):
        coverage, accuracy = values
        y = 275+i*98
        f.text(878, y, f'Confidence >= {float(threshold):.2f}', 18, weight='700')
        f.text(878, y+31, f'{coverage*100:.1f}% coverage', 17, fill=MUT)
        f.text(878, y+57, f'{accuracy*100:.1f}% accepted accuracy', 17, fill=MUT)
    f.footer('All curves and thresholds shown here use dev (1,996 items); they are not a held-out deployment guarantee.')
    f.save('risk-coverage')


def latency():
    f = Figure(4, 'Milliseconds for the head. Seconds for the channel.',
               'Historical JevBench public-231 run: feature readout, transport, normalisation and head.', 720)
    f.text(262, 214, 'Latency per decision (log scale)', 17, weight='700', fill=MUT)
    x0, w = 262, 596
    def xp(v):
        return x0+w*math.log10(v)/4
    for v, label in [(1,'1 ms'),(10,'10 ms'),(100,'100 ms'),(1000,'1 s'),(10000,'10 s')]:
        x = xp(v)
        f.line(x, 242, x, 482, GRID)
        f.text(x, 516, label, 16, 'middle', fill=MUT)
    for i, (label, value, display) in enumerate([('Head p50',4.8,'4.8 ms'),('Channel p50',1820,'1.82 s'),('Channel p95',7100,'7.10 s')]):
        y = 264+i*80
        f.text(230, y+26, label, 18, 'end', weight='700')
        f.rect(x0, y, xp(value)-x0, 38, ORANGE if i==0 else BLUE, 'none', radius=5)
        f.text(xp(value)+14, y+26, display, 18, weight='700', fill=ORANGE_INK if i==0 else INK)
    f.rect(950, 186, 282, 354, GRAY_BG, 'none')
    f.text(976, 225, 'CHANNEL p50', 16, weight='700', fill=MUT)
    for i,(tier,value) in enumerate([('Original','1.52 s'),('Easy','1.56 s'),('Hard','3.04 s')]):
        yy = 280+i*80
        f.text(976, yy, tier, 17, fill=MUT)
        f.text(1206, yy+32, value, 26, 'end', '700')
    f.rect(48, 558, 1184, 54, BLUE_BG, 'none', radius=8)
    f.text(72, 592, 'E5-2673 v4 | V100-SXM2 16 GB: 3 serving GPUs + GPU3 for the head | No NVLink', 17, fill=MUT)
    f.footer('231 items; 149,519 prompt tokens. Timings are specific to this hardware and serving implementation.',
             'Multi-layer readout does not reuse the prefix cache in this implementation.')
    f.save('latency-breakdown')


def backbone_swap():
    f = Figure(3, 'Frozen head, different backbone build',
               'Input tokens, spans and feature protocol are fixed. The extracted feature values are not.', 750)
    legend(f, 88, 209, [('Archived-feature baseline', BLUE, False), ('Second-build live readout', ORANGE, False)])
    x, w = 238, 584
    def xp(v):
        return x+w*(v-.70)/.15
    for v in [.70,.75,.80,.85]:
        f.line(xp(v), 282, xp(v), 578, GRID)
        f.text(xp(v), 615, f'{v*100:.0f}%', 16, 'middle', fill=MUT)
    rows = [('dev',.7455,.7380),('test',.7644,.7494),('OOD',.8378,.8343),
            ('cal',.7742,.7604),('all 7,718',.7806,.7708)]
    for i,(name,old,new) in enumerate(rows):
        yy = 302+i*62
        f.text(88, yy+6, name, 18, weight='700')
        f.line(xp(new), yy, xp(old), yy, '#c9d3df', 5)
        f.circle(xp(old), yy, 7, BLUE)
        f.circle(xp(new), yy, 7, ORANGE)
        f.text(xp(old)+14, yy-10, f'{old*100:.2f}', 16, fill=BLUE)
        f.text(xp(new)-14, yy+23, f'{new*100:.2f}', 16, 'end', fill=ORANGE_INK)
    f.rect(918, 188, 314, 436, ORANGE_BG, 'none')
    f.text(946, 230, 'ACROSS 7,718 ITEMS', 15, weight='700', fill=ORANGE_INK)
    f.text(946, 298, '-0.98 pp', 36, weight='700', fill=ORANGE_INK)
    f.text(946, 330, 'aggregate accuracy change', 17, fill=MUT)
    f.line(946, 357, 1204, 357, '#eacda6')
    f.text(946, 412, '8.94%', 36, weight='700', fill=ORANGE_INK)
    f.text(946, 444, 'items changed answer', 17, fill=MUT)
    f.text(946, 507, '+253 / -329 correct', 17, weight='700')
    f.text(946, 540, 'Nominal item-level', 16, fill=MUT)
    f.text(946, 567, 'p ~ 0.0016', 18, weight='700')
    f.footer('This is a comparison of two recorded systems, not a weights-only causal ablation.',
             'Engine, server and config identities also differ; archived and live feature tensors are compared separately.')
    f.save('backbone-swap')


def depth():
    f = Figure(7, 'Deeper heads did not improve dev NLL in this run',
               'Single-seed depth ablation (seed 0): same input layers, width and training-budget policy.', 700)
    x0,y0,w,h = 128,238,620,300
    xy = chart(f,x0,y0,w,h,[],[(v,f'{v:.2f}') for v in [.64,.66,.68,.70]], ymin=.64,ymax=.70)
    f.text(x0, 204, 'Dev NLL (lower is better)', 17, weight='700', fill=MUT)
    for i,(tag,blocks) in enumerate([('baseline-s0',2),('depth4-s0',4),('depth8-s0',8)]):
        value = METRICS[tag]['dev_nll']
        xx = x0+65+i*204
        yy = xy(0,value)[1]
        f.rect(xx,yy,100,y0+h-yy,ORANGE if i==0 else BLUE,'none',radius=6)
        f.text(xx+50,yy-15,f'{value:.4f}',21,'middle','700',ORANGE_INK if i==0 else INK)
        f.text(xx+50,y0+h+32,f'{blocks} blocks',18,'middle',weight='700')
    f.rect(844,190,388,394,GRAY_BG,'none')
    f.text(872,230,'TEST ACCURACY',16,weight='700',fill=MUT)
    for i,(tag,label) in enumerate([('baseline-s0','2 blocks'),('depth4-s0','4 blocks'),('depth8-s0','8 blocks')]):
        yy = 294+i*67
        f.text(872,yy,label,18,weight='700')
        f.text(1202,yy,f"{METRICS[tag]['test_acc']*100:.2f}%",22,'end','700')
    f.line(872,460,1202,460)
    f.text(872,502,'2 blocks: best dev NLL',18,weight='700',fill=ORANGE_INK)
    f.text(872,539,'No multi-seed depth estimate',16,fill=MUT)
    f.footer('Architecture is selected by dev NLL, never by the highest test accuracy.',
             'These three runs do not establish a general depth effect or quantify seed-to-seed noise.')
    f.save('head-depth-ablation')


def scaling():
    f = Figure(8, 'More labelled data helped this baseline',
               'Three input layers, 12.67 M parameters, seed 0; this is not the released five-layer head.', 740)
    xs = [1994,3998,7994,16000]
    xy = chart(f,116,232,754,326,[(v,label) for v,label in zip(xs,['1,994','3,998','7,994','16,000'])],
               [(v,f'{v*100:.0f}%') for v in [.65,.70,.75,.80]],xmax=17000,ymin=.65,ymax=.80)
    f.text(116,200,'Accuracy',17,weight='700',fill=MUT)
    tags = ['data2k-s0','data4k-s0','data8k-s0','baseline-s0']
    for key,color in [('dev_acc',BLUE),('test_acc',ORANGE)]:
        series(f,xy,[(n,METRICS[tag][key]) for n,tag in zip(xs,tags)],color,dots=True)
    f.text(493,618,'Labelled training items',17,'middle',fill=MUT)
    f.rect(946,190,286,396,GRAY_BG,'none')
    legend(f,972,236,[('Dev',BLUE,False),('Test',ORANGE,False)])
    f.text(972,337,'AT 16,000 ITEMS',15,weight='700',fill=MUT)
    f.text(972,385,'71.94%',32,weight='700',fill=BLUE)
    f.text(972,415,'Dev accuracy',17,fill=MUT)
    f.text(972,479,'75.99%',32,weight='700',fill=ORANGE_INK)
    f.text(972,509,'Test accuracy',17,fill=MUT)
    f.footer('Dev selects checkpoints; test is used only for reporting. Normalisation is re-fit per training subset.',
             'Epoch caps are matched, not optimiser-update counts; early stopping can differ between runs.')
    f.save('data-scaling')


if __name__ == '__main__':
    architecture()
    reliability()
    backbone_swap()
    latency()
    readout_layout()
    risk_coverage()
    depth()
    scaling()
