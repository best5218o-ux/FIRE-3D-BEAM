# -*- coding: utf-8 -*-
"""공개 데이터(CubiCasa5K, CC BY-NC-SA 4.0)로 벽·문 추출 정확도 측정
정답: model.svg 의 Wall / Door 다각형"""
import os, re, sys, json
import numpy as np, cv2
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
from fire3dbeam.vision import masks, grid_lines
from fire3dbeam import geom

NS = '{http://www.w3.org/2000/svg}'


def _pts(s):
    v = [float(x) for x in re.split(r'[ ,]+', s.strip()) if x]
    return list(zip(v[0::2], v[1::2]))


def parse_svg(path):
    root = ET.parse(path).getroot()
    W = float(root.get('width')); H = float(root.get('height'))
    walls, doors, spaces = [], [], []
    for g in root.iter(NS + 'g'):
        cls = g.get('class') or ''
        poly = g.find(NS + 'polygon')
        if poly is None or not poly.get('points'): continue
        pts = _pts(poly.get('points'))
        if len(pts) < 3: continue
        if cls.startswith('Wall'): walls.append(pts)
        elif cls.startswith('Door'): doors.append(pts)
        elif cls.startswith('Space'):
            lab = ' '.join(t.text or '' for t in g.iter(NS + 'text'))
            m = re.search(r'([\d.]+)\s*m\s*x\s*([\d.]+)\s*m', lab)
            if m and float(m.group(1)) > 0.5 and float(m.group(2)) > 0.5: spaces.append((pts, float(m.group(1)), float(m.group(2))))
    return W, H, walls, doors, spaces


def centerline(pts):
    rect = cv2.minAreaRect(np.array(pts, np.float32))
    (cx, cy), (w, h), ang = rect
    if w < h: w, h = h, w; ang += 90
    a = np.deg2rad(ang)
    dx, dy = np.cos(a) * w / 2, np.sin(a) * w / 2
    return LineString([(cx - dx, cy - dy), (cx + dx, cy + dy)]), h, (ang % 180)


def run_one(d, verbose=False):
    img = cv2.imread(os.path.join(d, 'F1_original.png'))
    W, H, walls, doors, spaces = parse_svg(os.path.join(d, 'model.svg'))
    k = img.shape[1] / W                             # svg → 이미지 px
    # 축척(px/m): 방 치수 표기(“7.09 m x 4.88 m”)와 다각형 크기로
    ss = []
    for pts, a, b in spaces:
        xs = [p[0] * k for p in pts]; ys = [p[1] * k for p in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        c1 = (w / a + h / b) / 2; c2 = (w / b + h / a) / 2
        ss.append(c1 if abs(w / a - h / b) < abs(w / b - h / a) else c2)
    if not ss: return None
    s = float(np.median(ss))
    M = masks(img)
    vl, hl = grid_lines(M, img.shape)
    wm = geom.wall_mask(M, s, vl, hl)
    pred = geom.clean(geom.merge(geom.segments(wm, s)))
    # 정답: 벽 다각형 중심선 (px → m, y 뒤집기)
    T = lambda x, y: (x * k / s, -y * k / s)
    gl, axis_len, tot_len = [], 0, 0
    wall_polys = []
    for pts in walls:
        ln, th, ang = centerline([(x, y) for x, y in pts])
        ln = LineString([T(*p) for p in ln.coords])
        wall_polys.append(Polygon([T(x, y) for x, y in pts]).buffer(0))
        tot_len += ln.length
        axis = min(ang % 90, 90 - ang % 90) < 3
        if axis: axis_len += ln.length
        gl.append((ln, axis))
    pl = [LineString(geom._ends(w)) for w in pred]
    P = unary_union(pl) if pl else None
    G = unary_union([l for l, a in gl]) if gl else None
    WP = unary_union(wall_polys)
    def covered(ln, other, tol):
        if other is None or ln.length == 0: return 0
        return ln.intersection(other.buffer(tol)).length
    tol = 0.3
    rec_all = sum(covered(l, P, tol) for l, a in gl) / max(1e-6, tot_len)
    rec_axis = sum(covered(l, P, tol) for l, a in gl if a) / max(1e-6, axis_len)
    prec = sum(covered(l, WP.buffer(0.15), 0.0) for l in pl) / max(1e-6, sum(l.length for l in pl)) if pl else 0
    # 문: 틈으로 찾은 문 vs 정답 문 중심
    fp = geom.footprint_from_walls(pred)
    pd = geom.gap_doors(pred, fp)
    gd = [Polygon([T(x, y) for x, y in pts]).centroid for pts in doors]
    hit = sum(1 for g in gd if any(Point(p['x'], p['y']).distance(g) < 0.8 for p in pd))
    dprec = sum(1 for p in pd if any(Point(p['x'], p['y']).distance(g) < 0.8 for g in gd))
    return dict(id=os.path.basename(d), s=round(s, 2), n_gt=len(walls), n_pred=len(pred), axis_share=round(axis_len / max(1e-6, tot_len), 2),
                wall_recall=round(rec_all, 3), wall_recall_axis=round(rec_axis, 3), wall_precision=round(prec, 3),
                door_recall=round(hit / max(1, len(gd)), 3), door_precision=round(dprec / max(1, len(pd)), 3), n_doors=len(gd), n_pred_doors=len(pd))


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'public/cubicasa'
    res = []
    for d in sorted(os.listdir(base)):
        r = run_one(os.path.join(base, d))
        if r: res.append(r)
    for r in res: print(r)
    keys = ['wall_recall', 'wall_recall_axis', 'wall_precision', 'door_recall', 'door_precision']
    print('MEAN', {k: round(float(np.mean([r[k] for r in res])), 3) for k in keys}, 'n', len(res))
    json.dump(res, open('out/cubicasa_eval.json', 'w'), indent=1)
