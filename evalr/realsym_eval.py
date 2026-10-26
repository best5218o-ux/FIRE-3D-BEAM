# -*- coding: utf-8 -*-
"""실제 도면 3종 검출기 채점 — 손으로 만든 근사 정답(evalr/realsym_gt.json) 기준.
python -m evalr.realsym_eval [dev|held|all] [--only c01,c02] [--det rooms,doors,stairs,elev] [--overlay DIR] [--model]
원천 이미지: AIHUB_OBJ2 환경변수(기본 ./aihub/obj2) — 공개 불가 자료, 로컬 전용."""
import os, sys, json, time, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebim import realsym as RS

SRC = os.environ.get('AIHUB_OBJ2', './aihub/obj2')
GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realsym_gt.json')


def match_points(det, gt, rad):
    """탐욕 최근접 1:1 매칭. det,gt=[(x,y)], 반환 (매칭쌍 list, 미매칭 det idx, 미매칭 gt idx)"""
    pairs = sorted([(np.hypot(d[0] - g[0], d[1] - g[1]), i, j) for i, d in enumerate(det) for j, g in enumerate(gt)])
    ud, ug, out = set(), set(), []
    for dist, i, j in pairs:
        if dist > rad: break
        if i in ud or j in ug: continue
        ud.add(i); ug.add(j); out.append((i, j))
    return out, [i for i in range(len(det)) if i not in ud], [j for j in range(len(gt)) if j not in ug]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def box_hit(d, g):
    cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return iou(d, g) > 0.2 or (g[0] <= cx <= g[2] and g[1] <= cy <= g[3])


def match_boxes(det, gt, ign=()):
    ud, ug, tp = set(), set(), 0
    for j, g in enumerate(gt):
        for i, d in enumerate(det):
            if i not in ud and box_hit(d, g): ud.add(i); ug.add(j); tp += 1; break
    fp = [i for i in range(len(det)) if i not in ud and not any(box_hit(det[i], z) for z in ign)]
    return tp, len(fp), len(gt) - tp, fp


def run_crop(c, dets, model=False):
    g = cv2.imread(os.path.join(SRC, c['file']), 0)
    box = c['box']
    r = dict(id=c['id'], split=c['split'])
    t0 = time.time()
    out = RS.analyze(g, box, dets=dets, model=model)
    r['sec'] = round(time.time() - t0, 1)
    rad_room = max(60, 1.3 * c['door_px'])
    if 'rooms' in dets:
        gtr = [x for x in c['rooms'] if x['cat'] != '_ign']; ign = [x for x in c['rooms'] if x['cat'] == '_ign']
        D = out['rooms']
        pairs, ud, ug = match_points([d['xy'] for d in D], [x['xy'] for x in gtr], rad_room)
        _, ud2, _ = match_points([D[i]['xy'] for i in ud], [x['xy'] for x in ign], rad_room)
        fp = [ud[k] for k in ud2]
        typed = sum(1 for i, j in pairs if D[i]['cat'] == gtr[j]['cat'])
        bed = [j for j, x in enumerate(gtr) if x['cat'] == '침실']
        r['rooms'] = dict(tp=len(pairs), fp=len(fp), fn=len(ug), typed=typed, n_gt=len(gtr),
                          bed_tp=sum(1 for i, j in pairs if j in bed and D[i]['cat'] == '침실'), bed_gt=len(bed),
                          miss=[gtr[j]['name'] for j in ug], false=[D[i]['text'] for i in fp],
                          wrong=[(D[i]['text'], gtr[j]['name']) for i, j in pairs if D[i]['cat'] != gtr[j]['cat']])
    if 'doors' in dets:
        D = out['doors']
        pairs, ud, ug = match_points([d['xy'] for d in D], c['doors'], 0.9 * c['door_px'])
        wr = [D[i]['r'] / c['door_px'] for i, j in pairs]
        r['doors'] = dict(tp=len(pairs), fp=len(ud), fn=len(ug), n_gt=len(c['doors']), w_ratio=round(float(np.median(wr)), 2) if wr else None)
    if 'stairs' in dets:
        tp, fp, fn, _ = match_boxes([d['box'] for d in out['stairs']], c['stairs'], c.get('stairs_ign', []))
        r['stairs'] = dict(tp=tp, fp=fp, fn=fn, n_gt=len(c['stairs']))
    if 'elev' in dets:
        tp, fp, fn, _ = match_boxes([d['box'] for d in out['elevators']], c['elev'])
        r['elev'] = dict(tp=tp, fp=fp, fn=fn, n_gt=len(c['elev']))
    return r, out, g


def overlay(g, c, out, fn):
    x0, y0, x1, y1 = c['box']
    v = cv2.cvtColor(g[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    o = np.array([x0, y0])
    P = lambda p: tuple(int(t) for t in (np.array(p[:2]) - o))
    for rm in c['rooms']: cv2.circle(v, P(rm['xy']), 10, (0, 200, 0), 2)                      # 정답 실(초록 원)
    for d in c['doors']: cv2.drawMarker(v, P(d), (0, 200, 0), cv2.MARKER_TILTED_CROSS, 24, 2)  # 정답 문(초록 X)
    for b in c['stairs'] + c['elev']: cv2.rectangle(v, P(b[:2]), P(b[2:]), (0, 200, 0), 2)
    for rm in out.get('rooms', []):
        cv2.rectangle(v, P(rm['box'][:2]), P(rm['box'][2:]), (255, 0, 0), 2)
        cv2.putText(v, rm['cat'][:2].encode('ascii', 'ignore').decode() or str(rm['priority']), P(rm['box'][2:]), 0, 0.8, (255, 0, 0), 2)
    for d in out.get('doors', []):
        h = P(d['hinge']); r = int(d['r'])
        a0 = d['a0']
        cv2.ellipse(v, h, (r, r), 0, a0, a0 + 90, (0, 0, 255), 3)
        cv2.circle(v, h, 5, (0, 0, 255), -1)
    for s in out.get('stairs', []): cv2.rectangle(v, P(s['box'][:2]), P(s['box'][2:]), (255, 0, 255), 3)
    for s in out.get('elevators', []): cv2.rectangle(v, P(s['box'][:2]), P(s['box'][2:]), (0, 140, 255), 3)
    sc = min(1.0, 1600 / max(v.shape[:2]))
    cv2.imwrite(fn, cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))


def prf(rows, key):
    R = [r[key] for r in rows if key in r]
    tp, fp, fn = sum(x['tp'] for x in R), sum(x['fp'] for x in R), sum(x['fn'] for x in R)
    P = tp / (tp + fp) if tp + fp else None; Rc = tp / (tp + fn) if tp + fn else None
    return dict(tp=tp, fp=fp, fn=fn, P=round(P, 3) if P is not None else None, R=round(Rc, 3) if Rc is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split', nargs='?', default='dev')
    ap.add_argument('--only', default='')
    ap.add_argument('--det', default='rooms,doors,stairs,elev')
    ap.add_argument('--overlay', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--model', action='store_true', help='realsym.analyze(model=True) — 규칙 + SPA 분할망(models/spa_seg.onnx) 합침')
    a = ap.parse_args()
    dets = a.det.split(',')
    G = json.load(open(GT))
    crops = [c for c in G['crops'] if a.split == 'all' or c['split'] == a.split]
    if a.only: crops = [c for c in crops if c['id'] in a.only.split(',')]
    rows = []
    for c in crops:
        r, out, g = run_crop(c, dets, a.model)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
        if a.overlay:
            os.makedirs(a.overlay, exist_ok=True); overlay(g, c, out, os.path.join(a.overlay, f"{c['id']}.png"))
        del g
    summ = {}
    for s in sorted(set(r['split'] for r in rows)) + (['all'] if len(set(r['split'] for r in rows)) > 1 else []):
        R = [r for r in rows if s == 'all' or r['split'] == s]
        d = dict(n=len(R))
        for k in ('rooms', 'doors', 'stairs', 'elev'):
            if k in dets: d[k] = prf(R, k)
        if 'rooms' in dets:
            typed = sum(r['rooms']['typed'] for r in R); tp = d['rooms']['tp']
            d['rooms']['type_acc'] = round(typed / tp, 3) if tp else None
            d['rooms']['typed_R'] = round(typed / max(1, sum(r['rooms']['n_gt'] for r in R)), 3)
            d['rooms']['bedroom_R'] = f"{sum(r['rooms']['bed_tp'] for r in R)}/{sum(r['rooms']['bed_gt'] for r in R)}"
        summ[s] = d
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if a.out: json.dump(dict(rows=rows, summary=summ), open(a.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
