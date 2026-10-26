# -*- coding: utf-8 -*-
"""규칙↔모델 교차검증(검수 대상) 채점 — 가상 정답(gt.json) 기준.
사용: python -m evalr.xcheck_eval [DATA=data] [OUT=out_model]   (먼저 FIREBIM_MODEL=1 OUT=out_model python run_all.py 6)
- 규칙만 본 벽 조각(rule_only): 정답 벽(±0.4 m)과 겹치는 길이 < 50% 이면 '실제 오류'(규칙 오검출 벽)
- 모델만 본 벽 조각(model_only): 정답 벽과 겹치는 길이 ≥ 50% 이면 '실제 오류'(규칙이 놓친 벽)
- 모델 보충 문(src=model): 정답 문 0.9 m 이내면 정답(외벽 개구부는 폭/2 허용)
- 포착률: 규칙 오검출 벽 길이 중 rule_only 항목이 덮는 비율, 규칙 누락 벽 길이 중 model_only 항목이 덮는 비율
  (전체 벽 길이 중 항목으로 올라간 비율과 함께 → 사람이 볼 곳이 얼마나 줄었나)"""
import json, os, sys, glob, math
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
from evalr.score import eng_map, gt_map

TOL = 0.4
try:
    from firebim.xcheck import CFG as _C; ITEM_P = _C['item_p']
except Exception:
    ITEM_P = 0.5


def offset(gt, res):
    T = gt['transforms']; E = res['floors']
    fids = [f for f in gt['floors'] if f in E and f in T]
    diffs = {}
    for f in fids:
        t = T[f]; pts = [(0, 0), (t['w'], 0), (0, t['h']), (t['w'], t['h']), (t['w'] / 2, t['h'] / 2)]
        diffs[f] = [np.subtract(eng_map(E[f], *p), gt_map(t, *p)) for p in pts]
    g = np.median([d[-1] for d in diffs.values()], axis=0) if diffs else np.zeros(2)
    resid = {f: float(max(np.hypot(*(d - g)) for d in ds)) for f, ds in diffs.items()}
    return float(g[0]), float(g[1]), resid


def seg(w, gx, gy):
    return (LineString([(w['a'] - gx, w['c'] - gy), (w['b'] - gx, w['c'] - gy)]) if w['o'] == 'h'
            else LineString([(w['c'] - gx, w['a'] - gy), (w['c'] - gx, w['b'] - gy)]))


def evaluate(gt, res):
    gx, gy, resid = offset(gt, res)
    X = res.get('xcheck') or {}
    items = []           # (kind, p_score, real_error, length)
    cap = dict(fp_len=0.0, fp_cov=0.0, miss_len=0.0, miss_cov=0.0, wall_len=0.0, ro_len=0.0, mo_len=0.0)
    doors = []
    for pi, rep in X.items():
        if 'error' in rep: continue
        fid = rep['floor']
        if fid not in gt['floors'] or resid.get(fid, 99) > 3: continue
        G = unary_union([LineString([(w['x0'], w['y0']), (w['x1'], w['y1'])]) for w in gt['floors'][fid]['walls']])
        GB = G.buffer(TOL)
        F = res['floors'][fid]
        P = [seg(w, gx, gy) for w in F['walls'] if w.get('src') != 'model']        # 규칙 벽(모델 보충 전)
        PU = unary_union(P).buffer(TOL) if P else None
        RO = [seg(x, gx, gy) for x in rep['rule_only']]; MO = [seg(x, gx, gy) for x in rep['model_only']]
        ROg = [l for x, l in zip(rep['rule_only'], RO) if x.get('p', 1) >= ITEM_P]; MOg = [l for x, l in zip(rep['model_only'], MO) if x.get('p', 1) >= ITEM_P]
        for x, l in zip(rep['rule_only'], RO):
            ok = l.intersection(GB).length / max(1e-6, l.length)
            items.append(dict(x, kind='rule_only', err=ok < 0.5, L=l.length))
        for x, l in zip(rep['model_only'], MO):
            ok = l.intersection(GB).length / max(1e-6, l.length)
            items.append(dict(x, kind='model_only', err=ok >= 0.5, L=l.length))
        # 포착률
        ROU = unary_union(ROg).buffer(0.05) if ROg else None           # 포착률은 검수 화면에 실제로 올라가는 항목(p ≥ ITEM_P)만
        MOU = unary_union(MOg).buffer(TOL) if MOg else None
        for l in P:
            cap['wall_len'] += l.length
            fp = l.difference(GB)
            cap['fp_len'] += fp.length
            if ROU is not None: cap['fp_cov'] += fp.intersection(ROU).length
        for l in (G.geoms if hasattr(G, 'geoms') else [G]):
            miss = l.difference(PU) if PU is not None else l
            cap['miss_len'] += miss.length
            if MOU is not None: cap['miss_cov'] += miss.intersection(MOU).length
        cap['ro_len'] += sum(l.length for l in ROg); cap['mo_len'] += sum(l.length for l in MOg)
        gd = gt['floors'][fid]['doors']
        ed = [(d['x'] - gx, d['y'] - gy) for d in F['doors'] if d.get('src') != 'model']
        for d in rep.get('doors_model', []):
            x, y = d['x'] - gx, d['y'] - gy
            hit = [g for g in gd if math.dist((g['x'], g['y']), (x, y)) < (max(0.9, g['w'] / 2) if d.get('ext') else 0.9)]
            # 새로 찾은 정답 문인가(규칙 문이 이미 맞힌 정답이면 중복)
            new = [g for g in hit if not any(math.dist((g['x'], g['y']), e) < 0.9 for e in ed)]
            added = any(e.get('src') == 'model' and abs(e['x'] - d['x']) < 1e-6 and abs(e['y'] - d['y']) < 1e-6 for e in F['doors'])
            doors.append(dict(d, ok=bool(hit), new=bool(new), added=added, ent=any(g['kind'] == 'entrance' for g in new)))
    return items, cap, doors


def summarize(allitems, caps, doors):
    out = {}
    for k in ('rule_only', 'model_only'):
        L = [i for i in allitems if i['kind'] == k]
        out[k] = dict(n=len(L), real=sum(i['err'] for i in L), prec=round(sum(i['err'] for i in L) / max(1, len(L)), 3),
                      len=round(sum(i['L'] for i in L), 1), len_real=round(sum(i['L'] for i in L if i['err']), 1))
    n = len(allitems); r = sum(i['err'] for i in allitems)
    out['all'] = dict(n=n, real=r, prec=round(r / max(1, n), 3))
    c = {k: sum(x[k] for x in caps) for k in caps[0]} if caps else {}
    if c:
        out['capture'] = dict(rule_fp_len=round(c['fp_len'], 1), rule_fp_covered=round(c['fp_cov'] / max(1e-6, c['fp_len']), 3),
                              rule_miss_len=round(c['miss_len'], 1), rule_miss_covered=round(c['miss_cov'] / max(1e-6, c['miss_len']), 3),
                              flagged_share_of_walls=round((c['ro_len'] + c['mo_len']) / max(1e-6, c['wall_len']), 3))
    A = [d for d in doors if d['added']]
    out['added_doors'] = dict(n=len(A), ok=sum(d['ok'] for d in A), new_gt=sum(d['new'] for d in A), ext=sum(d['ext'] for d in A),
                              ext_ok=sum(d['ok'] for d in A if d['ext']), entrance_new=sum(d['ent'] for d in A))
    for thr in (0.3, 0.5):
        L = [i for i in allitems if i.get('p', 1) >= thr]
        out[f'items_p>={thr}'] = dict(n=len(L), real=sum(i['err'] for i in L), prec=round(sum(i['err'] for i in L) / max(1, len(L)), 3),
                                      rule_only=sum(i['kind'] == 'rule_only' for i in L), model_only=sum(i['kind'] == 'model_only' for i in L))
    # 의심도 정렬 효과: 의심도 순 상위 절반의 실제 오류 비율
    return out


def main(data='data', outd='out_model', verbose=False):
    allitems, caps, doors = [], [], []
    for gp in sorted(glob.glob(os.path.join(data, '*', 'gt.json'))):
        name = os.path.basename(os.path.dirname(gp)); rp = os.path.join(outd, name + '.json')
        if not os.path.exists(rp): continue
        it, cap, dr = evaluate(json.load(open(gp)), json.load(open(rp)))
        for i in it: i['bld'] = name
        allitems += it; caps.append(cap); doors += dr
    s = summarize(allitems, caps, doors)
    if verbose: s['_items'] = allitems; s['_doors'] = doors
    return s


def logit_fit(X, y, l2=1e-2, it=50):
    """로지스틱 회귀(뉴턴법, 작은 L2) → [절편, 계수...]"""
    X = np.c_[np.ones(len(X)), np.asarray(X, float)]; y = np.asarray(y, float); w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + R
        w -= np.linalg.solve(H, X.T @ (p - y) + R @ w)
    return [round(float(v), 3) for v in w]


def fit(data='data', outd='out_fit'):
    """의심도 계수 맞추기 — 반드시 개발 세트(data/)에서만. 결과를 firebim/xcheck.py CFG['coef'] 에 넣는다.
    (먼저 FIREBIM_XCHECK_CFG='{"wall_fill":false,"door_add":false,"item_p":0}' FIREBIM_MODEL=1 OUT=out_fit python run_all.py 6)"""
    from firebim.xcheck import feats
    s = main(data, outd, verbose=True)
    co = {}
    for k in ('rule_only', 'model_only'):
        L = [i for i in s['_items'] if i['kind'] == k]
        co[k] = logit_fit([feats(k, i) for i in L], [i['err'] for i in L])
    D = s['_doors']
    co['door'] = logit_fit([feats('door', d) for d in D], [d['new'] for d in D])
    return co


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--fit':
        print(json.dumps(fit(*(a[1:] or ['data', 'out_fit'])))); sys.exit()
    s = main(a[0] if a else 'data', a[1] if len(a) > 1 else 'out_model')
    print(json.dumps(s, ensure_ascii=False, indent=1))
