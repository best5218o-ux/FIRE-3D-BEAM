# -*- coding: utf-8 -*-
"""선택 경로(model=True): 정보카드 평면도 → STR 분할 모델(fire3dbeam/str_seg) 을 규칙 추출기(geom.py)와 **독립된 두 번째 추출기**로 돌려
(1) 벽 교차검증 — 두 추출기가 모두 본 벽 = 높은 신뢰도, 한쪽만 본 벽 조각 = '검수 필요'(신뢰도·의심도 포함)
(2) 규칙이 놓친 문·외벽 개구부(진입구) 보충 — 모델이 확신할 때만
(3) (선택) 규칙 벽의 끊긴 틈 중 모델이 벽으로 보고 문으로 보지 않는 곳 메우기
기하 표현은 엔진 그대로: 벽 {o,c,a,b,t}(m), 문 {o,x,y,w,kind,ext}(m). 좌표는 도면 지역좌표(x=px/s, y=-py/s) ↔ 전역(+tx,ty).

모델 출력은 작업 해상도(1/100 기준 1 m ≈ 29 px)의 확률 영상으로 보관: 지역좌표 (x,y) → 작업 px (x·ppm, −y·ppm).
기본값(규칙만)에는 아무 영향 없음 — pipeline.run(model=False) 는 이 모듈을 부르지 않는다."""
import os, hashlib
import numpy as np, cv2
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

# ── 조정값(가상 개발 세트 data/ 에서만 조정; 보류 data_hold 는 마지막에 1회 측정) ──
CFG = dict(
    band=0.25,            # 규칙 벽 중심선 양옆 탐색 폭(m) = t/2 + band — 두 추출기의 중심선 차이 허용
    p_wall=0.5,           # 모델 벽 확률 임계
    step=0.1,             # 벽 표본 간격(m)
    min_piece=1.0,        # 검수 항목으로 올릴 최소 불일치 길이(m)
    gap_merge=0.3,        # 불일치 구간 사이 이 길이 이하 일치 구간은 하나로 묶음(m)
    tol_model=0.4,        # 모델 벽 조각이 규칙 벽과 '같은 벽'으로 인정되는 거리(m) — 채점 허용오차와 같음
    # 문 보충
    door_add=True, p_door=0.5, door_p=0.6, door_wmin=0.6, door_wmax=4.6, door_near=0.9,
    door_wall_tol=0.35,   # 문 중심이 벽선(규칙 또는 모델) 위에 있어야 함(m)
    ext_edge=0.8,         # 외곽선까지 이 거리 이내 → 외벽 개구부(진입구)
    # 벽 틈 메우기(선택)
    wall_fill=True, fill_p=0.7, keep_openings=True, open_min=0.7, open_max=7.0, corner=True, corner_gap=0.6,
    item_p=0.5,           # 검수 항목으로 올리는 최소 의심도
    # 의심도 로지스틱 계수 [절편, 특징...] — feats() 순서. 가상 개발 세트 data/ 에서 맞춤(evalr/xcheck_eval.py --fit)
    coef=dict(rule_only=[3.729, -11.4, -4.069, -0.788, -0.082], model_only=[-3.253, 0.906, 18.769, -3.547, -0.57],
              door=[-6.512, 11.66, -1.233, 0.223, -0.374]),
)
# 의심도 계수는 모델 출력에 맞춘 것 → 모델 .json 의 "xcheck_coef" 가 있으면 그것(위 coef = 이전 v3 모델용, models/str_seg_v3.json 에는 없음)
try:
    import json as _json
    from fire3dbeam.str_seg import MODEL as _M
    _mj = os.path.splitext(_M)[0] + '.json'
    if os.path.exists(_mj): CFG['coef'] = dict(CFG['coef'], **(_json.load(open(_mj)).get('xcheck_coef') or {}))
except Exception:
    pass
if os.environ.get('FIRE3DBEAM_XCHECK_CFG'):          # 조정 실험용(JSON 덮어쓰기)
    import json as _json
    CFG.update(_json.loads(os.environ['FIRE3DBEAM_XCHECK_CFG']))


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


_SEG = None


def _seg():
    global _SEG
    if _SEG is None:
        from fire3dbeam.str_seg import StrSeg
        _SEG = StrSeg(threads=int(os.environ.get('FIRE3DBEAM_MODEL_THREADS', '1')))
    return _SEG


def model_plan(img, s, tile=None, even=False, skip_blank=False, roi=None):
    """평면도 1장(BGR/회색, 축척 s px/m) → 모델 출력 {ppm, pw(벽 확률), pd(문 확률), lab, v(벡터)}.
    FIRE3DBEAM_MODEL_CACHE 폴더가 있으면 (영상·축척·모델) 지문으로 캐시.
    roi=[x0,y0,x1,y1](원본 px): 그 안만 모델을 돌리고 밖은 배경(실제 도면 경로 — 글자층 치수로 찾은 건물·치수 범위, 2026-10-03 2차)"""
    from fire3dbeam.str_seg import WORK_PPM_1_100, MODEL, vectorize, model_vec_param
    if roi is not None:
        g0 = _gray(img); H, W = g0.shape
        x0, y0, x1, y1 = [int(round(v)) for v in roi]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        k = WORK_PPM_1_100 / s
        Hw, Ww = max(32, round(H * k)), max(32, round(W * k))
        sub = model_plan(g0[y0:y1, x0:x1], s, tile=tile, even=even, skip_blank=skip_blank)
        lab = np.zeros((Hw, Ww), np.uint8); pw = np.zeros((Hw, Ww), np.float32); pd = np.zeros((Hw, Ww), np.float32)
        oy, ox = int(round(y0 * k)), int(round(x0 * k))
        h_, w_ = min(sub['lab'].shape[0], Hw - oy), min(sub['lab'].shape[1], Ww - ox)
        lab[oy:oy + h_, ox:ox + w_] = sub['lab'][:h_, :w_]; pw[oy:oy + h_, ox:ox + w_] = sub['pw'][:h_, :w_]; pd[oy:oy + h_, ox:ox + w_] = sub['pd'][:h_, :w_]
        return dict(ppm=WORK_PPM_1_100, pw=pw, pd=pd, lab=lab, v=vectorize(lab, WORK_PPM_1_100, P=model_vec_param(MODEL)), roi=[x0, y0, x1, y1])
    g = _gray(img)
    VP = model_vec_param(MODEL)                          # 모델별 벡터화 조정값(모델 .json "vec_param")
    cache = os.environ.get('FIRE3DBEAM_MODEL_CACHE')
    key = None
    if cache:
        h = hashlib.sha1(g.tobytes()); h.update((f'{g.shape}|{s:.5f}|{os.path.getmtime(MODEL):.0f}' + (f'|t{tile}' if tile else '') + ('|even' if even else '') + ('|sb' if skip_blank else '')).encode())
        key = os.path.join(cache, h.hexdigest()[:20] + '.npz')
        if os.path.exists(key):
            z = np.load(key)
            pw, pd = z['pw'].astype(np.float32) / 255, z['pd'].astype(np.float32) / 255
            lab = z['lab']
            return dict(ppm=WORK_PPM_1_100, pw=pw, pd=pd, lab=lab, v=vectorize(lab, WORK_PPM_1_100, P=VP))
    lab, prob, k = _seg().predict(g, px_per_m=s, tile=tile, even=even, skip_blank=skip_blank)
    pw, pd = prob[1], prob[3]
    if key:
        os.makedirs(cache, exist_ok=True)
        np.savez_compressed(key, pw=np.round(pw * 255).astype(np.uint8), pd=np.round(pd * 255).astype(np.uint8), lab=lab)
    return dict(ppm=WORK_PPM_1_100, pw=pw.astype(np.float32), pd=pd.astype(np.float32), lab=lab, v=vectorize(lab, WORK_PPM_1_100, P=VP))


# ───────────────────────── 좌표 ─────────────────────────
def _ends(w):
    return ((w['a'], w['c']), (w['b'], w['c'])) if w['o'] == 'h' else ((w['c'], w['a']), (w['c'], w['b']))


def _line(w):
    p, q = _ends(w); return LineString([p, q])


def _band_max(P, x, y, o, half):
    """지역좌표 점 (x,y)에서 벽 방향 o 에 수직으로 ±half(m) 안의 최대 확률"""
    ppm = P['ppm']; H, W = P['pw'].shape
    X, Y = x * ppm, -y * ppm
    r = max(1, int(round(half * ppm)))
    xi, yi = int(round(X)), int(round(Y))
    if o == 'h':
        if not (0 <= xi < W): return 0.0
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)
        return float(P['pw'][y0:y1, xi].max()) if y1 > y0 else 0.0
    if not (0 <= yi < H): return 0.0
    x0, x1 = max(0, xi - r), min(W, xi + r + 1)
    return float(P['pw'][yi, x0:x1].max()) if x1 > x0 else 0.0


def _runs(mask, merge_gap):
    """불리언 열 → (i,j) 참 구간, 짧은 거짓 틈(≤merge_gap 표본)은 이어 붙임"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            if runs and i - runs[-1][1] <= merge_gap: runs[-1] = (runs[-1][0], j)
            else: runs.append((i, j))
            i = j
        else: i += 1
    return runs


# ───────────────────────── (1) 벽 교차검증 ─────────────────────────
def wall_support(walls_loc, P, cfg=CFG):
    """규칙 벽(지역 m)마다 모델 지지율(0..1)과 모델이 지지하지 않는 구간 목록."""
    out = []
    st = cfg['step']
    for w in walls_loc:
        L = w['b'] - w['a']
        n = max(2, int(np.ceil(L / st)) + 1)
        ts = np.linspace(w['a'], w['b'], n)
        half = w.get('t', 0.15) / 2 + cfg['band']
        pv = np.array([_band_max(P, t, w['c'], 'h', half) if w['o'] == 'h' else _band_max(P, w['c'], t, 'v', half) for t in ts])
        ok = pv >= cfg['p_wall']
        bad = _runs(~ok, int(round(cfg['gap_merge'] / st)))
        pieces = []
        for i, j in bad:
            a, b = ts[i], ts[min(j, n) - 1]
            if b - a + st >= cfg['min_piece']:
                pieces.append(dict(a=round(float(a), 3), b=round(float(b), 3), pm=round(float(pv[i:j].mean()), 3)))
        out.append(dict(sup=round(float(ok.mean()), 3), pmean=round(float(pv.mean()), 3), bad=pieces))
    return out


def model_only_walls(walls_loc, P, cfg=CFG):
    """모델 벽 벡터 중 규칙 벽(±tol)으로 설명 안 되는 조각(지역 m) → [{o,c,a,b,t,pm}]"""
    V = P['v']['walls']
    if not V: return []
    R = unary_union([_line(w) for w in walls_loc]).buffer(cfg['tol_model']) if walls_loc else None
    out = []
    sat = P.get('sat'); s_ = P.get('s')
    for w in V:
        ln = _line(w)
        rest = ln.difference(R) if R is not None else ln
        for g in ([rest] if rest.geom_type == 'LineString' else list(getattr(rest, 'geoms', []))):
            if g.geom_type != 'LineString' or g.length < cfg['min_piece']: continue
            (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
            a, b = (min(x0, x1), max(x0, x1)) if w['o'] == 'h' else (min(y0, y1), max(y0, y1))
            ts = np.linspace(a, b, max(2, int(np.ceil((b - a) / cfg['step'])) + 1))
            pv = [_band_max(P, t, w['c'], 'h', w['t'] / 2) if w['o'] == 'h' else _band_max(P, w['c'], t, 'v', w['t'] / 2) for t in ts]
            col = 0.0
            if sat is not None:                          # 채색 주석(대피 방향선 등) 위 조각 비율
                xs = [(t, w['c']) if w['o'] == 'h' else (w['c'], t) for t in ts]
                H, W = sat.shape
                v = [sat[min(H - 1, max(0, int(round(-y * s_)))), min(W - 1, max(0, int(round(x * s_))))] for x, y in xs]
                col = float(np.mean(np.array(v) >= 60))
            out.append(dict(o=w['o'], c=w['c'], a=round(a, 3), b=round(b, 3), t=w['t'], pm=round(float(np.mean(pv)), 3), col=round(col, 2)))
    return out


# ───────────────────────── (2) 문 보충 ─────────────────────────
def model_doors(P, walls_loc, cfg=CFG):
    """모델 문 개체 → 엔진 문 후보(지역 m) + 신뢰도. 문 중심이 벽선(규칙 또는 모델) 위에 있는 것만."""
    ppm = P['ppm']
    lines = [_line(w) for w in walls_loc] + [_line(w) for w in P['v']['walls']]
    out = []
    for d in P['v']['doors']:
        b = d['box']
        pd = P['pd'][b[1]:b[3], b[0]:b[2]]
        conf = float(pd[pd >= cfg['p_door']].mean()) if (pd >= cfg['p_door']).any() else 0.0
        bw, bh = (b[2] - b[0]) / ppm, (b[3] - b[1]) / ppm
        o = 'h' if bw >= bh else 'v'
        w = max(bw, bh)
        pt = Point(d['x'], d['y'])
        dist = min((l.distance(pt) for l in lines), default=9.0)
        out.append(dict(o=o, x=round(d['x'], 3), y=round(d['y'], 3), w=round(w, 2), conf=round(conf, 3), on_wall=round(dist, 2)))
    return out


def _to_glob(w, tx, ty):
    w = dict(w)
    if w['o'] == 'h': w['c'] += ty; w['a'] += tx; w['b'] += tx
    else: w['c'] += tx; w['a'] += ty; w['b'] += ty
    return w


def _to_loc(w, tx, ty):
    return _to_glob(w, -tx, -ty)


def _sig(z): return 1.0 / (1.0 + np.exp(-z))


def suspicion(kind, x, cfg=CFG):
    """불일치 항목이 '실제 오류'일 확률(의심도 p) — 가상 개발 세트 data/ 정답으로 맞춘 로지스틱 회귀(CFG['coef']).
    rule_only: 규칙 벽이 틀렸을(오검출) 확률, model_only: 규칙이 벽을 놓쳤을 확률."""
    c = cfg['coef'][kind]
    f = feats(kind, x)
    return float(_sig(c[0] + sum(a * b for a, b in zip(c[1:], f))))


def feats(kind, x):
    if kind == 'door': return [x['conf'], x['w'], x['on_wall'], float(x['gap'])]
    L = np.log(max(0.1, x['b'] - x['a']))
    if kind == 'rule_only': return [x['pm'], x['sup'], L, x['t']]
    if kind == 'model_only': return [x['pm'], x['t'], x.get('col', 0.0), L]
    raise ValueError(kind)


def _fp_ok(fp, F):
    ca = (F.get('card') or {}).get('area')
    return fp is not None and (not ca or 0.7 <= fp.area / ca <= 1.5)


def cross_floor(F, P, tx, ty, fp, cfg=CFG):
    """층 1개: 규칙 결과 F(walls·doors 전역 m) × 모델 출력 P → 교차검증 결과(전역 m).
    반환 dict(sup=[벽별 모델 지지율], rule_only=[...], model_only=[...], door_cand=[...], agree=길이 가중 일치율)
    각 불일치 조각에 의심도 p(실제 오류일 확률)를 붙인다."""
    WL = [_to_loc(w, tx, ty) for w in F['walls']]
    sup = wall_support(WL, P, cfg)
    rule_only, agree_len, tot_len = [], 0.0, 0.0
    for k, (w, s) in enumerate(zip(F['walls'], sup)):
        L = w['b'] - w['a']; tot_len += L; agree_len += L * s['sup']
        for pc in s['bad']:
            r = dict(o=w['o'], c=w['c'], a=round(pc['a'] + (tx if w['o'] == 'h' else ty), 3), b=round(pc['b'] + (tx if w['o'] == 'h' else ty), 3),
                     t=w['t'], wall=k, pm=pc['pm'], sup=s['sup'])
            r['p'] = round(suspicion('rule_only', r, cfg), 3)
            rule_only.append(r)
    mo = [_to_glob(m, tx, ty) for m in model_only_walls(WL, P, cfg)]
    if _fp_ok(fp, F):                                   # 외곽 밖(범례·표·조경) 모델 벽은 제외 — 외곽을 믿을 만할 때만
        A = fp.buffer(1.0)
        mo = [m for m in mo if A.intersects(_line(m))]
    for m in mo: m['p'] = round(suspicion('model_only', m, cfg), 3)
    # 문 후보(규칙 문과 겹치지 않는 모델 문)
    cand = []
    ex = [(d['x'], d['y']) for d in F['doors']]
    for d in model_doors(P, WL, cfg):
        x, y = d['x'] + tx, d['y'] + ty
        if d['conf'] < cfg['p_door'] or not (cfg['door_wmin'] <= d['w'] <= cfg['door_wmax']) or d['on_wall'] > cfg['door_wall_tol']: continue
        if any(np.hypot(x - a, y - b) < cfg['door_near'] for a, b in ex): continue
        if fp is not None and _fp_ok(fp, F) and not fp.buffer(0.5).contains(Point(x, y)): continue
        on_edge = fp is not None and fp.boundary.distance(Point(x, y)) < cfg['ext_edge']
        # 규칙 벽 위(틈 없음)인가, 규칙 벽 틈 안인가
        gap = not any(_line(w).distance(Point(d['x'], d['y'])) < 0.2 for w in WL)
        nd = dict(o=d['o'], x=round(x, 2), y=round(y, 2), w=round(d['w'], 2), kind='opening' if on_edge else 'door', ext=bool(on_edge),
                  src='model', conf=d['conf'], on_wall=d['on_wall'], gap=gap)
        nd['p'] = round(1 - suspicion('door', nd, cfg), 3)          # 의심도 = 모델 문이 틀렸을 확률
        cand.append(nd); ex.append((x, y))
    return dict(sup=[s['sup'] for s in sup], rule_only=rule_only, model_only=mo, door_cand=cand,
                agree=round(agree_len / tot_len, 3) if tot_len else None)


def fill_walls(walls, mo, cfg=CFG):
    """(3) 벽 보충: 의심도(규칙이 벽을 놓쳤을 확률) ≥ fill_p 인 모델 벽 조각을 엔진 벽으로 추가(src='model').
    조각 양끝을 tol_model 만큼 늘려 이웃 규칙 벽과 겹치게 하고(가짜 틈=가짜 문 방지), 같은 선의 규칙 벽 중심에 맞춘다."""
    add = []
    tol = cfg['tol_model']
    for m in mo:
        if m['p'] < cfg['fill_p']: continue
        c = m['c']
        same = [w for w in walls if w['o'] == m['o'] and abs(w['c'] - c) < 0.3]
        # 규칙이 '두께 사라짐'으로 끊은 개구부(문·셔터 폭)를 모델이 벽으로 메우려 하면 → 자동 보충하지 않고 검수 항목으로만 남김
        lft = any(abs(w['b'] - m['a']) <= tol + 0.15 for w in same); rgt = any(abs(w['a'] - m['b']) <= tol + 0.15 for w in same)
        if cfg['keep_openings'] and lft and rgt and cfg['open_min'] <= (m['b'] - m['a']) + 2 * tol <= cfg['open_max']:
            m['bridge'] = True; continue
        if same: c = min(same, key=lambda w: abs(w['c'] - m['c']))['c']
        add.append(dict(o=m['o'], c=round(c, 3), a=round(m['a'] - tol, 3), b=round(m['b'] + tol, 3), t=round(max(0.15, min(0.5, m['t'])), 2), src='model'))
    return add


def extend_ends(walls, P, tx, ty, cfg=CFG):
    """(3') 모서리 틈 메우기: 벽 끝이 직교 벽 바로 앞(≤ corner_gap m)에서 끊겼고 그 사이를 모델이 벽으로 보면 직교 벽 중심선까지 늘린다.
    (외곽 모서리가 0.3 m 만 벌어져도 외곽 닫기가 실패해 외벽 개구부·진입구를 못 찾는 문제). 새 목록 반환, 바뀐 벽 수."""
    out, n = [], 0
    G = cfg['corner_gap']
    for w in walls:
        v = dict(w)
        perp = [u for u in walls if u['o'] != w['o'] and u['a'] - G <= w['c'] <= u['b'] + G]
        for end in ('a', 'b'):
            e = v[end]
            cands = [u['c'] for u in perp if (0.02 < (e - u['c']) <= G if end == 'a' else 0.02 < (u['c'] - e) <= G)]
            if not cands: continue
            tgt = max(cands) if end == 'a' else min(cands)
            ts = np.linspace(min(e, tgt), max(e, tgt), 5)
            loc_c = w['c'] - (ty if w['o'] == 'h' else tx); off = tx if w['o'] == 'h' else ty
            pv = [_band_max(P, t - off, loc_c, 'h', w['t'] / 2 + cfg['band']) if w['o'] == 'h' else _band_max(P, loc_c, t - off, 'v', w['t'] / 2 + cfg['band']) for t in ts]
            if np.mean(pv) >= cfg['p_wall']: v[end] = round(float(tgt), 3); n += 1
        out.append(v)
    return out, n


def still_present(x, walls, frac=0.5):
    """불일치 조각 x 가 (보충·외곽 자르기 뒤) 최종 벽 목록에 아직 있는가(같은 선 위 겹침 ≥ frac)"""
    L = x['b'] - x['a']; ov = 0.0
    for w in walls:
        if w['o'] == x['o'] and abs(w['c'] - x['c']) < 0.05 and w.get('src') != 'model':
            ov += max(0.0, min(w['b'], x['b']) - max(w['a'], x['a']))
    return ov >= frac * L
