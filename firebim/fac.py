# -*- coding: utf-8 -*-
"""층별 소방시설 목록 + 위치 확실성(located / estimated / present)
입력: 정보카드 '층별 소방시설 설치현황'(분류표 firebim/fac_catalog.py 로 해석), 도면 범례 기호(기존 소화전·수신기 + 추가 소화기·방수구·비상콘센트·송수구),
      계단·승강기(층간 클러스터), 문(벽 틈)
- located  : 도면 범례 기호로 찾은 위치
- estimated: 규칙 추정 위치(방수구 NFPC 502 제6조 · 비상콘센트 NFTC 504 2.1.5.2 → 계단 출입구 5 m 반경, 비상용승강기 → 건물 단위 점수: 승강기 앞 닫힌 승강장·부속실 구획 / 비상용 글자 / 계단 근접 / 전 층 운행 / 피난층 출구 30 m, 대수 = 정보카드 — emev_rank)
- present  : 정보카드 수량만(스프링클러 헤드·감지기 등 — 가짜 위치를 만들지 않고 층 단위 존재·수량만)
카드 수량 ↔ 도면 기호 수 교차검증 플래그 kind='fac_count' (방수구·비상콘센트·송수구: 도면에 그렸으면 수가 같아야 함 / 소화기: 도면 ≤ 카드)"""
import math, os
from firebim.fac_catalog import CATALOG, SYMBOL_CODE, RULES
from firebim.card import level_of

EXACT = {'STANDPIPE', 'OUTLET', 'SIAMESE'}          # 도면에 그렸으면 층의 전부를 그리는 항목(수 일치 요구)
OLD_SYM = {'hose': 'HOSE', 'panel': 'PANEL'}        # 기존 기호(수량 불일치는 기존 hose_count 플래그가 담당)


def _door_of(stair_box, doors):
    """계단 상자 둘레 1 m 안의 문 → (x, y, 바깥 법선) 목록"""
    x0, y0, x1, y1 = stair_box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for d in doors:
        dist = max(x0 - d['x'], d['x'] - x1, 0) + max(y0 - d['y'], d['y'] - y1, 0)
        inner = min(d['x'] - x0, x1 - d['x'], d['y'] - y0, y1 - d['y'])      # 상자 안쪽 깊이(양수 = 안)
        if dist <= 1.0 and inner <= 0.8:
            if d['o'] == 'h': n = (0.0, 1.0 if d['y'] >= cy else -1.0)
            else: n = (1.0 if d['x'] >= cx else -1.0, 0.0)
            out.append((dist, d['x'], d['y'], n))
    out.sort(key=lambda t: t[0])
    return out


def stair_anchor(S, doors, fp, exits=()):
    """계단 출입구 위치: ① 계단 앞 비상구(피난구) 표지(상자 3.5 m 안) ② 상자 둘레 문 ③ 외곽 중심 쪽 상자 변 가운데.
    도면의 계단 상자는 디딤판 영역이라 실제 계단실 문과 1 m 넘게 떨어질 수 있어 표지를 먼저 쓴다"""
    x0, y0, x1, y1 = S['box']; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ex = [(max(x0 - e['x'], e['x'] - x1, 0, y0 - e['y'], e['y'] - y1), e) for e in exits]
    ex = [t for t in ex if t[0] <= 3.5]
    if ex:
        e = min(ex, key=lambda t: t[0])[1]
        dx, dy = e['x'] - cx, e['y'] - cy
        if abs(dx) / max(0.1, (x1 - x0)) >= abs(dy) / max(0.1, (y1 - y0)): n = (1.0 if dx > 0 else -1.0, 0.0)
        else: n = (0.0, 1.0 if dy > 0 else -1.0)
        return e['x'], e['y'], n, 'exit_sign'
    ds = _door_of(S['box'], doors)
    if ds:
        _, x, y, n = ds[0]; return x, y, n, 'door'
    if fp is not None:
        c = fp.centroid; cx, cy = c.x, c.y
    cand = [((x0 + x1) / 2, y0, (0, -1)), ((x0 + x1) / 2, y1, (0, 1)), (x0, (y0 + y1) / 2, (-1, 0)), (x1, (y0 + y1) / 2, (1, 0))]
    x, y, n = min(cand, key=lambda t: math.hypot(t[0] - cx, t[1] - cy))
    return x, y, n, 'box'


# ───────────────────────────── 비상용승강기(EMEV) 추정 — 건물 단위(승강로는 전 층 하나로 이어짐)
# 근거 법령: 건축법 제64조②(높이 31 m 초과), 시행령 제90조(대수: 31 m 넘는 층 최대 바닥면적 1,500㎡ 이하 1대, 넘으면 3,000㎡마다 1대 추가),
# 「건축물의 설비기준 등에 관한 규칙」 제10조(승강장: 다른 부분과 내화구조로 구획·각 층 내부와 연결·바닥면적 1대당 6㎡ 이상(옥외 승강장 제외)·
# 특별피난계단 부속실 겸용 가능·피난층 승강장 출입구에서 도로·공지까지 30 m 이하·비상용승강기 표지 / 승강로: 각 층에서 피난층까지 단일 구조로 연결).
# 도면·정보카드에서 볼 수 있는 단서만 쓴다: ① 승강기 옆 글자(비상·부속실·전실·승강장·피난용) ② 승강기 앞 작은 구획 공간(승강장·부속실, 벽망 실 분할)
# ③ 계단 출입구 거리(부속실 겸용이면 계단 옆) ④ 전 층 운행·피난층 정차 ⑤ 피난층 옥외 출구 거리 30 m ⑥ 대수 = 정보카드 ⑩ 층별 '비상용승강기' 수 · ⑦ 승강기(비상).
EMEV_W = dict(label=3.0, lobby=3.0, stair=0.05, allfloor=0.3, no_ground=-2.0, ground_far=-1.0, decisive=0.5)   # decisive: 1·2위 점수 차 문턱(개발 세트에서만 정함)
EMEV_WORDS = [('비상', 3.0), ('피난용', 2.5), ('부속실', 2.0), ('전실', 2.0), ('승강장', 2.0)]
EMEV_LOBBY = dict(tol=0.2, side=2.0, dmin=1.6, dmax=4.0, front=(0.4, 2.2), hw=1.15, ext=1.6)   # 앞 구역: 선 허용(m)·옆벽 확인 길이·끝벽 깊이 범위·승강로 앞벽 거리·기본 반폭·옆벽 바깥 탐색(부속실 겸용)
EMEV_OCR_MIN_S = float(os.environ.get('FIREBIM_EMEV_OCR_MIN_S', '14'))   # 글자 OCR 은 도면 축척 ≥ 이 값(px/m)일 때만: 실명 글자(≈0.5~0.75 m)가 10 px 미만이면 판독 불가


def _ecen(E): return ((E['box'][0] + E['box'][2]) / 2, (E['box'][1] + E['box'][3]) / 2)


def _row_axis(k, cen):
    """승강기 k 가 줄지어 선 방향('x'|'y')과 반폭(이웃 중심 간격/2) — 이웃 없으면 (None, 기본값)"""
    cx, cy = cen[k]
    nb = [(abs(x - cx), 'x') for j, (x, y) in enumerate(cen) if j != k and abs(y - cy) < 0.8 and abs(x - cx) < 3.5] + \
         [(abs(y - cy), 'y') for j, (x, y) in enumerate(cen) if j != k and abs(x - cx) < 0.8 and abs(y - cy) < 3.5]
    if not nb: return None, EMEV_LOBBY['hw']
    d, ax = min(nb)
    return ax, max(0.8, min(1.6, d / 2))


def _ink(p):
    """도면 p 의 선 잉크(무채색) — 글자 덩어리·작은 조각·그리드선(얇은 구간) 제외. 벽 벡터화가 놓치는 얇은 칸막이(1~2 px)까지 본다"""
    import cv2, numpy as np
    from firebim.vision import masks
    M = p.get('M') or masks(p['img']); s = p['s']
    ink = ((M['lum'] < 215) & (M['sat'] < 45)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    w_, h_, a_ = st[1:, 2], st[1:, 3], st[1:, 4]
    keep = np.zeros(n, bool)
    keep[1:] = (np.maximum(w_, h_) >= 1.2 * s) & ~((np.minimum(w_, h_) >= 0.45 * s) & (np.maximum(w_, h_) <= 5.0 * s) & (a_ >= 0.3 * w_ * h_))   # 글자(뭉친 덩어리) 제외
    ink = keep[lab].astype(np.uint8)
    H, W = ink.shape
    for x in p.get('vlines', []):
        x = int(round(x))
        if 3 <= x < W - 3:
            thin = (ink[:, x - 3] == 0) & (ink[:, x + 3] == 0)
            for dx in (-1, 0, 1): ink[thin, x + dx] = 0
    for y in p.get('hlines', []):
        y = int(round(y))
        if 3 <= y < H - 3:
            thin = (ink[y - 3, :] == 0) & (ink[y + 3, :] == 0)
            for dy in (-1, 0, 1): ink[y + dy, thin] = 0
    return ink


def _cov(ink, o, c, a, b, tol):
    """잉크 영상에서 선(o='h': 행 c, 열 a..b / 'v': 열 c, 행 a..b, ±tol px) 위 잉크 덮임 비율"""
    H, W = ink.shape
    a, b = int(round(min(a, b))), int(round(max(a, b)))
    c0, c1 = int(round(c - tol)), int(round(c + tol)) + 1
    if o == 'h':
        seg = ink[max(0, c0):min(H, c1), max(0, a):min(W, b)] if c1 > 0 and c0 < H and b > a else None
        return float(seg.max(0).mean()) if seg is not None and seg.size else 0.0
    seg = ink[max(0, a):min(H, b), max(0, c0):min(W, c1)] if c1 > 0 and c0 < W and b > a else None
    return float(seg.max(1).mean()) if seg is not None and seg.size else 0.0


def _front_zone(ink, p, c, hw, axis):
    """승강기(중심 c, 반폭 hw) 문 앞 구역이 벽으로 닫힌 정도 0~1 → (정도, 깊이 m, 폭 m).
    앞 구역 = 승강로 앞벽(중심에서 0.4~2.2 m 나란한 선) 바깥 폭 2·hw(옆벽은 바깥 1.6 m 까지 — 특별피난계단 부속실 겸용이면 계단 벽) × 깊이 d(1.6~4 m).
    정도 = min(두 옆벽 덮임, 끝벽 덮임) — 세 변이 다 있어야 높음(이웃 승강기는 공유 옆벽 한 변뿐), 앞·뒤 두 방향 중 큰 쪽.
    비상용승강기 승강장(설비기준규칙 제10조: 다른 부분과 내화구조 구획·1대당 6㎡ 이상) 단서."""
    import numpy as np
    s, tx, ty = p['s'], p['tx'], p['ty']
    tol = max(1.0, EMEV_LOBBY['tol'] * s); L = EMEV_LOBBY
    if axis == 'x': U, V, cu, cv, op, oq = (lambda x: (x - tx) * s), (lambda y: (ty - y) * s), c[0], c[1], 'h', 'v'
    else: U, V, cu, cv, op, oq = (lambda y: (ty - y) * s), (lambda x: (x - tx) * s), c[1], c[0], 'v', 'h'
    best = (0.0, None, None)
    for sg in (1, -1):
        u0, u1 = sorted((U(cu - hw), U(cu + hw)))
        fr = [(_cov(ink, op, V(cv + sg * dv), u0, u1, tol), dv) for dv in np.arange(L['front'][0], L['front'][1] + 1e-6, 0.1)]
        f0, vf = max(fr)
        if f0 < 0.5: vf = 1.3
        lo, hi = sorted((V(cv + sg * vf), V(cv + sg * (vf + L['side']))))
        e_ = L['ext'] * s
        s0 = max((_cov(ink, oq, u, lo, hi, tol), -abs(u - u0), u) for u in np.arange(u0 - e_, u0 + 0.3 * s, 1.0))
        s1 = max((_cov(ink, oq, u, lo, hi, tol), -abs(u - u1), u) for u in np.arange(u1 + e_, u1 - 0.3 * s, -1.0))
        ua, ub = s0[2], s1[2]
        e, d = max((_cov(ink, op, V(cv + sg * (vf + d)), ua, ub, tol), d) for d in np.arange(L['dmin'], L['dmax'] + 1e-6, 0.2))
        v = min(s0[0], s1[0], e)
        if v > best[0]: best = (v, round(float(d), 1), round((ub - ua) / s, 1))
    return best


def _word_hit(t):
    """OCR 글자 → (어휘, 가중) 또는 None. '승강기홀'·'승강기' 는 승강장이 아님(자모 유사 매칭에서 제외)"""
    from firebim.vision import _jsim
    t = ''.join(ch for ch in t if '가' <= ch <= '힣')
    if not t: return None
    for w, v in EMEV_WORDS:
        if w in t: return w, v
    if '승강기' in t or '홀' in t: return None
    best = None
    for w, v in EMEV_WORDS:
        if len(w) < 3: continue                                # 두 글자 어휘는 정확히 일치할 때만
        for i in range(0, max(1, len(t) - len(w) + 1)):
            seg = t[i:i + len(w)]
            sc = _jsim(seg, w) if len(seg) == len(w) else 0
            if sc >= 0.8 and (best is None or sc > best[2]): best = (w, v, sc)
    return best[:2] if best else None


def _ocr_words(p, boxes, pad=5.0):
    """도면 p 에서 승강기 상자들 둘레(pad m) 글자 → [(전역 x, y, 글자)]. 3배 확대 tesseract kor"""
    import cv2
    from firebim.vision import ocr_tsv
    s, tx, ty, img = p['s'], p['tx'], p['ty'], p['img']
    H, W = img.shape[:2]
    X0 = min(b[0] for b in boxes) - pad; X1 = max(b[2] for b in boxes) + pad
    Y0 = min(b[1] for b in boxes) - pad; Y1 = max(b[3] for b in boxes) + pad
    c0, c1 = max(0, int((X0 - tx) * s)), min(W, int((X1 - tx) * s))
    r0, r1 = max(0, int((ty - Y1) * s)), min(H, int((ty - Y0) * s))
    if c1 - c0 < 10 or r1 - r0 < 10: return []
    g = cv2.cvtColor(img[r0:r1, c0:c1], cv2.COLOR_BGR2GRAY)
    L = p.get('legend')
    if L:                                                      # 범례 상자 글자 제외
        g = g.copy(); g[max(0, L['y'] - r0 - 3):max(0, L['y'] + L['h'] - r0 + 3), max(0, L['x'] - c0 - 3):max(0, L['x'] + L['w'] - c0 + 3)] = 255
    k = 3.0
    up = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    try: rows = ocr_tsv(up, psm=11, lang='kor', wl=None)
    except Exception: return []
    toks = []                                                  # 같은 줄에 붙은 글자 조각('부' '속' '실') → 한 낱말(왼쪽부터)
    for r in sorted(rows, key=lambda r: r['x']):
        q = next((q for q in toks if abs((r['y'] + r['h'] / 2) - (q['y'] + q['h'] / 2)) < 0.5 * max(r['h'], q['h'])
                  and -0.2 * r['h'] <= r['x'] - (q['x'] + q['w']) < 1.0 * max(r['h'], q['h'])), None)
        if q: q['text'] += r['text']; q['h'] = max(q['h'], r['h']); q['w'] = r['x'] + r['w'] - q['x']
        else: toks.append(dict(r))
    return [((c0 + (r['x'] + r['w'] / 2) / k) / s + tx, -(r0 + (r['y'] + r['h'] / 2) / k) / s + ty, r['text']) for r in toks]


def emev_rank(info, floors, stairs, elevs, plans=None, n_card=0):
    """비상용승강기 후보 점수(건물 단위) → dict(order=[승강기 번호], score, why{k: [근거]}, n, decisive, cues)"""
    n_em = max(1, n_card or 0, int(((info or {}).get('elev') or {}).get('emergency') or 0))
    fids = sorted(floors, key=level_of)
    lv_all = [level_of(f) for f in fids]
    ground = '1F' if '1F' in floors else (fids[0] if fids else None)
    cen = [_ecen(E) for E in elevs]
    score = [0.0] * len(elevs); why = {k: [] for k in range(len(elevs))}; cues = {k: {} for k in range(len(elevs))}
    by_plan = {}
    for fid in fids: by_plan.setdefault(floors[fid].get('plan'), []).append(fid)
    P = {p['i']: p for p in (plans or [])}
    lob = [[] for _ in elevs]; lab_hit = [dict() for _ in elevs]; n_plans = [0] * len(elevs); n_ocr = 0
    for pi, pf in by_plan.items():
        on = [k for k, E in enumerate(elevs) if any(f in E['floors'] for f in pf)]
        if not on or pi not in P or len(elevs) <= n_em: continue
        ink = _ink(P[pi])
        for k in on:
            n_plans[k] += 1
            ax, hw = _row_axis(k, cen)
            lob[k].append(max(_front_zone(ink, P[pi], cen[k], hw, a) for a in ((ax,) if ax else ('x', 'y'))))
        if len(elevs) > n_em and pi in P and P[pi]['s'] >= EMEV_OCR_MIN_S:
            n_ocr += 1
            for x, y, t in _ocr_words(P[pi], [elevs[k]['box'] for k in on]):
                h = _word_hit(t)
                if not h: continue
                d = sorted((math.hypot(x - cen[k][0], y - cen[k][1]), k) for k in on)
                if d[0][0] > 4.0 or (len(d) > 1 and d[1][0] - d[0][0] < 0.5): continue      # 가장 가까운 승강기(4 m 안, 둘째와 0.5 m 이상 차)
                k = d[0][1]
                lab_hit[k][h[0]] = (h[1], lab_hit[k].get(h[0], (0, 0))[1] + 1); cues[k].setdefault('ocr', []).append(t)
    # 계단 출입구 거리(층별 최소 → 건물 최소)
    dst = [None] * len(elevs)
    for fid in fids:
        F = floors[fid]
        anchors = [stair_anchor(S, F.get('doors', []), F.get('_fp'))[:2] for S in stairs if fid in S['floors']]
        for k, E in enumerate(elevs):
            if fid not in E['floors'] or not anchors: continue
            d = min(math.hypot(cen[k][0] - a[0], cen[k][1] - a[1]) for a in anchors)
            dst[k] = d if dst[k] is None else min(dst[k], d)
    gx = [d for d in (floors.get(ground) or {}).get('doors', []) if d.get('ext')] if ground else []
    for k, E in enumerate(elevs):
        m = max(1, n_plans[k])
        if lab_hit[k]:
            w, (v, c) = max(lab_hit[k].items(), key=lambda t: (t[1][0], t[1][1]))
            f = min(1.0, 2.0 * c / max(1, n_ocr))              # 글자는 도면 몇 장에서만 읽혀도 인정(OCR 누락 많음)
            score[k] += EMEV_W['label'] * (v / 3.0) * f
            why[k].append(f"도면 글자 '{w}' ({c}/{n_ocr}장 — 비상용승강기 표지·승강장 명칭)")
        if lob[k]:
            v = sum(q[0] for q in lob[k]) / len(lob[k]); cues[k]['lobby'] = round(v, 2)
            score[k] += EMEV_W['lobby'] * v
            if v >= 0.3:                                        # (면적은 저해상도 도면에서 깊이 오차가 커 표시하지 않음)
                why[k].append(f"승강기 앞 벽으로 닫힌 공간(세 변 덮임 {v:.0%}, 도면 {len(lob[k])}장 평균) — "
                              "승강장·부속실 추정(설비기준규칙 제10조: 다른 부분과 내화구조 구획·1대당 6㎡ 이상·특별피난계단 부속실 겸용 가능)")
        if dst[k] is not None:
            score[k] -= EMEV_W['stair'] * dst[k]; cues[k]['stair_m'] = round(dst[k], 1)
        lv = [level_of(f) for f in E['floors']]
        if lv and lv_all and min(lv) <= min(lv_all) and max(lv) >= max(lv_all):
            score[k] += EMEV_W['allfloor']; cues[k]['all_floors'] = True
        if ground and ground not in E['floors']:
            score[k] += EMEV_W['no_ground']; why[k].append('피난층 미정차(제10조 승강로는 피난층까지 연결 — 비상용 아닐 가능성)')
        elif gx:
            d = min(math.hypot(cen[k][0] - d_['x'], cen[k][1] - d_['y']) for d_ in gx); cues[k]['ground_exit_m'] = round(d, 1)
            if d > 30.0: score[k] += EMEV_W['ground_far']; why[k].append(f'피난층 옥외 출구까지 직선 {d:.0f} m(제10조 30 m 초과)')
    order = sorted(range(len(elevs)), key=lambda k: (-score[k], dst[k] if dst[k] is not None else 1e9, k))
    decisive = len(elevs) <= n_em or (score[order[n_em - 1]] - score[order[n_em]] >= EMEV_W['decisive'])
    if not decisive:                  # 단서 차이가 문턱 미만(잡음) → 계단 출입구 최근접(건물 단위로 같은 승강기) 후보
        order = sorted(range(len(elevs)), key=lambda k: (dst[k] if dst[k] is not None else 1e9, -score[k], k))
    for k in range(len(elevs)):
        if dst[k] is not None: why[k].append(f'계단 출입구 {dst[k]:.1f} m')
        if cues[k].get('all_floors'): why[k].append('전 층 운행(제10조 승강로 단일 연결)')
        if 'ground_exit_m' in cues[k] and cues[k]['ground_exit_m'] <= 30: why[k].append(f"피난층 옥외 출구까지 직선 {cues[k]['ground_exit_m']:.0f} m(≤30 m)")
    if n_em > 1: why = {k: v + [f'정보카드 비상용 {n_em}대'] for k, v in why.items()}
    return dict(order=order, score=[round(v, 2) for v in score], why=why, n=n_em, decisive=decisive, cues=cues)


def build(info, floors, stairs, elevs, plans=None):
    """→ (fac 절 {floors: {fid: [item]}, rules, catalog}, flags). plans(선택): 도면 영상(비상용승강기 글자 OCR)"""
    out, flags = {}, []
    EM = None
    for fid in sorted(floors, key=level_of):
        F = floors[fid]
        card = (F.get('card') or {}).get('fac_items') or []
        syms = {}
        for q in F.get('fac_symbols', []): syms.setdefault(SYMBOL_CODE.get(q['type']), []).append(q)
        for q in F.get('symbols', []):
            if q['type'] in OLD_SYM: syms.setdefault(OLD_SYM[q['type']], []).append(q)
        fst = [S for S in stairs if fid in S['floors']]
        fev = [E for E in elevs if fid in E['floors']]
        exits = [q for q in F.get('symbols', []) if q['type'] == 'exit']
        items = []
        seen = set()
        for it in card:
            c = it['code']; seen.add(c)
            C = CATALOG.get(c)
            item = dict(code=c, name=it['name'], cat=it['category'], group=it['group'], unit=C['unit'] if C else '', raw=it['raw'],
                        card=it['count'], drawing=None, loc=[], est=[], present=0, src=[], note='')
            if C and C['symbol']:
                ds = syms.get(c, [])
                item['drawing'] = len(ds)
                item['loc'] = [dict(x=q['x'], y=q['y'], conf=q.get('conf')) for q in ds]
                if c in EXACT and ds and len(ds) != it['count']:
                    flags.append(dict(kind='fac_count', code=c, floor=fid, card=it['count'], drawing=len(ds), name=C['name']))
                elif c == 'EXT' and len(ds) > it['count']:
                    flags.append(dict(kind='fac_count', code=c, floor=fid, card=it['count'], drawing=len(ds), name=C['name']))
            rest = max(0, it['count'] - len(item['loc']))
            if rest and C and C['rule'] in ('nfpc502', 'nftc504') and fst:
                taken = [(p['x'], p['y']) for p in item['loc']]
                cands = []
                for S in fst:
                    ax, ay, n, how = stair_anchor(S, F.get('doors', []), F.get('_fp'), exits)
                    if any(math.hypot(ax - x, ay - y) <= 5.0 + 1.0 for x, y in taken): continue    # 이미 도면 기호가 있는 계단
                    cands.append(dict(x=round(ax + n[0] * 1.5, 2), y=round(ay + n[1] * 1.5, 2), ax=round(ax, 2), ay=round(ay, 2), r=RULES[C['rule']]['radius'],
                                      anchor=how, stair=[round(v, 1) for v in S['box']]))
                if cands:
                    alt = len(cands) > rest            # 규칙이 '계단 중 1개(2개)'만 요구 → 어느 계단인지 모름: 후보 전부 표시
                    for q in cands: q['alt'] = alt
                    item['est'] = cands
                    item['present'] = max(0, rest - len(cands))
                    item['note'] = RULES[C['rule']]['ref'] + (f' — 계단 {len(cands)}곳 중 {rest}곳(후보)' if alt else '')
                else: item['present'] = rest
            elif rest and c == 'EMEV' and fev:
                # 건물 단위 점수(글자·승강장 구획·계단 거리·전층 운행·피난층 출구) → 상위 N대(N = 정보카드 수). 단서가 없으면 계단 출입구 최근접(후보)
                if EM is None:
                    n_card = max([i2['count'] for f2 in floors.values() for i2 in ((f2.get('card') or {}).get('fac_items') or []) if i2['code'] == 'EMEV'] or [0])
                    EM = emev_rank(info, floors, stairs, elevs, plans, n_card)
                on = {id(E) for E in fev}
                ks = [k for k in EM['order'] if id(elevs[k]) in on]
                for k in ks[:rest]:
                    E = elevs[k]
                    item['est'].append(dict(x=round((E['box'][0] + E['box'][2]) / 2, 2), y=round((E['box'][1] + E['box'][3]) / 2, 2), r=None, anchor='elevator',
                                            box=[round(v, 2) for v in E['box']], alt=len(fev) > rest and not EM['decisive'],
                                            score=EM['score'][k], why=' · '.join(EM['why'][k])))
                top = item['est'][0]['why'] if item['est'] else ''
                item['note'] = RULES['elevator']['ref'] + (' — 근거: ' + top if EM['decisive'] and top else
                                                           ' — 도면 단서(글자·승강장 구획)로 가리지 못함 → 계단 출입구 최근접 후보' if len(fev) > rest else '')
                item['present'] = max(0, rest - len(item['est']))
            else:
                item['present'] = rest
            if c == 'STANDPIPE' and level_of(fid) >= 10: item['note'] = (item['note'] + ' · ' if item['note'] else '') + '11층 이상: 쌍구형(NFPC 502 제6조)'
            item['src'] = (['도면 기호'] if item['loc'] else []) + (['규칙 추정'] if item['est'] else []) + (['정보카드 수량'] if item['card'] else [])
            item['cert'] = 'located' if item['loc'] and not item['est'] and not item['present'] else 'estimated' if item['est'] and not item['present'] and not item['loc'] \
                else 'mixed' if (item['loc'] or item['est']) else 'present'
            items.append(item)
        # 카드에 없는데 도면에만 있는 기호(수신기 등) → 도면 기호로만
        for c, ds in syms.items():
            if c is None or c in seen or not ds: continue
            C = CATALOG[c]
            items.append(dict(code=c, name=C['name'], cat=C['cat'], group=C['group'], unit=C['unit'], raw='(정보카드 항목 없음)', card=None, drawing=len(ds),
                              loc=[dict(x=q['x'], y=q['y'], conf=q.get('conf')) for q in ds], est=[], present=0, src=['도면 기호'], note='정보카드에 수량 없음 — 도면 기호만', cert='located'))
        out[fid] = items
    return dict(floors=out, rules={k: v['ref'] for k, v in RULES.items()},
                legend='located=도면 범례 기호 위치 · estimated=규칙 추정 위치(반경) · present=정보카드 수량만(위치 없음)'), flags
