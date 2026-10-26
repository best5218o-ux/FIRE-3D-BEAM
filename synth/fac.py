# -*- coding: utf-8 -*-
"""가상 건물 소방시설(추가분) — 정보카드 '층별 소방시설 설치현황' 항목 + 평면도 새 범례 기호(소화기·방수구·비상콘센트·송수구) + 정답.
기존 기호·정답·난수 흐름은 건드리지 않는다(별도 난수, 기존 요소를 피해 빈 자리에만 그림).
설치 규칙(가상): 시행령 [별표 4](2023.12.1.) — 스프링클러 6층 이상 전층, 연결송수관 5층 이상&연면적 6천㎡ 이상 또는 지하 포함 7층 이상,
비상콘센트 11층 이상 층 / NFPC 502 제6조(방수구 계단 5 m 이내, 1,000㎡ 이상 층은 계단 2개, 11층 이상 쌍구형, 아파트 1·2층 제외 가능)
/ NFTC 504 2.1.5.2(비상콘센트 계단 출입구 5 m 이내) / 건축법 제64조(높이 31 m 초과 → 비상용승강기, 가상: 지상 11층 이상)"""
import math, random

HS = 0.7           # 새 기호 반폭(m) — 도면 기호 한 변 1.4 m
FAC_TYPES = [('ext', '소화기'), ('outlet', '방수구'), ('socket', '비상콘센트'), ('siamese', '송수구')]


def _area(f): return sum((r[2] - r[0]) * (r[3] - r[1]) for r in f['rects'])


def _floor_bounds(b, f):
    xs, ys = [], []
    for r in f['rects']: xs += [r[0], r[2]]; ys += [r[1], r[3]]
    for S in b.stairs + b.elevs:
        if f['id'] in S['floors']: xs += [S['box'][0], S['box'][2]]; ys += [S['box'][1], S['box'][3]]
    return min(xs), min(ys), max(xs), max(ys)


def _sep(x, y, bx0, by0, bx1, by1):
    """기호 사각(반폭 HS)과 상자 사이 간격(m, 겹치면 음수)"""
    dx = max(bx0 - (x + HS), (x - HS) - bx1); dy = max(by0 - (y + HS), (y - HS) - by1)
    return max(dx, dy)


def obstacles(b, f):
    """기호가 피해야 할 것들: (상자, 여유 m)"""
    ob = []
    for w in f['walls']:
        t = w['t'] / 2
        ob.append(((w['x0'] - t, w['y0'] - t, w['x1'] + t, w['y1'] + t), 0.4))
    for d in f['doors']:
        w = d['w']
        bx = (d['x'] - w / 2 - 0.1, d['y'] - w - 0.1, d['x'] + w / 2 + 0.1, d['y'] + w + 0.1) if d['o'] == 'h' else \
             (d['x'] - w - 0.1, d['y'] - w / 2 - 0.1, d['x'] + w + 0.1, d['y'] + w / 2 + 0.1)
        ob.append((bx, 0.3))
    for s in f['symbols']: ob.append(((s['x'] - 1.05, s['y'] - 1.05, s['x'] + 1.05, s['y'] + 1.05), 0.35))
    for S in b.stairs + b.elevs:
        if f['id'] in S['floors']: ob.append((tuple(S['box']), 0.35))
    for c in f['columns']: ob.append(((c[0] - 0.35, c[1] - 0.35, c[0] + 0.35, c[1] + 0.35), 0.4))
    for r in f['racks']: ob.append((tuple(r), 0.3))
    for rm in f['rooms']:
        xs = [p[0] for p in rm['poly']]; ys = [p[1] for p in rm['poly']]
        if rm['type'] == 'critical': ob.append(((min(xs), min(ys), max(xs), max(ys)), 0.3)); continue
        h = 1.5 if rm['type'] == 'critical' else (0.95 if rm['type'] == 'ward' else 0.75)
        tw = sum(h if '가' <= ch <= '힣' else 0.6 * h for ch in rm['name'])
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        ob.append(((cx - tw / 2, cy - h * 0.6, cx + tw / 2, cy + h * 0.6), 0.25))
    x0, y0, x1, y1 = _floor_bounds(b, f)
    bay = b.meta.get('bay', 8.4)
    gx = [x0 + k * bay for k in range(int((x1 - x0) / bay) + 1)]
    gy = sorted(set([y0, y1] + [y0 + k * bay for k in range(int((y1 - y0) / bay) + 1)]))
    return ob, gx, gy


def free(x, y, ob, gx, gy, placed, inside=None):
    if any(abs(x - g) < HS + 0.4 for g in gx) or any(abs(y - g) < HS + 0.4 for g in gy): return False
    if any(math.hypot(x - p[0], y - p[1]) < 2.2 for p in placed): return False
    if inside is not None and not inside(x, y): return False
    return all(_sep(x, y, *bx) >= m for bx, m in ob)


def _inside_fn(f, margin=0.9):
    def g(x, y):
        return any(r[0] + margin <= x <= r[2] - margin and r[1] + margin <= y <= r[3] - margin for r in f['rects'])
    return g


def search(tx, ty, R, ob, gx, gy, placed, inside, anchor=None, rmax=None):
    """목표점 근처 나선 탐색 → 첫 빈 자리. anchor 가 있으면 anchor 로부터 rmax 이내만"""
    cands = []
    step = 0.25
    n = int(R / step)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = tx + i * step, ty + j * step
            d = math.hypot(i * step, j * step)
            if d > R: continue
            if anchor is not None and math.hypot(x - anchor[0], y - anchor[1]) > rmax: continue
            cands.append((d, x, y))
    cands.sort()
    for d, x, y in cands:
        if free(x, y, ob, gx, gy, placed, inside): return (round(x, 2), round(y, 2))
    return None


def stair_door(f, S):
    """계단 상자에 가장 가까운 '계단실 문'(정답) → (x, y, 바깥쪽 법선)"""
    x0, y0, x1, y1 = S['box']
    best = None
    for d in f['doors']:
        if '계단' not in d['name']: continue
        dist = max(x0 - d['x'], d['x'] - x1, 0) + max(y0 - d['y'], d['y'] - y1, 0)
        if dist < 0.6 and (best is None or dist < best[0]): best = (dist, d)
    if best is None: return None
    d = best[1]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if d['o'] == 'h': nx, ny = 0.0, (1.0 if d['y'] >= cy else -1.0)
    else: nx, ny = (1.0 if d['x'] >= cx else -1.0), 0.0
    return d['x'], d['y'], nx, ny


def plan(b, groups, seed):
    """→ dict(draw={type:bool}, pos={fid:[{code,x,y,drawn,twin,stair}]}, sym={group_first_fid:[(type,x,y)]}, items={fid:[(카드 항목명, 수)]}, flags=[...], emev=...)"""
    rng = random.Random(seed * 13 + 7)
    fl = sorted(b.floors.values(), key=lambda f: f['level'])
    nb = sum(1 for f in fl if f['level'] < 0); nu = sum(1 for f in fl if f['level'] >= 0)
    gfa = sum(_area(f) for f in fl)
    apt = b.use == '공동주택'
    standpipe = (nu >= 5 and gfa >= 6000) or (nb + nu >= 7)
    sp_all = nu >= 6
    sp_use = b.use in ('의료시설', '노유자시설', '숙박시설') or (b.use == '창고시설' and gfa >= 5000)
    tall = nu >= 11
    draw = dict(ext=rng.random() < 0.65, outlet=standpipe and rng.random() < 0.6, socket=tall and rng.random() < 0.7, siamese=standpipe and rng.random() < 0.7)
    sp_name = rng.choice(['스프링클러헤드', 'S/P헤드', '스프링클러설비(헤드)', '스프링클러'])
    sp_twin_note = rng.random() < 0.3
    pos, items, flags = {}, {}, []
    # 층마다 설치 대상·개수(정답)
    need = {}
    for f in fl:
        lv, a = f['level'], _area(f)
        st = [S for S in b.stairs if f['id'] in S['floors']]
        n_st = len(st)
        k_sp = (1 if (apt or a < 1000) else min(2, n_st)) if standpipe and not (apt and lv in (0, 1)) else 0
        k_oc = (1 if a < 1000 else min(2, n_st)) if (tall and lv >= 10) else 0
        need[f['id']] = dict(STANDPIPE=min(k_sp, n_st), OUTLET=min(k_oc, n_st))
    # 기호 위치: 도면 묶음(기준층)마다 한 번 계산해 같은 도면 층에 복사
    sym = {}
    for g in groups:
        f = b.floors[g[0]]
        ob, gx, gy = obstacles(b, f)
        inside = _inside_fn(f)
        placed = []
        st = [S for S in b.stairs if f['id'] in S['floors']]
        doors = [(S, stair_door(f, S)) for S in st]
        doors = [(S, d) for S, d in doors if d]
        gpos = {'STANDPIPE': [], 'OUTLET': [], 'EXT': [], 'SIAMESE': []}
        kmax = {c: max(need[fid][c] for fid in g) for c in ('STANDPIPE', 'OUTLET')}
        for code, off in (('STANDPIPE', 1.6), ('OUTLET', 2.4)):
            for S, (dx, dy, nx, ny) in doors[:kmax[code]]:
                # 문 앞(바닥 쪽) 1.6~2.4 m, 문에서 4.3 m 이내 — 기호가 계단 5 m 반경 안에 들도록
                side = rng.choice([-1, 1])
                tx, ty = dx + nx * off + (ny * side * 1.2), dy + ny * off + (nx * side * 1.2)
                p = search(tx, ty, 3.5, ob, gx, gy, placed, inside, anchor=(dx, dy), rmax=4.3)
                if p is None:            # 기호 놓을 빈 자리가 없으면 정답 위치만(도면 미표기)
                    p = (round(dx + nx * 1.2, 2), round(dy + ny * 1.2, 2)); ok = False
                else: ok = True; placed.append(p)
                gpos[code].append(dict(x=p[0], y=p[1], stair=S['id'], fits=ok))
        # 소화기: 흩어진 빈 자리 2~4곳
        if draw['ext']:
            x0, y0, x1, y1 = _floor_bounds(b, f)
            want = rng.randint(2, 4); tries = 0
            while len(gpos['EXT']) < want and tries < 400:
                tries += 1
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if free(x, y, ob, gx, gy, placed, inside) and all(math.hypot(x - q['x'], y - q['y']) > 8 for q in gpos['EXT']):
                    placed.append((x, y)); gpos['EXT'].append(dict(x=round(x, 2), y=round(y, 2), fits=True))
        # 송수구: 1층 외벽 바깥(남측) — 주출입구 옆
        if standpipe and f['level'] == 0:
            ents = [d for d in f['doors'] if d['kind'] == 'entrance']
            x0, y0, x1, y1 = _floor_bounds(b, f)
            ex = ents[0]['x'] if ents else (x0 + x1) / 2
            ob2 = [(bx, m) for bx, m in ob]
            p = search(ex + 4.0, y0 - 1.7, 6.0, ob2, gx, gy, placed, lambda x, y: y < y0 - 1.1)
            if p: placed.append(p); gpos['SIAMESE'].append(dict(x=p[0], y=p[1], fits=True))
        for fid in g:
            P = []
            for code in ('STANDPIPE', 'OUTLET'):
                for q in gpos[code][:need[fid][code]]:
                    P.append(dict(code=code, x=q['x'], y=q['y'], stair=q['stair'], twin=code == 'STANDPIPE' and b.floors[fid]['level'] >= 10, fits=q['fits']))
            for code in ('EXT', 'SIAMESE'):
                for q in gpos[code]: P.append(dict(code=code, x=q['x'], y=q['y'], fits=True))
            pos[fid] = P
        # 이 도면에 그릴 기호: 묶음 전 층이 같은 개수이고 빈 자리를 찾은 것만
        S_ = []
        for code, t in (('STANDPIPE', 'outlet'), ('OUTLET', 'socket')):
            same = len(set(need[fid][code] for fid in g)) == 1
            if draw[t] and same and gpos[code][:need[g[0]][code]] and all(q['fits'] for q in gpos[code][:need[g[0]][code]]):
                S_ += [(t, q['x'], q['y']) for q in gpos[code][:need[g[0]][code]]]
        if draw['ext']: S_ += [('ext', q['x'], q['y']) for q in gpos['EXT']]
        if draw['siamese']: S_ += [('siamese', q['x'], q['y']) for q in gpos['SIAMESE']]
        sym[g[0]] = S_
        for fid in g:
            drawn = {(t, x, y) for t, x, y in S_}
            for q in pos[fid]:
                t = {'STANDPIPE': 'outlet', 'OUTLET': 'socket', 'EXT': 'ext', 'SIAMESE': 'siamese'}[q['code']]
                q['drawn'] = (t, q['x'], q['y']) in drawn
            if not draw['ext']: pos[fid] = [q for q in pos[fid] if q['code'] != 'EXT']
    # 비상용승강기(정답): 계단 문에 가장 가까운 승강기
    emev = None
    if tall and b.elevs:
        f1 = b.floors['1F']
        sd = [stair_door(f1, S) for S in b.stairs if '1F' in S['floors']]
        sd = [d for d in sd if d]
        def dmin(E):
            cx, cy = (E['box'][0] + E['box'][2]) / 2, (E['box'][1] + E['box'][3]) / 2
            return min(math.hypot(cx - d[0], cy - d[1]) for d in sd) if sd else 0
        E = min(b.elevs, key=dmin)
        emev = dict(id=E['id'], x=round((E['box'][0] + E['box'][2]) / 2, 2), y=round((E['box'][1] + E['box'][3]) / 2, 2), floors=list(E['floors']))
        me = b.meta.get('emev')
        if me:                               # SYNTH_EMEV: 모델이 승강장까지 그린 비상용승강기(1~2대) — 계단 최근접 동률(대칭 코어) 대신 실제 배치
            Es = [E for E in b.elevs if E['id'] in me['ids']]
            C = lambda E: dict(id=E['id'], x=round((E['box'][0] + E['box'][2]) / 2, 2), y=round((E['box'][1] + E['box'][3]) / 2, 2), floors=list(E['floors']))
            emev = dict(C(Es[0]), all=[C(E) for E in Es], mode=me['mode'], label=me['label'])
    return dict(draw=draw, pos=pos, sym=sym, flags=flags, emev=emev, standpipe=standpipe, tall=tall, _rng=rng, sp_name=sp_name, sp_twin_note=sp_twin_note, gfa=gfa, nb=nb, nu=nu, apt=apt, sp_all=sp_all, sp_use=sp_use)


def card_items(b, P):
    """렌더 뒤(도면에 실제로 그려진 기호 확정 후) 정보카드 추가 항목(층별) + 심은 수량 불일치"""
    rng, pos, flags, emev = P['_rng'], P['pos'], P['flags'], P['emev']
    sp_name, sp_twin_note, gfa, nb, nu, apt, sp_all, sp_use, tall = (P[k] for k in ('sp_name', 'sp_twin_note', 'gfa', 'nb', 'nu', 'apt', 'sp_all', 'sp_use', 'tall'))
    fl = sorted(b.floors.values(), key=lambda f: f['level'])
    items = {}
    for f in fl:
        fid, lv, a = f['id'], f['level'], _area(f)
        it = []
        if sp_all or sp_use or (lv < 0 and nu >= 6):
            it.append((sp_name, max(1, int(round(a / 9.0 * rng.uniform(0.9, 1.1))))))
        elif lv < 0:
            it.append(('포소화설비(헤드)', max(1, int(round(a / 9.0 * rng.uniform(0.9, 1.1))))))
        if lv == -1 and b.btype != 'hall':
            it.append((rng.choice(['이산화탄소소화설비', '할로겐화합물및불활성기체소화설비', '청정소화약제소화설비']), 1))
            it.append(('자동확산소화기', rng.randint(1, 3)))
        if apt and lv >= 1: it.append(('주거용주방자동소화장치', 4))
        if tall or gfa >= 3500: it.append((rng.choice(['비상방송(스피커)', '비상방송설비']), max(2, int(a // 60))))
        if b.use in ('의료시설', '노유자시설', '숙박시설', '업무시설', '교육연구시설') and lv >= 0:
            it.append(('시각경보기', max(1, int(a // 150))))
        if lv == 0 and b.use in ('의료시설', '노유자시설'): it.append(('자동화재속보설비', 1))
        if apt and lv >= 1: it.append(('가스누설경보기', 4))
        if 2 <= lv <= 9 and b.btype != 'hall':
            nm = {'의료시설': '구조대', '노유자시설': '구조대', '숙박시설': '완강기', '공동주택': '하향식피난구', '교육연구시설': '완강기'}.get(b.use, '완강기')
            it.append((nm, rng.randint(1, 3) if nm != '하향식피난구' else 4))
        if lv == 0 and nb + nu >= 7 and b.use in ('의료시설', '숙박시설'): it.append(('공기호흡기', 2))
        if (tall and lv >= 0) or (b.use == '의료시설' and lv >= 1):
            it.append((rng.choice(['제연설비(부속실)', '부속실제연']), len([S for S in b.stairs if fid in S['floors']])))
        n_sp = sum(1 for q in pos.get(fid, []) if q['code'] == 'STANDPIPE')
        if n_sp:
            c = n_sp
            if any(q['drawn'] for q in pos[fid] if q['code'] == 'STANDPIPE') and rng.random() < 0.15:
                c += 1; flags.append(dict(kind='fac_count', code='STANDPIPE', floor=fid))
            it.append((rng.choice(['연결송수관(방수구)', '방수구', '연결송수관설비(방수구)']) + ('(쌍구형)' if lv >= 10 and sp_twin_note else ''), c))
        nd = sum(1 for q in pos.get(fid, []) if q['code'] == 'STANDPIPE' and q['drawn'])
        if n_sp and 0 < nd != n_sp and not any(f_['floor'] == fid and f_['code'] == 'STANDPIPE' for f_ in flags):
            flags.append(dict(kind='fac_count', code='STANDPIPE', floor=fid, cause='도면 일부 미표기'))        # 빈 자리가 없어 일부만 그려짐 → 실제로도 불일치
        n_oc = sum(1 for q in pos.get(fid, []) if q['code'] == 'OUTLET')
        if n_oc: it.append(('비상콘센트', n_oc))
        n_si = sum(1 for q in pos.get(fid, []) if q['code'] == 'SIAMESE')
        if n_si: it.append(('연결송수관(송수구)', n_si))
        if lv < 0 and b.btype == 'core': it.append(('무선통신보조설비', max(1, int(a // 800))))
        if emev and fid in emev['floors']: it.append(('비상용승강기', len(emev.get('all') or [emev])))
        if b.btype == 'hall' and rng.random() < 0.5: it.append(('방화셔터', rng.randint(1, 4)))       # 소방시설 아님(건축 방화설비) → '기타'
        items[fid] = it
    return items
