# -*- coding: utf-8 -*-
"""가상 건물 모델 생성기 — 정답(ground truth)을 가진 건물을 구조 유형별로 만든다.
좌표: m, x=동쪽, y=북쪽. 모든 이름·연락처는 가상."""
import math, os, random

# SYNTH_EMEV=1: 고층(지상 11층 이상) 코어형에 비상용승강기 승강장(전용 6㎡ 이상 또는 특별피난계단 부속실 겸용)을 도면에 그림.
# 기본(끔)은 기존 data/·data_hold/ 와 바이트 단위 동일(별도 난수, 기존 난수 흐름 불변).
EMEV_SYNTH = os.environ.get('SYNTH_EMEV', '') == '1'

FLOOR_NAME = lambda lv: (f'B{-lv}' if lv < 0 else f'{lv + 1}F')     # level 0 = 1F
FLOOR_KO = lambda lv: (f'지하{-lv}층' if lv < 0 else f'지상{lv + 1}층')


class B:
    def __init__(self, name, use, btype):
        self.name, self.use, self.btype = name, use, btype
        self.floors = {}          # fid -> dict
        self.stairs, self.elevs = [], []
        self.meta = {}

    def floor(self, lv, use):
        fid = FLOOR_NAME(lv)
        f = dict(id=fid, level=lv, ko=FLOOR_KO(lv), use=use, rects=[], walls=[], doors=[], rooms=[],
                 symbols=[], columns=[], racks=[], evac=[], entrances=[])
        self.floors[fid] = f
        return f

    def to_json(self):
        return dict(name=self.name, use=self.use, btype=self.btype, meta=self.meta,
                    floors=self.floors, stairs=self.stairs, elevs=self.elevs)


def wall(f, x0, y0, x1, y1, t, kind):
    """축방향 벽. kind: ext(외벽) rc(콘크리트) part(칸막이)"""
    if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
        f['walls'].append(dict(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1), t=t, kind=kind))


def wall_with_doors(f, x0, y0, x1, y1, t, kind, doors):
    """doors: [(pos_along, width, dkind, name)] — 벽을 문 위치에서 끊는다"""
    horiz = abs(y1 - y0) < 1e-6
    a0, a1 = (min(x0, x1), max(x0, x1)) if horiz else (min(y0, y1), max(y0, y1))
    cur = a0
    for pos, w, dkind, name in sorted(doors):
        s, e = pos - w / 2, pos + w / 2
        if s > cur + 0.05:
            if horiz: wall(f, cur, y0, s, y0, t, kind)
            else: wall(f, x0, cur, x0, s, t, kind)
        dx, dy = (pos, y0) if horiz else (x0, pos)
        f['doors'].append(dict(x=round(dx, 3), y=round(dy, 3), o='h' if horiz else 'v', w=w, kind=dkind, name=name))
        cur = e
    if a1 > cur + 0.05:
        if horiz: wall(f, cur, y0, a1, y0, t, kind)
        else: wall(f, x0, cur, x0, a1, t, kind)


def rect_walls(f, x0, y0, x1, y1, t, kind, doors_by_side=None):
    d = doors_by_side or {}
    wall_with_doors(f, x0, y0, x1, y0, t, kind, d.get('S', []))
    wall_with_doors(f, x0, y1, x1, y1, t, kind, d.get('N', []))
    wall_with_doors(f, x0, y0, x0, y1, t, kind, d.get('W', []))
    wall_with_doors(f, x1, y0, x1, y1, t, kind, d.get('E', []))


def room(f, x0, y0, x1, y1, name, rtype='room'):
    f['rooms'].append(dict(poly=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], name=name, type=rtype))


def sym(f, t, x, y):
    f['symbols'].append(dict(type=t, x=round(x, 2), y=round(y, 2)))


# ------------------------------------------------------------------ 복도형 (병원·숙박·요양·학교)
def gen_corridor(rng, idx):
    kind = rng.choice(['의료시설', '노유자시설', '숙박시설', '교육연구시설'])
    names = {'의료시설': '가상병원', '노유자시설': '가상요양원', '숙박시설': '가상호텔', '교육연구시설': '가상학교'}
    b = B(f'{names[kind]}{idx:03d}', kind, 'corridor')
    bay = rng.choice([7.2, 7.5, 8.1, 8.4])
    nb = rng.randint(7, 12)
    L = bay * nb
    Dr = rng.choice([6.6, 7.2, 7.5])
    C = rng.choice([2.4, 2.7, 3.0])
    depth = 2 * Dr + C
    nbase = rng.choice([1, 1, 2])
    nup = rng.randint(3, 7)
    slope = rng.random() < 0.35              # 경사지: 지하1층도 지상 진입
    cores = [1, nb - 2]
    stair_w = 3.2
    ev_w, ev_d = 2.4, 2.6
    b.meta.update(bay=bay, nb=nb, L=L, depth=depth, slope=slope)
    y_c0, y_c1 = Dr, Dr + C
    for ci, cb in enumerate(cores):
        sx = cb * bay + 0.4
        b.stairs.append(dict(id=f'ST{ci + 1}', box=[sx, y_c1, sx + stair_w, depth], floors=[]))
        b.elevs.append(dict(id=f'EV{ci + 1}', box=[sx + stair_w, y_c1, sx + stair_w + ev_w, y_c1 + ev_d], floors=[], cars=1))
    crit_floor = rng.randint(1, nup - 1) if kind == '의료시설' else None
    for lv in range(-nbase, nup):
        use = ('기계실·주차장' if lv < 0 else '로비·외래' if lv == 0 else '병동' if kind == '의료시설' else '객실' if kind == '숙박시설' else '생활실' if kind == '노유자시설' else '교실')
        f = b.floor(lv, use)
        x_min, x_max = 0.0, L
        if lv < 0:                                   # 지하는 조금 더 넓게(층별 윤곽이 다름)
            x_max = L + bay
        f['rects'].append([x_min, 0.0, x_max, depth])
        # 외벽 + 끝단 비상구
        ex_doors_W = [(Dr + C / 2, 1.8, 'exit', '비상구')]
        ex_doors_E = [(Dr + C / 2, 1.8, 'exit', '비상구')]
        south = []
        if lv == 0:
            south.append((L / 2, 2.4, 'entrance', '주출입구'))
        if lv == -1 and slope:
            south.append((bay * 2.5, 2.4, 'entrance', '지하1층 출입구'))
        rect_walls(f, x_min, 0, x_max, depth, 0.3, 'ext', {'W': ex_doors_W, 'E': ex_doors_E, 'S': south})
        for d in f['doors']:
            if d['kind'] == 'entrance':
                f['entrances'].append(dict(x=d['x'], y=d['y'], name=d['name']))
        # 복도 벽 (문 포함)
        n_doors, s_doors = [], []
        rn = 1
        room_list = []
        for k in range(int(round((x_max - x_min) / bay))):
            bx0 = x_min + k * bay
            for side in ('S', 'N'):
                is_core = side == 'N' and k in cores
                if is_core:
                    sx = k * bay + 0.4
                    n_doors.append((sx + stair_w / 2, 1.2, 'door', '계단실 문'))
                    n_doors.append((sx + stair_w + ev_w / 2, 1.0, 'door', '승강기 문'))
                    continue
                split = rng.random() < 0.3
                parts = [(bx0, bx0 + bay / 2), (bx0 + bay / 2, bx0 + bay)] if split else [(bx0, bx0 + bay)]
                for (r0, r1) in parts:
                    dpos = r0 + (r1 - r0) * rng.choice([0.25, 0.5, 0.75])
                    lst = s_doors if side == 'S' else n_doors
                    if lv == 0 and side == 'S' and abs((r0 + r1) / 2 - L / 2) < bay * 0.6:
                        lst.append(((r0 + r1) / 2, 2.4, 'door', '로비 출입'))
                        room_list.append((r0, r1, side, '로비', 'room'))
                        continue
                    lst.append((dpos, 1.0, 'door', '실 출입문'))
                    room_list.append((r0, r1, side, None, 'room'))
                    if split and r0 != parts[0][0]:
                        pass
                if split:
                    y0, y1 = (0, y_c0) if side == 'S' else (y_c1, depth)
                    wall(f, bx0 + bay / 2, y0, bx0 + bay / 2, y1, 0.15, 'part')
            if k > 0:
                wall(f, bx0, 0, bx0, y_c0, 0.15, 'part')
                if not (k in cores or (k - 1) in cores):
                    wall(f, bx0, y_c1, bx0, depth, 0.15, 'part')
        wall_with_doors(f, x_min, y_c0, x_max, y_c0, 0.2, 'rc', s_doors)
        wall_with_doors(f, x_min, y_c1, x_max, y_c1, 0.2, 'rc', n_doors)
        # 코어(계단·승강기)
        for ci, cb in enumerate(cores):
            S, E = b.stairs[ci], b.elevs[ci]
            sx0, sy0, sx1, sy1 = S['box']
            wall(f, sx0, sy0, sx0, sy1, 0.25, 'rc'); wall(f, sx1, sy0, sx1, sy1, 0.25, 'rc')
            ex0, ey0, ex1, ey1 = E['box']
            wall(f, ex1, ey0, ex1, ey1, 0.25, 'rc'); wall(f, ex0, ey1, ex1, ey1, 0.25, 'rc')
            wall(f, cb * bay + bay, y_c1, cb * bay + bay, depth, 0.2, 'rc')
            if f['id'] not in S['floors']: S['floors'].append(f['id'])
            if f['id'] not in E['floors']: E['floors'].append(f['id'])
            sym(f, 'exit', sx0 + stair_w / 2, y_c1 - 0.9)
            sym(f, 'ev', (ex0 + ex1) / 2, (ey0 + ey1) / 2)
        sym(f, 'exit', x_min + 1.2, y_c0 + C / 2)
        sym(f, 'exit', x_max - 1.2, y_c0 + C / 2)
        # 옥내소화전: 복도를 따라 수평거리 25m 이내가 되도록
        n_h = max(2, math.ceil((x_max - x_min) / 42))
        for i in range(n_h):
            hx = x_min + (x_max - x_min) * (i + 0.5) / n_h
            sym(f, 'hose', hx, y_c0 + 0.45 if i % 2 == 0 else y_c1 - 0.45)
        if kind == '의료시설' and lv >= 0:
            for i in range(rng.randint(1, 3)):
                sym(f, 'gas', x_min + (x_max - x_min) * rng.uniform(0.2, 0.8), y_c1 - 0.5)
        if lv == 0:
            sym(f, 'panel', x_min + bay * 0.5, Dr * 0.5)
        # 실 이름
        num = 1
        for (r0, r1, side, nm, rt) in room_list:
            y0, y1 = (0, y_c0) if side == 'S' else (y_c1, depth)
            if nm is None:
                if lv < 0:
                    nm = rng.choice(['기계실', '전기실', '창고', '저수조실', '주차장'])
                elif lv == 0:
                    nm = rng.choice(['원무과', '사무실', '약국', '진료실', '방재실' if num == 1 else '상담실'])
                else:
                    nm = f'{lv + 1}{num:02d}호'; rt = 'ward'
                num += 1
            room(f, r0, y0, r1, y1, nm, rt)
        if crit_floor is not None and lv == crit_floor:
            r = rng.choice([r for r in f['rooms'] if r['type'] == 'ward'])
            r['name'] = rng.choice(['중환자실', '수술실', '음압격리병실']); r['type'] = 'critical'
        # 대피 방향(빨간 화살표): 복도 중심선
        f['evac'].append([[x_min + 1.2, y_c0 + C / 2], [x_max - 1.2, y_c0 + C / 2]])
    return b


# ------------------------------------------------------------------ 대공간형 (창고·공장)
def gen_hall(rng, idx):
    kind = rng.choice(['창고시설', '공장'])
    b = B(f'가상{"물류센터" if kind == "창고시설" else "공장"}{idx:03d}', kind, 'hall')
    bay = rng.choice([9.0, 10.0, 12.0])
    nx, ny = rng.randint(6, 10), rng.randint(4, 6)
    W, D = nx * bay, ny * bay
    nup = rng.randint(1, 3)
    b.meta.update(bay=bay, W=W, D=D)
    stairs = [[0.5, D - 7.5, 4.0, D - 0.5], [W - 4.0, 0.5, W - 0.5, 7.5]]
    if rng.random() < 0.5: stairs.append([W - 4.0, D - 7.5, W - 0.5, D - 0.5])
    for i, s in enumerate(stairs):
        b.stairs.append(dict(id=f'ST{i + 1}', box=s, floors=[]))
    b.elevs.append(dict(id='EV1', box=[4.5, D - 4.0, 8.0, D - 0.5], floors=[], cars=1))
    for lv in range(0, nup):
        f = b.floor(lv, '보관·작업장')
        f['rects'].append([0, 0, W, D])
        south = [(bay * 1.5, 4.0, 'entrance', '하역장 셔터'), (bay * 3.5, 4.0, 'entrance', '하역장 셔터')] if lv == 0 else []
        west = [(D / 2, 1.8, 'entrance' if lv == 0 else 'exit', '주출입구' if lv == 0 else '비상구')]
        rect_walls(f, 0, 0, W, D, 0.3, 'ext', {'S': south, 'W': west})
        for d in f['doors']:
            if d['kind'] == 'entrance': f['entrances'].append(dict(x=d['x'], y=d['y'], name=d['name']))
        # 기둥
        for i in range(1, nx):
            for j in range(1, ny):
                f['columns'].append([i * bay, j * bay])
        # 사무실 띠(동측)
        ox0 = W - bay
        n_off = rng.randint(2, 4)
        doors = []
        for k in range(n_off):
            y0 = bay + k * (D - 2 * bay) / n_off; y1 = bay + (k + 1) * (D - 2 * bay) / n_off
            if k > 0: wall(f, ox0, y0, W, y0, 0.15, 'part')
            doors.append(((y0 + y1) / 2, 1.0, 'door', '사무실 문'))
            room(f, ox0, y0, W, y1, rng.choice(['사무실', '휴게실', '관리실', '전기실']))
        wall(f, ox0, bay, W, bay, 0.15, 'part'); wall(f, ox0, D - bay, W, D - bay, 0.15, 'part')
        wall_with_doors(f, ox0, bay, ox0, D - bay, 0.15, 'part', doors)
        room(f, 0, 0, ox0, D, '보관장' if kind == '창고시설' else '작업장', 'hall')
        # 계단·승강기
        for S in b.stairs:
            x0, y0, x1, y1 = S['box']
            side = 'E' if x0 < W / 2 else 'W'
            rect_walls(f, x0, y0, x1, y1, 0.25, 'rc', {side: [((y0 + y1) / 2, 1.2, 'door', '계단실 문')]})
            S['floors'].append(f['id'])
            sym(f, 'exit', x1 + 0.9 if side == 'E' else x0 - 0.9, (y0 + y1) / 2)
        E = b.elevs[0]; x0, y0, x1, y1 = E['box']
        rect_walls(f, x0, y0, x1, y1, 0.25, 'rc', {'S': [((x0 + x1) / 2, 1.4, 'door', '승강기 문')]})
        E['floors'].append(f['id']); sym(f, 'ev', (x0 + x1) / 2, (y0 + y1) / 2)
        # 랙(얇은 선) — 벽으로 오인하면 안 됨
        if kind == '창고시설':
            for r in range(rng.randint(3, 6)):
                ry = bay * 1.2 + r * bay * 0.7
                if ry < D - bay * 1.2:
                    f['racks'].append([bay * 0.8, ry, ox0 - bay * 0.8, ry + 1.2])
        # 소화전: 외벽·기둥열을 따라
        for i in range(max(2, int(W // 40))):
            sym(f, 'hose', (i + 0.5) * W / max(2, int(W // 40)), 0.6)
            sym(f, 'hose', (i + 0.5) * W / max(2, int(W // 40)), D - 0.6)
        if lv == 0: sym(f, 'panel', W - bay / 2, bay * 0.6)
        f['evac'].append([[bay, D / 2], [0.6, D / 2]])
    return b


# ------------------------------------------------------------------ 코어형 (업무·공동주택, 반복층)
def gen_core(rng, idx):
    kind = rng.choice(['업무시설', '공동주택'])
    b = B(f'가상{"타워" if kind == "업무시설" else "아파트"}{idx:03d}', kind, 'core')
    W, D = rng.choice([30.0, 33.6, 36.0, 40.0]), rng.choice([24.0, 27.0, 30.0])
    nbase = rng.choice([1, 2]); nup = rng.randint(6, 12)
    cw, cd = 15.0, 8.4
    cx0, cy0 = (W - cw) / 2, (D - cd) / 2
    cx1, cy1 = cx0 + cw, cy0 + cd
    b.meta.update(W=W, D=D, repeat=True)
    b.stairs += [dict(id='ST1', box=[cx0, cy0, cx0 + 3.0, cy1], floors=[]), dict(id='ST2', box=[cx1 - 3.0, cy0, cx1, cy1], floors=[])]
    nev = 2 if kind == '공동주택' else 3
    ew = 2.3
    ex0 = (W - nev * ew) / 2
    for i in range(nev):
        b.elevs.append(dict(id=f'EV{i + 1}', box=[ex0 + i * ew, cy1 - 2.6, ex0 + (i + 1) * ew, cy1], floors=[], cars=1))
    em = None
    if EMEV_SYNTH and nup >= 11:        # 건축법 제64조②(높이 31 m 초과) — 가상: 지상 11층 이상(synth/fac.py tall 과 같음)
        r2 = random.Random(idx * 101 + 3)
        shared = kind == '업무시설' and r2.random() < 0.5          # 공동주택은 세대 현관이 승강기홀에 붙어 겸용 배치 안 함
        if shared:                                                  # 특별피난계단 부속실 겸용: 계단 옆 끝 승강기, 25% 는 양쪽 끝 2대
            sides = [0, 1] if r2.random() < 0.25 else [r2.choice([0, 1])]
            ids = [0 if sd == 0 else nev - 1 for sd in sides]
        else:                                                       # 전용 승강장: 승강기 중 아무거나 1대
            ids = [r2.randrange(nev)]; sides = [None]
        label = r2.choice(['부속실', '전실', '승강장', '비상E/V', '비상용']) if r2.random() < 0.6 else ''
        em = dict(ids=ids, sides=sides, label=label, dep=2.8)       # 승강장 깊이 2.8 m × 폭 2.3 m = 6.44㎡ (설비기준규칙 제10조 6㎡ 이상)
        b.meta['emev'] = dict(ids=[f'EV{i + 1}' for i in ids], mode='부속실겸용' if shared else '전용승강장', label=label)
    PW, PD = W + 24, D + 20             # 지하 주차장 윤곽
    px0, py0 = -12.0, -10.0
    for lv in range(-nbase, nup):
        use = '주차장' if lv < 0 else ('로비' if lv == 0 else ('사무실' if kind == '업무시설' else '세대'))
        f = b.floor(lv, use)
        if lv < 0:
            f['rects'].append([px0, py0, px0 + PW, py0 + PD])
            ent = [(px0 + PW * 0.25, 6.0, 'entrance', '주차장 차량진입로')] if lv == -1 else []
            rect_walls(f, px0, py0, px0 + PW, py0 + PD, 0.35, 'ext', {'S': ent})
            for i in range(1, int(PW // 8.4)):
                for j in range(1, int(PD // 8.4)):
                    c = [px0 + i * 8.4, py0 + j * 8.4]
                    if not (cx0 - 1 < c[0] < cx1 + 1 and cy0 - 1 < c[1] < cy1 + 1):
                        f['columns'].append(c)
            room(f, px0, py0, px0 + PW, py0 + PD, '주차장', 'hall')
        else:
            f['rects'].append([0, 0, W, D])
            ent = [(W / 2, 2.4, 'entrance', '주출입구')] if lv == 0 else []
            rect_walls(f, 0, 0, W, D, 0.3, 'ext', {'S': ent})
            if kind == '공동주택' and lv > 0:           # 세대 4개
                for (x0, y0, x1, y1, nm) in [(0, 0, W / 2, cy0, '01호'), (W / 2, 0, W, cy0, '02호'), (0, cy1, W / 2, D, '03호'), (W / 2, cy1, W, D, '04호')]:
                    room(f, x0, y0, x1, y1, f'{lv + 1}{nm}', 'ward')
                wall_with_doors(f, 0, cy0, W, cy0, 0.2, 'rc', [(W / 2 - 3, 1.0, 'door', '세대 현관'), (W / 2 + 3, 1.0, 'door', '세대 현관')])
                wall_with_doors(f, 0, cy1, W, cy1, 0.2, 'rc', [(W / 2 - 3, 1.0, 'door', '세대 현관'), (W / 2 + 3, 1.0, 'door', '세대 현관')])
                wall(f, W / 2, 0, W / 2, cy0, 0.2, 'rc'); wall(f, W / 2, cy1, W / 2, D, 0.2, 'rc')
                wall(f, 0, cy0, 0, cy1, 0.3, 'ext')
                wall(f, cx0, cy0, cx0, cy1, 0.2, 'rc'); wall(f, cx1, cy0, cx1, cy1, 0.2, 'rc')
                room(f, cx0 + 3, cy0, cx1 - 3, cy1 if not em else cy1 - 2.6 - em['dep'], '승강기홀')     # 승강장이 있으면 홀 이름은 그 남쪽(글자 겹침 방지)
            else:
                room(f, 0, 0, W, D, '로비' if lv == 0 else '사무공간', 'hall')
                rect_walls(f, cx0, cy0, cx1, cy1, 0.25, 'rc', {'S': [(cx0 + 5, 1.2, 'door', '코어 출입'), (cx1 - 5, 1.2, 'door', '코어 출입')]})
                if kind == '업무시설' and lv > 0:
                    for k in range(rng.randint(1, 3)):
                        x0 = rng.choice([0.0, W - 6.0]); y0 = rng.uniform(0, D - 6)
                        rect_walls(f, x0, y0, x0 + 6, y0 + 6, 0.12, 'part', {'E' if x0 == 0 else 'W': [(y0 + 3, 0.9, 'door', '회의실 문')]})
                        room(f, x0, y0, x0 + 6, y0 + 6, '회의실')
        for S in b.stairs:
            x0, y0, x1, y1 = S['box']
            side = 'E' if x0 < W / 2 else 'W'
            wall_with_doors(f, x1 if side == 'E' else x0, y0, x1 if side == 'E' else x0, y1, 0.25, 'rc', [((y0 + y1) / 2, 1.2, 'door', '계단실 문')])
            S['floors'].append(f['id'])
            sym(f, 'exit', x1 + 0.8 if side == 'E' else x0 - 0.8, (y0 + y1) / 2 + 1.5)
        for E in b.elevs:
            x0, y0, x1, y1 = E['box']
            wall(f, x0, y0, x1, y0, 0.2, 'rc')
            E['floors'].append(f['id']); sym(f, 'ev', (x0 + x1) / 2, (y0 + y1) / 2)
        if em:                                                      # 비상용승강기 승강장(승강기 앞, 문 1개, 방화구획 벽)
            for i, sd in zip(em['ids'], em['sides']):
                x0, y0, x1, y1 = b.elevs[i]['box']
                ya = y0 - em['dep']
                xa, xb = (cx0 + 3.0, x1) if sd == 0 else (x0, cx1 - 3.0) if sd == 1 else (x0, x1)
                wall_with_doors(f, xa, ya, xb, ya, 0.2, 'rc', [((xa + xb) / 2, 1.0, 'door', '승강장 문')])
                if sd != 0: wall(f, xa, ya, xa, y0, 0.2, 'rc')
                if sd != 1: wall(f, xb, ya, xb, y0, 0.2, 'rc')
                if sd is not None:                                  # 겸용: 승강로 옆벽(계단~승강기 틈이 승강로 띠로 새지 않게) + 지하(코어 벽 없음) 틈 북측 벽
                    wall(f, x0 if sd == 0 else x1, y0, x0 if sd == 0 else x1, y1, 0.2, 'rc')
                    if lv < 0: wall(f, xa, y1, x0, y1, 0.2, 'rc') if sd == 0 else wall(f, x1, y1, xb, y1, 0.2, 'rc')
                room(f, xa, ya, xb, y0, em['label'])
        sym(f, 'hose', cx0 + 4.0, cy0 - 0.5)
        sym(f, 'hose', cx1 - 4.0, cy1 + 0.5)
        if lv < 0:
            sym(f, 'hose', px0 + 4, py0 + 4); sym(f, 'hose', px0 + PW - 4, py0 + PD - 4)
        if lv == 0: sym(f, 'panel', 2.0, 2.0)
    return b


GENS = {'corridor': gen_corridor, 'hall': gen_hall, 'core': gen_core}


def generate(btype, seed):
    rng = random.Random(seed)
    return GENS[btype](rng, seed)
