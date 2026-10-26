# -*- coding: utf-8 -*-
"""인명검색 우선순위 — 실 용도 범주 추정 + 기준값(한 곳) + 화점 미정 시 층별 정적 순서.

화점이 정해진 뒤의 순서(화재층 대피 곤란 분류·공격계단 기준 보행거리)는 3D 지휘화면 viewer/src/search.js 가 계산하고,
여기 SEARCH 기준값(D['sop_search'])과 실마다 붙인 용도 범주(D['evac'].floors[f].rooms[k].cat/cw/why)를 그대로 쓴다
→ 가중치는 이 파일 한 곳에서만 바꾼다.

정렬 = 사전식(계층) 키 — 점수 합이 아니라 아래 순서로 비교한다(설명 가능·결정적):
  ① 층 순위: 화재층 → 직상층 → 그 위 층(floors_above 개 층까지)
  ② 대피 곤란 분류 순위(화재층: 고립 → 화재실 → 공격계단 공용 → 화점 인접 → 우회 증가 → 정상,
     위층: 경로 없음 → 화점 직상부 → 평시 대피 곤란(기준 초과·한 방향) → 정상)
  ③ 용도·시간대 가중치 등급(높음/보통/낮음 — cat_weight[범주][주간|야간] + 거동 곤란 층 가산)
  ④ 공격계단(위층은 공격계단 또는 그 층 계단)에서 실 중심까지 이동 비용 격자 보행거리
  ⑤ 실 번호(동률 고정)
모든 수치는 **임시값** — 기관 SOP(인명검색 지침)로 정해야 한다. 실 이름이 없으면(가상 정보카드 도면은 실명 OCR 안 함) 정보카드 ⑩ 층 용도 + 실 모양으로 추정."""
import math, re

from firebim.sop import SEARCH            # 기준값은 sop.py 한 곳(SEARCH · ACT · TRAIN)

CLASS_KO = {'iso': '고립(대피로 없음)', 'fire': '화재실(화점 구역)', 'shared': '공격계단 공용', 'near': '화점 인접 경로', 'detour': '우회 증가',
            'ok': '정상', 'none': '대피 경로 없음(평시)', 'above': '화점 직상부', 'hard': '평시 대피 곤란(기준 초과·한 방향)'}
CAT_KO = {'crit': '거동 불가 실', 'sleep': '취침 가능 실', 'occupied': '주간 재실 실', 'hall': '대공간', 'corr': '복도·홀', 'svc': '설비·창고', 'unk': '용도 모름'}

# 실 이름 어휘(실명이 있을 때 — 실제 도면 실명 OCR·검수 입력). 앞 항목 우선
NAME_VOCAB = [
    ('crit', ('중환자', '수술', '격리', '신생아', '분만', '회복실', '투석', '음압', '집중치료')),
    ('svc', ('기계실', '전기실', '발전기', '창고', '저수조', '주차', '화장실', '욕실', '샤워', '탕비', 'PS', 'EPS', 'AD', '덕트', '쓰레기', '방재실', '저장',
             # (2026-10-03 비주거 실명 추가)
             'TPS', 'DA', '펌프', '물탱크', '전산실', 'UPS', '정화조', '보일러', '공조', '설비', '청소', '세탁', '오수')),
    ('corr', ('복도', '홀', '현관', '전실', '부속실', '승강장', '통로', '로비', '라운지', '풍실')),
    ('sleep', ('침실', '병실', '객실', '숙직', '생활실', '세대', '안방', '방', '숙소', '기숙', '요양실', '입원', '당직실', '입소')),
    ('hall', ('작업장', '보관장', '공장', '물류', '하역')),
    ('occupied', ('사무', '교실', '강의', '진료', '상담', '약국', '원무', '회의', '거실', '주방', '식당', '매장', '판매', '휴게', '관리실', '외래', '대기',
                  # (2026-10-03 비주거 실명 추가: 복지·교육·의료 주간 시설)
                  '강당', '프로그램', '치료', '재활', '교육', '도서', '열람', '체력', '헬스', '다목적', '놀이', '보육', '집회', '예배', '공연', '전시', '카페', '매점', '조리')),
]
# 노유자시설 세부 용도(2026-10-03): 주간 이용(복지관·주간보호 — 취침 아님) / 숙박형(요양원 — 취침). 둘 다 없으면 '노유자' = 취침 가능 가정(보수적)
CARE_RES = ('요양원', '요양시설', '노인요양', '요양병원', '양로', '그룹홈', '공동생활', '보육원', '단기보호', '생활시설', '입소', '쉼터', '숙소')
CARE_DAY = ('주간보호', '주야간보호', '데이케어', '복지관', '복지센터', '장애인복지', '노인복지관', '경로당', '노인교실', '어린이집', '유치원', '아동센터',
            '지역아동', '재활', '프로그램', '교육', '상담', '식당', '로비', '강당', '사무')


def care_kind(*texts):
    """노유자시설 세부 용도 → 'res'(숙박형·취침) | 'day'(주간 이용) | None(모름)"""
    t = ' '.join(x or '' for x in texts)
    if any(w in t for w in CARE_RES): return 'res'
    if any(w in t for w in CARE_DAY): return 'day'
    return None
# 정보카드 ⑩ 층 용도 어휘 → 층 범주
FLOOR_VOCAB = [
    ('svc', ('주차', '기계', '전기', '설비', '저수', '물탱크')),
    ('sleep', ('병동', '병실', '객실', '숙박', '세대', '주거', '공동주택', '아파트', '생활실', '침실', '기숙', '요양', '숙소', '입원', '노유자')),
    ('hall', ('작업장', '보관', '창고', '공장', '물류')),
    ('occupied', ('사무', '업무', '교실', '강의', '학원', '판매', '근린', '외래', '로비', '진료', '영업', '집회', '식당', '교육')),
]
MOBILITY = ('병동', '병실', '요양', '노유자', '중환자', '입원', '생활실', '의료')


def _match(text, vocab):
    t = text or ''
    for cat, words in vocab:
        for w in words:
            if w in t: return cat, w
    return None, None


def _shape(poly):
    """(외접 사각형 길이비, 짧은 변, 평균 폭 2A/P, 반둘레 P/2) — 복도 판정용"""
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    A = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]))) / 2
    P = sum(math.dist(p, q) for p, q in zip(poly, poly[1:] + poly[:1]))
    return max(w, h) / max(0.1, min(w, h)), min(w, h), 2 * A / max(P, 1e-6), P / 2


def classify(room, floor_use, bldg_use='', floor_area=None, bldg_name=''):
    """실 1개 → (범주, 근거 문장). 실명 우선, 없으면 층 용도(정보카드 ⑩) + 모양(긴 실 = 복도, 큰 실 = 대공간).
    노유자시설은 세부 용도(층 용도·주용도·상호 글자)로 주간 이용(복지관·주간보호 → 주간 재실) / 숙박형(요양원 → 취침)을 나눔"""
    name = (room.get('name') or '').strip()
    if name and not re.fullmatch(r'실\s*\d+', name):
        cat, w = _match(name, NAME_VOCAB)
        if cat: return cat, f"실명 '{name}'('{w}') → {CAT_KO[cat]}"
    elong, minw, mw, half = _shape(room['poly']); area = room.get('area') or 0
    fcat, fw = _match(floor_use, FLOOR_VOCAB)
    src = f"정보카드 ⑩ 층 용도 '{floor_use}'" if floor_use else ''
    if not fcat and bldg_use:
        fcat, fw = _match(bldg_use, FLOOR_VOCAB); src = f"주용도 '{bldg_use}'"
    if (elong >= 4.0 and minw <= 3.5 and elong * minw >= 8.0) or (mw <= 4.0 and half >= 20.0):     # 곧은 복도 · 꺾이거나 홀이 붙은 복도(평균 폭 2A/P 좁고 김)
        return 'corr', f"가늘고 긴 실(평균 폭 {mw:.1f}m·길이 약 {half:.0f}m) → 복도"
    if fcat == 'svc':
        return 'svc', f"{src} → 설비·주차"
    if fcat == 'hall':                                   # 대공간 층: 층 실 면적 합의 30% 이상(또는 1,000㎡ 이상)인 실만 대공간, 나머지는 사무·휴게
        big = area >= 1000 or (floor_area and area >= 0.3 * floor_area)
        if not big: return 'occupied', f"{src} 층의 부속 실({area:.0f}㎡) → 사무·휴게 추정"
        return 'hall', f"{src} · {area:.0f}㎡ → 대공간"
    if fcat == 'sleep' and fw == '노유자':
        ck = care_kind(floor_use, bldg_use, bldg_name)
        if ck == 'day': return 'occupied', f"{src} · 노유자시설 주간 이용(복지관·주간보호 등, 취침 아님) · {area:.0f}㎡ 실 → {CAT_KO['occupied']} 추정"
        if ck is None: return 'sleep', f"{src} · 노유자시설(세부 용도 미상 → 취침 가능 가정) · {area:.0f}㎡ 실 → {CAT_KO['sleep']} 추정"
    if fcat in ('sleep', 'occupied'):
        return fcat, f"{src} · {area:.0f}㎡ 실 → {CAT_KO[fcat]} 추정"
    if area >= 300:
        return 'hall', f"용도 모름 · {area:.0f}㎡ → 대공간"
    return 'unk', '층 용도·실명 없음'


def weight(cat, mode, floor_use='', cfg=SEARCH, day_care=False):
    w = cfg['cat_weight'].get(cat, cfg['cat_weight']['unk'])[1 if mode == 'night' else 0]
    if any(k in (floor_use or '') for k in MOBILITY) and cat in ('sleep', 'crit', 'unk'):
        w += cfg['mobility_bonus']
    elif day_care and cat == 'occupied' and mode == 'day':          # 주간 이용 노유자(장애인·노인 복지관 등): 낮에는 거동 곤란 재실자
        w += cfg['mobility_bonus']
    return round(w, 3)


def band(w, cfg=SEARCH):
    b0, b1 = cfg['bands']
    return 0 if w >= b0 else 1 if w >= b1 else 2


def annotate(D, cfg=SEARCH):
    """지휘화면 데이터 D(evac.to_viewer 결과 포함)의 실마다 cat·cw[주간,야간]·why 추가 + D['sop_search'] = 기준값.
    대피로(D['evac'])가 없으면 아무것도 안 함."""
    EV = D.get('evac')
    D['sop_search'] = cfg
    if not EV: return D
    use = {f['id']: f.get('use') or '' for f in D.get('floors') or []}
    bu = (D.get('building') or {}).get('use') or ''
    bn = (D.get('building') or {}).get('name') or ''
    for fid, rec in EV['floors'].items():
        fa = sum(r.get('area') or 0 for r in rec['rooms'])
        dc = ('노유자' in use.get(fid, '') or '노유자' in bu) and care_kind(use.get(fid, ''), bu, bn) == 'day'
        for r in rec['rooms']:
            cat, why = classify(r, use.get(fid, ''), bu, fa, bn)
            r['cat'] = cat; r['why'] = why
            r['cw'] = [weight(cat, 'day', use.get(fid, ''), cfg, dc), weight(cat, 'night', use.get(fid, ''), cfg, dc)]
    return D


def static_class(r, n_targets):
    """화점 없이 알 수 있는 대피 곤란(위층 분류와 같은 규칙, 'above' 제외)"""
    if r.get('d') is None: return 'none'
    if r.get('over') == 'high' or (r.get('n_reach') == 1 and n_targets >= 2): return 'hard'
    return 'ok'


def static_order(D, mode='night', cfg=SEARCH, top=8):
    """화점 미정 사전 활동계획용: 층마다 (평시 대피 곤란 분류, 용도 등급, 계단까지 보행거리) 순 상위 실.
    화점이 정해지면 화재층 분류·공격계단 기준 거리로 3D 지휘화면이 다시 계산한다."""
    EV = D.get('evac') or {}
    use = {f['id']: f.get('use') or '' for f in D.get('floors') or []}
    order = cfg['class_order_upper']
    out = {}
    for fid, rec in (EV.get('floors') or {}).items():
        rows = []
        for k, r in enumerate(rec['rooms']):
            if 'cat' not in r: continue
            cl = static_class(r, len(rec.get('targets') or []))
            w = r['cw'][1 if mode == 'night' else 0]
            d = r.get('d')
            rows.append(dict(k=k, cls=cl, cat=r['cat'], w=w, band=band(w, cfg), d=d, area=r.get('area'), c=r.get('c'), why=r.get('why'),
                             key=(order.index(cl), band(w, cfg), d if d is not None else 1e9, k)))
        rows.sort(key=lambda z: z['key'])
        out[fid] = rows[:top]
    return out
