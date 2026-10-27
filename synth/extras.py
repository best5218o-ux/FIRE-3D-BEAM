# -*- coding: utf-8 -*-
"""가상 건축물대장(daejang.json) + 가상 부지(site.json) 생성 — 기존 card.pdf·정답은 그대로 두고 추가만 한다.
사용: python -m synth.extras            (data/*/ 전부)
      python -m synth.extras data/core_1404 --pdf   (대장 출력물 PDF 도 생성)

daejang.json 은 공공데이터포털 '건축HUB 건축물대장정보 서비스' 응답 항목명을 그대로 쓴다.
  표제부(getBrTitleInfo): bldNm, platPlc, mainPurpsCdNm, strctCdNm, totArea, archArea, grndFlrCnt, ugrndFlrCnt, heit,
                          useAprDay, rideUseElvtCnt, emgenUseElvtCnt
  층별개요(getBrFlrOulnInfo): flrGbCdNm(지상/지하/옥탑), flrNo, flrNoNm, area, mainPurpsCdNm, etcPurps
일부러 심은 불일치는 gt.json 의 'daejang_flags'(엔진이 내야 할 플래그 전체), 'daejang_planted'(심은 원인)에 기록.
site.json 좌표는 가상 건물 정답 좌표계(m). 엔진은 건물 외곽(GIS)↔도면 외곽 정합으로 맞춘다."""
import json, math, os, random, sys, glob
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fire3dbeam.sop import CFG

KO = lambda lv: (f'지하{-lv}층' if lv < 0 else f'지상{lv + 1}층')


def _area(f): return sum((r[2] - r[0]) * (r[3] - r[1]) for r in f['rects'])


def make_daejang(gt, rng):
    fl = sorted(gt['floors'].values(), key=lambda f: f['level'])
    card = gt['card']
    nb = sum(1 for f in fl if f['level'] < 0); nu = sum(1 for f in fl if f['level'] >= 0)
    sh = rng.choice([3.0, 3.3, 3.6, 4.2]) if gt['btype'] != 'hall' else rng.choice([6.0, 7.5, 9.0])
    rows = []
    for f in fl:
        lv = f['level']
        rows.append(dict(flrGbCdNm='지하' if lv < 0 else '지상', flrNo=-lv if lv < 0 else lv + 1, flrNoNm=KO(lv),
                         area=round(_area(f) * rng.uniform(0.995, 1.005), 2),
                         mainPurpsCdNm=('자동차관련시설' if '주차' in f['use'] else gt['use']), etcPurps=f['use'], _fid=f['id']))
    if 'RF' in card['areas']:                                  # 표준양식 가상 카드의 옥탑 행(카드·대장 모두 있음)
        rows.append(dict(flrGbCdNm='옥탑', flrNo=1, flrNoNm='옥탑1층', area=card['areas']['RF'], mainPurpsCdNm=gt['use'], etcPurps='계단실·기계실', _fid='RF'))
    title = dict(bldNm=gt['name'], platPlc=f"가상시 가상구 가상동 1-{gt['name'][-3:]}", mainPurpsCdNm=gt['use'], strctCdNm='철근콘크리트구조',
                 archArea=round(_area(gt['floors']['1F']) * 1.0, 2), grndFlrCnt=nu, ugrndFlrCnt=nb, heit=round(nu * sh, 1),
                 useAprDay=f"{rng.randint(1995, 2022)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
                 rideUseElvtCnt=len(gt['elevs']), emgenUseElvtCnt=0)
    planted = []
    ups = [r for r in rows if r['flrGbCdNm'] == '지상']
    if nu >= 2 and rng.random() < 0.3:                        # 증축(최상층) 미반영 대장
        top = ups[-1]; rows.remove(top); title['grndFlrCnt'] -= 1; title['heit'] = round(title['heit'] - sh, 1)
        planted.append(dict(kind='extension', floor=top['_fid']))
    elif rng.random() < 0.25:                                 # 표제부 층수만 틀림
        k = rng.choice(['grndFlrCnt', 'ugrndFlrCnt']) if nb else 'grndFlrCnt'
        title[k] += rng.choice([1, -1]) if title[k] > 1 else 1
        planted.append(dict(kind='count', field=k))
    if rng.random() < 0.3:                                    # 층 면적 변경 미반영(또는 카드 오기)
        r = rng.choice(rows); r['area'] = round(r['area'] * rng.choice([0.78, 1.25]), 2)
        planted.append(dict(kind='area', floor=r['_fid']))
    if rng.random() < 0.15 and 'RF' not in card['areas']:     # 대장에만 있는 옥탑층
        rows.append(dict(flrGbCdNm='옥탑', flrNo=1, flrNoNm='옥탑1층', area=round(ups[-1]['area'] * 0.12, 2), mainPurpsCdNm=gt['use'],
                         etcPurps='계단실·기계실', _fid='RF'))
        planted.append(dict(kind='extra', floor='RF'))
    if rng.random() < 0.15:                                   # 주용도 불일치
        title['mainPurpsCdNm'] = rng.choice(['제2종근린생활시설', '판매시설', '운동시설'])
        planted.append(dict(kind='use'))
    title['totArea'] = round(sum(r['area'] for r in rows), 2)
    # 엔진이 내야 할 플래그(규칙 정의 그대로 정답 계산)
    exp = []
    card_ids = [f['id'] for f in fl] + (['RF'] if 'RF' in card['areas'] else [])
    dj = {r['_fid']: r for r in rows}
    for fid in card_ids:
        if fid not in dj: exp.append(dict(kind='daejang_floor_missing', floor=fid))
        elif abs(dj[fid]['area'] - card['areas'][fid]) / dj[fid]['area'] > CFG['daejang_area_rel']: exp.append(dict(kind='daejang_area', floor=fid))
    for fid in dj:
        if fid not in card_ids: exp.append(dict(kind='daejang_floor_extra', floor=fid))
    if (title['grndFlrCnt'], title['ugrndFlrCnt']) != (nu, nb): exp.append(dict(kind='daejang_floor_count'))
    gfa = sum(card['areas'].values())
    if abs(title['totArea'] - gfa) / title['totArea'] > CFG['daejang_gfa_rel']: exp.append(dict(kind='daejang_gfa'))
    if title['mainPurpsCdNm'] != gt['use']: exp.append(dict(kind='daejang_use'))
    for r in rows: r.pop('_fid')
    dj = dict(source='건축HUB 건축물대장정보 서비스 형식(가상)', brTitleInfo=[title], brFlrOulnInfo=rows)
    return dj, planted, exp


def make_site(gt, rng):
    f1 = gt['floors']['1F']
    x0, y0, x1, y1 = unary_union([box(*r) for r in f1['rects']]).bounds
    W, D = x1 - x0, y1 - y0; cx = (x0 + x1) / 2
    roads, obst, flags = [], [], []
    # 전면(남측) 간선도로 — 소방차 진입 가능
    s1, w1 = rng.uniform(3, 9), rng.choice([10.0, 12.0, 15.0, 20.0])
    yR1 = y0 - s1 - w1 / 2
    roads.append(dict(id='R1', name='가상대로', width=w1, line=[[cx - 420, yR1], [cx + 420, yR1]]))
    # 측면 이면도로(동 또는 서) — 폭이 좁으면 사다리차 전개 불가
    side = rng.choice(['E', 'W'])
    s2, w2 = rng.uniform(2, 24), rng.choice([3.0, 4.0, 6.0, 8.0])
    xR2 = x1 + s2 + w2 / 2 if side == 'E' else x0 - s2 - w2 / 2
    roads.append(dict(id='R2', name='가상로12길', width=w2, line=[[xR2, yR1], [xR2, y1 + 160]]))
    # 후면(북측) 골목 — 있을 때도 있음
    n_road = rng.random() < 0.35
    if n_road:
        w3 = rng.choice([2.5, 3.0, 4.0]); s3 = rng.uniform(1.5, 6)
        yR3 = y1 + s3 + w3 / 2
        roads.append(dict(id='R3', name='가상로12길 안길', width=w3, line=[[xR2, yR3], [cx - (xR2 - cx), yR3]]))
    # 전면 도로의 사다리 전개 장애(가로수·전선) 구간
    if rng.random() < 0.6:
        a = rng.uniform(x0 - 10, x1 - 20); L = rng.uniform(15, 35)
        obst.append(dict(kind='가로수·가공전선', poly=[[a, yR1 - w1 / 2], [a + L, yR1 - w1 / 2], [a + L, yR1 + w1 / 2], [a, yR1 + w1 / 2]]))
    # 소방용수: 정보카드 ⑥ '소화전(지상식)-가상-1 : 0.150Km / 소화전(지하식)-가상-2 : 0.320Km'
    bpoly = box(x0, y0, x1, y1)
    def on_R1(dist, sgn):
        yy = yR1 + sgn * (w1 / 2 - 0.5)
        lo, hi = 0, 500
        for _ in range(50):
            m = (lo + hi) / 2
            if bpoly.distance(Point(cx + sgn * m, yy)) < dist: lo = m
            else: hi = m
        return [round(cx + sgn * lo, 2), round(yy, 2)]
    d2 = 320.0
    if rng.random() < 0.25:
        d2 = rng.choice([470.0, 520.0]); flags.append(dict(kind='water_dist', id='가상-2'))
    sg = rng.choice([1, -1])
    h1 = on_R1(150.0, sg); h2 = on_R1(d2, -sg)
    hyd = [dict(id='가상-1', type='지상식', x=h1[0], y=h1[1]), dict(id='가상-2', type='지하식', x=h2[0], y=h2[1])]
    # 송수구(연결송수관): 전면 외벽, 주출입구 옆
    ent = [e for e in f1.get('entrances', []) if abs(e['y'] - y0) < 0.5]
    ex = ent[0]['x'] if ent else cx
    fdc = [dict(name='연결송수관 송수구', x=round(min(x1 - 1, ex + rng.uniform(3, 6)), 2), y=round(y0, 2))]
    # 정답: 면별 펌프차 접근 가능(도로 폭·이격 기준)
    def acc(width, setback): return width >= CFG['engine_min_road_width_m'] and setback <= CFG['engine_max_setback_m']
    facade = {'S': acc(w1, s1), 'N': bool(n_road and acc(w3, s3)), 'E': side == 'E' and acc(w2, s2), 'W': side == 'W' and acc(w2, s2)}
    site = dict(source='가상 부지(도로 구간·소방용수시설 GIS 형식 모사)', crs='local_m',
                building_outline=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], roads=roads, obstacles=obst, hydrants=hyd, fdc=fdc)
    return site, dict(facade_engine=facade, flags=flags)


def extend(d, pdf=False):
    gp = os.path.join(d, 'gt.json')
    gt = json.load(open(gp, encoding='utf-8'))
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(os.path.basename(d.rstrip('/'))))
    rng = random.Random(seed)
    dj, planted, exp = make_daejang(gt, rng)
    site, sgt = make_site(gt, random.Random(seed + 1))
    json.dump(dj, open(os.path.join(d, 'daejang.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(site, open(os.path.join(d, 'site.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    gt.update(daejang_planted=planted, daejang_flags=exp, site_gt=sgt)
    json.dump(gt, open(gp, 'w', encoding='utf-8'), ensure_ascii=False)
    if pdf: daejang_pdf(dj, os.path.join(d, 'daejang.pdf'))
    return planted, exp


def daejang_pdf(dj, out):
    """건축물대장(갑) 출력물 모양의 가상 PDF — fire3dbeam.daejang.parse_pdf 시험용"""
    import asyncio
    t = dj['brTitleInfo'][0]
    rows = ''.join(f"<tr><td>{'주' if i == 0 else '주'}1</td><td>{r['flrNoNm']}</td><td>{r['strct'] if 'strct' in r else '철근콘크리트구조'}</td>"
                   f"<td>{r['mainPurpsCdNm']}({r['etcPurps']})</td><td class=r>{r['area']:,.2f}</td></tr>" for i, r in enumerate(dj['brFlrOulnInfo']))
    d = t['useAprDay']
    html = f"""<!doctype html><html><head><meta charset=utf-8><style>body{{font-family:'Noto Sans CJK KR';font-size:9pt}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #444;padding:2px 4px}}.r{{text-align:right}}h1{{text-align:center;font-size:14pt}}</style></head><body>
<div>■ 건축물대장의 기재 및 관리 등에 관한 규칙 [별지 제1호서식] (가상 출력물 — 시험용)</div><h1>일반건축물대장(갑)</h1>
<table><tr><td>대지위치 {t['platPlc']}</td><td>명칭 {t['bldNm']}</td></tr>
<tr><td>건축면적 {t['archArea']:,.2f}㎡</td><td>연면적 {t['totArea']:,.2f}㎡</td></tr>
<tr><td>주구조 {t['strctCdNm']}</td><td>주용도 {t['mainPurpsCdNm']}</td></tr>
<tr><td>층수 지하: {t['ugrndFlrCnt']}층, 지상: {t['grndFlrCnt']}층</td><td>높이 {t['heit']}m</td></tr>
<tr><td>승강기 승용 {t['rideUseElvtCnt']}대 비상용 {t['emgenUseElvtCnt']}대</td><td>사용승인일 {d[:4]}.{d[4:6]}.{d[6:]}</td></tr></table>
<br><b>건축물 현황</b><table><tr><th>구분</th><th>층별</th><th>구조</th><th>용도</th><th>면적(㎡)</th></tr>{rows}</table></body></html>"""
    h = out.replace('.pdf', '.html'); open(h, 'w', encoding='utf-8').write(html)
    async def go():
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(); pg = await b.new_page()
            await pg.goto('file://' + os.path.abspath(h)); await pg.pdf(path=out, format='A4'); await b.close()
    asyncio.run(go()); os.remove(h)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dirs = args or sorted(glob.glob(os.path.join(ROOT, 'data', '*/')))
    for d in dirs:
        pl, ex = extend(d, '--pdf' in sys.argv)
        print(os.path.basename(d.rstrip('/')), [p['kind'] for p in pl], [(e['kind'], e.get('floor')) for e in ex])
