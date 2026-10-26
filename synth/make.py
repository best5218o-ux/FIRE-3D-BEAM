# -*- coding: utf-8 -*-
"""가상 정보카드 PDF + 정답 JSON 생성.  사용: python -m synth.make 12  (유형별 n동)"""
import base64, io, json, os, random, sys, asyncio
from PIL import Image
from synth.model import generate
from synth.render import render_floor, draw_fac_overlay
from synth import fac as SF
from synth import stdcard as SC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get('SYNTH_DATA') or os.path.join(ROOT, 'data'))     # 보류(held-out) 세트: SYNTH_DATA=data_hold SYNTH_SEED_OFS=5000
# 정보카드 양식: html(기존 근사 배치 — data/·data_hold/ 재현, 기본값) | std(표준양식 docx 채움 → LibreOffice PDF) | hwp(표준양식 + 한글→PDF 흉내) | mix(std·hwp 섞음)
FORM = os.environ.get('SYNTH_FORM', 'html')


def sitemap_image(b, ctx=None):
    """⑪ [0 진압작전도] 이미지 공급 훅. PIL.Image 를 돌려주면 그 그림을, None 이면 자리표시 그림을 넣는다(표준양식 경로만).
    ctx = dict(outdir, gt, site(site.json dict), fields(①~⑩ 정답)). 진압작전도 생성기(별도 작업)는 이 함수를 바꿔 끼우거나
    환경변수 SYNTH_SITEMAP='모듈:함수' 로 연결한다(같은 시그니처)."""
    prov = os.environ.get('SYNTH_SITEMAP')
    if prov:
        import importlib
        m, f = prov.split(':'); return getattr(importlib.import_module(m), f)(b, ctx)
    return None


def area(f):
    return sum((r[2] - r[0]) * (r[3] - r[1]) for r in f['rects'])


def card_html(b, plans, card):
    fl = sorted(b.floors.values(), key=lambda f: f['level'])
    nb = sum(1 for f in fl if f['level'] < 0); nu = sum(1 for f in fl if f['level'] >= 0)
    gfa = sum(card['areas'][f['id']] for f in fl)
    rows = []
    for f in reversed(fl):
        fac = card['fac'][f['id']]
        fac_txt = ' '.join(f'{k}:{v}' for k, v in fac.items())
        rows.append(f"<tr><td class=c>{f['ko'].replace('층', '')}</td><td>{f['use']}</td><td class=r>{card['areas'][f['id']]:.2f}㎡</td><td class=fac>{fac_txt}</td><td></td></tr>")
    st = card['stairs']
    imgs = []
    for i, (cap, im) in enumerate(plans):
        buf = io.BytesIO(); im.save(buf, 'PNG')
        imgs.append(f"<div class=fig><div class=cap>{cap}</div><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'></div>")
    return f"""<!doctype html><html><head><meta charset=utf-8><style>
@page{{size:A4;margin:12mm}} body{{font-family:'Noto Sans CJK KR';font-size:9pt}}
h1{{text-align:center;font-size:15pt;letter-spacing:6px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #555;padding:2px 4px;vertical-align:top}}
td.c{{text-align:center;width:14mm}} td.r{{text-align:right;width:22mm}} td.fac{{width:62mm}} .fig{{page-break-inside:avoid;margin:6mm 0}} .fig img{{max-width:100%;max-height:115mm}} .cap{{font-weight:bold;margin-bottom:2mm}}
.pb{{page-break-before:always}}</style></head><body>
<div>출력일시 : 2026-09-29 10:00:00   출력자 : 가상사용자</div>
<h1>소방활동 정보카드</h1>
<b>1. 건 물 개 황</b>
<table><tr><td>① 관 리 번 호 00-000-2026-{b.name[-3:]}</td><td>② 주 용 도 {b.use}</td><td>③상호 {b.name}</td></tr>
<tr><td colspan=3>④ 소재지 가상시 가상구 가상동 1-{b.name[-3:]} ⑤ 소방대상물과의 거리 119안전센터 2.10Km</td></tr>
<tr><td colspan=3>⑥ 소방용수시설 소화전(지상식)-가상-1 : 0.150Km / 소화전(지하식)-가상-2 : 0.320Km</td></tr>
<tr><td colspan=3>⑦ 건물구조 양식(옥)/철근콘크리트조/슬라브가/지하{nb}층/지상{nu}층/동명:{b.name}/연면적:{gfa:,.2f} ㎡/바닥면적:{max(card['areas'].values()):,.2f} ㎡<br>
특별피난계단수 : {st['special']}  피난계단수 : {st['evac']}  일반계단수 : {st['normal']}<br>옥외계단수 : {st['outdoor']}  승강기(비상) : {st['ev_em']}  승강기(일반) : {st['ev']}</td></tr>
<tr><td colspan=3>⑧ 관계인 등 · 소유자 성명 : 가상소유자 / 연락처 : 000-000-0000 · 관리자 성명 : 가상관리자 / 연락처 : 000-000-0000</td></tr>
<tr><td colspan=3>⑨ 위험물 시설 {card['hazmat']}</td></tr></table>
<br><b>⑩ 층별 현황</b>
<table><tr><th>층별</th><th>용도</th><th>면적</th><th>소방시설 설치현황</th><th>기타</th></tr>{''.join(rows)}</table>
<div class=pb></div><b>2. 동별 위치도, 사진 및 층별 평면도</b>
{''.join(imgs)}
</body></html>"""


def build_one(btype, seed, outdir, form=None):
    form = form or FORM
    rng = random.Random(seed * 7 + 1)
    b = generate(btype, seed)
    fl = sorted(b.floors.values(), key=lambda f: f['level'])
    legend = [('exit', '비상구'), ('arrow', '대피방향'), ('hose', '소화전'), ('panel', '수신기')]
    if b.use == '의료시설': legend.append(('gas', '의료가스밸브'))
    # 기준층 묶기 (코어형 공동주택·업무: 2층~최상층 동일)
    groups = []
    if b.btype == 'core' and rng.random() < 0.6:
        ups = [f['id'] for f in fl if f['level'] >= 1]
        others = [f['id'] for f in fl if f['level'] < 1]
        groups = [[x] for x in others] + [ups]
    else:
        groups = [[f['id']] for f in fl]
    FP = SF.plan(b, groups, seed)                 # 소방시설 추가분(별도 난수 — 기존 난수 흐름·정답 불변)
    used = {t for v in FP['sym'].values() for t, _, _ in v}
    fac_legend = [(t, lb) for t, lb in SF.FAC_TYPES if t in used]
    plans, tfs = [], {}
    # 위치도·사진(도면 아님) — 캡션 매칭 시험용
    dummy = Image.new('RGB', (900, 600), (200, 214, 190))
    plans.append(('0. 위치도', dummy)); plans.append(('00정면사진', Image.new('RGB', (800, 600), (150, 160, 175))))
    for n, g in enumerate(groups):
        meta = {}
        img, tf = render_floor(b, g, rng, legend, meta)
        drawn = draw_fac_overlay(img, meta, FP['sym'].get(g[0]), fac_legend, b, g) or []     # 소방시설 기호·범례(스캔 열화 뒤 덧그림)
        dset = set(drawn)
        for fid in g:
            for q in FP['pos'][fid]:
                t = {'STANDPIPE': 'outlet', 'OUTLET': 'socket', 'EXT': 'ext', 'SIAMESE': 'siamese'}[q['code']]
                q['drawn'] = (t, q['x'], q['y']) in dset
        f0 = b.floors[g[0]]
        cap = f"{n + 1}. 평면도({f0['ko']})" if len(g) == 1 else f"{n + 1}. 평면도({f0['ko']}~{b.floors[g[-1]]['ko']} 기준층)"
        plans.append((cap, img))
        for fid in g: tfs[fid] = dict(tf, group=g, caption=cap)
    # 정보카드 값 (일부러 틀린 값을 넣어 교차검증 시험)
    flags = []
    card = dict(areas={}, fac={}, stairs={}, hazmat='옥내탱크저장소(제4류) 제2석유류 1000ℓ' if rng.random() < 0.4 else '')
    for f in fl:
        a = area(f) * rng.uniform(0.97, 1.02)
        if rng.random() < 0.08:
            a *= 1.35; flags.append(dict(kind='area', floor=f['id']))
        card['areas'][f['id']] = round(a, 2)
        nh = sum(1 for s in f['symbols'] if s['type'] == 'hose')
        if rng.random() < 0.2:
            nh += rng.choice([2, 3, 5]); flags.append(dict(kind='hose_count', floor=f['id']))
        card['fac'][f['id']] = {'옥내소화전': nh, '유도등': rng.randint(10, 60), '자동화재탐지기(감)': rng.randint(20, 150),
                                '자동화재탐지기(회)': max(1, int(area(f) // 550) + rng.randint(0, 2)), '수동식소화기': rng.randint(5, 50),
                                '비상조명등': rng.randint(5, 80)}
    FP['items'] = SF.card_items(b, FP)
    for fid, it in FP['items'].items():           # 정보카드 소방시설 항목 추가(기존 6개 항목 뒤에)
        for k, v in it: card['fac'][fid][k] = card['fac'][fid].get(k, 0) + v
    n_st = len(b.stairs)
    card['stairs'] = dict(special=0, evac=0, normal=n_st, outdoor=0, ev_em=0, ev=sum(e.get('cars', 1) for e in b.elevs))
    if rng.random() < 0.2:
        card['stairs']['normal'] += 1; flags.append(dict(kind='stair_count'))
    os.makedirs(outdir, exist_ok=True)
    std = form != 'html'
    if not std:
        html = card_html(b, plans, card)
        with open(os.path.join(outdir, 'card.html'), 'w', encoding='utf-8') as fp: fp.write(html)
    else:                                         # 표준양식: 추가 값은 별도 난수(기존 도면·정답 난수 흐름 그대로)
        rng2 = random.Random(seed * 31 + 7)
        variant = 'hwp' if form == 'hwp' or (form == 'mix' and rng2.random() < 0.35) else 'std'
        if rng2.random() < 0.3: SC.rf_row(b, card, rng2)             # 옥탑 행(정보카드에만)
        st = card['stairs']; ev_tot = st['ev']
        if FP.get('emev') and st['ev'] >= 1:                           # 승강기(비상)·(일반) 나눔(합은 그대로)
            k_em = len(FP['emev'].get('all') or [1]); st['ev_em'], st['ev'] = k_em, st['ev'] - k_em
        nfl = len(fl)
        if b.btype == 'core' and nfl >= 5:                            # 계단 종류 나눔(합은 그대로 — 심은 +1 은 일반계단에)
            k = 'special' if nfl >= 11 else 'evac'
            st[k], st['normal'] = n_st, st['normal'] - n_st
    gt = b.to_json(); gt.update(transforms=tfs, card=card, flags=flags, groups=groups)
    gt['fac_gt'] = dict(drawn_plans=[g[0] for g in groups if any(q['drawn'] for q in FP['pos'][g[0]])], pos=FP['pos'], emev=FP['emev'], draw=FP['draw'], legend=[t for t, _ in fac_legend], standpipe=FP['standpipe'], tall=FP['tall'])
    gt['fac_flags'] = FP['flags']
    with open(os.path.join(outdir, 'gt.json'), 'w', encoding='utf-8') as fp: json.dump(gt, fp, ensure_ascii=False)
    from synth.extras import extend          # 기능 18·24: 가상 건축물대장·부지(추가 파일, 기존 정답 불변)
    extend(outdir)
    if not std: return os.path.join(outdir, 'card.html')
    return build_std(b, card, plans[2:], seed, outdir, variant, rng2, groups)


def build_std(b, card, plan_figs, seed, outdir, variant, rng2, groups):
    """표준양식 docx 채움(①~⑪) + 정답 gt['card_std'] · gt['card_std_figs'] 추가 → card.docx 경로"""
    gp = os.path.join(outdir, 'gt.json')
    gt = json.load(open(gp, encoding='utf-8'))
    multi = rng2.random() < 0.3
    d0 = SC.fields(b, card, rng2, seed, dong='가동' if multi else '주동', multi=multi)
    dongs = [dict(fields=d0, figs=[], b=b)]
    if multi:                                     # 부속동(①~⑩ 반복 + 도면 1~2장) — 채점·3D 는 첫 동
        from synth.model import generate
        b2 = generate(rng2.choice(['corridor', 'hall']), seed + 50000)
        r3 = random.Random(seed * 13 + 5)
        c2 = dict(areas={}, fac={}, hazmat='' if r3.random() < 0.7 else 'x')
        fl2 = sorted(b2.floors.values(), key=lambda f: f['level'])
        for f in fl2:
            c2['areas'][f['id']] = round(area(f) * r3.uniform(0.97, 1.02), 2)
            c2['fac'][f['id']] = {'옥내소화전': sum(1 for q in f['symbols'] if q['type'] == 'hose'), '유도등': r3.randint(5, 30), '자동화재탐지기(감)': r3.randint(10, 60),
                                  '자동화재탐지기(회)': max(1, int(area(f) // 550)), '수동식소화기': r3.randint(4, 20)}
        c2['stairs'] = dict(special=0, evac=0, normal=len(b2.stairs), outdoor=r3.choice([0, 0, 1]), ev_em=0, ev=sum(e.get('cars', 1) for e in b2.elevs))
        d1 = SC.fields(b2, c2, rng2, seed + 50000, dong='나동', multi=True)
        d1['관리번호'] = d0['관리번호'][:-3] + '-나동'; d1['소재지'] = d0['소재지']; d1['거리'] = d0['거리']; d1['소방용수'] = d0['소방용수']
        figs2 = []
        leg2 = [('exit', '비상구'), ('arrow', '대피방향'), ('hose', '소화전'), ('panel', '수신기')]
        for n, f in enumerate(fl2[:2]):
            img, _ = render_floor(b2, [f['id']], r3, leg2, {})
            figs2.append(('평면도', f"{n + 1}. 평면도({f['ko']})", img, [f['id']]))
        dongs.append(dict(fields=d1, figs=figs2, b=b2))
    site = json.load(open(os.path.join(outdir, 'site.json'), encoding='utf-8')) if os.path.exists(os.path.join(outdir, 'site.json')) else None
    for k, D in enumerate(dongs):
        pre = [('위치도', '0. 위치도', SC.img_location(D['fields'], rng2), []), ('사진', '00 정면사진', SC.img_photo(D['b'], rng2), [])]
        if rng2.random() < 0.4: pre.append(('사진', '00 측면사진', SC.img_photo(D['b'], rng2, side=True), []))
        sm = sitemap_image(D['b'], dict(outdir=outdir, gt=gt, site=site if k == 0 else None, fields=D['fields'], dong=k))
        pre.append(('진압작전도', '0 진압작전도', sm if sm is not None else SC.img_sitemap_placeholder(D['fields']), []))
        if k == 0: D['figs'] = pre + [('평면도', cap, im, g) for (cap, im), g in zip(plan_figs, groups)]
        else: D['figs'] = pre + D['figs']
    for D in dongs:
        for x in D['fields']['층별현황']: x['층_원문'] = SC.floor_txt(x['층'], variant == 'hwp')
    gt['card_std'] = [D['fields'] for D in dongs]
    gt['card_std_figs'] = [dict(동=k, 종류=kind, 캡션=cap, floors=list(g)) for k, D in enumerate(dongs) for kind, cap, im, g in D['figs']]
    gt['card_form'] = dict(form='std', variant=variant, dongs=len(dongs))
    json.dump(gt, open(gp, 'w', encoding='utf-8'), ensure_ascii=False)
    dp = os.path.join(outdir, 'daejang.json')          # 대장 주구조를 카드 ⑦ 구조와 맞춤(가상 — 내화 판단 경로 시험)
    if os.path.exists(dp):
        dj = json.load(open(dp, encoding='utf-8'))
        dj['brTitleInfo'][0]['strctCdNm'] = {'철근콘크리트조': '철근콘크리트구조', '철골철근콘크리트조': '철골철근콘크리트구조', '철골조': '일반철골구조'}.get(d0['건물구조']['구조'], d0['건물구조']['구조'])
        json.dump(dj, open(dp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out = os.path.join(outdir, 'card.docx')
    SC.build_docx([dict(fields=D['fields'], figs=[(k_, c_, i_) for k_, c_, i_, _ in D['figs']]) for D in dongs], out, variant, rng2)
    return out


async def html_to_pdf(paths):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        br = await p.chromium.launch()
        pg = await br.new_page()
        for h in paths:
            await pg.goto('file://' + h); await pg.wait_for_timeout(200)
            await pg.pdf(path=h.replace('card.html', 'card.pdf'), format='A4', print_background=True)
            os.remove(h)
        await br.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    types = sys.argv[2].split(',') if len(sys.argv) > 2 else ['corridor', 'hall', 'core']
    htmls = []
    tall_only = os.environ.get('SYNTH_TALL_ONLY') == '1'      # 비상용승강기 평가 세트: 지상 11층 이상 건물만 n동(씨앗 순서대로 건너뜀)
    for t in types:
        i = k = 0
        while k < n:
            seed = 1000 + i * 17 + len(t) * 101 + int(os.environ.get('SYNTH_SEED_OFS', '0')); i += 1
            if tall_only and sum(1 for f in generate(t, seed).floors.values() if f['level'] >= 0) < 11: continue
            htmls.append(build_one(t, seed, os.path.join(DATA, f'{t}_{seed}'))); k += 1
    hs = [h for h in htmls if h.endswith('.html')]
    if hs: asyncio.run(html_to_pdf(hs))
    ds = [h for h in htmls if h.endswith('.docx')]
    SC.to_pdf([(d, d.replace('card.docx', 'card.pdf')) for d in ds])       # LibreOffice headless
    if not os.environ.get('SYNTH_KEEP_DOCX'):
        for d in ds: os.remove(d)
    print('generated', len(htmls))
