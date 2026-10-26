# -*- coding: utf-8 -*-
"""검수 흐름 자동 시험(개발 PC, playwright 필요): 검수 HTML 생성 → 헤드리스 브라우저에서 편집 → 저장(다운로드) → 반영 → 3D·IFC 재생성 → 라벨.
사용: python offline/selftest_review.py out/core_1404.json [스크린샷_폴더]
      python offline/selftest_review.py out_xrev/hall_1404.json [스크린샷_폴더] --xcheck
        (모델 경로 결과: run(pdf, model=True) 로 만든 JSON — 교차검증 xwall/xdoor 판정 + 계단 추가 → 반영 → IFC·3D·채점 → 라벨 → strseg.review_import)"""
import asyncio, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


async def browser_edit(html, corr_out, shots):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={'width': 1600, 'height': 900}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('console', lambda m: errs.append(m.type + ': ' + m.text) if m.type in ('error', 'warning') else None)
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        await pg.goto('file://' + os.path.abspath(html)); await pg.wait_for_timeout(600)
        await pg.screenshot(path=os.path.join(shots, 'review_0_open.png'))
        info = await pg.evaluate('''() => {
          const R = window.__review, D = R.D, log = [];
          const its = D.items;
          // 1) 의심 기호 항목: 첫째 '맞음', 둘째 '틀림(삭제)'
          const sy = its.filter(i => i.kind === 'symbol');
          if (sy[0]) { R.decide(sy[0].id, true); log.push('yes ' + sy[0].ref); }
          if (sy[1]) { R.decide(sy[1].id, false); log.push('no ' + sy[1].ref); }
          // 2) 교차검증 플래그: 첫째 확인, 둘째 문제없음
          const fl = its.filter(i => i.kind === 'flag');
          if (fl[0]) R.decide(fl[0].id, true); if (fl[1]) R.decide(fl[1].id, false);
          // 3) 첫 도면: 소화전 추가, 문 추가, 벽 1개 삭제, 기호 종류 변경, 정합 정상, 완료
          const p0 = D.plans[0]; R.gotoPlan(p0.i);
          R.addSym(p0.w * 0.3, p0.h * 0.6, 'hose');
          const w0 = p0.walls[0]; R.addDoor(w0.o === 'h' ? (w0.a + w0.b) / 2 : w0.c, w0.o === 'h' ? w0.c : (w0.a + w0.b) / 2);
          R.select('wall', p0.walls[1].id); R.delSel();
          const pe = D.plans.find(p => p.symbols.some(q => q.type === 'exit'));
          const ex = R.S.plans[pe.i].symbols.find(q => q.type === 'exit'); R.gotoPlan(pe.i); R.select('sym', ex.id); R.setType('gas'); R.setType('exit'); R.setType('gas'); R.gotoPlan(p0.i);
          R.doUndo(); R.doRedo();                       // 되돌리기·다시하기
          R.setAlign('ok'); R.toggleDone();
          // 4) 둘째 도면: 위치 보정 +0.3m, 문 하나 삭제, 완료
          const p1 = D.plans[1]; R.gotoPlan(p1.i); R.shiftPlan(0.3, 0);
          if (R.S.plans[p1.i].doors[0]) { R.select('door', R.S.plans[p1.i].doors[0].id); R.delSel(); }
          R.toggleDone();
          // 5) 승강기 하나 오검출 표시
          if (D.elevs.length > 1) { R.select('elevs', D.elevs.length - 1); R.delSel(); }
          R.gotoPlan(p0.i);
          return { log, items: its.length, plans: D.plans.length };
        }''')
        # 키보드: 선택 기호 이동·종류 바꾸기
        await pg.keyboard.press('Escape'); await pg.keyboard.press('j'); await pg.wait_for_timeout(150)
        await pg.evaluate("() => { const R = window.__review; const p = R.D.plans[R.D.plans.length - 1]; R.gotoPlan(p.i); R.select('sym', R.S.plans[p.i].symbols[0].id); }")
        await pg.keyboard.press('Shift+ArrowRight'); await pg.keyboard.press('Shift+ArrowRight'); await pg.keyboard.press('4')
        await pg.keyboard.press('j'); await pg.wait_for_timeout(150)
        # 마우스: 기호 추가(A→2→클릭) → 끌어서 이동 → 휠 확대
        n0 = await pg.evaluate('() => window.__review.S.plans[window.__review.U.plan].symbols.length')
        box = await pg.locator('#svg').bounding_box()
        cx, cy = box['x'] + box['width'] * 0.45, box['y'] + box['height'] * 0.55
        await pg.keyboard.press('a'); await pg.keyboard.press('2'); await pg.mouse.click(cx, cy); await pg.wait_for_timeout(100)
        await pg.keyboard.press('v')
        added = await pg.evaluate('() => { const R = window.__review; const L = R.S.plans[R.U.plan].symbols; return L[L.length - 1]; }')
        r = await pg.locator(f'rect[data-id="{added["id"]}"]').bounding_box()
        await pg.mouse.move(r['x'] + r['width'] / 2, r['y'] + r['height'] / 2); await pg.mouse.down()
        await pg.mouse.move(r['x'] + r['width'] / 2 + 60, r['y'] + r['height'] / 2 + 30, steps=6); await pg.mouse.up()
        moved = await pg.evaluate(f'() => window.__review.S.plans[window.__review.U.plan].symbols.find(q => q.id === "{added["id"]}")')
        vb0 = await pg.get_attribute('#svg', 'viewBox'); await pg.mouse.move(cx, cy); await pg.mouse.wheel(0, -400); await pg.wait_for_timeout(100)
        vb1 = await pg.get_attribute('#svg', 'viewBox')
        n1 = await pg.evaluate('() => window.__review.S.plans[window.__review.U.plan].symbols.length')
        print('마우스: 추가', n0, '→', n1, added['type'], '| 끌기', (added['px'], added['py']), '→', (moved['px'], moved['py']), '| 휠', vb0, '→', vb1)
        assert n1 == n0 + 1 and added['type'] == 'hose' and moved['px'] > added['px'] and vb0 != vb1
        ime = await pg.evaluate('''() => { const R = window.__review; const f = (k, c) => window.dispatchEvent(new KeyboardEvent('keydown', {key: k, code: c, bubbles: true}));
          f('ㅁ', 'KeyA'); const m1 = R.U.mode; f('ㅍ', 'KeyV'); return [m1, R.U.mode]; }''')
        print('한글 IME 단축키(ㅁ→A, ㅍ→V):', ime); assert ime == ['sym', 'select']
        await pg.keyboard.press('g'); await pg.wait_for_timeout(300)
        await pg.screenshot(path=os.path.join(shots, 'review_1_edited.png'))
        await pg.keyboard.press('f'); await pg.wait_for_timeout(200)
        await pg.screenshot(path=os.path.join(shots, 'review_2_plan.png'))
        async with pg.expect_download() as dl:
            await pg.click('#bSave')
        d = await dl.value; await d.save_as(corr_out)
        await pg.wait_for_timeout(300)
        await pg.screenshot(path=os.path.join(shots, 'review_3_saved.png'))
        await pg.keyboard.press('?'); await pg.wait_for_timeout(200)
        await pg.screenshot(path=os.path.join(shots, 'review_4_help.png'))
        await b.close()
        return info, errs


async def reopen(html, shots):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1600, 'height': 900}); errs = []
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        await pg.goto('file://' + os.path.abspath(html)); await pg.wait_for_timeout(600)
        st = await pg.evaluate('() => window.__review.buildCorrections().summary')
        await pg.screenshot(path=os.path.join(shots, 'review_5_reopened.png'))
        await b.close(); return st, errs


def main(result, shots):
    os.makedirs(shots, exist_ok=True)
    from review.make import make
    stem = os.path.splitext(result)[0]
    html = make(result, out_html=stem + '_review.html')
    corr = stem + '_corrections.json'
    info, errs = asyncio.run(browser_edit(html, corr, shots))
    print('편집:', info, '브라우저 오류:', errs)
    assert not [e for e in errs if 'PAGEERR' in e], errs
    c = json.load(open(corr, encoding='utf-8'))
    print('교정 요약:', c['summary'], '작업기록', len(c['ops']))
    subprocess.run([sys.executable, '-m', 'review.apply', result, corr, '--labels', os.path.join(os.path.dirname(result), 'labels_selftest')], check=True, cwd=ROOT)
    new = json.load(open(stem + '.corrected.json', encoding='utf-8'))
    import ifcopenshell
    f = ifcopenshell.open(stem + '.corrected.ifc')
    print('IFC:', len(f.by_type('IfcWall')), '벽', len(f.by_type('IfcDoor')), '문', len(f.by_type('IfcFireSuppressionTerminal')), '소화전',
          [p.HasProperties[0].NominalValue.wrappedValue for p in f.by_type('IfcPropertySet') if p.Name == 'Pset_소방대상물'][:1])
    print('검수 반영 통계:', new['review']['stats'])
    html2 = make(result, out_html=stem + '_review_reload.html', corrections=corr)
    st, errs2 = asyncio.run(reopen(html2, shots))
    print('다시 열기:', st, errs2)
    assert st['items_done'] == c['summary']['items_done'], (st, c['summary'])


# ───────────────────────── 교차검증 판정 + 계단 추가 (모델 경로 결과) ─────────────────────────
async def browser_xcheck(html, corr_out, shots, stair_px, elev_px):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={'width': 1600, 'height': 900}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        await pg.goto('file://' + os.path.abspath(html)); await pg.wait_for_timeout(600)
        its = await pg.evaluate("() => window.__review.D.items.filter(i => i.kind === 'xwall' || i.kind === 'xdoor')")
        ro = [i for i in its if i['sub'] == 'rule_only']; mo = [i for i in its if i['kind'] == 'xwall' and i['sub'] == 'model_only']
        pick = []                                             # (항목, 버튼) — 화면의 실제 버튼을 누른다
        if ro: pick.append((ro[0], 'y'))                      # 규칙만 본 벽: 모델이 맞음 → 벽 구간 삭제 + 음성 라벨
        if len(ro) > 1: pick.append((ro[1], 'n'))             #               규칙이 맞음 → 유지(확정)
        if len(ro) > 2: pick.append((ro[2], 'l'))             #               나중에
        plain = [i for i in mo if not i.get('filled') and not i.get('bridge')]
        if plain: pick.append((plain[0], 'y'))                # 모델만 본 벽: 모델이 맞음 → 벽 추가
        if len(plain) > 1: pick.append((plain[1], 'n'))       #               규칙이 맞음 → 음성 라벨
        filled = [i for i in mo if i.get('filled')]
        if filled: pick.append((filled[0], 'n'))              # 모델로 보충한 벽: 규칙이 맞음 → 보충 벽 삭제
        if len(filled) > 1: pick.append((filled[1], 'y'))
        br = [i for i in mo if i.get('bridge')]
        if br: pick.append((br[0], 'y'))                      # 개구부↔벽: 모델이 맞음 → 벽으로 막음(그 자리 문 삭제)
        xd = [i for i in its if i['kind'] == 'xdoor']
        if xd: pick.append((xd[0], 'y' if xd[0]['sub'] == 'model_only' else 'n'))
        if len(xd) > 1: pick.append((xd[1], 'n'))
        for it, act in pick:
            await pg.click(f'[data-it="{it["id"]}"] .h')      # 항목으로 이동(도면 확대)
            if act == 'y' and it is pick[0][0]: await pg.screenshot(path=os.path.join(shots, 'xrev_1_item.png'))
            await pg.click(f'[data-it="{it["id"]}"] button[data-act="{act}"]')
        # 키보드: 다음 미처리 교차검증 항목에 L(나중에)
        rest = [i for i in its if i['id'] not in {x[0]['id'] for x in pick}]
        if rest:
            await pg.click(f'[data-it="{rest[0]["id"]}"] .h'); await pg.keyboard.press('l')
        dec = await pg.evaluate('() => window.__review.S.xdec')
        await pg.evaluate(f"() => window.__review.gotoItem('{pick[0][0]['id']}')")      # 판정한 항목(규칙만 벽 → 모델이 맞음 = 삭제 ✗)
        await pg.wait_for_timeout(150)
        await pg.screenshot(path=os.path.join(shots, 'xrev_2_decided.png'))
        await pg.keyboard.press('f'); await pg.wait_for_timeout(150)
        await pg.screenshot(path=os.path.join(shots, 'xrev_2b_plan.png'))
        # 계단 추가: 도면 0 에서 S → 계단 → 상자 끌어 그리기(마우스)
        await pg.evaluate('() => window.__review.gotoPlan(window.__review.D.plans[0].i)')
        await pg.keyboard.press('f'); await pg.keyboard.press('s'); await pg.click('[data-vtype="stairs"]')
        scr = await pg.evaluate("""(b) => { const svg = document.getElementById('svg'); const m = svg.getScreenCTM(); const f = (x, y) => { const p = svg.createSVGPoint(); p.x = x; p.y = y; const q = p.matrixTransform(m); return [q.x, q.y]; };
                                 return [f(b[0], b[1]), f(b[2], b[3])]; }""", stair_px)
        (ax, ay), (bx, by) = scr
        await pg.mouse.move(ax, ay); await pg.mouse.down(); await pg.mouse.move((ax + bx) / 2, (ay + by) / 2, steps=4)
        await pg.mouse.move(bx, by, steps=4); await pg.mouse.up(); await pg.wait_for_timeout(100)
        va = await pg.evaluate('() => window.__review.S.vadd')
        assert len(va) == 1 and va[0]['kind'] == 'stairs', va
        await pg.click('#vfPlan'); await pg.click('#vfAll')             # 연결층 버튼: 이 도면 층만 → 전체 층
        vf = await pg.evaluate('() => window.__review.S.vadd[0].floors')
        await pg.mouse.move((ax + bx) / 2, (ay + by) / 2)
        for _ in range(5): await pg.mouse.wheel(0, -120); await pg.wait_for_timeout(40)
        await pg.screenshot(path=os.path.join(shots, 'xrev_3_stair_added.png'))
        # 승강기 추가(API): 도면 1, 연결층 = 그 도면 층 + 종류 바꾸기 왕복
        if elev_px:
            await pg.evaluate(f"""() => {{ const R = window.__review, p = R.D.plans[1]; R.gotoPlan(p.i);
                const id = R.addVert({elev_px[0]}, {elev_px[1]}, {elev_px[2]}, {elev_px[3]}, 'stairs', p.floors.map(f => f.id));
                R.setVaddKind(id, 'elevs'); }}""")
        await pg.evaluate('() => { const R = window.__review; R.D.plans.forEach(p => { R.gotoPlan(p.i); R.toggleDone(); }); R.gotoPlan(R.D.plans[0].i); }')
        await pg.keyboard.press('?'); await pg.wait_for_timeout(150)
        await pg.screenshot(path=os.path.join(shots, 'xrev_5_help.png')); await pg.keyboard.press('Escape')
        async with pg.expect_download() as dl:
            await pg.click('#bSave')
        d = await dl.value; await d.save_as(corr_out)
        await b.close()
        return dict(picked=[(i['id'], i['kind'], i['sub'], bool(i.get('filled')), bool(i.get('bridge')), a) for i, a in pick], xdec=dec, stair_floors=vf), errs


def _cov(walls, x, tol=0.12):
    """구간 x(o,c,a,b; m)가 벽 목록에 덮인 비율"""
    L = x['b'] - x['a']; ov = 0.0
    for w in walls:
        if w['o'] == x['o'] and abs(w['c'] - x['c']) < tol: ov += max(0.0, min(w['b'], x['b']) - max(w['a'], x['a']))
    return min(1.0, ov / L) if L > 0 else 0.0


def main_xcheck(result, shots):
    import tempfile
    from review.make import make
    from review.common import plan_T, g2p, p2g
    from review import xdec
    os.makedirs(shots, exist_ok=True)
    res = json.load(open(result, encoding='utf-8'))
    assert res.get('xcheck'), '모델 경로 결과(xcheck 절)가 아님 — run(pdf, model=True) 결과를 쓰세요'
    stem = os.path.splitext(result)[0]; name = os.path.basename(stem)
    html = make(result, out_html=stem + '_review.html')
    # 계단을 그릴 자리: 가상 정답(gt)에서 엔진이 놓친 계단(없으면 도면 가운데)
    gtp = os.path.join(ROOT, 'data', name, 'gt.json'); stair_px = elev_px = None; gt = None
    T0 = plan_T(res, 0)
    if os.path.exists(gtp):
        from evalr.score import score, gt_map, eng_map
        gt = json.load(open(gtp, encoding='utf-8'))
        s0 = score(gt, res)
        # 전역 오프셋 g(엔진 − 정답) = 층정합 채점과 같은 방법(첫 층 도면 중심)
        f0 = next(f for f in gt['transforms'] if f in res['floors']); t = gt['transforms'][f0]
        g = [a - b for a, b in zip(eng_map(res['floors'][f0], t['w'] / 2, t['h'] / 2), gt_map(t, t['w'] / 2, t['h'] / 2))]
        import math
        miss = [S for S in gt['stairs'] if not any(math.hypot((S['box'][0] + S['box'][2]) / 2 + g[0] - e['x'], (S['box'][1] + S['box'][3]) / 2 + g[1] - e['y']) < 3 for e in res['stairs'])]
        if miss:
            bx = miss[0]['box']; (x0, y0), (x1, y1) = g2p(T0, bx[0] + g[0], bx[3] + g[1]), g2p(T0, bx[2] + g[0], bx[1] + g[1])
            stair_px = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        print('채점(반영 전): 계단', s0['stairs_found'], '승강기', s0['elevs_found'], '벽 R/P', s0['wall_recall'], s0['wall_precision'], '문 R/P', s0['door_recall'], s0['door_precision'])
    if stair_px is None:
        P0 = res['plans'][0]; s = T0[0]; stair_px = [100, 100, 100 + 3.5 * s, 100 + 7 * s]
    if len(res['plans']) > 1:
        s1 = res['plans'][1]['s']; elev_px = [60, 60, 60 + 2.2 * s1, 60 + 2.4 * s1]
    corr = stem + '_corrections.json'
    info, errs = asyncio.run(browser_xcheck(html, corr, shots, stair_px, elev_px))
    print('판정:', json.dumps(info, ensure_ascii=False)); print('브라우저 오류:', errs)
    assert not errs, errs
    c = json.load(open(corr, encoding='utf-8'))
    print('교정 요약:', c['summary'], '| xcheck', len(c['xcheck']), '| 계단·승강기 추가', [(v['kind'], v['floors'], v['box_px']) for v in c['vertical_added']])
    assert len(c['xcheck']) == len(info['xdec']) and c['vertical_added']
    labels = os.path.join(os.path.dirname(result), 'labels_xcheck')
    if os.path.exists(os.path.join(labels, 'annotations.json')): os.unlink(os.path.join(labels, 'annotations.json'))
    subprocess.run([sys.executable, '-m', 'review.apply', result, corr, '--labels', labels], check=True, cwd=ROOT)
    new = json.load(open(stem + '.corrected.json', encoding='utf-8'))
    st = new['review']['stats']; print('검수 반영 통계:', {k: v for k, v in st.items() if v}); print('교차검증 판정 집계:', new['review']['xcheck'])
    # ① 벽·문 반영 확인(결과 JSON, 전역 m)
    pf = {}
    for fid, F in res['floors'].items(): pf.setdefault(F['plan'], []).append(fid)
    chk = []
    for e in c['xcheck']:
        if e['kind'] != 'xwall' or e['decision'] == 'later': continue
        i = e['plan']; T = plan_T(res, i); w = xdec.seg_wall(e['seg'], e['o'])
        if w['o'] == 'h': (a, cc), (b, _) = p2g(T, w['a'], w['c']), p2g(T, w['b'], w['c'])
        else: (cc, a), (_, b) = p2g(T, w['c'], w['a']), p2g(T, w['c'], w['b'])
        x = dict(o=w['o'], c=cc, a=min(a, b) + 0.05, b=max(a, b) - 0.05)
        fid = sorted(pf[i])[0]; tol = 0.35 if e['sub'] == 'model_only' else 0.12
        before, after = _cov(res['floors'][fid]['walls'], x, tol), _cov(new['floors'][fid]['walls'], x, tol)
        chk.append((e['id'], e['sub'], e.get('filled'), e.get('bridge'), e['decision'], round(before, 2), round(after, 2)))
        eff = ('remove' if (e['sub'] == 'rule_only') == (e['decision'] == 'model') else 'keep') if e['sub'] == 'rule_only' or e.get('filled') else ('add' if e['decision'] == 'model' else 'none')
        if eff == 'remove': assert after < 0.1, chk[-1]
        if eff == 'add': assert after > 0.9, chk[-1]
        if eff == 'keep': assert after >= before - 0.05, chk[-1]
    print('벽 구간 덮임(전→후):', chk)
    assert st['xwall_cut'] >= 1 and st['xwall_added'] >= 1 and st['stairs_added'] == 1, st
    assert len(new['stairs']) == len(res['stairs']) + 1 and any(S.get('review') == 'added' for S in new['stairs'])
    # ② IFC·3D 지휘화면 데이터
    import ifcopenshell
    f = ifcopenshell.open(stem + '.corrected.ifc')
    added_st = [e for e in f.by_type('IfcStair') if any(r.RelatingPropertyDefinition.Name == 'Pset_소방활동' and any(getattr(pp.NominalValue, 'wrappedValue', '') == '검수자 추가(검수 화면)' for pp in r.RelatingPropertyDefinition.HasProperties) for r in e.IsDefinedBy)]
    added_ev = [e for e in f.by_type('IfcTransportElement') if any(r.RelatingPropertyDefinition.Name == 'Pset_소방활동' and any(getattr(pp.NominalValue, 'wrappedValue', '') == '검수자 추가(검수 화면)' for pp in r.RelatingPropertyDefinition.HasProperties) for r in e.IsDefinedBy)]
    walls_ifc = len(f.by_type('IfcWall'))
    print('IFC:', walls_ifc, '벽', len(f.by_type('IfcDoor')), '문', len(f.by_type('IfcStair')), '계단(층별)', '— 검수자 추가 계단', len(added_st), '/ 승강기', len(added_ev))
    assert len(added_st) == len(next(S for S in new['stairs'] if S.get('review') == 'added')['floors'])
    vh = open(stem + '.corrected.html', encoding='utf-8').read()
    assert '검수자 추가(검수 화면)' in vh, '3D 지휘화면 데이터에 추가 계단 없음'
    n_w_new = sum(len(F['walls']) for F in new['floors'].values()); n_w_old = sum(len(F['walls']) for F in res['floors'].values())
    print('3D 지휘화면: 추가 계단 포함 ✓ · 벽 수', n_w_old, '→', n_w_new)
    if gt:
        s1 = score(gt, new)
        print('채점(반영 후): 계단', s1['stairs_found'], '승강기', s1['elevs_found'], '벽 R/P', s1['wall_recall'], s1['wall_precision'], '문 R/P', s1['door_recall'], s1['door_precision'])
        a0, b0 = map(int, s0['stairs_found'].split('/')); a1, _ = map(int, s1['stairs_found'].split('/'))
        assert a1 == a0 + 1 or a0 == b0, (s0['stairs_found'], s1['stairs_found'])
        oracle_check(result, res, c, gt, stem + '_review.html', s0)
    # ③ 학습 라벨(COCO) → STR 학습 캐시
    coco = json.load(open(os.path.join(labels, 'annotations.json'), encoding='utf-8'))
    cid = {x['name']: x['id'] for x in coco['categories']}
    from collections import Counter
    cnt = Counter(next(k for k, v in cid.items() if v == a['category_id']) for a in coco['annotations'])
    conf = sum(1 for a in coco['annotations'] if a['category_id'] == cid['wall'] and a['attributes'].get('confirmed'))
    xw = sum(1 for a in coco['annotations'] if a['category_id'] == cid['wall'] and a['attributes'].get('src') == 'xcheck')
    print('라벨:', dict(cnt), '| 확정 벽', conf, '· 교차검증으로 추가한 벽', xw)
    n_neg = sum(1 for e in c['xcheck'] if (e['kind'] == 'xwall' and ((e['sub'] == 'rule_only') == (e['decision'] == 'model')) and e['decision'] != 'later'))
    assert cnt['wall_rejected'] == n_neg and cnt['stair'] >= 1 and cnt['elevator_shaft'] >= 1 and conf >= 1 and xw >= 1, (cnt, n_neg)
    met = json.load(open(os.path.join(labels, 'review_metrics.json'), encoding='utf-8'))
    print('검수 기반 성능(교차검증):', met[-1].get('xcheck'), met[-1].get('vertical'))
    cache = tempfile.mkdtemp(prefix='revcache_')
    subprocess.run([sys.executable, '-m', 'strseg.review_import', labels, '--cache', cache], check=True, cwd=ROOT)
    rs = json.load(open(os.path.join(cache, 'review_import_stats.json'), encoding='utf-8'))
    print('review_import:', rs)
    assert rs and sum(r['neg_model_px'] for r in rs) > 0, '음성 영역에서 모델 벽 화소가 0 으로 들어가지 않음'
    # ④ 다시 열기: 판정·추가 계단이 복원되는가
    html2 = make(result, out_html=stem + '_review_reload.html', corrections=corr)
    st2, errs2 = asyncio.run(reopen_x(html2, shots))
    print('다시 열기:', st2, errs2)
    assert st2['xdec'] == len(c['xcheck']) and st2['vadd'] == len(c['vertical_added']) and not errs2


def oracle_check(result, res, c, gt, html, s0):
    """(참고) 가상 정답으로 판정하는 '완벽한 검수자' — 모든 xwall/xdoor 항목을 정답대로 판정해 반영하면 채점이 어떻게 바뀌는가.
    화면 조작과 무관한 반영 로직(review.apply + review/xdec.py) 확인용."""
    import re, math, copy, numpy as np
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    from evalr.score import score, gt_map, eng_map
    from review.common import plan_T, p2g
    from review import xdec
    from review.apply import apply
    D = json.loads(re.search(r'const D = (\{.*?\});</script>', open(html, encoding='utf-8').read(), re.S).group(1).replace('<\\/', '</'))
    T_ = gt['transforms']; fids = [f for f in gt['floors'] if f in res['floors'] and f in T_]
    g = np.median([np.subtract(eng_map(res['floors'][f], T_[f]['w'] / 2, T_[f]['h'] / 2), gt_map(T_[f], T_[f]['w'] / 2, T_[f]['h'] / 2)) for f in fids], axis=0)
    fl_of = {}
    for f, F in res['floors'].items(): fl_of.setdefault(F['plan'], []).append(f)
    xs = []
    for it in D['items']:
        if it['kind'] not in ('xwall', 'xdoor'): continue
        T = plan_T(res, it['plan']); f = sorted(fl_of[it['plan']])[0]
        if f not in gt['floors']: continue
        (x0, y0), (x1, y1) = p2g(T, it['seg'][0], it['seg'][1]), p2g(T, it['seg'][2], it['seg'][3])
        if it['kind'] == 'xwall':
            G = unary_union([LineString([(w['x0'], w['y0']), (w['x1'], w['y1'])]) for w in gt['floors'][f]['walls']]).buffer(0.4)
            L = LineString([(x0 - g[0], y0 - g[1]), (x1 - g[0], y1 - g[1])]); cov = L.intersection(G).length / max(1e-6, L.length)
            dec = ('model' if cov < 0.5 else 'rule') if it['sub'] == 'rule_only' else ('model' if cov >= 0.5 else 'rule')
        else:
            px, py = p2g(T, *it['loc']); real = any(math.dist((px - g[0], py - g[1]), (d['x'], d['y'])) < 0.9 for d in gt['floors'][f]['doors'])
            dec = 'model' if real else 'rule'
        e = {k: it.get(k) for k in ('id', 'kind', 'sub', 'plan', 'p', 'seg', 'loc', 'o', 'ref', 'tpx', 'filled', 'bridge', 'wpx', 'ext')}; e['decision'] = dec; xs.append(e)
    oc = copy.deepcopy(c); oc['xcheck'] = xs; oc['vertical_added'] = []
    new, st = apply(res, oc, log=lambda *a: None)
    s1 = score(gt, new)
    from collections import Counter
    print('(참고) 정답 기반 판정', dict(Counter((e['kind'], e['sub'], e['decision']) for e in xs)))
    print('   → 벽 R/P', s0['wall_recall'], s0['wall_precision'], '→', s1['wall_recall'], s1['wall_precision'], '| 문 R/P', s0['door_recall'], s0['door_precision'], '→', s1['door_recall'], s1['door_precision'],
          '| 진입구', s0.get('entrance_found'), '→', s1.get('entrance_found'), '|', {k: v for k, v in st.items() if k.startswith('x') and v})
    return s1


async def reopen_x(html, shots):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1600, 'height': 900}); errs = []
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        await pg.goto('file://' + os.path.abspath(html)); await pg.wait_for_timeout(600)
        st = await pg.evaluate('() => ({ xdec: Object.keys(window.__review.S.xdec).length, vadd: window.__review.S.vadd.length })')
        await pg.screenshot(path=os.path.join(shots, 'xrev_4_reopened.png'))
        await b.close(); return st, errs


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if '--xcheck' in sys.argv: main_xcheck(a[0] if a else 'out_xrev/hall_1404.json', a[1] if len(a) > 1 else 'out/shots')
    else: main(a[0] if a else 'out/core_1404.json', a[1] if len(a) > 1 else 'out/shots')
