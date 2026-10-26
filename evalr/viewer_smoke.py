# -*- coding: utf-8 -*-
"""3D 지휘화면 인명검색·훈련·활동계획 표시 — 헤드리스 Chromium 클릭 시험(화면 조작 그대로).

  python -m evalr.viewer_smoke 결과.html [스크린샷 접두어]

확인: 페이지 오류 0 · (v2 화면) 3단 배치·머리줄·범례 칩·층 버튼·보기 전환(전체/층 분리/단층 평면)·확인 필요 → 확인사항 탭·지시선 말풍선 ·
화점 지정 버튼 → 캔버스 클릭으로 화점 · 인명검색 번호 표시/표 · 주간↔야간 전환 · 호스 1본 눈금 ·
귀환 한계점(용기 용량을 줄여 화점 앞에서 걸리게) · 문 미검출 구간 수 문구 · 훈련 모드(버튼·선택·캔버스 실 클릭·제출·JSON/HTML 저장).
진입 경로(계단)가 없는 건물(계단 미검출)은 공기호흡기 두 항목을 '해당 없음'으로 통과 처리(이유를 함께 출력).
v3(18~34): 벽 두께·문 개구부·문짝·외벽 유리 띠·승강로 · 단면(가로·세로 + 채움) · 투시 · 층 격리 · 확산 시뮬레이션 재생 · 대원 이동 · 진입로 따라가기 ·
미니맵 클릭 · 거리 재기 · 실 면적 · 화면 저장(PNG) · 브리핑(HTML) · 단축키 · 터치(두 손가락 확대·비틀기) · 화면 속도(그리기 횟수·저사양) · 2D 대체 보기(WebGL 없음) · 확인사항 문구.
v3.1(35~): 화면 겹침 자동 점검(1280×720·1600×900·1920×1080·태블릿 1024×768 × 전체/층/화재 가정(확산 6분·대원)/단면+거리 안내 —
고정 UI 끼리 · 라벨·말풍선 ↔ 고정 UI · 라벨 끼리 사각형이 겹치면 실패) · 출동 전 30초 요약 · 접기(점진 공개) · 인명검색 번호 묶음·설명 ·
테마(야간·주간·고대비)·글자 크기(크게 해도 겹침 없음) · 인쇄(Ctrl+P → A4 1~2쪽 PDF) · 시작 화면·WebGL 중단 → 2D 전환 · 접근성(아이콘 단추 이름·키보드 초점 표시)."""
import sys, json, os, re


def _boot(pg, max_ms=20000):
    """시작 화면(불러오는 중)이 걷힐 때까지 — 그 전에는 시작 화면이 마우스·터치를 받음(무거운 실제 도면 건물은 첫 화면이 늦음)"""
    for _ in range(max_ms // 250):
        if pg.evaluate("() => !!(window.__viewer && window.__viewer.booted && window.__viewer.booted()) && !document.querySelector('#boot:not(.done)')"): return True
        pg.wait_for_timeout(250)
    return False


def main(html, shot=None):
    from playwright.sync_api import sync_playwright
    R = []
    def chk(name, ok, note=''): R.append((name, bool(ok), note)); print(('PASS' if ok else 'FAIL'), name, note, flush=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        pg = b.new_page(viewport={'width': 1400, 'height': 860}, accept_downloads=True)
        errs = []; net = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
        pg.on('request', lambda r: net.append(r.url) if not r.url.startswith(('file:', 'data:', 'blob:')) else None)
        pg.goto('file://' + os.path.abspath(html), timeout=120000); _boot(pg); pg.wait_for_timeout(1500)
        # 0) v2 화면 구성: 3단(왼쪽 탭·가운데 3D·오른쪽) · 범례 칩 · 층 버튼 · 보기 전환 · 확인 필요 버튼 · 말풍선
        wl, ws, wr = (pg.evaluate(f"() => document.querySelector('{q}').getBoundingClientRect().width") for q in ('#left', '#stage', '#right'))
        nchip = len(pg.query_selector_all('#layers .lay')); nfb = len(pg.query_selector_all('#fbar .fbtn[data-f]')); nfl = pg.evaluate("() => window.__DATA__.D.floors.length")
        foot = pg.inner_text('#foot')
        chk('v2 배치: 3단·범례 칩·층 버튼·바닥줄', wl > 200 and ws > 500 and wr > 200 and nchip >= 10 and nfb == nfl and '자동 변환' in foot and pg.inner_text('#bname').strip(),
            f"왼쪽 {wl:.0f}·3D {ws:.0f}·오른쪽 {wr:.0f}px · 칩 {nchip} · 층 버튼 {nfb}/{nfl}")
        st0 = pg.evaluate("() => window.__viewer.state()")
        nco = pg.evaluate("() => [...document.querySelectorAll('#labels .co')].filter(e => e.style.display !== 'none' && !e.classList.contains('culled')).length")
        home_txt = pg.inner_text('#rpHead')
        g = pg.evaluate("() => window.__DATA__.D.floors.find(f => f.level === 0).id")
        pg.click(f'#fbar .fbtn[data-f="{g}"]'); pg.wait_for_timeout(900)
        st1 = pg.evaluate("() => window.__viewer.state()"); big = pg.inner_text('#rpHead .fid.big') if pg.query_selector('#rpHead .fid.big') else ''
        chk('층 버튼 → 층 선택(위층 들림·오른쪽 층 상세)', not st0['focus'] and '층을 선택' in home_txt and st1['focus'] and st1['selId'] == g and big.strip() == g,
            f"말풍선 {nco}개 · 선택 {st1['selId']}")
        k0 = pg.evaluate("() => window.__viewer.state().layers.stair")
        pg.click('#layers .lay[data-l="stair"]'); pg.wait_for_timeout(200)
        k1 = pg.evaluate("() => window.__viewer.state().layers.stair"); on1 = pg.evaluate("() => document.querySelector('#layers .lay[data-l=\"stair\"]').classList.contains('on')")
        pg.click('#layers .lay[data-l="stair"]'); pg.wait_for_timeout(200)
        chk('범례 칩 → 레이어 켜고 끄기', k0 and not k1 and not on1 and pg.evaluate("() => window.__viewer.state().layers.stair"), '계단 칩')
        pg.click('#vExp'); pg.wait_for_timeout(500); m1 = pg.evaluate("() => window.__viewer.state().viewMode")
        pg.click('#vOne'); pg.wait_for_timeout(500); m2 = pg.evaluate("() => window.__viewer.state().viewMode")
        vis = len(pg.evaluate("() => window.__viewer.state().visibleFloors"))
        pg.click('#vAll'); pg.wait_for_timeout(500); st3 = pg.evaluate("() => window.__viewer.state()")
        chk('보기 전환: 층 분리 → 단층 평면 → 전체', m1 == 'exp' and m2 == 'one' and st3['viewMode'] == 'all' and not st3['focus'] and vis == 1, f"{m1} → {m2} → {st3['viewMode']}")
        pg.click('.tab[data-t="basis"]'); pg.wait_for_timeout(150); pg.click('#bissue'); pg.wait_for_timeout(200)
        chk("'확인 필요' 버튼 → 확인사항 탭", pg.is_visible('#tp-issues') and not pg.is_visible('#tp-basis') and len(pg.query_selector_all('#issues .issue')) == pg.evaluate("() => window.__DATA__.D.issues.length"),
            f"카드 {len(pg.query_selector_all('#issues .issue'))}건 · 표시 숫자 {pg.inner_text('#nIssues')}")
        # 1) 화점: 지상층 하나 고르고 '화점 지정' → 캔버스에서 그 층 실 중심 클릭
        info = pg.evaluate("""() => { const D = window.__DATA__.D, EV = D.evac; const fl = D.floors.filter(f => f.level >= 1 && EV && EV.floors[f.id]).sort((a, b) => a.level - b.level);
          const f = (fl[1] || fl[0] || D.floors[0]).id; const rr = EV.floors[f].rooms.filter(r => r.area > 12 && r.cat !== 'corr'); return { f, c: (rr[0] || EV.floors[f].rooms[0]).c }; }""")
        pg.evaluate("(f) => window.__viewer.selectFloor(f, false)", info['f']); pg.click('#vOne'); pg.wait_for_timeout(300)
        pg.evaluate("([f,x,y]) => window.__viewer.cam(x, y, 0, 0.1, 60, 0.1)", [info['f'], info['c'][0], info['c'][1]]); pg.wait_for_timeout(400)
        pg.click('#bFire'); pg.wait_for_timeout(200)
        sx, sy = pg.evaluate("([f,x,y]) => window.__viewer.project(f, x, y)", [info['f'], info['c'][0], info['c'][1]])
        pg.mouse.click(sx, sy); pg.wait_for_timeout(800)
        P = pg.evaluate("() => window.__viewer.plan()")
        chk('화점 지정(버튼 → 캔버스 클릭)', P and P['fid'] == info['f'], f"{P and P['fid']} ({P and round(P['x'], 1)}, {P and round(P['y'], 1)})")
        n_lab = pg.evaluate("() => [...document.querySelectorAll('.lb.srch')].filter(e => e.style.display !== 'none').length")
        S = pg.evaluate("() => window.__viewer.search()")
        chk('인명검색 우선순위: 번호 표시 + 표', n_lab > 0 and pg.query_selector('#srPanel') is not None and S and len(S['items']) > 0, f"번호 {n_lab}개 · 순위 {len(S['items']) if S else 0}실 · 모드 {S and S['mode']}")
        m0 = S['mode']; other = 'day' if m0 == 'night' else 'night'
        pg.select_option('#srMode', other); pg.wait_for_timeout(600)
        S2 = pg.evaluate("() => window.__viewer.search()")
        chk('주간/야간 전환', S2['mode'] == other, f"{m0} → {S2['mode']}")
        if shot: pg.screenshot(path=shot + '_search.png')
        # 2) 호스 눈금·귀환 한계점·불확실 구간
        pg.evaluate("() => window.__viewer.setAct({ hose_slack_m: 20 })"); pg.wait_for_timeout(500)
        nt = pg.evaluate("() => document.querySelectorAll('.lb.tick').length"); P2 = pg.evaluate("() => window.__viewer.plan()")
        chk('호스 여유 길이 α → 본수·1본 눈금', P2['hoseN'] is None or (P2['hoseN'] == -(-(P2['hoseL'] + 20) // 15) and P2['hoseTicks'] == int(P2['hoseL'] // 15)), f"L={P2['hoseL'] and round(P2['hoseL'], 1)} N={P2['hoseN']} 눈금 {P2['hoseTicks']}(화면 표시 {nt})")
        pg.evaluate("() => window.__viewer.setAct({ hose_slack_m: 0, scba_cyl_l: 0.75 })"); pg.wait_for_timeout(500); pg.evaluate("() => window.__viewer.tidy()")   # 라벨(HTML)은 다음 그리기 때 붙으므로 한 번 그림
        P3 = pg.evaluate("() => window.__viewer.plan()"); ns = pg.evaluate("() => document.querySelectorAll('.lb.scba').length")
        txt = pg.inner_text('#fireRes')
        na = not P3['scba'] and not P3['attack'] and P3['fid'] != P3['ground']        # 진입 경로 없음(이 층과 1층을 잇는 계단 미검출) → 공기호흡기 계산 대상 아님
        NA = 'N/A — 진입 경로 없음(계단 미검출)'
        chk('귀환 한계점(용기 0.75 L 가정 → 화점 앞)', na or (P3['scba'] and P3['scba']['pt'] and ns == 1 and '왕복 불가' in txt), NA if na else f"T_A={P3['scba'] and round(P3['scba']['TA'], 2)}분 · 진입 {P3['scba'] and round(P3['scba']['tFire'], 2)}분")
        if shot: pg.screenshot(path=shot + '_scba.png')
        pg.evaluate("() => window.__viewer.setAct({ scba_cyl_l: 6.8 })"); pg.wait_for_timeout(400)
        P4 = pg.evaluate("() => window.__viewer.plan()")
        chk('귀환 한계(기본 예시값) — 활동 가능 시간', na or (P4['scba'] and P4['scba']['pt'] is None and P4['scba']['act'] > 0), NA if na else f"활동 가능 {P4['scba'] and round(P4['scba']['act'], 1)}분")
        nw = P4['wallSeg']; txt4 = pg.inner_text('#fireRes')
        chk('문 미검출 불확실 구간 수 표시', (nw == 0) or (f'불확실 구간 {nw}곳' in txt4), f"{nw}곳")
        # 3) 훈련 모드(화면 조작)
        pg.select_option('#trMode', 'night'); pg.click('#trCur'); pg.wait_for_timeout(800)
        hid = pg.evaluate("() => window.__viewer.trainer().hidden()"); vis = pg.evaluate("() => [...document.querySelectorAll('.lb.srch')].filter(e => e.style.display !== 'none').length")
        chk('훈련 시작: 산출안 숨김', hid and vis == 0 and '산출안은 제출 후' in pg.inner_text('#fireRes'), f"보이는 검색 번호 {vis}")
        Q = pg.evaluate("() => window.__viewer.plan()")
        pg.select_option('#trEnt', '0')
        if pg.query_selector('#trSt'): pg.select_option('#trSt', Q['attack'])
        pg.fill('#trHose', str(Q['hoseN'] if Q['hoseN'] is not None else 1)); pg.dispatch_event('#trHose', 'change')
        pg.click('#trPick'); pg.wait_for_timeout(1000)
        pg.evaluate("([x,y]) => window.__viewer.cam(x, y, 0, 0.1, 60, 0.1)", [Q['x'], Q['y']]); pg.wait_for_timeout(500)
        cs = pg.evaluate("""() => { const Q = window.__viewer.plan(), rr = window.__DATA__.D.evac.floors[Q.fid].rooms; return Q.search.filter(z => z.fid === Q.fid).slice(0, 3).map(z => rr[z.k].c); }""")
        for c in cs:
            x, y = pg.evaluate("([f,x,y]) => window.__viewer.project(f, x, y)", [Q['fid'], c[0], c[1]]); pg.mouse.click(x, y); pg.wait_for_timeout(250)
        pg.wait_for_timeout(500)
        npick = len(pg.evaluate("() => window.__viewer.trainer().answers().order"))
        chk('훈련: 캔버스에서 실 클릭으로 검색 순서', npick == len(cs), f"{npick}/{len(cs)}")
        if shot: pg.screenshot(path=shot + '_train.png')
        pg.click('#trGo'); pg.wait_for_timeout(800)
        T = pg.evaluate("() => window.__viewer.trainer().score()")
        chk('훈련: 제출 → 채점·산출안 공개', T and T['score100'] > 0 and not pg.evaluate("() => window.__viewer.trainer().hidden()"), f"{T and T['score100']}/100 · " + ' '.join(f"{z['key']}={z['pts']}/{z['max']}" for z in (T or {}).get('items', [])))
        with pg.expect_download() as dl: pg.click('#trJson')
        j = json.load(open(dl.value.path(), encoding='utf-8'))
        with pg.expect_download() as dl2: pg.click('#trHtml')
        h = open(dl2.value.path(), encoding='utf-8').read()
        chk('훈련 결과 저장(JSON·HTML, 오프라인)', j.get('score100') == T['score100'] and '<script' not in h and 'http' not in h, f"{dl.value.suggested_filename} · {dl2.value.suggested_filename}")
        if shot: pg.screenshot(path=shot + '_result.png')
        pg.click('#trQuit'); pg.wait_for_timeout(300)
        v3(pg, chk, shot)
        pg.close()                                   # 다음 시험 쪽에 CPU 를 넘김(소프트웨어 GL)
        v3_extra(b, html, chk, net)
        v31(b, html, chk, net, errs, shot)
        chk('페이지 오류·외부 요청 없음', not errs and not net, f"오류 {errs[:3]} 외부 {net[:3]}")
        b.close()
    ok = all(r[1] for r in R)
    print(f"viewer_smoke: {sum(r[1] for r in R)}/{len(R)}")
    return 0 if ok else 1


# ───────────────────────── v3 (18~34) ─────────────────────────
def _settle(pg, max_ms=9000):
    """카메라 이동(보간)이 끝날 때까지 기다린 뒤 한 번 그림 — 소프트웨어 GL 은 화면 1장이 1초를 넘기도 해 고정 대기만으로는 화면 좌표가 어긋남"""
    last = None
    for _ in range(max_ms // 300):
        cur = pg.evaluate("() => { const c = window.__viewer._three.camera.position; return [...window.__viewer.target(), c.x, c.y, c.z].map((v) => Math.round(v * 100) / 100); }")
        if cur == last: break
        last = cur; pg.wait_for_timeout(300)
    pg.evaluate("() => window.__viewer.tidy()")


def _cap_count(pg):
    return pg.evaluate("""() => { const T = window.__viewer._three; let n = 0, v = 0, hv = 0; T.scene.traverse((o) => { if (!o.userData.cap) return; for (let q = o; q; q = q.parent) if (!q.visible) return; n++; const c = o.geometry.attributes.position.count; v += c; if (o.material.clippingPlanes && o.material.clippingPlanes.length === 1 && o.material.clippingPlanes[0].normal.y === 0) hv += c; }); return { n, v }; }""")


def v3(pg, chk, shot):
    E = lambda s, *a: pg.evaluate(s, *a)
    D = E("() => ({ nSt: window.__DATA__.BIM.stairs.length, nEv: window.__DATA__.BIM.elevs.length, sp: (window.__DATA__.D.building.stairs || {})['특별피난계단'] || 0, nDoor: Object.values(window.__DATA__.BIM.floors).reduce((a, f) => a + f.doors.length, 0) })")
    S = E("() => window.__viewer.stats()")
    leaves = sum(w['leaves'] for w in S['walls'].values()); segs = sum(w['segs'] for w in S['walls'].values()); glass = sum(1 for w in S['walls'].values() if w['glass'])
    nsp = sum(1 for z in S['shafts'] if z['special'] and z['special']['kind'] == 'sp')
    chk('v3 형상: 두께 벽·문 개구부·문짝·외벽 유리 띠·승강로(특별피난계단 표시)', segs > 0 and leaves >= D['nDoor'] and len(S['shafts']) == D['nSt'] + D['nEv'] and (D['sp'] == 0 or nsp == min(D['sp'], D['nSt'])),
        f"벽 조각 {segs} · 문짝 {leaves}/문 {D['nDoor']} · 유리 띠 층 {glass} · 승강로 {len(S['shafts'])} · 특별피난 {nsp}(정보카드 {D['sp']})")
    # 단면: S 키 → 가로 단면(고른 층) → 세로 단면 슬라이더
    f = E("() => { const fl = window.__DATA__.D.floors.filter((f) => f.level >= 1).sort((a, b) => a.level - b.level); return (fl[1] || fl[0] || window.__DATA__.D.floors[0]).id; }")
    E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(900)
    pg.keyboard.press('s'); pg.wait_for_timeout(500)
    pg.fill('#secHv', '380'); pg.dispatch_event('#secHv', 'input'); pg.wait_for_timeout(500)
    c1 = _cap_count(pg); st = E("() => window.__viewer.state().section")
    pg.check('#secV'); pg.fill('#secVv', '500'); pg.dispatch_event('#secVv', 'input'); pg.wait_for_timeout(600)
    c2 = _cap_count(pg)
    if shot: pg.screenshot(path=shot + '_v3_section.png')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(400)
    st2 = E("() => window.__viewer.state().section"); htxt = st.get('text') or ''
    chk('단면: 가로(층 바닥 기준)·세로 + 잘린 면 채움 → Esc 끔', st['active'] and st['h'] and c1['n'] > 0 and c2['v'] > c1['v'] and not st2['active'], f"가로 채움 {c1['n']}개 · 세로 추가 꼭짓점 {c2['v'] - c1['v']}")
    # 투시
    op0 = E("() => { const P = window.__viewer._three.G[window.__viewer.state().selId]; return P.wallMat.length ? P.wallMat[0].opacity : null; }")
    pg.click('#dock [data-tool="xray"]'); pg.wait_for_timeout(500)
    op1 = E("() => { const P = window.__viewer._three.G[window.__viewer.state().selId]; return P.wallMat.length ? P.wallMat[0].opacity : null; }"); x1 = E("() => window.__viewer.state().xray")
    if shot: pg.screenshot(path=shot + '_v3_xray.png')
    pg.click('#dock [data-tool="xray"]'); pg.wait_for_timeout(300)
    chk('투시(X-ray): 벽 거의 투명 · 다시 누르면 원래대로', x1 and (op0 is None or op1 < op0 * 0.5) and not E("() => window.__viewer.state().xray"), f"벽 불투명도 {op0 and round(op0, 3)} → {op1 and round(op1, 3)}")
    # 층 격리
    pg.keyboard.press('i'); pg.wait_for_timeout(700)
    gh = E("() => { const G = window.__viewer._three.G; return Object.entries(G).filter(([k, P]) => P.state === 'ghost').length; }"); vis = len(E("() => window.__viewer.state().visibleFloors"))
    if shot: pg.screenshot(path=shot + '_v3_isolate.png')
    pg.keyboard.press('i'); pg.wait_for_timeout(400)
    nfl = E("() => window.__DATA__.D.floors.length")
    chk('층 격리: 고른 층만 진하게 · 나머지 윤곽(흐림)', gh == nfl - 1 and vis == nfl and not E("() => window.__viewer.state().isolate"), f"흐린 층 {gh}/{nfl - 1}")
    # 화재 가정 → 확산 시뮬레이션 재생
    fc = E("(f) => { const r = window.__DATA__.D.evac.floors[f].rooms.filter((r) => r.area > 12 && r.cat !== 'corr'); return (r[0] || window.__DATA__.D.evac.floors[f].rooms[0]).c; }", f)
    E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, fc[0], fc[1]]); pg.wait_for_timeout(1200)
    tlv = pg.is_visible('#tl')
    pg.click('#tlPlay'); pg.wait_for_timeout(1800); t1 = E("() => window.__viewer.state().t"); pg.click('#tlPlay'); pg.wait_for_timeout(200)
    pg.fill('#tlR', '600'); pg.dispatch_event('#tlR', 'input'); pg.wait_for_timeout(800)
    Sm = E("() => window.__viewer.sim.state()")
    ttu = E("() => window.__viewer.sim.ttu()")
    nroom = E("(f) => window.__DATA__.D.evac.floors[f].rooms.length", f)
    ev = Sm['evac'] or {}
    smokeVis = E("() => { let n = 0; window.__viewer._three.scene.traverse((o) => { if (o.geometry && o.geometry.userData && o.geometry.userData.idx && o.visible) n++; }); return n; }")
    ttuTxt = pg.inner_text('#simPanel') if pg.query_selector('#simPanel') else ''
    chk('확산 시뮬레이션: 재생 → 시각 이동 · 화점 실 곤란 · 시각별 대피로 재평가', tlv and t1 > 0 and Sm['t'] == 600 and ttu[Sm['src']] <= 60 and Sm['summary']['nHit'] >= 1 and (ev.get('ok', 0) + ev.get('cut', 0) + ev.get('smoke', 0)) == nroom and smokeVis > 0 and '곤란' in ttuTxt,
        f"재생 {t1:.0f}초 · 10분: 곤란 실 {Sm['summary']['nHit']} · 대피 가능 {ev.get('ok')}/고립 {ev.get('cut')}/연기 {ev.get('smoke')} (실 {nroom}) · 공격계단 곤란 {Sm['summary']['atkTTU']}초")
    if shot: pg.screenshot(path=shot + '_v3_sim.png')
    # 대원 이동
    C = E("() => window.__viewer.crew()")
    if C['info'] and not C['info'].get('none'):
        a = C['info']['arrive']
        E("(t) => window.__viewer.sim.seek(t)", a + 20); pg.wait_for_timeout(500); C1 = E("() => window.__viewer.crew()")
        E("(t) => window.__viewer.sim.seek(t)", min(900, a + C['info']['tEnd'] + 5)); pg.wait_for_timeout(500); C2 = E("() => window.__viewer.crew()")
        t1s = {z['id']: z for z in C1['teams']}; t2s = {z['id']: z for z in C2['teams']}
        lab = E("() => [...document.querySelectorAll('.lb.crew')].filter((e) => e.offsetParent && e.style.display !== 'none').length")
        ok = t1s['atk1']['state'] == 'move' and t1s['atk1']['p'] and (a + C['info']['tEnd'] + 5 > 900 or t2s['atk1']['state'] == 'done') and t2s['atk1']['bar'] < t1s['atk1']['bar'] and lab >= 1
        chk('대원 이동: 도착 → 진입로 이동 · 시각·잔압 표시 · 호스 전개', ok, f"도착 {a:.0f}초 · 화점까지 +{C['info']['tEnd']:.0f}초 · 진압1 {t1s['atk1']['at']} → {t2s['atk1']['state']} · 잔압 {t1s['atk1']['bar']:.0f}→{t2s['atk1']['bar']:.0f}bar · 호스 {C2['hose'] and round(C2['hose']['d'], 1)}m · 라벨 {lab}")
    else: chk('대원 이동: 도착 → 진입로 이동 · 시각·잔압 표시 · 호스 전개', True, 'N/A — 진입 경로 없음(계단 미검출)')
    if shot: pg.screenshot(path=shot + '_v3_crew.png')
    # 진입로 따라가기
    P = E("() => window.__viewer.plan()")
    if P and P.get('legs'):
        pg.keyboard.press('g'); pg.wait_for_timeout(600); F0 = E("() => window.__viewer.followState()"); pg.wait_for_timeout(1800); F1 = E("() => window.__viewer.followState()")
        hud = pg.inner_text('#fwTxt') if pg.is_visible('#follow') else ''
        if shot: pg.screenshot(path=shot + '_v3_follow.png')
        pg.keyboard.press('Escape'); pg.wait_for_timeout(400)
        off = not E("() => window.__viewer.state().follow")
        chk('진입로 따라가기(1인칭): 카메라 이동 · 남은 거리/시간 · Esc 종료', F0 and F1 and F1['t'] > F0['t'] and F1['cam'] != F0['cam'] and '남은 거리' in hud and off, f"{F0 and round(F0['t'], 1)}→{F1 and round(F1['t'], 1)}초/{F1 and round(F1['T'], 0)}초 · 남은 {F1 and round(F1['remain'], 1)}m")
    else: chk('진입로 따라가기(1인칭): 카메라 이동 · 남은 거리/시간 · Esc 종료', True, 'N/A — 진입 경로 없음')
    # 미니맵 클릭
    q = E("(f) => { const b = window.__DATA__.D.floors.find((x) => x.id === f).bb; return [b[0] + (b[2] - b[0]) * 0.3, b[1] + (b[3] - b[1]) * 0.6]; }", f)
    mx, my = E("([x, y]) => window.__viewer.miniPx(x, y)", q)
    hitEl = E("([x, y]) => { const e = document.elementFromPoint(x, y); return e ? (e.id || e.className || e.tagName) : null; }", [mx, my])
    pg.mouse.click(mx, my)
    tg = None
    for _ in range(20):                                   # 카메라 이동(0.45초 보간)이 끝날 때까지(최대 10초) — 소프트웨어 GL 은 화면 1장이 1초 넘게 걸리기도 함
        pg.wait_for_timeout(500); tg = E("() => window.__viewer.target()")
        if abs(tg[0] - q[0]) < 0.3 and abs(tg[1] - q[1]) < 0.3: break
    chk('미니맵: 누른 지점으로 카메라 이동(시야 표시)', abs(tg[0] - q[0]) < 1.0 and abs(tg[1] - q[1]) < 1.0, f"목표 {q[0]:.1f},{q[1]:.1f} → {tg[0]:.1f},{tg[1]:.1f} (누른 곳: {hitEl})")
    # 거리 재기 · 면적 (화재 해제 후 고른 층)
    pg.click('#bClr'); pg.wait_for_timeout(500); E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1000)
    pg.keyboard.press('m'); pg.wait_for_timeout(300); _settle(pg)
    pts = E("(f) => { const r = window.__DATA__.D.evac.floors[f].rooms.filter((r) => r.area > 12).sort((a, b) => b.area - a.area)[0], xs = r.poly.map((p) => p[0]), ys = r.poly.map((p) => p[1]); const cy = r.c[1]; return [[r.c[0] - 2, cy], [r.c[0] + 2.5, cy + 1.5]]; }", f)
    for (x, y) in pts:
        sx, sy = E("([x, y]) => window.__viewer.floorPx(x, y)", [x, y]); pg.mouse.click(sx, sy); pg.wait_for_timeout(400)
    M = E("() => window.__viewer.tools().measures")
    exp = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
    chk('거리 재기: 바닥 두 점 → 직선·보행 거리(m)', M and abs(M[-1]['straight'] - exp) < 0.35 and (M[-1]['walk'] is None or M[-1]['walk'] >= M[-1]['straight'] - 0.5), f"직선 {M and round(M[-1]['straight'], 2)}m(기대 {exp:.2f}) · 보행 {M and M[-1]['walk'] and round(M[-1]['walk'], 1)}m")
    if shot: pg.screenshot(path=shot + '_v3_measure.png')
    pg.keyboard.press('a'); pg.wait_for_timeout(300)
    rc = E("(f) => { const r = window.__DATA__.D.evac.floors[f].rooms.filter((r) => r.area > 12).sort((a, b) => b.area - a.area)[0]; return [r.c, r.area]; }", f)
    _settle(pg)
    sx, sy = E("([x, y]) => window.__viewer.floorPx(x, y)", rc[0]); pg.mouse.click(sx, sy); pg.wait_for_timeout(500)
    A = E("() => window.__viewer.tools().area")
    chk('실 면적: 실을 누르면 면적·둘레', A and abs(A['area'] - rc[1]) / rc[1] < 0.03 and abs(A.get('polyArea', A['area']) - rc[1]) / rc[1] < 0.08,
        f"{A and round(A['area'], 1)}㎡ (실 자료 {rc[1]}㎡ · 표시 다각형 {A and round(A.get('polyArea', 0), 1)}㎡) · 둘레 {A and round(A['perim'], 1)}m")
    if shot: pg.screenshot(path=shot + '_v3_area.png')
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    # 화면 저장 · 브리핑
    E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, fc[0], fc[1]]); pg.wait_for_timeout(1000); E("() => window.__viewer.sim.seek(300)"); pg.wait_for_timeout(400)
    with pg.expect_download() as d1: pg.keyboard.press('p')
    png = open(d1.value.path(), 'rb').read()
    chk('화면 저장(PNG): 3D + 라벨 합성', png[:8] == b'\x89PNG\r\n\x1a\n' and len(png) > 20000, f"{d1.value.suggested_filename} {len(png) // 1024}KB")
    with pg.expect_download() as d2: pg.click('#dock [data-tool="brief"]')
    h = open(d2.value.path(), encoding='utf-8').read(); nm = E("() => window.__DATA__.D.building.name")
    if shot: open(shot + '_v3_briefing.html', 'w', encoding='utf-8').write(h)
    chk('한 장 브리핑(인쇄용 HTML, 오프라인)', nm in h and '진입구' in h and '계단' in h and '화재 가정' in h and '확산 가정' in h and '<script' not in h and not re.search(r'https?://', h) and '@page' in h,
        f"{d2.value.suggested_filename} {len(h) // 1024}KB")
    # 단축키
    lv = E("() => { const L = window.__DATA__.D.floors.map((f) => f.level); return [L.includes(2) ? 2 : L.includes(1) ? 1 : null, L.includes(-1)]; }")
    pg.click('#bClr'); pg.wait_for_timeout(300); pg.mouse.move(5, 5)
    ok = True; note = []
    if lv[0]: pg.keyboard.press(str(lv[0])); pg.wait_for_timeout(500); s1 = E("() => window.__viewer.state()"); f1 = E("(l) => window.__DATA__.D.floors.find((f) => f.level === l).id", lv[0]); ok &= s1['selId'] == f1 and s1['focus']; note.append(f"{lv[0]}→{s1['selId']}")
    if lv[1]: pg.keyboard.press('0'); pg.wait_for_timeout(500); s0 = E("() => window.__viewer.state()"); f0 = E("() => window.__DATA__.D.floors.find((f) => f.level === -1).id"); ok &= s0['selId'] == f0; note.append(f"0→{s0['selId']}")
    pg.keyboard.press('x'); pg.wait_for_timeout(200); ok &= E("() => window.__viewer.state().xray"); pg.keyboard.press('Escape'); pg.wait_for_timeout(200); ok &= not E("() => window.__viewer.state().xray")
    pg.keyboard.press('f'); pg.wait_for_timeout(400); ok &= E("() => window.__viewer.state().picking"); pg.keyboard.press('Escape'); pg.wait_for_timeout(300); ok &= not E("() => window.__viewer.state().picking")
    pg.keyboard.press('?'); pg.wait_for_timeout(200); ok &= pg.is_visible('#help'); pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
    chk('단축키: 숫자 층 · X 투시 · F 화재 · ? 도움말 · Esc', ok, ' · '.join(note))
    # 화면 속도
    pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
    Bn = E("() => window.__viewer.bench(12)")
    pg.keyboard.press('l'); pg.wait_for_timeout(400); lq = E("() => window.__viewer.state().lowQ"); Bl = E("() => window.__viewer.bench(12)"); pg.keyboard.press('l'); pg.wait_for_timeout(300)
    chk('화면 속도: 그리기 횟수·저사양 모드', Bn['calls'] < 900 and lq and Bl['pr'] <= 1 and Bl['calls'] <= Bn['calls'], f"보통 {Bn['ms']:.0f}ms/화면 · 그리기 {Bn['calls']}회 · 삼각형 {Bn['tris']:,} → 저사양 {Bl['ms']:.0f}ms · {Bl['calls']}회 (SwiftShader 상대값)")
    # 확인사항 문구(현장 말)
    txt = E("() => [...document.querySelectorAll('#issues .issue')].map((e) => { const c = e.cloneNode(true); c.querySelectorAll('details').forEach((d) => d.remove()); return c.innerText; }).join('\\n')")
    chk('확인사항 문구: 개발용 표기(px/m·real:·grid:) 없음 — 상세는 접힘', not re.search(r'px/m|real:|grid:|ppm=', txt), f"{len(txt)}자")


def v3_extra(b, html, chk, net):
    # 터치: 두 손가락 벌리기(확대) · 비틀기(수평 회전)
    ctx = b.new_context(viewport={'width': 1200, 'height': 800}, has_touch=True); pg = ctx.new_page()
    pg.on('request', lambda r: net.append(r.url) if not r.url.startswith(('file:', 'data:', 'blob:')) else None)
    pg.goto('file://' + os.path.abspath(html), timeout=120000); _boot(pg); pg.wait_for_timeout(1500)
    cdp = ctx.new_cdp_session(pg)
    r = pg.evaluate("() => document.getElementById('c').getBoundingClientRect().toJSON()"); cx, cy = r['x'] + r['width'] / 2, r['y'] + r['height'] / 2
    def T(kind, pts): cdp.send('Input.dispatchTouchEvent', {'type': kind, 'touchPoints': [{'x': x, 'y': y, 'id': i} for i, (x, y) in enumerate(pts)]})
    dist = lambda: pg.evaluate("() => { const t = window.__viewer.target(); const c = window.__viewer._three.camera.position; return Math.hypot(c.x - t[0], c.y - t[2], c.z + t[1]); }")
    az = lambda: pg.evaluate("() => { const t = window.__viewer.target(); const c = window.__viewer._three.camera.position; return Math.atan2(c.x - t[0], c.z + t[1]); }")
    D0 = dist()
    T('touchStart', [(cx - 40, cy), (cx + 40, cy)])
    for k in range(1, 9): T('touchMove', [(cx - 40 - 15 * k, cy), (cx + 40 + 15 * k, cy)]); pg.wait_for_timeout(30)
    T('touchEnd', []); pg.wait_for_timeout(700); D1 = dist()
    for _ in range(30):                                   # 확대는 다음 화면 그리기(controls.update) 때 반영 — 소프트웨어 GL 은 1장이 1초를 넘기도 함
        if D1 < D0 * 0.95: break
        pg.wait_for_timeout(300); D1 = dist()
    A0 = az(); import math
    T('touchStart', [(cx - 80, cy), (cx + 80, cy)])
    for k in range(1, 9):
        a = k * 0.08; T('touchMove', [(cx - 80 * math.cos(a), cy - 80 * math.sin(a)), (cx + 80 * math.cos(a), cy + 80 * math.sin(a))]); pg.wait_for_timeout(30)
    T('touchEnd', []); pg.wait_for_timeout(700); A1 = az()
    chk('터치: 두 손가락 벌리기 = 확대 · 비틀기 = 수평 회전', D1 < D0 * 0.95 and abs(A1 - A0) > 0.15, f"거리 {D0:.0f}→{D1:.0f} · 방위 {math.degrees(A0):.0f}°→{math.degrees(A1):.0f}°")
    ctx.close()
    # 2D 대체 보기(WebGL 없음): ?nogl
    pg = b.new_page(viewport={'width': 1400, 'height': 860}); e2 = []
    pg.on('pageerror', lambda e: e2.append(str(e))); pg.on('console', lambda m: e2.append(m.text) if m.type == 'error' else None)
    pg.on('request', lambda r: net.append(r.url) if not r.url.startswith(('file:', 'data:', 'blob:')) else None)
    pg.goto('file://' + os.path.abspath(html) + '?nogl=1', timeout=120000); _boot(pg); pg.wait_for_timeout(1200)
    f = pg.evaluate("() => { const fl = window.__DATA__.D.floors.filter((f) => f.level >= 1).sort((a, b) => a.level - b.level); return (fl[0] || window.__DATA__.D.floors[0]).id; }")
    pg.click(f'#fbar .fbtn[data-f="{f}"]'); pg.wait_for_timeout(500); pg.click('#bFire'); pg.wait_for_timeout(500)
    xy = pg.evaluate("""(f) => { const cv = document.getElementById('fallback'), T = cv.__T, r = cv.getBoundingClientRect(); const R = window.__DATA__.D.evac.floors[f].rooms.filter((q) => q.area > 8); const c = (R[0] || window.__DATA__.D.evac.floors[f].rooms[0]).c; return [r.left + T.X(c[0]) / cv.__dpr, r.top + T.Y(c[1]) / cv.__dpr]; }""", f)
    pg.mouse.click(xy[0], xy[1]); pg.wait_for_timeout(1200)
    P = pg.evaluate("() => window.__viewer.plan()"); st = pg.evaluate("() => window.__viewer.state()")
    chk('2D 대체 보기(WebGL 없는 PC): 평면 그림 · 층 고르기 · 화점 지정', not st['gl'] and pg.is_visible('#fallback') and pg.is_visible('#glmsg') and P and P['fid'] == f and not e2, f"화점 {P and P['fid']} · 오류 {e2[:2]}")
    pg.close()


# ───────────────────────── v3.1 (35~) ─────────────────────────
# 겹침 점검: 고정 UI(범례 칩·+N·도구 열·보기 줄·층 열·나침반·미니맵·시간축·실 용도 범례·단면 창·따라가기·안내 띠·객체 정보·WebGL 안내)
#   ① UI 끼리 ② 보이는 라벨·말풍선 상자 ↔ UI ③ 라벨 끼리 — 사각형이 1px 넘게 겹치면 실패. 라벨 정리(tidy)를 한 번 돌린 뒤 잼.
OVERLAP_JS = """() => { window.__viewer.tidy(); const ui = window.__viewer.ui().rects; const st = document.getElementById('stage').getBoundingClientRect();
 const shown = (el) => { if (el.closest('.culled')) return false; const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none') return false; for (let p = el; p && p.id !== 'labels'; p = p.parentElement) if (p.style && p.style.display === 'none') return false; return true; };
 const labs = [...document.querySelectorAll('#labels .lb, #labels .co .bx, #labels .ftag')].filter(shown).map((el) => { const r = el.getBoundingClientRect(); const q = { n: String(el.className).replace(/\\s+/g, '.') + ':' + (el.textContent || '').trim().slice(0, 16), l: r.left, t: r.top, r: r.right, b: r.bottom }; if (el.dataset && el.dataset.n) { q.r += 16; q.t -= 10; } return q; })
   .filter((q) => q.r - q.l > 1 && q.b - q.t > 1 && q.r > st.left && q.l < st.right && q.b > st.top && q.t < st.bottom);
 const X = (a, b, e = 1) => a.l < b.r - e && b.l < a.r - e && a.t < b.b - e && b.t < a.b - e;
 const uu = [], ul = [], ll = [], out = [];
 for (let i = 0; i < ui.length; i++) { const u = ui[i]; if (u.l < st.left - 1 || u.t < st.top - 1 || u.r > st.right + 1 || u.b > st.bottom + 1) out.push(u.id); for (let j = i + 1; j < ui.length; j++) if (X(u, ui[j])) uu.push(u.id + '×' + ui[j].id); }
 for (const a of labs) for (const u of ui) if (X(a, u)) ul.push(a.n + '×' + u.id);
 for (let i = 0; i < labs.length; i++) for (let j = i + 1; j < labs.length; j++) if (X(labs[i], labs[j])) ll.push(labs[i].n + '×' + labs[j].n);
 return { nUi: ui.length, nLab: labs.length, uu, ul, ll, out, clusters: document.querySelectorAll('#labels .lb.srch[data-n]').length };
}"""
PICK_JS = """() => { const D = window.__DATA__.D, EV = D.evac || { floors: {} };
  const fl = D.floors.filter((f) => f.level >= 1 && EV.floors[f.id]).sort((a, b) => EV.floors[b.id].rooms.length - EV.floors[a.id].rooms.length || a.level - b.level), f = (fl[0] || D.floors[0]);
  const rr = (EV.floors[f.id] ? EV.floors[f.id].rooms : []).filter((r) => r.area > 12 && r.cat !== 'corr').sort((a, b) => a.area - b.area);
  const r = rr[Math.floor(rr.length / 2)] || (EV.floors[f.id] || { rooms: [{ c: [(f.bb[0] + f.bb[2]) / 2, (f.bb[1] + f.bb[3]) / 2] }] }).rooms[0];
  return { f: f.id, bb: f.bb, fire: r.c, span: Math.max(f.bb[2] - f.bb[0], f.bb[3] - f.bb[1]) }; }"""


def _shot(pg, path):                                       # 화면 저장은 참고용 — 소프트웨어 GL 이 느려도 시험은 계속
    try: pg.screenshot(path=path, timeout=120000)
    except Exception as e: print('  (화면 저장 건너뜀)', path, str(e)[:60])


def _launch(b, html, W, H, net, errs, q=''):
    pg = b.new_page(viewport={'width': W, 'height': H}, accept_downloads=True)
    pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    pg.on('request', lambda r: net.append(r.url) if not r.url.startswith(('file:', 'data:', 'blob:')) else None)
    pg.goto('file://' + os.path.abspath(html) + q, timeout=180000)
    for _ in range(40):
        if pg.evaluate("() => !!(window.__viewer && window.__viewer.booted && window.__viewer.booted())"): break
        pg.wait_for_timeout(250)
    pg.wait_for_timeout(600)
    return pg


def _states(pg, info, which):
    """화면 상태를 차례로 만들고 (이름, 겹침 결과) 를 돌려줌"""
    E = pg.evaluate; f, bb, S = info['f'], info['bb'], info['span']; cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    for st in which:
        if st == 'home': E("() => window.__viewer.home()"); pg.wait_for_timeout(1300)
        elif st == 'floor': E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1300)
        elif st == 'fire':
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1300)
            z = E("(f) => window.__viewer._three.G[f].g.position.y", f)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.5, -s * 0.6, s * 0.8)", [cx, cy, z, S]); E("() => window.__viewer.sim.seek(360)"); pg.wait_for_timeout(1300)
        elif st == 'tools':                               # 화재 해제 → 층 선택 → 단면 창 + 거리 재기 안내 띠(위 가운데)
            E("() => document.getElementById('bClr').click()"); pg.wait_for_timeout(300); E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1000)   # 태블릿은 오른쪽 패널이 닫혀 있어 단추를 직접 누름
            E("() => window.__viewer.section({ h: true })"); pg.keyboard.press('m'); pg.wait_for_timeout(700)
        yield st, E(OVERLAP_JS)
        if st == 'tools': pg.keyboard.press('Escape'); pg.keyboard.press('Escape'); E("() => window.__viewer.section(false)"); pg.wait_for_timeout(300)


def v31(b, html, chk, net, errs, shot):
    import re as _re
    for (W, H) in ((1280, 720), (1600, 900), (1920, 1080), (1024, 768)):
        pg = _launch(b, html, W, H, net, errs); info = pg.evaluate(PICK_JS)
        bad, notes = [], []
        for st, r in _states(pg, info, ('home', 'floor', 'fire', 'tools')):
            n = len(r['uu']) + len(r['ul']) + len(r['ll']) + len(r['out'])
            notes.append(f"{st} UI{r['nUi']}·라벨{r['nLab']}{'·묶음' + str(r['clusters']) if r['clusters'] else ''}")
            if n: bad.append(f"{st}: UI×UI {r['uu'][:3]} 라벨×UI {r['ul'][:3]} 라벨×라벨 {r['ll'][:3]} 화면밖 {r['out'][:3]}")
            if shot and st == 'fire': _shot(pg, f"{shot}_v31_{W}x{H}_fire.png")
        U = pg.evaluate("() => window.__viewer.ui()")
        chk(f'겹침 없음 {W}×{H}: 고정 UI·라벨·말풍선 (전체·층·화재 가정·도구)', not bad, ' | '.join(bad) if bad else ' · '.join(notes) + f" · 도구 열 {'2열' if U['dock']['two'] else '1열'}{'(작게)' if U['dock']['compact'] else ''} · 범례 +{U['layMore']} · 미니맵 {'표시' if U['mini'] else '자리 없음'}")
        if (W, H) == (1600, 900):
            # 출동 전 30초 요약 · 점진 공개 · 인명검색 묶음 설명(현재 화면 = 'tools' 뒤 → 화점 다시)
            E = pg.evaluate
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [info['f'], info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1300)
            ks = E("() => [...document.querySelectorAll('#b30 li small')].map((e) => e.textContent)")
            want = ['진입구', '공격계단', '위험물', '인명', '확인 필요']
            okk = len(ks) == 5 and all(any(w in k for k in ks) for w in want)
            pg.click('#b30 li:last-child button'); pg.wait_for_timeout(300)
            chk('출동 전 30초 요약: 5항목(진입구·공격계단·위험물·인명 우선·확인 필요) · 누르면 확인사항', okk and pg.is_visible('#tp-issues'), ' / '.join(E("() => [...document.querySelectorAll('#b30 li')].map((e) => e.querySelector('small').textContent + ': ' + e.querySelector('b').textContent)")))
            dfire = E("() => document.querySelectorAll('#fireRes details.more').length"); fk = E("() => document.querySelectorAll('#fireRes .fk > div').length")
            vis_len = E("() => document.getElementById('fireRes').innerText.length"); all_len = E("() => document.getElementById('fireRes').textContent.length")
            chk('점진 공개: 화재 결과 = 요약 카드 + 접힌 상세(계산 근거·대피로)', dfire >= 1 and fk >= 5 and vis_len < all_len * 0.8, f"요약 {fk}칸 · 접힘 {dfire}개 · 보이는 글 {vis_len}/{all_len}자")
            tip = None
            z = E("(f) => window.__viewer._three.G[f].g.position.y", info['f']); bb = info['bb']; S = info['span']
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.5, -s * 0.6, s * 0.8)", [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, z, S]); pg.wait_for_timeout(900); E("() => window.__viewer.tidy()")
            el = pg.query_selector('#labels .lb.srch:not(.culled)')
            if el and el.is_visible():
                el.hover(); pg.wait_for_timeout(300); tip = pg.inner_text('#mkTip') if pg.is_visible('#mkTip') else None
                el.click(); pg.wait_for_timeout(200); pinned = pg.is_visible('#mkTip'); pg.keyboard.press('Escape'); pg.wait_for_timeout(200)
            nclu = E("() => document.querySelectorAll('#labels .lb.srch[data-n]').length")
            chk('인명검색 번호: 겹치면 묶음(+N) · 올림/누름 = 설명', tip is not None and '인명검색' in tip and pinned, f"묶음 {nclu}개 · 설명: {(tip or '').splitlines()[0] if tip else '-'}")
            # 테마·글자
            bg = lambda: E("() => getComputedStyle(document.getElementById('right')).backgroundColor")
            lum = lambda c: (lambda v: (0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]) / 255)([int(x) for x in _re.findall(r'\d+', c)[:3]])
            b0 = bg(); E("() => window.__viewer.theme('day')"); pg.wait_for_timeout(200); b1 = bg(); E("() => window.__viewer.theme('hc')"); pg.wait_for_timeout(200); b2 = bg()
            if shot: _shot(pg, f"{shot}_v31_hc.png")
            pg.keyboard.press('t'); pg.wait_for_timeout(200); t3 = E("() => window.__viewer.theme()")
            chk('테마: 야간(어두움) · 주간(밝은 패널) · 고대비(검정) · T 키 순환', lum(b0) < 0.15 and lum(b1) > 0.9 and lum(b2) < 0.02 and t3 == 'night', f"패널 밝기 {lum(b0):.2f} → {lum(b1):.2f} → {lum(b2):.2f} · T → {t3}")
            h0 = E("() => document.querySelector('#b30 .hd').getBoundingClientRect().height")
            pg.click('#fzUp'); pg.click('#fzUp'); pg.wait_for_timeout(300); fzv = E("() => window.__viewer.fontScale()")
            h1 = E("() => document.querySelector('#b30 .hd').getBoundingClientRect().height"); lz = E("() => parseFloat(getComputedStyle(document.querySelector('#labels .lb, #labels .co .bx') || document.body).fontSize)")
            r = E(OVERLAP_JS); nov = len(r['uu']) + len(r['ul']) + len(r['ll'])
            pg.click('#fzDn'); pg.click('#fzDn'); pg.wait_for_timeout(200)
            chk('글자 크기: 크게(패널·라벨) · 크게 해도 겹침 없음', fzv > 1.1 and h1 > h0 * 1.1 and nov == 0 and E("() => window.__viewer.fontScale()") == 1, f"배율 {fzv} · 요약 머리 {h0:.0f}→{h1:.0f}px · 라벨 글자 {lz:.1f}px · 겹침 {nov} {(r['uu'] + r['ul'] + r['ll'])[:2]}")
            # 인쇄(Ctrl+P): 인쇄 직전 이벤트 → A4 PDF 1~2쪽
            E("() => window.dispatchEvent(new Event('beforeprint'))"); pg.wait_for_timeout(400)
            sh = E("() => document.getElementById('printSheet').innerText")
            pdf = pg.pdf(format='A4', print_background=True)
            npg = len(_re.findall(rb'/Type\s*/Page[^s]', pdf))
            if shot: open(f"{shot}_v31_print.pdf", 'wb').write(pdf)
            nm = E("() => window.__DATA__.D.building.name")
            chk('인쇄(Ctrl+P): A4 지휘 요약 1~2쪽 · 30초 요약·진입구·계단·확인사항·화재 가정', 1 <= npg <= 2 and nm in sh and '진입구' in sh and '계단' in sh and '확인사항' in sh and '화재 가정' in sh, f"{npg}쪽 · PDF {len(pdf) // 1024}KB")
            # 접근성: 아이콘 단추 이름 · 탭 역할 · 키보드 초점 표시
            noname = E("() => [...document.querySelectorAll('#stage button, header button, #right button, #layers button')].filter((b) => b.offsetParent && !(b.getAttribute('aria-label') || b.textContent.trim() || b.title)).map((b) => b.id || b.className)")
            tabs = E("() => [...document.querySelectorAll('.tab')].every((t) => t.getAttribute('role') === 'tab' && t.hasAttribute('aria-selected'))")
            pressed = E("() => document.querySelectorAll('#layers .lay[aria-pressed], #dock [aria-pressed]').length")
            pg.click('#bname'); [pg.keyboard.press('Tab') for _ in range(3)]
            foc = E("() => { const a = document.activeElement, cs = getComputedStyle(a); return [a.id || a.className, cs.outlineStyle, parseFloat(cs.outlineWidth)]; }")
            chk('접근성: 아이콘 단추 이름(aria-label) · 탭 역할 · 켜짐 상태 · 키보드 초점 테두리', not noname and tabs and pressed >= 10 and foc[1] != 'none' and foc[2] >= 2, f"이름 없는 단추 {noname[:3]} · 켜짐 상태 {pressed}개 · Tab 초점 {foc[0]} 테두리 {foc[1]} {foc[2]}px")
        pg.close()
    # 시작 화면 · WebGL 중단 → 2D
    pg = _launch(b, html, 1400, 860, net, errs)
    raw = open(html, encoding='utf-8').read()
    gone = pg.evaluate("() => !document.getElementById('boot') || document.getElementById('boot').classList.contains('done')")
    lost = pg.evaluate("() => { const gl = window.__viewer._three.renderer.getContext(); const x = gl.getExtension('WEBGL_lose_context'); if (!x) return 'no-ext'; x.loseContext(); return 'ok'; }"); pg.wait_for_timeout(1200)
    st = pg.evaluate("() => window.__viewer.state()"); f = pg.evaluate("() => window.__DATA__.D.floors.find((q) => q.level >= 1).id")
    pg.click(f'#fbar .fbtn[data-f="{f}"]'); pg.wait_for_timeout(500)
    msg = pg.inner_text('#glmsg') if pg.is_visible('#glmsg') else ''
    chk('시작 화면(불러오는 중) · 3D 그래픽 중단 → 2D 대체 보기로 이어 씀', 'id="boot"' in raw and gone and (lost == 'no-ext' or (not st['gl'] and pg.is_visible('#fallback') and '2D' in msg)), f"그래픽 중단 {lost} · gl={st['gl']} · 안내 '{msg[:30]}'")
    pg.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
