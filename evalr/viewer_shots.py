# -*- coding: utf-8 -*-
"""3D 지휘화면 v3 기능별 화면 저장(설명·검토용) — 헤드리스 Chromium(SwiftShader).

  python -m evalr.viewer_shots 결과.html 출력폴더 접두어 [1600x900,1920x1080]

저장: 01 전체 · 02 층 상세(두께 벽·문짝·실 용도 입체) · 03 승강로(투시) · 04 단면 · 05 투시 · 06 층 격리 · 07 확산 시뮬레이션 · 08 대원 이동 ·
09 진입로 따라가기 · 10 미니맵 · 11 거리·면적 · 12 브리핑(인쇄용 HTML) · 13 2D 대체 보기 · 14 확인사항 문구 · 15 단축키·화면 속도
v3.1: 16 인명검색 번호 묶음 설명 · 17 범례 +N 펼침 · 18 주간 테마 · 19 고대비 테마 · 20 글자 크게 · 21 인쇄(Ctrl+P, A4 PDF 첫 쪽 그림) · 22 3D 그래픽 중단 → 2D
  + 다른 화면 크기(1280×720 · 1024×768 태블릿 · 1920×1080)마다 전체·화재 가정 2장"""
import sys, os, json


def _boot(pg):                                            # 시작 화면이 걷힐 때까지(그 전엔 시작 화면이 입력을 받음)
    for _ in range(80):
        if pg.evaluate("() => !!(window.__viewer && window.__viewer.booted && window.__viewer.booted()) && !document.querySelector('#boot:not(.done)')"): return
        pg.wait_for_timeout(250)


def shots(html, out, pre, sizes=((1600, 900),), extra=((1280, 720), (1024, 768), (1920, 1080))):
    from playwright.sync_api import sync_playwright
    os.makedirs(out, exist_ok=True)
    url = 'file://' + os.path.abspath(html)
    done = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        for W, H in sizes:
            tag = f'{W}x{H}'
            pg = b.new_page(viewport={'width': W, 'height': H}, accept_downloads=True)
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto(url, timeout=180000); _boot(pg); pg.wait_for_timeout(1500)
            E = lambda s, *a: pg.evaluate(s, *a)
            def snap(name, wait=900):
                pg.wait_for_timeout(wait); E("() => window.__viewer.invalidate()"); pg.wait_for_timeout(500)
                f = os.path.join(out, f'{pre}_{tag}_{name}.png'); pg.screenshot(path=f); done.append(f); print(f, flush=True)
            info = E("""() => { const D = window.__DATA__.D, EV = D.evac || { floors: {} };
              const fl = D.floors.filter((f) => f.level >= 1 && EV.floors[f.id]).sort((a, b) => EV.floors[b.id].rooms.length - EV.floors[a.id].rooms.length || a.level - b.level), f = (fl[0] || D.floors[0]);   // 실이 가장 많은 지상층
              const rr = (EV.floors[f.id] ? EV.floors[f.id].rooms : []).filter((r) => r.area > 12 && r.cat !== 'corr').sort((a, b) => a.area - b.area);
              const r = rr[Math.floor(rr.length / 2)] || (EV.floors[f.id] || { rooms: [{ c: [(f.bb[0] + f.bb[2]) / 2, (f.bb[1] + f.bb[3]) / 2] }] }).rooms[0];
              const big = (EV.floors[f.id] ? EV.floors[f.id].rooms : []).filter((r) => r.area > 12).sort((a, b) => b.area - a.area)[0];
              return { f: f.id, bb: f.bb, fire: r.c, big: big ? big.c : r.c, span: Math.max(f.bb[2] - f.bb[0], f.bb[3] - f.bb[1]) }; }""")
            f, bb, S = info['f'], info['bb'], info['span']
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            # 01 전체
            E("() => window.__viewer.home()"); snap('01_overview', 1400)
            # 02 층 상세(가까이)
            E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1300)
            z = E("(f) => window.__viewer._three.G[f].g.position.y", f)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.42, -s * 0.55, s * 0.5)", [cx, cy, z, S]); snap('02_floor_detail')
            # 03 승강로(전체 + 투시)
            E("() => window.__viewer.home()"); pg.wait_for_timeout(900); E("() => window.__viewer.xray(true)"); snap('03_vertical_shafts', 1200); E("() => window.__viewer.xray(false)")
            # 04 단면
            E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1200)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.55, -s * 0.7, s * 0.6)", [cx, cy, z, S])
            E("() => window.__viewer.section({ h: true, hv: 0.38, v: true, axis: 'x', vv: 0.55 })"); snap('04_section', 1200)
            E("() => window.__viewer.section(false)")
            # 05 투시 · 06 격리
            E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1000); E("() => window.__viewer.xray(true)"); snap('05_xray', 1000); E("() => window.__viewer.xray(false)")
            E("() => window.__viewer.isolate(true)"); snap('06_isolate', 1300); E("() => window.__viewer.isolate(false)")
            # 07 확산 시뮬레이션(6분) — 화재층을 비스듬히 위에서
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1500)
            z = E("(f) => window.__viewer._three.G[f].g.position.y", f)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.5, -s * 0.6, s * 0.8)", [cx, cy, z, S])
            E("() => window.__viewer.sim.seek(360)"); E("() => window.__viewer.tab('issues')"); snap('07_fire_simulation', 1500)
            # 08 대원 이동(도착 + 진입 중)
            C = E("() => window.__viewer.crew()")
            if C['info'] and not C['info'].get('none'):
                E("(t) => window.__viewer.sim.seek(t)", min(890, C['info']['arrive'] + C['info']['tFloor'] + 12))
            snap('08_crew_movement', 1200)
            # 09 따라가기
            E("() => window.__viewer.follow(true)"); snap('09_follow_camera', 2600); E("() => window.__viewer.follow(false)"); pg.wait_for_timeout(400)
            # 10 미니맵 클릭
            mx, my = E("([x, y]) => window.__viewer.miniPx(x, y)", [bb[0] + (bb[2] - bb[0]) * 0.3, bb[1] + (bb[3] - bb[1]) * 0.6]); pg.mouse.click(mx, my)
            snap('10_minimap', 1200)
            # 11 거리·면적
            pg.click('#bClr'); pg.wait_for_timeout(500); E("(f) => window.__viewer.selectFloor(f, true)", f); pg.wait_for_timeout(1300)
            pg.keyboard.press('a'); pg.wait_for_timeout(300)
            sx, sy = E("([x, y]) => window.__viewer.floorPx(x, y)", info['big']); pg.mouse.click(sx, sy); pg.wait_for_timeout(400)
            pg.keyboard.press('m'); pg.wait_for_timeout(300)
            for (x, y) in [(info['big'][0] - 3, info['big'][1] - 1), (info['fire'][0], info['fire'][1])] if abs(info['big'][0] - info['fire'][0]) + abs(info['big'][1] - info['fire'][1]) > 3 else [(info['big'][0] - 4, info['big'][1]), (info['big'][0] + 3, info['big'][1] + 2)]:
                sx, sy = E("([x, y]) => window.__viewer.floorPx(x, y)", [x, y]); pg.mouse.click(sx, sy); pg.wait_for_timeout(400)
            pg.mouse.move(5, 5); snap('11_measure_area', 900); pg.keyboard.press('Escape')
            # 12 브리핑(화재 가정 + 6분)
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1500); E("() => window.__viewer.sim.seek(360)"); pg.wait_for_timeout(600)
            with pg.expect_download() as d: pg.click('#dock [data-tool="brief"]')
            bh = os.path.join(out, f'{pre}_{tag}_12_briefing.html'); d.value.save_as(bh)
            q = b.new_page(viewport={'width': 900, 'height': 1270}); q.goto('file://' + os.path.abspath(bh)); q.wait_for_timeout(800)
            f12 = os.path.join(out, f'{pre}_{tag}_12_briefing.png'); q.screenshot(path=f12, full_page=True); q.close(); done.append(f12); print(f12)
            with pg.expect_download() as d: pg.click('#dock [data-tool="shot"]')
            d.value.save_as(os.path.join(out, f'{pre}_{tag}_12b_screenshot_export.png'))
            # 14 확인사항(현장 문구, 상세 펼침) · 15 도움말
            pg.click('#bClr'); pg.wait_for_timeout(400); pg.keyboard.press('Delete'); E("() => window.__viewer.home()"); E("() => window.__viewer.tab('issues')")
            pg.evaluate("() => { const d = document.querySelector('#issues details'); if (d) d.open = true; }"); snap('14_issue_wording', 900)
            pg.keyboard.press('?'); snap('15_shortcuts_perf', 700); pg.keyboard.press('Escape')
            # v3.1 ── 16 인명검색 번호 묶음 설명(올림) · 17 범례 +N · 18 주간 · 19 고대비 · 20 글자 크게 · 21 인쇄 · 22 그래픽 중단 → 2D
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1500)
            z = E("(f) => window.__viewer._three.G[f].g.position.y", f)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.5, -s * 0.6, s * 0.8)", [cx, cy, z, S]); pg.wait_for_timeout(1200); E("() => window.__viewer.tidy()")
            el = pg.query_selector('#labels .lb.srch[data-n]:not(.culled)') or pg.query_selector('#labels .lb.srch:not(.culled)')
            if el:
                try: el.hover(timeout=3000)
                except Exception: pass
            snap('16_marker_cluster_tip', 400); pg.mouse.move(5, 5)
            if pg.is_visible('#layMore'): pg.click('#layMore')
            snap('17_layers_more', 400)
            if pg.is_visible('#layMore'): pg.click('#layMore')
            E("() => window.__viewer.sim.seek(360)"); E("() => window.__viewer.theme('day')"); snap('18_theme_day', 800)
            E("() => window.__viewer.theme('hc')"); snap('19_theme_high_contrast', 800); E("() => window.__viewer.theme('night')")
            E("() => window.__viewer.fontScale(1.25)"); snap('20_font_large', 900); E("() => window.__viewer.fontScale(1)")
            E("() => window.dispatchEvent(new Event('beforeprint'))"); pg.wait_for_timeout(400)
            pdf = os.path.join(out, f'{pre}_{tag}_21_print_A4.pdf'); open(pdf, 'wb').write(pg.pdf(format='A4', print_background=True)); done.append(pdf); print(pdf)
            pg.emulate_media(media='print'); pg.set_viewport_size({'width': 794, 'height': 1123}); pg.wait_for_timeout(500)
            f21 = os.path.join(out, f'{pre}_{tag}_21_print_A4.png'); pg.screenshot(path=f21, full_page=True); done.append(f21); print(f21)
            pg.emulate_media(media='screen'); pg.set_viewport_size({'width': W, 'height': H}); pg.wait_for_timeout(600)
            lost = E("() => { const x = window.__viewer._three.renderer.getContext().getExtension('WEBGL_lose_context'); if (x) x.loseContext(); return !!x; }")
            if lost: snap('22_webgl_lost_2d', 1500)
            pg.close()
            # 13 2D 대체 보기
            pg = b.new_page(viewport={'width': W, 'height': H}); pg.goto(url + '?nogl=1', timeout=120000); _boot(pg); pg.wait_for_timeout(1000)
            pg.evaluate("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1200); pg.evaluate("() => window.__viewer.sim.seek(360)")
            pg.wait_for_timeout(800); f13 = os.path.join(out, f'{pre}_{tag}_13_fallback_2d.png'); pg.screenshot(path=f13); done.append(f13); print(f13); pg.close()
            if errs: print('page errors', errs[:3])
        # 다른 화면 크기: 전체 · 화재 가정(확산 6분)
        for W, H in extra:
            tag = f'{W}x{H}'
            pg = b.new_page(viewport={'width': W, 'height': H}); pg.goto(url, timeout=180000); _boot(pg); pg.wait_for_timeout(1500)
            E = lambda s, *a: pg.evaluate(s, *a)
            def snap2(name, wait=900):
                pg.wait_for_timeout(wait); E("() => window.__viewer.invalidate()"); pg.wait_for_timeout(500)
                f = os.path.join(out, f'{pre}_{tag}_{name}.png'); pg.screenshot(path=f); done.append(f); print(f, flush=True)
            E("() => window.__viewer.home()"); snap2('01_overview', 1400)
            E("([f, x, y]) => window.__viewer.setFire(f, x, y)", [f, info['fire'][0], info['fire'][1]]); pg.wait_for_timeout(1500)
            z = E("(f) => window.__viewer._three.G[f].g.position.y", f)
            E("([x, y, z, s]) => window.__viewer.cam(x, y, z, s * 0.5, -s * 0.6, s * 0.8)", [cx, cy, z, S]); E("() => window.__viewer.sim.seek(360)"); snap2('07_fire_simulation', 1500)
            pg.close()
        b.close()
    return done


if __name__ == '__main__':
    sz = tuple(tuple(int(v) for v in s.split('x')) for s in (sys.argv[4] if len(sys.argv) > 4 else '1600x900').split(','))
    shots(sys.argv[1], sys.argv[2], sys.argv[3], sz)
