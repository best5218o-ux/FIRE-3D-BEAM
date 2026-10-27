# -*- coding: utf-8 -*-
"""인명검색 우선순위·훈련 채점 평가 (가상 건물).

  python -m evalr.search_eval data out_search_dev [--n 18] [--floors 3] [--pts 2]
  python -m evalr.search_eval data_hold out_search_hold

건물마다: 정보카드 변환(run keep=True) → 지휘화면 데이터(D) → 단일 HTML(임시) → 헤드리스 Chromium 에서 화점 시나리오를 넣고
3D 지휘화면이 실제로 계산한 인명검색 순서(window.__viewer.search())를 받아 검사한다. 임시 HTML 은 지운다(디스크 절약).

(1) 규칙 준수(구현 검사 — 정렬 키가 설계대로 적용되는지):
    R1 층 순서(화재층 → 직상층 → 위층), R2 같은 층 안 대피 곤란 분류 순서, R3 화재층 '고립' 실이 다른 실보다 먼저,
    R4 같은 (층, 분류, 용도 등급) 안 공격계단 기준 보행거리 단조 증가, R5 같은 (층, 분류) 안 용도 등급 단조, R6 화재층 실 누락·중복 없음,
    R7 같은 시나리오 반복 시 같은 순서(결정성)
(2) 안정성: 화점을 ±1 m(4방향) 옮겼을 때 화재층 순서의 Kendall τ(공통 실)와 1순위 유지율, 상위 5 겹침
(3) 실 용도 범주 추정(정답 있음): 생성기가 심은 실 이름·유형(gt.floors[f].rooms) + 층 용도 → 기대 범주, 엔진 실 대표점이 든 정답 실과 비교.
    ※ 층 용도 → 범주 규칙은 추정기와 같은 표를 쓰므로 이 지표는 주로 실 분할·모양 규칙(복도·대공간·작은 실)을 잰다.
    ※ 거동 불가 실(중환자실 등)은 실명 OCR 없이는 구별 불가 → 재현율을 따로 보고(0 이면 0).
(4) 훈련 채점 기능 시험: 산출안과 같은 답 → 경로·계단·호스·검색 만점, 틀린 답(다른 계단·호스 +3·검색 역순) → 감점.
재실자 위치 정답은 생성기에 없으므로 '검색 순서 정확도'는 만들지 않는다(정답 없는 지표를 지어내지 않음)."""
import json, os, sys, glob, math, random, shutil, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def gt_category(room, floor_use):
    """생성기 정답 실 → 기대 범주(인명검색 범주표)"""
    t, nm = room.get('type'), room.get('name') or ''
    if t == 'critical': return 'crit'
    if t == 'ward':
        return 'occupied' if '교실' in floor_use else 'sleep'
    if any(w in nm for w in ('기계실', '전기실', '창고', '저수조', '주차장')): return 'svc'
    if any(w in nm for w in ('보관장', '작업장')): return 'hall'
    if '승강기홀' in nm: return 'corr'
    return 'occupied'          # 로비·사무실·원무과·약국·진료실·상담실·방재실·휴게실·관리실·사무공간


def offset(gt, res):
    from evalr.score import eng_map, gt_map
    T = gt['transforms']; E = res['floors']
    diffs = {}
    for f in [f for f in gt['floors'] if f in E and f in T]:
        t = T[f]; pts = [(0, 0), (t['w'], 0), (0, t['h']), (t['w'], t['h']), (t['w'] / 2, t['h'] / 2)]
        diffs[f] = [np.subtract(eng_map(E[f], *p), gt_map(t, *p)) for p in pts]
    g = np.median([d[-1] for d in diffs.values()], axis=0) if diffs else np.zeros(2)
    resid = {f: float(max(np.hypot(*(d - g)) for d in ds)) for f, ds in diffs.items()}
    return float(g[0]), float(g[1]), resid


def cat_accuracy(gt, res, D):
    """엔진 실 대표점이 든 정답 실 → 기대 범주. 정답 실 밖이면: 생성기 대피 동선(복도 중심선 gt.floors[f].evac)이 5 m 이상 지나는 실 = 복도,
    그 밖(생성기가 이름을 붙이지 않은 남는 공간 등)은 정답 없음 → 채점 제외('unlabeled' 수만 보고)"""
    from shapely.geometry import Polygon, Point, LineString
    gx, gy, resid = offset(gt, res)
    conf = {}
    for f, rec in (D.get('evac') or {}).get('floors', {}).items():
        if f not in gt['floors'] or resid.get(f, 9) > 3: continue
        Fg = gt['floors'][f]; use = Fg.get('use') or ''
        G = [(Polygon(r['poly']), r) for r in Fg['rooms']]
        lines = [LineString(a) for a in Fg.get('evac') or [] if len(a) >= 2]
        for r in rec['rooms']:
            p = Point(r['c'][0] - gx, r['c'][1] - gy)
            hit = next((gr for pg, gr in G if pg.contains(p)), None)
            if hit: exp = gt_category(hit, use)
            else:
                pe = Polygon([(x - gx, y - gy) for x, y in r['poly']]).buffer(0)
                exp = 'corr' if any(pe.intersection(l).length >= 5.0 for l in lines) else 'unlabeled'
            if exp == 'unlabeled': conf.setdefault('_unlabeled', {}); conf['_unlabeled'][r['cat']] = conf['_unlabeled'].get(r['cat'], 0) + 1; continue
            conf.setdefault(exp, {}); conf[exp][r['cat']] = conf[exp].get(r['cat'], 0) + 1
    return conf


def make_html(d, tmp):
    from fire3dbeam.pipeline import run
    from fire3dbeam.export import build, to_viewer_bim
    from build_viewer import make
    res = run(os.path.join(d, 'card.pdf'), keep=True)
    D, BIM = build(res)
    for S in BIM['stairs'] + BIM['elevs']: S.setdefault('gids', [])          # IFC 를 만들지 않으므로 GlobalId 없음
    name = os.path.basename(d.rstrip('/'))
    dj = os.path.join(tmp, name + '.data.json'); html = os.path.join(tmp, name + '.html')
    json.dump(dict(D=D, BIM=to_viewer_bim(BIM)), open(dj, 'w'), ensure_ascii=False)
    os.environ['FIRE3DBEAM_PREBUILT'] = '1'
    make(dj, html)
    os.remove(dj)
    return res, D, html


JS_RUN = """([fid, x, y, mode]) => { const V = window.__viewer; const s = V.evalFire(fid, x, y, mode); const p = V.plan();
  return { items: s ? s.items.map(z => [z.fid, z.k, z.fr, z.ci, z.band, z.d, z.cls, z.cat]) : [], floors: s ? s.floors : [], attack: p && p.attack, mode: s && s.mode }; }"""


def check_rules(R, n_fire_rooms):
    it = R['items']; bad = {k: 0 for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')}
    for a, b in zip(it, it[1:]):
        if b[2] < a[2]: bad['R1'] += 1
        if b[2] == a[2] and b[3] < a[3]: bad['R2'] += 1
        if b[2] == a[2] and b[3] == a[3] and b[4] < a[4]: bad['R5'] += 1
        if b[2] == a[2] and b[3] == a[3] and b[4] == a[4] and (a[5] if a[5] is not None else 1e9) > (b[5] if b[5] is not None else 1e9) + 1e-9: bad['R4'] += 1
    fire = [z for z in it if z[2] == 0]
    seen_non = False
    for z in fire:
        if z[6] != 'iso': seen_non = True
        elif seen_non: bad['R3'] += 1
    ks = [z[1] for z in fire]
    if len(ks) != len(set(ks)) or len(ks) != n_fire_rooms: bad['R6'] += 1
    return bad


def tau(a, b):
    """두 순서(실 번호 목록)의 공통 실 Kendall τ-a"""
    common = [k for k in a if k in set(b)]
    rb = {k: i for i, k in enumerate(b)}
    c = d = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s = rb[common[j]] - rb[common[i]]
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (c + d) if c + d else None


def train_test(pg, R):
    """훈련 채점 시험: 같은 시나리오에서 (가) 산출안과 같은 답, (나) 틀린 답"""
    out = {}
    for tag in ('ref', 'wrong'):
        r = pg.evaluate("""([tag]) => { const V = window.__viewer, T = V.trainer(), P = V.plan(); T.quit(); T.start('cur');
          const Q = V.plan(), ents = window.__DATA__.BIM.floors[Q.ground].doors.filter(d => d[0] === 'entrance' || (d[0] === 'exit' && d[6]));
          let ei = Q.entry ? ents.findIndex(d => Math.abs(d[2] - Q.entry.x) < 1e-6 && Math.abs(d[3] - Q.entry.y) < 1e-6) : 0; if (ei < 0) ei = 0;
          const fireRooms = Q.search.filter(z => z.fid === Q.fid).map(z => z.k);
          let st = Q.attack, hose = Q.hoseN, order = fireRooms.slice(0, 6);
          if (tag === 'wrong') { st = Q.cand.find(c => c !== Q.attack) || Q.attack; hose = (Q.hoseN || 0) + 3; order = fireRooms.slice(0, 6).reverse(); }
          T.answer(ei, st, hose, order); T.submit(); const R = T.score(); return { score: R.score100, items: R.items.map(z => [z.key, z.pts, z.max]), hidden_before: true }; }""", [tag])
        out[tag] = r
    pg.evaluate("() => window.__viewer.trainer().quit()")
    return out


def main(data, outdir, n=18, nfl=3, npts=2):
    from playwright.sync_api import sync_playwright
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='srch_')
    dirs = sorted(glob.glob(os.path.join(data, '*/')))[:n]
    if os.environ.get('ONLY'): dirs = [d for d in dirs if os.path.basename(d.rstrip('/')) in os.environ['ONLY'].split(',')]
    rows = []
    conf_all = {}
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        for d in dirs:
            name = os.path.basename(d.rstrip('/')); t0 = time.time()
            gt = json.load(open(os.path.join(d, 'gt.json')))
            try:
                res, D, html = make_html(d, tmp)
            except Exception as e:
                rows.append(dict(name=name, error=repr(e))); continue
            conf = cat_accuracy(gt, res, D)
            for e, row in conf.items():
                for c, v in row.items(): conf_all.setdefault(e, {}); conf_all[e][c] = conf_all[e].get(c, 0) + v
            pg = b.new_page(viewport={'width': 480, 'height': 320})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            pg.goto('file://' + html); pg.wait_for_timeout(1500)
            EV = D.get('evac') or {}
            rng = random.Random(name)
            fl = [f['id'] for f in sorted(D['floors'], key=lambda f: f['level']) if f['level'] >= 0 and f['id'] in EV.get('floors', {})]
            fl = sorted(rng.sample(fl, min(nfl, len(fl))), key=lambda f: [x['id'] for x in sorted(D['floors'], key=lambda f: f['level'])].index(f))
            bad = {k: 0 for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7')}
            nsc = 0; taus = []; top1 = []; top5 = []; sp_cmp = []; cls_cnt = {}; cat_top = {'day': {}, 'night': {}}; nrooms = []
            for f in fl:
                rooms = EV['floors'][f]['rooms']
                cand = [k for k, r in enumerate(rooms) if (r['area'] or 0) > 12 and r.get('cat') != 'corr']
                for k in rng.sample(cand, min(npts, len(cand))):
                    x, y = rooms[k]['c']
                    for mode in ('night', 'day'):
                        R = pg.evaluate(JS_RUN, [f, x, y, mode]); nsc += 1
                        bb = check_rules(R, len(rooms))
                        for kk, v in bb.items(): bad[kk] += v
                        fire_o = [z[1] for z in R['items'] if z[2] == 0]
                        nrooms.append(len(R['items']))
                        for z in R['items']: cls_cnt[z[6]] = cls_cnt.get(z[6], 0) + 1
                        t1 = R['items'][0] if R['items'] else None
                        if t1: cat_top[mode][t1[7]] = cat_top[mode].get(t1[7], 0) + 1
                        if mode == 'night':
                            # 방수구 가점(ACT.atk_w_standpipe_m = 10 m)을 켜면 공격계단이 바뀌는가(기본 0)
                            pg.evaluate("() => window.__viewer.setAct({ atk_w_standpipe_m: 10 })")
                            Rs = pg.evaluate(JS_RUN, [f, x, y, mode]); pg.evaluate("() => window.__viewer.setAct({ atk_w_standpipe_m: 0 })")
                            sp_cmp.append(Rs['attack'] == R['attack'])
                            R2 = pg.evaluate(JS_RUN, [f, x, y, mode])
                            if R2['items'] != R['items']: bad['R7'] += 1
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                Rp = pg.evaluate(JS_RUN, [f, x + dx, y + dy, mode])
                                po = [z[1] for z in Rp['items'] if z[2] == 0]
                                t = tau(fire_o, po)
                                if t is not None: taus.append(t)
                                if fire_o and po: top1.append(fire_o[0] == po[0]); top5.append(len(set(fire_o[:5]) & set(po[:5])) / max(1, min(5, len(fire_o))))
            tr = train_test(pg, None) if nsc else None
            pg.close(); os.remove(html)
            rows.append(dict(name=name, scenarios=nsc, bad=bad, tau=taus, top1=top1, top5=top5, cls=cls_cnt, top_cat=cat_top, rooms_mean=float(np.mean(nrooms)) if nrooms else 0, standpipe_same=sp_cmp,
                             cat=conf, train=tr, page_errors=errs[:5], sec=round(time.time() - t0, 1)))
            print(name, nsc, bad, f"tau={np.mean(taus):.3f}" if taus else '', f"top1={np.mean(top1):.2f}" if top1 else '', 'train', tr and {k: v['score'] for k, v in tr.items()}, 'err', len(errs), f"{time.time() - t0:.0f}s", flush=True)
        b.close()
    shutil.rmtree(tmp, ignore_errors=True)
    ok = [r for r in rows if 'error' not in r]
    T = lambda k: [x for r in ok for x in r[k]]
    unl = conf_all.pop('_unlabeled', {})
    cats = sorted({c for row in conf_all.values() for c in row} | set(conf_all))
    tot = sum(v for row in conf_all.values() for v in row.values()); hit = sum(conf_all.get(c, {}).get(c, 0) for c in cats)
    summ = dict(buildings=len(ok), errors=[r['name'] for r in rows if 'error' in r], scenarios=sum(r['scenarios'] for r in ok),
                rule_violations={k: sum(r['bad'][k] for r in ok) for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7')},
                stability=dict(perturb_m=1.0, n=len(T('tau')), tau_mean=round(float(np.mean(T('tau'))), 3) if T('tau') else None,
                               tau_median=round(float(np.median(T('tau'))), 3) if T('tau') else None, top1_kept=round(float(np.mean(T('top1'))), 3) if T('top1') else None,
                               top5_overlap=round(float(np.mean(T('top5'))), 3) if T('top5') else None),
                class_counts={k: sum(r['cls'].get(k, 0) for r in ok) for k in sorted({k for r in ok for k in r['cls']})},
                category=dict(unlabeled=unl, n=tot, acc=round(hit / tot, 3) if tot else None, confusion=conf_all,
                              recall={c: f"{conf_all.get(c, {}).get(c, 0)}/{sum(conf_all.get(c, {}).values())}" for c in cats if c in conf_all}),
                train=dict(ref_scores=[r['train']['ref']['score'] for r in ok if r.get('train')], wrong_scores=[r['train']['wrong']['score'] for r in ok if r.get('train')],
                           ref_items_full=sum(1 for r in ok if r.get('train') and all(p >= m - 1e-6 for k, p, m in r['train']['ref']['items'] if k != 'time')),
                           n=sum(1 for r in ok if r.get('train'))),
                standpipe_bonus_same_attack=f"{sum(T('standpipe_same'))}/{len(T('standpipe_same'))}",
                page_errors=sum(len(r['page_errors']) for r in ok))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(outdir, 'search_eval.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ


def cat_only(data, outdir):
    """범주 추정 정확도만(브라우저 없이) — 변환 → D → 정답 비교"""
    from fire3dbeam.pipeline import run
    from fire3dbeam.export import build
    conf_all = {}; rows = []
    for d in sorted(glob.glob(os.path.join(data, '*/'))):
        gt = json.load(open(os.path.join(d, 'gt.json')))
        res = run(os.path.join(d, 'card.pdf'), keep=True); D, _ = build(res)
        c = cat_accuracy(gt, res, D); rows.append(dict(name=os.path.basename(d.rstrip('/')), cat=c))
        for e, row in c.items():
            for k, v in row.items(): conf_all.setdefault(e, {}); conf_all[e][k] = conf_all[e].get(k, 0) + v
        print(rows[-1], flush=True)
    unl = conf_all.pop('_unlabeled', {})
    tot = sum(v for row in conf_all.values() for v in row.values()); hit = sum(conf_all.get(c, {}).get(c, 0) for c in conf_all)
    S = dict(unlabeled=unl, n=tot, acc=round(hit / tot, 3) if tot else None, confusion=conf_all,
             recall={c: f"{conf_all[c].get(c, 0)}/{sum(conf_all[c].values())}" for c in conf_all},
             precision={c: f"{conf_all.get(c, {}).get(c, 0)}/{sum(row.get(c, 0) for row in conf_all.values())}" for c in {k for row in conf_all.values() for k in row}})
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(summary=S, rows=rows), open(os.path.join(outdir, 'search_cat.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False))


if __name__ == '__main__':
    if '--cat-only' in sys.argv: sys.argv.remove('--cat-only'); cat_only(sys.argv[1], sys.argv[2]); sys.exit(0)
    a = sys.argv[1:]
    kw = {}
    for k in ('--n', '--floors', '--pts'):
        if k in a: i = a.index(k); kw[{'--n': 'n', '--floors': 'nfl', '--pts': 'npts'}[k]] = int(a[i + 1]); del a[i:i + 2]
    main(a[0], a[1], **kw)
