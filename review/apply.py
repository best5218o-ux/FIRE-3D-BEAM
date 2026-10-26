# -*- coding: utf-8 -*-
"""검수 결과(교정 JSON) → 수정된 결과 JSON → 3D 지휘화면 HTML + IFC 재생성.
사용: python -m review.apply out/X.json X_corrections.json [--pdf card.pdf] [-o out/X.corrected] [--no-build] [--labels 학습데이터]
교정 JSON 형식은 review/SCHEMA.md 참고. 기호·문·벽은 도면 이미지 px 좌표로 저장돼 있어 엔진 축척·정합이 바뀌어도 다시 적용된다."""
import argparse, copy, datetime, hashlib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review.common import (SCHEMA, load_json, save_json, result_sha, stem_of, find_pdf, plan_images, plan_T, p2g, plan_floors, build_outputs)
from review import xdec

WALL_TOL_M = 0.35


def _shift_floor(F, dx, dy):
    F['tx'] = round(F['tx'] + dx, 3); F['ty'] = round(F['ty'] + dy, 3)
    for w in F['walls']:
        if w['o'] == 'h': w['c'] += dy; w['a'] += dx; w['b'] += dx
        else: w['c'] += dx; w['a'] += dy; w['b'] += dy
    for q in F['symbols']:
        q['x'] = round(q['x'] + dx, 2); q['y'] = round(q['y'] + dy, 2)
        if q.get('box'): q['box'] = [q['box'][0] + dx, q['box'][1] + dy, q['box'][2] + dx, q['box'][3] + dy]
    for d in F.get('doors') or []:
        d['x'] = round(d['x'] + dx, 2); d['y'] = round(d['y'] + dy, 2)


def wall_px2g(T, w):
    s = T[0]
    if w['o'] == 'h':
        (a, c), (b, _) = p2g(T, w['a'], w['c']), p2g(T, w['b'], w['c'])
    else:
        (c, a), (_, b) = p2g(T, w['c'], w['a']), p2g(T, w['c'], w['b'])
    return dict(o=w['o'], c=c, a=min(a, b), b=max(a, b), t=round(max(0.1, w.get('t', 1.5) / s), 2))


def same_wall(u, v, tol=WALL_TOL_M):
    if u['o'] != v['o'] or abs(u['c'] - v['c']) > tol: return False
    ov = min(u['b'], v['b']) - max(u['a'], v['a'])
    return ov > 0.6 * min(u['b'] - u['a'], v['b'] - v['a'])


def sym_px2g(T, q):
    x, y = p2g(T, q['px'], q['py'])
    d = dict(type=q['type'], x=round(x, 2), y=round(y, 2))
    if q.get('box') and not q.get('box_est'):
        (x0, y1), (x1, y0) = p2g(T, q['box'][0], q['box'][1]), p2g(T, q['box'][2], q['box'][3])
        d['box'] = [x0, y0, x1, y1]
    if q.get('conf') is not None: d['conf'] = q['conf']
    src = q.get('src') or 'auto'
    if src != 'auto' or q.get('verified'): d['review'] = 'verified' if src == 'auto' else src
    return d


def door_px2g(T, d):
    x, y = p2g(T, d['px'], d['py'])
    o = dict(o=d['o'], x=round(x, 2), y=round(y, 2), w=round(d['w'] / T[0], 2), kind='door', ext=bool(d.get('ext')))
    if (d.get('src') or 'auto') != 'auto': o['review'] = d['src']
    if d.get('via') == 'xcheck': o['review'] = 'xcheck'
    return o


def recompute_elevs(res):
    """승강기 기호를 고쳤으면 층간 반복 클러스터를 다시 만든다(엔진과 같은 규칙)"""
    import numpy as np
    from firebim.card import level_of
    from firebim.pipeline import cluster_vertical
    fl = res['floors']; n_fl = len(fl)
    pts = [dict(q, floor=fid) for fid, F in fl.items() for q in F['symbols'] if q['type'] == 'ev']
    out = []
    for c in cluster_vertical(pts, 1.6):
        if len(c['floors']) < min(2, n_fl): continue
        bxs = [m['box'] for m in c['m'] if 'box' in m]
        box = [float(np.median([b[k] for b in bxs])) for k in range(4)] if bxs else [c['x'] - 1.2, c['y'] - 1.2, c['x'] + 1.2, c['y'] + 1.2]
        lv = [level_of(f) for f in c['floors']]
        floors = [f for f in sorted(fl, key=level_of) if min(lv) <= level_of(f) <= max(lv)]
        out.append(dict(x=round(c['x'], 2), y=round(c['y'], 2), floors=floors, box=box))
    return out


def apply(res, corr, log=print):
    if corr.get('schema') != SCHEMA: log(f"  주의: 교정 파일 형식 {corr.get('schema')} (기대 {SCHEMA})")
    if (corr.get('building') or {}).get('result_sha') not in (None, result_sha(res)):
        log('  주의: 교정 파일이 다른 변환 결과에서 만들어졌습니다 — 도면 px 좌표 기준으로 적용합니다')
    res = copy.deepcopy(res)
    pf = plan_floors(res)
    stat = dict(sym_added=0, sym_deleted=0, sym_moved=0, sym_retyped=0, doors_added=0, doors_deleted=0, walls_added=0, walls_deleted=0,
                plans_done=0, align_bad=[], shifted=[], xwall_cut=0, xwall_added=0, xdoor_deleted=0, xdoor_added=0, xdoor_dropped=0,
                stairs_added=0, elevs_added=0, vertical_merged=0)
    ev_changed = False
    for key, C in (corr.get('plans') or {}).items():
        i = int(key)
        if i >= len(res['plans']) or i not in pf:
            log(f'  건너뜀: 도면 {i} 없음'); continue
        P = res['plans'][i]
        dx, dy = (C.get('shift_m') or [0, 0])[:2]
        if dx or dy:
            P['tx'] += dx; P['ty'] += dy
            for fid in pf[i]: _shift_floor(res['floors'][fid], dx, dy)
            stat['shifted'].append(dict(plan=i, dx=dx, dy=dy))
        T = plan_T(res, i)
        syms = [sym_px2g(T, q) for q in C.get('symbols') or []]
        xops = xdec.plan_ops(corr, i)                      # 교차검증(규칙↔모델) 판정 → 벽·문 연산(px)
        doors_px, dst = xdec.run_door_ops(C.get('doors') or [], xops, T[0])
        stat['xdoor_deleted'] += dst['door_deleted']; stat['xdoor_added'] += dst['door_added']; stat['xdoor_dropped'] += dst['door_dropped']
        doors = [door_px2g(T, d) for d in doors_px]
        wdel = [wall_px2g(T, w) for w in C.get('walls_deleted') or []]
        wadd = [dict(wall_px2g(T, w), review='added') for w in C.get('walls_added') or []]
        for q in C.get('symbols') or []:
            src = q.get('src') or 'auto'
            stat['sym_added'] += src == 'added'; stat['sym_moved'] += 'moved' in src; stat['sym_retyped'] += 'retyped' in src
            if q['type'] == 'ev' and src != 'auto': ev_changed = True
            if src == 'retyped' and (q.get('orig') or {}).get('type') == 'ev': ev_changed = True
        stat['sym_deleted'] += len(C.get('symbols_deleted') or [])
        ev_changed |= any(q['type'] == 'ev' for q in C.get('symbols_deleted') or [])
        stat['doors_added'] += sum(1 for d in C.get('doors') or [] if d.get('src') == 'added')
        stat['doors_deleted'] += len(C.get('doors_deleted') or [])
        stat['walls_added'] += len(wadd); stat['walls_deleted'] += len(wdel)
        st = C.get('status') or {}
        stat['plans_done'] += bool(st.get('done'))
        has_xw = any(op['op'] in ('cut', 'add') for op in xops)
        for n_f, fid in enumerate(pf[i]):
            F = res['floors'][fid]
            F['symbols'] = copy.deepcopy(syms)
            F['doors'] = copy.deepcopy(doors)
            if wdel or wadd or has_xw:
                F['walls'] = [w for w in F['walls'] if not any(same_wall(w, d) for d in wdel)] + copy.deepcopy(wadd)
                if has_xw:
                    F['walls'], wst = xdec.run_wall_ops(F['walls'], xops, 1.0, conv=lambda w: wall_px2g(T, w))
                    if n_f == 0: stat['xwall_cut'] += wst['cut']; stat['xwall_added'] += wst['added']
                try:
                    from firebim import geom
                    fp = geom.footprint_from_walls(F['walls'])
                    F['footprint_area'] = round(fp.area, 1) if fp is not None else None
                except Exception as e:
                    log('  외곽 재계산 생략:', e)
            F['review'] = dict(align=st.get('align'), scale=st.get('scale'), done=bool(st.get('done')), shift_m=[dx, dy])
            if st.get('align') == 'bad': stat['align_bad'].append(fid)
    # 수직 동선
    if ev_changed:
        try: res['elevs'] = recompute_elevs(res); log(f"  승강기 다시 묶음: {len(res['elevs'])}대")
        except Exception as e: log('  승강기 재계산 생략:', e)
    rej = [v for v in corr.get('vertical') or [] if v.get('decision') == 'reject']
    for kind in ('stairs', 'elevs'):
        keep = []
        for S in res.get(kind) or []:
            cx, cy = (S['box'][0] + S['box'][2]) / 2, (S['box'][1] + S['box'][3]) / 2
            hit = any(v['kind'] == kind and math.hypot(cx - (v['box_m'][0] + v['box_m'][2]) / 2, cy - (v['box_m'][1] + v['box_m'][3]) / 2) < 1.5 for v in rej)
            if not hit: keep.append(S)
        if len(keep) != len(res.get(kind) or []): log(f"  {kind} 오검출 제거 {len(res[kind]) - len(keep)}")
        res[kind] = keep
    add_vertical(res, corr, pf, stat, log)
    # 교차검증 다시 → 검수자 판정 반영
    try:
        from firebim.pipeline import checks
        flags = checks(res['card'], res['floors'], res['stairs'], res['elevs'])
    except Exception as e:
        log('  교차검증 재계산 생략:', e); flags = res.get('flags') or []
    for fid in stat['align_bad']:
        if not any(f['kind'] == 'align' and f.get('floor') == fid for f in flags):
            flags.append(dict(kind='align', floor=fid, conf='검수자 불량 판정'))
    dec = {f['key']: f['decision'] for f in corr.get('flags') or []}
    kept, dismissed = [], []
    for f in flags:
        d = dec.get(f"{f['kind']}|{f.get('floor', '')}" + (f"|{f['field']}" if f.get('field') else ''))     # = review.make.flag_key
        if d == 'reject': dismissed.append(f)
        else:
            if d == 'confirm': f['review'] = 'confirmed'
            kept.append(f)
    res['flags'] = kept
    ch = hashlib.sha256(json.dumps(corr, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
    res['review'] = dict(schema=SCHEMA, applied=datetime.datetime.now().isoformat(timespec='seconds'), corrections_sha=ch,
                         base_result_sha=(corr.get('building') or {}).get('result_sha'), summary=corr.get('summary'), stats=stat,
                         dismissed_flags=dismissed, vertical=corr.get('vertical') or [], vertical_added=corr.get('vertical_added') or [],
                         xcheck=xdec.summary(corr))
    return res, stat


def add_vertical(res, corr, pf, stat, log=print):
    """검수자가 추가한 계단·승강기(도면 px 상자 + 연결층) → res['stairs'|'elevs'].
    엔진의 층간 클러스터와 같은 규칙: 같은 종류가 이미 가까이(중심 1.5 m) 있으면 연결층만 합치고, 연결층은 사이 층까지 채운다."""
    from firebim.card import level_of
    allf = sorted(res['floors'], key=level_of)
    for v in corr.get('vertical_added') or []:
        kind = v.get('kind')
        if kind not in ('stairs', 'elevs'): continue
        i = int(v.get('plan', -1))
        if i not in pf: log(f"  건너뜀: 추가 {kind} 도면 {i} 없음"); continue
        T = plan_T(res, i); b = v['box_px']
        (x0, y1), (x1, y0) = p2g(T, min(b[0], b[2]), min(b[1], b[3])), p2g(T, max(b[0], b[2]), max(b[1], b[3]))
        fl = [f for f in (v.get('floors') or pf[i]) if f in res['floors']] or pf[i]
        lv = [level_of(f) for f in fl]
        floors = [f for f in allf if min(lv) <= level_of(f) <= max(lv)]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        L = res.setdefault(kind, [])
        near = [S for S in L if math.hypot(cx - (S['box'][0] + S['box'][2]) / 2, cy - (S['box'][1] + S['box'][3]) / 2) < 1.5]
        if near:
            S = near[0]; lv2 = [level_of(f) for f in S['floors'] + floors]
            S['floors'] = [f for f in allf if min(lv2) <= level_of(f) <= max(lv2)]; S['review'] = 'merged'
            stat['vertical_merged'] += 1; continue
        L.append(dict(x=round(cx, 2), y=round(cy, 2), floors=floors, box=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)], review='added'))
        stat[kind + '_added'] += 1
    if stat['stairs_added'] or stat['elevs_added']: log(f"  계단·승강기 추가: 계단 {stat['stairs_added']} · 승강기 {stat['elevs_added']}")


def main(result_path, corr_path, pdf=None, out_stem=None, build=True, labels=None):
    res = load_json(result_path); corr = load_json(corr_path)
    out_stem = out_stem or os.path.join(os.path.dirname(os.path.abspath(result_path)), stem_of(result_path) + '.corrected')
    new, st = apply(res, corr)
    new['source'] = dict(res.get('source') or {}, corrections=os.path.abspath(corr_path))
    save_json(new, out_stem + '.json')
    print('검수 반영:', json.dumps({k: v for k, v in st.items() if v}, ensure_ascii=False))
    print(' →', out_stem + '.json')
    imgs = None
    if build or labels:
        pdf = find_pdf(result_path, res, pdf or (corr.get('source') or {}).get('pdf'))
        if not pdf: raise SystemExit('원본 PDF를 찾을 수 없음 — --pdf 로 지정하세요')
        imgs = plan_images(pdf, res)
    if build:
        s = corr.get('summary') or {}
        note = f"검수 반영 {datetime.date.today().isoformat()} (완료 {s.get('plans_done', '?')}/{s.get('plans_total', '?')}개 도면)"
        D, _ = build_outputs(new, imgs, out_stem + '.html', out_stem + '.ifc', note=note)
        try:                                   # 사전 활동계획도 검수 반영본으로(review/plan_sheet.py)
            from review import plan_sheet
            plan_sheet.write(new, D, out_stem + '_활동계획.html', html_name=os.path.basename(out_stem + '.html'), src_note=note)
        except Exception as e:
            print('  (활동계획 생략:', repr(e), ')')
        print(' →', out_stem + '.html', '/', out_stem + '.ifc', '/', out_stem + '_활동계획.html')
    if labels:
        from review.labels import export as export_labels
        export_labels([(corr, res, imgs)], labels)
    return new


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='검수 결과 반영 → 수정 결과·3D·IFC 재생성')
    ap.add_argument('result'); ap.add_argument('corrections')
    ap.add_argument('--pdf'); ap.add_argument('-o', '--out', help='출력 이름(확장자 없이). 기본: <결과>.corrected')
    ap.add_argument('--no-build', action='store_true'); ap.add_argument('--labels', help='학습데이터(COCO) 폴더')
    a = ap.parse_args()
    main(a.result, a.corrections, a.pdf, a.out, not a.no_build, a.labels)
