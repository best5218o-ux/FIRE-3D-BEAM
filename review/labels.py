# -*- coding: utf-8 -*-
"""⑤ 학습 순환: 검수 결과(교정 JSON) → 학습 라벨(COCO 형식) + 검수 기반 성능(자동 vs 사람 확정).
사용: python -m review.labels X_corrections.json [Y_corrections.json ...] --out 학습데이터 [--all]
- 기본은 '검수 완료'로 표시한 도면만 정답으로 사용(--all 이면 전부).
- 출력: <out>/images/<건물>_p<도면>.png, <out>/annotations.json(COCO, 누적), <out>/review_metrics.json(누적)
- 실제 정보카드에서 나온 이미지·라벨은 내부 PC 밖으로 반출 금지."""
import argparse, datetime, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review.common import TYPES, KO, load_json, save_json, find_pdf, plan_images, slug
from review import xdec

CATS = [dict(id=i + 1, name=t, name_ko=KO[t], supercategory='symbol') for i, t in enumerate(TYPES)] + \
       [dict(id=6, name='door', name_ko='문', supercategory='opening'), dict(id=7, name='wall', name_ko='벽', supercategory='structure'),
        # 8~11 (추가, 1~7 번호는 그대로): 교차검증 판정의 명시적 음성(여기는 벽/문이 아님) · 계단·승강기 샤프트 영역
        dict(id=8, name='wall_rejected', name_ko='벽 아님(검수 판정)', supercategory='negative'),
        dict(id=9, name='door_rejected', name_ko='문 아님(검수 판정)', supercategory='negative'),
        dict(id=10, name='stair', name_ko='계단', supercategory='circulation'), dict(id=11, name='elevator_shaft', name_ko='승강기 샤프트', supercategory='circulation')]
CID = {c['name']: c['id'] for c in CATS}


def _bbox(x0, y0, x1, y1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def plan_annotations(C):
    """교정 도면 1장 → (기호·문·벽) COCO 주석 목록(이미지 id 미정)"""
    s = C.get('s') or 10.0; out = []
    for q in C.get('symbols') or []:
        b = q['box']
        out.append(dict(category_id=CID[q['type']], bbox=_bbox(*b), attributes=dict(src=q.get('src', 'auto'), conf=q.get('conf'), verified=q.get('verified', False),
                                                                                   box_estimated=bool(q.get('box_est')), center=[q['px'], q['py']])))
    for d in (C['doors_final'] if 'doors_final' in C else C.get('doors') or []):
        hw, ht = d['w'] / 2, max(2.0, 0.15 * s)
        b = (d['px'] - hw, d['py'] - ht, d['px'] + hw, d['py'] + ht) if d['o'] == 'h' else (d['px'] - ht, d['py'] - hw, d['px'] + ht, d['py'] + hw)
        out.append(dict(category_id=CID['door'], bbox=_bbox(*b), attributes=dict(src=d.get('src', 'auto'), orientation=d['o'], external=d.get('ext', False),
                                                                                   segment=[[b[0], d['py']], [b[2], d['py']]] if d['o'] == 'h' else [[d['px'], b[1]], [d['px'], b[3]]])))
    del_ids = {w['id'] for w in C.get('walls_deleted') or []}
    walls = [w for w in (C.get('walls_final') or []) if w.get('id') not in del_ids]
    for w in walls:
        t = max(1.5, w.get('t', 1.5)) / 2
        if w['o'] == 'h': x0, y0, x1, y1 = w['a'], w['c'] - t, w['b'], w['c'] + t; seg = [[w['a'], w['c']], [w['b'], w['c']]]
        else: x0, y0, x1, y1 = w['c'] - t, w['a'], w['c'] + t, w['b']; seg = [[w['c'], w['a']], [w['c'], w['b']]]
        at = dict(src=w.get('src', 'auto'), segment=seg, thickness_px=round(2 * t, 1))
        if w.get('review') == 'xcheck': at['src'] = 'xcheck'
        if w.get('confirmed'): at['confirmed'] = True          # 교차검증 불일치에서 검수자가 벽으로 확정
        out.append(dict(category_id=CID['wall'], bbox=_bbox(x0, y0, x1, y1), segmentation=[[x0, y0, x1, y0, x1, y1, x0, y1]], attributes=at))
    # 교차검증 판정의 음성 라벨: 규칙만 본 벽을 '모델이 맞음'(벽 아님) · 모델만 본 벽/문을 '규칙이 맞음'(벽·문 아님)
    for n in C.get('negatives') or []:
        if n['cat'] == 'wall_rejected':
            w = n['w']; t = max(w.get('t', 1.5), xdec.NEG_T_M * s) / 2
            if w['o'] == 'h': x0, y0, x1, y1 = w['a'], w['c'] - t, w['b'], w['c'] + t
            else: x0, y0, x1, y1 = w['c'] - t, w['a'], w['c'] + t, w['b']
            out.append(dict(category_id=CID['wall_rejected'], bbox=_bbox(x0, y0, x1, y1), segmentation=[[x0, y0, x1, y0, x1, y1, x0, y1]],
                            attributes=dict(src='xcheck', item=n.get('id'), decision=n.get('decision'))))
        else:
            d = n['d']; hw, ht = max(d['w'], 0.6 * s) / 2, max(2.0, 0.15 * s)
            b = (d['px'] - hw, d['py'] - ht, d['px'] + hw, d['py'] + ht) if d['o'] == 'h' else (d['px'] - ht, d['py'] - hw, d['px'] + ht, d['py'] + hw)
            out.append(dict(category_id=CID['door_rejected'], bbox=_bbox(*b), attributes=dict(src='xcheck', item=n.get('id'), orientation=d['o'])))
    for v in C.get('vertical_final') or []:              # 계단·승강기 샤프트(자동 − 오검출 + 검수자 추가)
        b = v['box_px']
        out.append(dict(category_id=CID['stair' if v['kind'] == 'stairs' else 'elevator_shaft'], bbox=_bbox(*b),
                        attributes=dict(src=v['src'], floors=v['floors'], name=v.get('name'))))
    for a in out:
        a['area'] = round(a['bbox'][2] * a['bbox'][3], 1); a['iscrowd'] = 0
    return out


def _prf(tp, n_auto, n_final):
    return dict(precision=round(tp / n_auto, 3) if n_auto else None, recall=round(tp / n_final, 3) if n_final else None, n_auto=n_auto, n_final=n_final)


def review_metrics(corr, only_done=True):
    """자동 인식 vs 사람이 확정한 결과(검수 완료 도면) → 종류별 정밀도·재현율. 정답 없는 실제 건물에서의 성능 지표."""
    agg = {t: [0, 0, 0] for t in TYPES + ['door']}; wall = [0.0, 0.0, 0.0]; n = 0
    for C in (corr.get('plans') or {}).values():
        if only_done and not (C.get('status') or {}).get('done'): continue
        n += 1
        final = C.get('symbols') or []
        auto = [q for q in final if q.get('src', 'auto') != 'added'] + (C.get('symbols_deleted') or [])
        for t in TYPES:
            a_t = [q for q in auto if ((q.get('orig') or {}).get('type') or q['type']) == t]
            f_t = [q for q in final if q['type'] == t]
            tp = sum(1 for q in final if q['type'] == t and q.get('src', 'auto') in ('auto', 'moved') and ((q.get('orig') or {}).get('type') or q['type']) == t)
            agg[t][0] += tp; agg[t][1] += len(a_t); agg[t][2] += len(f_t)
        fd = C.get('doors') or []
        tp = sum(1 for d in fd if d.get('src', 'auto') in ('auto', 'edited'))
        agg['door'][0] += tp; agg['door'][1] += tp + len(C.get('doors_deleted') or []); agg['door'][2] += len(fd)
        L = lambda w: w['b'] - w['a']
        wall[0] += sum(L(w) for w in C.get('walls_deleted') or []); wall[1] += sum(L(w) for w in C.get('walls_added') or [])
        wall[2] += sum(L(w) for w in C.get('walls_final') or [])
    out = dict(building=(corr.get('building') or {}).get('name'), id=(corr.get('building') or {}).get('id'), plans_used=n,
               symbols={t: _prf(*agg[t]) for t in TYPES if agg[t][1] or agg[t][2]}, doors=_prf(*agg['door']))
    if wall[2]:
        tot = wall[2]
        out['walls'] = dict(precision=round(1 - wall[0] / tot, 3), recall=round((tot - wall[0]) / max(1e-6, tot - wall[0] + wall[1]), 3), deleted_px=round(wall[0]), added_px=round(wall[1]))
    used = {k for k, C in (corr.get('plans') or {}).items() if not only_done or (C.get('status') or {}).get('done')}
    out['xcheck'] = xdec.summary(corr, used)
    va = [v for v in corr.get('vertical_added') or [] if str(v.get('plan')) in used]
    out['vertical'] = dict(rejected=sum(1 for v in corr.get('vertical') or [] if v.get('decision') == 'reject'),
                           added_stairs=sum(1 for v in va if v.get('kind') == 'stairs'), added_elevs=sum(1 for v in va if v.get('kind') == 'elevs'))
    fl = corr.get('flags') or []
    out['flags'] = dict(confirmed=sum(1 for f in fl if f['decision'] == 'confirm'), dismissed=sum(1 for f in fl if f['decision'] == 'reject'))
    st = [C.get('status') or {} for C in (corr.get('plans') or {}).values()]
    out['align'] = dict(ok=sum(1 for s in st if s.get('align') == 'ok'), bad=sum(1 for s in st if s.get('align') == 'bad'),
                        shifted=sum(1 for C in (corr.get('plans') or {}).values() if any(C.get('shift_m') or [0, 0])))
    sm = corr.get('summary') or {}
    out['review_load'] = dict(items_total=sm.get('items_total'), items_done=sm.get('items_done'), auto_total=sm.get('auto_total'))
    return out


def with_walls(corr, res):
    """교정 파일에는 벽 '변경분'만 있으므로 원 결과에서 최종 벽 목록(px)을 복원.
    + 교차검증 판정(xcheck: 벽 자르기·추가, 문 삭제·추가) 반영 → walls_final / doors_final / negatives(음성 라벨)
    + 계단·승강기: 자동(오검출 판정 제외) + 검수자 추가 → vertical_final (도면 px 상자)"""
    import math
    from review.common import plan_T, plan_floors, g2p, p2g
    from review.make import walls_px
    pf = plan_floors(res)
    rej = [v for v in corr.get('vertical') or [] if v.get('decision') == 'reject']
    for key, C in (corr.get('plans') or {}).items():
        i = int(key)
        if i not in pf: continue
        T = plan_T(res, i); s = T[0]
        base = [dict(w, id=f'p{i}w{k}') for k, w in enumerate(walls_px(res['floors'][pf[i][0]]['walls'], T))]
        W = base + list(C.get('walls_added') or [])
        ops = xdec.plan_ops(corr, i)
        W, _ = xdec.run_wall_ops(W, ops, s)
        for op in ops:                                    # 확정 표시(같은 선·겹침)
            if op['op'] != 'confirm': continue
            w = op['w']
            for v in W:
                if v['o'] == w['o'] and abs(v['c'] - w['c']) < xdec.SNAP_M * s and min(v['b'], w['b']) - max(v['a'], w['a']) > 0: v['confirmed'] = True
        C['walls_final'] = W
        C['doors_final'], _ = xdec.run_door_ops(C.get('doors') or [], ops, s)
        C['negatives'] = [dict(op, decision=next((e['decision'] for e in xdec.entries(corr, i) if e['id'] == op['id']), None)) for op in ops if op['op'] == 'neg']
        vf = []
        for kind in ('stairs', 'elevs'):
            for k, S in enumerate(res.get(kind) or []):
                if not set(S['floors']) & set(pf[i]): continue
                cx, cy = (S['box'][0] + S['box'][2]) / 2, (S['box'][1] + S['box'][3]) / 2
                if any(v['kind'] == kind and math.hypot(cx - (v['box_m'][0] + v['box_m'][2]) / 2, cy - (v['box_m'][1] + v['box_m'][3]) / 2) < 1.5 for v in rej): continue
                (x0, y1), (x1, y0) = g2p(T, S['box'][0], S['box'][1]), g2p(T, S['box'][2], S['box'][3])
                vf.append(dict(kind=kind, src='auto', floors=S['floors'], name=(f'계단{k + 1}' if kind == 'stairs' else f'승강기 {chr(65 + k)}'),
                               box_px=[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]))
        for v in corr.get('vertical_added') or []:           # 그린 도면 px → 전역 m → 연결층이 겹치는 다른 도면 px 에도
            j = int(v.get('plan', -1))
            if v.get('kind') not in ('stairs', 'elevs') or j not in pf or not (set(v.get('floors') or pf[j]) & set(pf[i])): continue
            b = v['box_px']
            if j != i:
                Tj = plan_T(res, j); (gx0, gy0), (gx1, gy1) = p2g(Tj, b[0], b[1]), p2g(Tj, b[2], b[3])
                (x0, y0), (x1, y1) = g2p(T, gx0, gy0), g2p(T, gx1, gy1); b = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            vf.append(dict(kind=v['kind'], src='added', floors=v.get('floors'), name=v.get('name'), box_px=[round(x, 1) for x in b]))
        C['vertical_final'] = vf
    return corr


def export(entries, out_dir, all_plans=False):
    """entries: [(corrections, result, imgs)] → COCO 누적 저장"""
    import cv2
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    ap = os.path.join(out_dir, 'annotations.json')
    coco = load_json(ap) if os.path.exists(ap) else dict(info=dict(description='소방활동 정보카드 평면도 — 검수 확정 라벨(내부 PC 전용, 반출 금지)', version='1'),
                                                          categories=CATS, images=[], annotations=[])
    mp = os.path.join(out_dir, 'review_metrics.json')
    mets = load_json(mp) if os.path.exists(mp) else []
    n_img = n_ann = 0
    for corr, res, imgs in entries:
        corr = with_walls(json.loads(json.dumps(corr)), res)
        bid = slug((corr.get('building') or {}).get('id') or 'building')
        for key, C in (corr.get('plans') or {}).items():
            if not all_plans and not (C.get('status') or {}).get('done'): continue
            i = int(key); fn = f'images/{bid}_p{i}.png'
            cv2.imencode('.png', imgs[i])[1].tofile(os.path.join(out_dir, fn))       # 한글 경로 안전
            old = [im['id'] for im in coco['images'] if im['file_name'] == fn]
            if old:
                coco['images'] = [im for im in coco['images'] if im['file_name'] != fn]
                coco['annotations'] = [a for a in coco['annotations'] if a['image_id'] not in old]
            iid = max([im['id'] for im in coco['images']] + [0]) + 1
            coco['images'].append(dict(id=iid, file_name=fn, width=C['img_w'], height=C['img_h'], building=bid, plan=i, floors=C.get('floors'),
                                       caption=C.get('caption'), px_per_m=C.get('s'), reviewed=bool((C.get('status') or {}).get('done')),
                                       align=(C.get('status') or {}).get('align'), pdf_sha256=(corr.get('building') or {}).get('pdf_sha256'),
                                       date=datetime.date.today().isoformat()))
            aid = max([a['id'] for a in coco['annotations']] + [0])
            for a in plan_annotations(C):
                aid += 1; a.update(id=aid, image_id=iid); coco['annotations'].append(a); n_ann += 1
            n_img += 1
        m = review_metrics(corr, only_done=not all_plans); m['date'] = datetime.datetime.now().isoformat(timespec='seconds')
        mets = [x for x in mets if x.get('id') != m['id']] + [m]
    save_json(coco, ap, indent=None); save_json(mets, mp, indent=1)
    print(f'학습 라벨: 이미지 {n_img}장, 주석 {n_ann}개 → {ap}  (누적 이미지 {len(coco["images"])}장)')
    return coco, mets


def load_entry(corr_path, result=None, pdf=None):
    corr = load_json(corr_path)
    rp = result or (corr.get('source') or {}).get('result_json')
    if not rp or not os.path.exists(rp): raise SystemExit(f'결과 JSON을 찾을 수 없음: {rp} (--result 로 지정)')
    res = load_json(rp)
    pdf = find_pdf(rp, res, pdf or (corr.get('source') or {}).get('pdf'))
    if not pdf: raise SystemExit('원본 PDF를 찾을 수 없음 (--pdf 로 지정)')
    return corr, res, plan_images(pdf, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='검수 결과 → 학습 라벨(COCO)')
    ap.add_argument('corrections', nargs='+'); ap.add_argument('--out', default='학습데이터')
    ap.add_argument('--all', action='store_true', help='검수 완료 표시 안 한 도면도 포함'); ap.add_argument('--result'); ap.add_argument('--pdf')
    a = ap.parse_args()
    files = [f for c in a.corrections for f in (glob.glob(c) or [c])]
    export([load_entry(f, a.result, a.pdf) for f in files], a.out, a.all)
