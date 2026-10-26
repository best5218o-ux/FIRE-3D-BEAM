# -*- coding: utf-8 -*-
"""검수 화면(오프라인 HTML 1개) 생성.
사용: python -m review.make out/X.json [--pdf card.pdf] [-o out/X_review.html] [--corrections X_corrections.json]
- 도면(평면도) 이미지 위에 자동 인식한 벽·문·소방기호·계단/승강기와 교차검증 플래그를 겹쳐 보여 준다.
- '검수 필요' 목록은 오류 가능성(의심도) 높은 순 — 사람은 틀렸을 법한 곳만 본다."""
import argparse, datetime, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review.common import (ROOT, TYPES, KO, SCHEMA, load_json, sha256_file, result_sha, stem_of, find_pdf, plan_images,
                           plan_T, g2p, plan_floors, img_data_uri)

HERE = os.path.dirname(os.path.abspath(__file__))
SYM_DEFAULT_M = 1.0          # 상자 정보가 없는 기호의 표시 크기(m)
CONF_REVIEW = 0.76           # 이 값 미만 신뢰도 기호는 검수 대상
WEAK_TYPES = {'panel': 0.5, 'gas': 0.45}   # 인식 정확도가 낮은 종류(가상 채점 기준) — 신뢰도와 무관하게 최소 의심도


def ko_floor(fid):
    if fid.startswith('B'): return f'지하{fid[1:]}층'
    if fid in ('RF', 'PIT'): return {'RF': '옥탑층', 'PIT': 'PIT층'}[fid]
    return f'지상{fid[:-1]}층' if fid.endswith('F') else fid


def floor_range(fids):
    return fids[0] if len(fids) == 1 else f'{fids[0]}~{fids[-1]}'


def flag_text(f):
    k = f['kind']
    from firebim.flagtext import TITLE, explain
    if f.get('msg'):                       # 쉬운 말 문구(firebim/flagtext.py) — 기술 정보는 detail 로 따로
        return TITLE.get(k, k), f['msg']
    try:                                   # JSON 에 msg 가 없는 결과(가상 카드 등)도 같은 쉬운 말로
        m_, _ = explain(f)
        if m_: return TITLE.get(k, k), m_
    except Exception:
        pass
    if k == 'area': return '층 면적 불일치', f"{ko_floor(f['floor'])}: 정보카드 {f['card']:,.0f}㎡ / 도면 외곽 {f['drawing']:,.0f}㎡ (비율 {f['ratio']})"
    if k == 'hose_count': return '옥내소화전 수량 불일치', f"{ko_floor(f['floor'])}: 정보카드 {f['card']}개 / 도면 기호 {f['drawing']}개 — 누락·오검출 기호 확인"
    if k == 'stair_count': return '계단 수 불일치', f"정보카드 {f['card']}개소 / 도면에서 찾은 계단 {f['drawing']}개소"
    if k == 'align': return '층 위치 맞춤 신뢰도 낮음', f"{ko_floor(f.get('floor', ''))}: 기호 부족으로 그리드선만으로 위치를 맞춤"
    try:                                   # 기능 12·18·24 플래그: 3D 화면과 같은 문구 사용
        from firebim.export import ext_issue
        e = ext_issue(f)
        if e: return e['t'], e['d']
    except Exception:
        pass
    from firebim.flagtext import TITLE
    return TITLE.get(k, k), '자동 검사 결과 — 검수 화면에서 확인'


def flag_key(f): return f"{f['kind']}|{f.get('floor', '')}" + (f"|{f['field']}" if f.get('field') else '')


def walls_px(walls, T):
    s = T[0]; out = []
    for w in walls:
        if w['o'] == 'h':
            (a, c), (b, _) = g2p(T, w['a'], w['c']), g2p(T, w['b'], w['c'])
        else:
            (c, a), (_, b) = g2p(T, w['c'], w['a']), g2p(T, w['c'], w['b'])
            a, b = min(a, b), max(a, b)
        out.append(dict(o=w['o'], c=round(c, 1), a=round(a, 1), b=round(b, 1), t=round(max(1.5, w.get('t', 0.15) * s), 1)))
    return out


def build_data(res, result_path, pdf, imgs, preload=None, prior=None):
    from firebim.card import level_of
    pf = plan_floors(res)
    card = res.get('card') or {}
    name = card.get('name') or stem_of(result_path)
    D = dict(schema=SCHEMA, generated=datetime.datetime.now().isoformat(timespec='seconds'),
             building=dict(id=stem_of(result_path), name=name, use=card.get('use') or '', result_sha=result_sha(res),
                           pdf_name=os.path.basename(pdf) if pdf else None, pdf_sha256=sha256_file(pdf) if pdf else None),
             source=dict(result_json=os.path.abspath(result_path), pdf=os.path.abspath(pdf) if pdf else None),
             types={t: KO[t] for t in TYPES}, plans=[], stairs=[], elevs=[], flags=[], items=[], preload=preload)
    items = []
    lvl_of_plan = {i: min(level_of(f) for f in fl) for i, fl in pf.items()}
    order = sorted(pf, key=lambda i: lvl_of_plan[i])
    for i in order:
        P = res['plans'][i]; fl = pf[i]; T = plan_T(res, i); s = T[0]
        F0 = res['floors'][fl[0]]
        img = imgs[i]; h, w = img.shape[:2]
        syms = []
        for k, q in enumerate(F0['symbols']):
            px, py = g2p(T, q['x'], q['y'])
            if q.get('box'):
                (x0, y1), (x1, y0) = g2p(T, q['box'][0], q['box'][1]), g2p(T, q['box'][2], q['box'][3])
                bx = [round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)]
            else:
                r = SYM_DEFAULT_M * s / 2; bx = [round(px - r, 1), round(py - r, 1), round(px + r, 1), round(py + r, 1)]
            syms.append(dict(id=f'p{i}s{k}', type=q['type'], px=round(px, 1), py=round(py, 1), box=bx, conf=q.get('conf'), box_est=not q.get('box')))
        doors = []
        for k, d in enumerate(F0.get('doors') or []):
            px, py = g2p(T, d['x'], d['y'])
            doors.append(dict(id=f'p{i}d{k}', o=d['o'], px=round(px, 1), py=round(py, 1), w=round(d['w'] * s, 1), ext=bool(d.get('ext'))))
        walls = [dict(wd, id=f'p{i}w{k}') for k, wd in enumerate(walls_px(F0['walls'], T))]
        # 겹쳐보기(층정합 확인): 바로 아래(없으면 위) 도면의 벽을 이 도면 px 로
        j = order.index(i); nb = order[j - 1] if j > 0 else (order[j + 1] if len(order) > 1 else None)
        ghost = walls_px(res['floors'][pf[nb][0]]['walls'], T) if nb is not None else []
        floors = []
        for fid in fl:
            F = res['floors'][fid]; c = F.get('card') or {}
            floors.append(dict(id=fid, name=ko_floor(fid), align=F.get('align'), card_area=c.get('area'), fp_area=F.get('footprint_area'),
                               hose_card=(c.get('fac') or {}).get('옥내소화전'), use=c.get('use') or ''))
        D['plans'].append(dict(i=i, caption=P.get('caption') or '', floors=floors, label=floor_range(fl), s=round(s, 4), tx=round(T[1], 4), ty=round(T[2], 4),
                               align=P.get('align'), scale_src=P.get('scale_src'), scale_basis=P.get('scale_basis'), w=w, h=h, img=img_data_uri(img),
                               symbols=syms, doors=doors, walls=walls, ghost=ghost, ghost_of=(floor_range(pf[nb]) if nb is not None else None)))
        # ── 검수 필요 항목(의심도 p = 틀렸을 가능성의 대략값) ──
        al = P.get('align')
        if al in ('low', 'none', 'mid'):
            items.append(dict(kind='align', plan=i, p={'none': 0.9, 'low': 0.8, 'mid': 0.5}[al],
                              t='층 위치 맞춤 확인', d=f"{floor_range(fl)} 정합 신뢰도 '{al}' — 겹쳐보기(G)로 아래층 벽과 맞는지 확인"))
        ss = P.get('scale_src')
        if ss and ss.startswith('real:') and P.get('scale_basis') and '근거가 약함' in P['scale_basis']:      # 실제 도면: 시트·치수·창호 근거 없이 정한 축척
            items.append(dict(kind='scale', plan=i, p=0.55, t='축척 확인', d=f"{floor_range(fl)} {P['scale_basis']}"))
        if ss in ('grid', 'area', 'default'):
            items.append(dict(kind='scale', plan=i, p={'grid': 0.35, 'area': 0.6, 'default': 0.9}[ss],
                              t='축척 확인', d=f"{floor_range(fl)} 치수 글자를 못 읽어 축척을 {'다른 층 그리드' if ss == 'grid' else ('정보카드 면적' if ss == 'area' else '기본값')}으로 추정"))
        for q in syms:
            c = q['conf']; pw = None
            if c is not None and c < CONF_REVIEW: pw = round(1 - c, 2)
            if q['type'] in WEAK_TYPES: pw = max(pw or 0, WEAK_TYPES[q['type']])
            if prior and q['type'] in prior and prior[q['type']] < 0.6: pw = max(pw or 0, round(1 - prior[q['type']], 2))
            if pw:
                items.append(dict(kind='symbol', plan=i, ref=q['id'], p=pw, loc=[q['px'], q['py']], t=f"{KO[q['type']]} 기호 확인",
                                  d=f"{floor_range(fl)} · 신뢰도 {c if c is not None else '-'}" + (" · 인식 정확도 낮은 종류" if q['type'] in WEAK_TYPES else '')))
        if not any(q['type'] == 'exit' for q in syms):
            items.append(dict(kind='missing', plan=i, p=0.55, t='비상구 기호 없음', d=f'{floor_range(fl)} — 누락된 비상구 기호가 있는지 확인(A→1로 추가)'))
        items += xcheck_items(res, i, fl, T, walls)
        if doors:
            items.append(dict(kind='doors', plan=i, p=0.4, t=f'문 {len(doors)}개 자동 추정', d=f'{floor_range(fl)} — 문(틈) 인식은 정확도가 낮음: 빠진 문 추가(D), 잘못된 문 삭제'))
    # 수직 동선(계단·승강기) — 각 도면 px 상자
    for kind, L in (('stairs', res.get('stairs') or []), ('elevs', res.get('elevs') or [])):
        for k, S in enumerate(L):
            boxes = {}
            for i in pf:
                if not set(pf[i]) & set(S['floors']): continue
                T = plan_T(res, i)
                (x0, y1), (x1, y0) = g2p(T, S['box'][0], S['box'][1]), g2p(T, S['box'][2], S['box'][3])
                boxes[str(i)] = [round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)]
            D[kind].append(dict(k=k, name=(f'계단{k + 1}' if kind == 'stairs' else f'승강기 {chr(65 + k)}'), floors=S['floors'],
                                label=floor_range(S['floors']), boxes=boxes, box_m=S['box']))
    D['floors_all'] = [dict(id=f, name=ko_floor(f), plan=res['floors'][f]['plan']) for f in sorted(res['floors'], key=level_of)]
    fplan = {fid: F['plan'] for fid, F in res['floors'].items()}
    for f in res.get('flags') or []:
        t, d = flag_text(f)
        key = flag_key(f)
        _e = None
        try:
            from firebim.export import ext_issue; _e = ext_issue(f)
        except Exception: pass
        lv = _e['lv'] if _e and _e.get('lv') in ('high', 'mid') else ('high' if f['kind'] in ('stair_count', 'align') else 'mid')
        D['flags'].append(dict(key=key, kind=f['kind'], floor=f.get('floor'), plan=fplan.get(f.get('floor')), t=t, d=d, lv=lv, raw=f, **({'detail': f['detail']} if f.get('detail') else {})))
        items.append(dict(kind='flag', ref=key, plan=fplan.get(f.get('floor')), p=0.9 if lv == 'high' else 0.75, t=t, d=d, fk=f['kind']))
    ce = card.get('elev') or {}
    if ce and sum(ce.values()) != len(D['elevs']):
        items.append(dict(kind='elev_count', plan=None, p=0.6, t='승강기 수 불일치',
                          d=f"정보카드 {sum(ce.values())}대 / 도면에서 찾은 승강기 {len(D['elevs'])}대(층간 반복 기준)"))
    items.sort(key=lambda it: -it['p'])
    for n, it in enumerate(items): it['id'] = f'it{n}'
    D['items'] = items
    n_auto = sum(len(p['symbols']) + len(p['doors']) + len(p['walls']) for p in D['plans']) + len(D['stairs']) + len(D['elevs']) + len(D['flags'])
    D['stats'] = dict(n_auto=n_auto, n_items=len(items), n_walls=sum(len(p['walls']) for p in D['plans']))
    return D


XC_ITEM_P = 0.5             # 교차검증 불일치를 검수 항목으로 올리는 최소 의심도(firebim/xcheck.py CFG['item_p'] 와 같게)


def xcheck_items(res, i, fl, T, walls):
    """선택 경로(모델) 결과의 '두 독립 추출기 교차검증' → 검수 항목.
    규칙만 본 벽(오검출 의심) · 모델만 본 벽(누락 의심 / 모델로 보충함 / 규칙은 개구부) · 모델이 보충한 문.
    의심도 p = 그 자리의 자동 결과가 틀렸을 확률(가상 개발 세트로 맞춘 로지스틱) → 다른 항목과 함께 의심도 순 정렬."""
    X = (res.get('xcheck') or {}).get(str(i))
    if not X or 'error' in X: return []
    try:
        from firebim.xcheck import CFG
        thr = CFG.get('item_p', XC_ITEM_P)
    except Exception:
        thr = XC_ITEM_P
    out = []
    rng = floor_range(fl)

    def seg_px(x):
        if x['o'] == 'h': (x0, y0), (x1, y1) = g2p(T, x['a'], x['c']), g2p(T, x['b'], x['c'])
        else: (x0, y0), (x1, y1) = g2p(T, x['c'], x['a']), g2p(T, x['c'], x['b'])
        return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]

    def wall_ref(x):                    # 규칙 벽 조각이 속한 벽(검수 화면에서 선택·삭제용)
        F0 = res['floors'][fl[0]]['walls']
        for k, w in enumerate(F0):
            if w['o'] == x['o'] and abs(w['c'] - x['c']) < 0.05 and w['a'] <= x['a'] + 0.05 and w['b'] >= x['b'] - 0.05: return walls[k]['id']
        return None

    for x in X.get('rule_only', []):
        if x['p'] < thr: continue
        sp = seg_px(x)
        out.append(dict(kind='xwall', sub='rule_only', plan=i, p=x['p'], ref=wall_ref(x), seg=sp, loc=[(sp[0] + sp[2]) / 2, (sp[1] + sp[3]) / 2],
                        o=x['o'], tpx=round(max(1.5, x.get('t', 0.15) * T[0]), 1),
                        t='벽 오검출 의심(규칙만 벽으로 봄)',
                        d=f"{rng} · 길이 {x['b'] - x['a']:.1f}m · AI 모델 벽 확률 {x['pm']:.2f} · 이 벽의 모델 지지율 {x['sup'] * 100:.0f}% — 벽이 아니면 삭제"))
    for x in X.get('model_only', []):
        if x['p'] < thr: continue
        sp = seg_px(x)
        if x.get('filled'): t, d = '모델로 보충한 벽 확인', '규칙이 놓친 벽을 AI 모델로 보충함 — 벽이 아니면 삭제'
        elif x.get('bridge'): t, d = '개구부↔벽 판단 불일치', '규칙은 개구부(문·셔터), AI 모델은 벽으로 봄 — 진입구 여부 확인'
        else: t, d = '벽 누락 의심(모델만 벽으로 봄)', '벽이면 「모델이 맞음」(벽 추가)'
        out.append(dict(kind='xwall', sub='model_only', plan=i, p=x['p'], seg=sp, loc=[(sp[0] + sp[2]) / 2, (sp[1] + sp[3]) / 2], t=t,
                        o=x['o'], tpx=round(max(1.5, x.get('t', 0.15) * T[0]), 1), filled=bool(x.get('filled')), bridge=bool(x.get('bridge')),
                        ref=wall_ref(x) if x.get('filled') else None,
                        d=f"{rng} · 길이 {x['b'] - x['a']:.1f}m · 모델 벽 확률 {x['pm']:.2f} · 두께 {x['t']:.2f}m — {d}"))
    for d in X.get('doors_model', []):
        q = 1 - d['p']
        try:
            from firebim.xcheck import CFG as _C; added = q >= _C['door_p']
        except Exception: added = q >= 0.6
        if not added and q < thr: continue
        px, py = g2p(T, d['x'], d['y']); hw = d['w'] * T[0] / 2
        sp = [round(px - hw, 1), round(py, 1), round(px + hw, 1), round(py, 1)] if d['o'] == 'h' else [round(px, 1), round(py - hw, 1), round(px, 1), round(py + hw, 1)]
        out.append(dict(kind='xdoor', sub='added' if added else 'model_only', plan=i, p=round(d['p'] if added else q, 3), seg=sp, loc=[round(px, 1), round(py, 1)],
                        o=d['o'], wpx=round(2 * hw, 1), ext=bool(d.get('ext')),
                        t=('모델로 보충한 ' + ('외벽 개구부' if d.get('ext') else '문') + ' 확인') if added else '문 누락 의심(모델만 문으로 봄)',
                        d=f"{rng} · 폭 {d['w']:.1f}m · 모델 문 신뢰도 {q:.2f}" + (' — 문이 아니면 「규칙이 맞음」(삭제)' if added else ' — 문이면 「모델이 맞음」(추가)')))
    return out


def symbol_prior():
    """가상 채점 요약이 있으면 종류별 정밀도를 의심도 사전값으로 사용"""
    p = os.path.join(ROOT, 'out', 'summary.json')
    try:
        allr = load_json(p).get('ALL', {}).get('symbols', {})
        return {t: float(v.split('/P')[1]) for t, v in allr.items()}
    except Exception:
        return None


def render(D, out_html):
    tpl = open(os.path.join(HERE, 'review_template.html'), encoding='utf-8').read()
    js = open(os.path.join(HERE, 'review_app.js'), encoding='utf-8').read()
    data = json.dumps(D, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    title = f"{D['building']['name']} 검수"
    html = tpl.replace('__TITLE__', title.replace('<', '')).replace('/*@@DATA@@*/null', data).replace('/*@@APP@@*/', js.replace('</script', '<\\/script'))
    with open(out_html, 'w', encoding='utf-8') as f: f.write(html)
    return out_html


def make(result_path, pdf=None, out_html=None, corrections=None, res=None, imgs=None):
    res = res or load_json(result_path)
    pdf = find_pdf(result_path, res, pdf)
    if imgs is None:
        if not pdf: raise SystemExit(f'원본 PDF를 찾을 수 없음 — --pdf 로 지정하세요 ({result_path})')
        imgs = plan_images(pdf, res)
    pre = load_json(corrections) if corrections else None
    D = build_data(res, result_path, pdf, imgs, preload=pre, prior=symbol_prior())
    out_html = out_html or os.path.join(os.path.dirname(os.path.abspath(result_path)), stem_of(result_path) + '_review.html')
    render(D, out_html)
    print(out_html, f"{os.path.getsize(out_html) / 1e6:.2f}MB", '검수 필요', D['stats']['n_items'], '/ 자동인식', D['stats']['n_auto'])
    return out_html


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='검수 화면 HTML 생성')
    ap.add_argument('result'); ap.add_argument('--pdf'); ap.add_argument('-o', '--out'); ap.add_argument('--corrections')
    a = ap.parse_args()
    make(a.result, a.pdf, a.out, a.corrections)
