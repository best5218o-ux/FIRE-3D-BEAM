# -*- coding: utf-8 -*-
"""스캔본(글자층 없는 이미지 PDF) 정보카드 OCR 채점 — synth/scan.py 로 만든 세트(data_scan_dev · data_scan_hold)
python -m evalr.scan_eval fields data_scan_dev              항목(필드) 정확도: 스캔본 OCR 해석 vs 가상 정답(evalr/card_eval.score_std 와 같은 규칙)
python -m evalr.scan_eval e2e data_scan_dev [N]             파이프라인 종단 비교: 스캔본 vs 원본(글자층) 카드(scan.json 의 src) — 결과 캐시 out_scan/<세트>_{ocr,text}/
python -m evalr.scan_eval calib data_scan_dev               신뢰도 보정: 필드 신뢰도 구간별 정답률(검수 플래그 문턱 OCR_FLAG_CONF 정하기)
개발(dev)에서만 조정하고 보류(hold)는 마지막에 1회만 돌린다. 한 번에 한 프로세스(가벼운 직렬 실행)."""
import glob, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalr.card_eval import score_std, GROUPS, _eq, _ns


def _dirs(D, n=None):
    ds = sorted(d.rstrip('/') for d in glob.glob(os.path.join(D, '*/')) if os.path.exists(os.path.join(d, 'card.pdf')))
    return ds[:n] if n else ds


def fields(D, n=None, quiet=False):
    from fire3dbeam.card import parse_card, extract_figures
    acc = {g: [0, 0] for g in GROUPS}; rows = []
    by_dpi = {}
    for d in _dirs(D, n):
        t0 = time.time()
        info, doc = parse_card(os.path.join(d, 'card.pdf'), dong=0)
        figs = extract_figures(doc)
        dt = time.time() - t0
        gt = json.load(open(os.path.join(d, 'gt.json'), encoding='utf-8'))
        R, bad = score_std(gt, info, figs)
        meta = json.load(open(os.path.join(d, 'scan.json'))) if os.path.exists(os.path.join(d, 'scan.json')) else {}
        for g, (a, b) in R.items(): acc[g][0] += a; acc[g][1] += b
        tot = [sum(a for a, b in R.values()), sum(b for a, b in R.values())]
        k = meta.get('dpi'); by_dpi.setdefault(k, [0, 0]); by_dpi[k][0] += tot[0]; by_dpi[k][1] += tot[1]
        oc = info.get('ocr') or {}
        for f, c in (oc.get('conf') or {}).items():                  # 신뢰도 보정용 점(동1)
            ok = field_ok(gt['card_std'][0], info, f)
            if ok is not None: CAL.append((c, ok, f))
        rows.append(dict(name=os.path.basename(d), dpi=meta.get('dpi'), variant=gt.get('card_form', {}).get('variant'), ok=tot[0], n=tot[1], sec=round(dt, 1),
                         pages=oc.get('pages'), per_page=oc.get('per_page'), low=len(oc.get('low') or []),
                         bad=[b if '관계인' not in str(b.get('field')) else dict(field=b['field']) for b in bad]))     # 관계인 값은 남기지 않음
        if not quiet:
            print(f"{os.path.basename(d)} dpi{meta.get('dpi')} {rows[-1]['variant']} {tot[0]}/{tot[1]} {dt:.0f}s 낮은신뢰 {rows[-1]['low']}", flush=True)
            for b in rows[-1]['bad'][:8]: print('   ', json.dumps(b, ensure_ascii=False)[:200])
    out = {g: f'{a}/{b}' for g, (a, b) in acc.items()}
    T = [sum(a for a, b in acc.values()), sum(b for a, b in acc.values())]
    out['전체'] = f'{T[0]}/{T[1]} ({T[0] / max(1, T[1]):.4f})'
    out['dpi별'] = {str(k): f'{a}/{b}' for k, (a, b) in sorted(by_dpi.items(), key=lambda kv: kv[0] or 0)}
    out['변형별'] = {}
    for r in rows:
        v = out['변형별'].setdefault(r['variant'], [0, 0]); v[0] += r['ok']; v[1] += r['n']
    out['변형별'] = {k: f'{a}/{b}' for k, (a, b) in out['변형별'].items()}
    secs = [r['sec'] for r in rows]; pp = [x for r in rows for x in (r['per_page'] or [])]
    out['속도'] = dict(카드당_초_중앙=sorted(secs)[len(secs) // 2] if secs else None, 카드당_초_최대=max(secs) if secs else None,
                     쪽당_초_평균=round(sum(pp) / max(1, len(pp)), 2), 표쪽당_초_최대=max(pp) if pp else None)
    out['낮은신뢰_필드_평균'] = round(sum(r['low'] for r in rows) / max(1, len(rows)), 1)
    out['신뢰도보정'] = calib_table(CAL)
    return out, rows


CAL = []


def calib_table(pts):
    bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    tab = []
    for a, b in zip(bins, bins[1:]):
        v = [ok for c, ok, _ in pts if a <= c < b]
        tab.append(f'{a:.2f}~{b:.2f}: {sum(v)}/{len(v)}')
    wrong = sum(1 for c, ok, _ in pts if not ok)
    thr = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        fl = [(c, ok) for c, ok, _ in pts if c < t]
        thr[str(t)] = f"플래그 {len(fl)} · 잡은 오답 {sum(1 for c, ok in fl if not ok)}/{wrong}"
    miss = [f for c, ok, f in pts if not ok and c >= 0.6]
    return dict(n=len(pts), 구간=tab, 문턱=thr, 놓친오답_예=[m for m in miss if not m.startswith('관계인')][:15])


def calib(D, n=None):
    """필드 신뢰도 ↔ 정답 여부(동1 기준): 구간별 정답률 + 문턱별 (플래그 수, 플래그가 잡는 오답 비율)"""
    from fire3dbeam.card import parse_card
    pts = []
    for d in _dirs(D, n):
        info, _ = parse_card(os.path.join(d, 'card.pdf'), dong=0)
        gt = json.load(open(os.path.join(d, 'gt.json'), encoding='utf-8'))['card_std']
        G = gt[0]; conf = (info.get('ocr') or {}).get('conf') or {}
        for f, c in conf.items():
            ok = field_ok(G, info, f)
            if ok is not None: pts.append((c, ok, f.split('.')[0] if not f.startswith('층별현황') else '층별현황.' + f.split('.')[-1]))
    bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    tab = []
    for a, b in zip(bins, bins[1:]):
        v = [ok for c, ok, _ in pts if a <= c < b]
        tab.append(dict(구간=f'{a:.2f}~{b:.2f}', n=len(v), 정답률=round(sum(v) / len(v), 3) if v else None))
    thr = []
    wrong = sum(1 for c, ok, _ in pts if not ok)
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        fl = [(c, ok) for c, ok, _ in pts if c < t]
        thr.append(dict(문턱=t, 플래그=len(fl), 잡은오답=f"{sum(1 for c, ok in fl if not ok)}/{wrong}", 플래그중오답=round(sum(1 for c, ok in fl if not ok) / max(1, len(fl)), 3)))
    return dict(n=len(pts), 오답=wrong, 구간=tab, 문턱=thr)


def field_ok(G, info, f):
    """ocr_meta 필드 이름 → 정답과 같나(동1). 비교 못 하면 None"""
    P = info
    p = f.split('.')
    try:
        if p[0] in ('관리번호', '주용도', '상호'): return _eq(G[p[0]], P.get(p[0]))
        if p[0] == '소재지': return all(_eq(G['소재지'][k], (P.get('소재지') or {}).get(k)) for k in ('지번', '도로명'))
        if p[0] == '거리': return all(_eq(G['거리'][k], (P.get('거리') or {}).get(k), 0.05) for k in ('소방서', '관할센터', '최단센터'))
        if p[0] == '소방용수':
            a = [(w['종류'], w['관서'], w['동'], w['번호'], w['거리km']) for w in G['소방용수']]
            b = [(w['종류'], w['관서'], w['동'], w['번호'], w['거리km']) for w in P.get('소방용수') or []]
            return a == b
        if p[0] == '건물구조':
            v = G['건물구조'].get(p[1]); return _eq(v, (P.get('건물구조') or {}).get(p[1]), 0.005 if p[1] in ('연면적', '바닥면적') else (0 if isinstance(v, int) else None))
        if p[0] == '관계인':
            g = next((r for r in G['관계인'] if r['구분'] == p[1]), None); q = next((r for r in P.get('관계인') or [] if r['구분'] == p[1]), None)
            if g is None or q is None: return None
            return _eq(g[p[2]], q.get(p[2]))
        if p[0] == '위험물':
            g = [h for h in G['위험물'] if _ns(h['구분']) == p[1]]; q = [h for h in P.get('위험물') or [] if _ns(h['구분']) == p[1]]
            if not g or not q: return None
            return _eq(g[0][p[2]], q[0].get(p[2]))
        if p[0] == '층별현황':
            g = next((r for r in G['층별현황'] if r['층'] == p[1]), None); q = next((r for r in P.get('층별현황') or [] if r['층'] == p[1]), None)
            if g is None or q is None: return False if q is not None else None
            from fire3dbeam.card import parse_fac
            k = p[2]
            if k == '층': return _eq(g['층_원문'], q.get('층_원문'))
            if k == '면적': return _eq(g['면적'], q.get('면적'), 0.005)
            if k == '소방시설': return parse_fac(g['소방시설_원문']) == parse_fac(q.get('소방시설_원문') or '')
            return _eq(g[k], q.get(k))
    except Exception:
        return None
    return None


# ───────────── 종단(파이프라인) 비교
def _run(pdf, out_json):
    if os.path.exists(out_json): return json.load(open(out_json, encoding='utf-8'))
    from fire3dbeam.pipeline import run
    t0 = time.time(); r = run(pdf); r['_sec'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(r, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False)
    return r


def _frac(v):
    if not v or '/' not in str(v): return (0, 0)
    a, b = str(v).split('/'); return int(a), int(b)


def _plan_frames(text_pdf, scan_pdf, dong=0):
    """스캔본 평면도 잘라낸 그림(화소) ↔ 원본 카드 평면도 그림(화소) 대응 — 정답 변환(gt transforms)은 원본 그림 화소 기준이므로
    채점 전에 스캔본 결과의 층 변환(s, tx, ty)을 원본 화소 기준으로 바꾼다. 다중 배율 형판 맞춤(원본 → 잘라낸 그림).
    → {층: (a, bx, by)}  (잘라낸 그림 화소 = a·원본 화소 + b)
    dong: 파이프라인이 고른 동(기본 0)의 평면도만 — (2026-10-01 수정) 예전에는 여러 동 카드에서 둘째 동의 같은 층 이름(1F 등) 그림이
    첫 동 층의 대응을 덮어써, 그 층은 축척이 맞아도 정합 실패로 채점됐다(dev 9층·원본 카드와 무관한 채점기 오류). None = 예전 방식"""
    import cv2, numpy as np, pymupdf
    from fire3dbeam.card import extract_figures
    T = [f for f in extract_figures(pymupdf.open(text_pdf)) if f['kind'] == '평면도' and (dong is None or f.get('dong', 0) == dong)]
    S = [f for f in extract_figures(pymupdf.open(scan_pdf)) if f['kind'] == '평면도']
    out = {}
    for t in T:
        s = next((x for x in S if x['floors'] == t['floors'] and x.get('dong', 0) == t.get('dong', 0)), None)
        if s is None: continue
        o = cv2.cvtColor(t['img'], cv2.COLOR_BGR2GRAY); c = cv2.cvtColor(s['img'], cv2.COLOR_BGR2GRAY)
        ppt_o = o.shape[1] / (t['bbox'][2] - t['bbox'][0]); ppt_c = c.shape[1] / (s['bbox'][2] - s['bbox'][0]) * 1.0
        # 잘라낸 그림의 pt 당 화소 = 쪽 스캔 해상도(틀 안쪽을 잘랐으므로 bbox 폭보다 약간 좁음 → 거의 같음)
        a0 = ppt_c / ppt_o
        f = min(1.0, 700.0 / c.shape[1])
        cs = cv2.GaussianBlur(cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1.0).astype(np.float32)
        best = None
        for k in np.arange(0.95, 1.051, 0.01):
            a = a0 * k * f
            os_ = cv2.resize(o, None, fx=a, fy=a, interpolation=cv2.INTER_AREA)
            if os_.shape[0] >= cs.shape[0] or os_.shape[1] >= cs.shape[1]: continue
            r = cv2.matchTemplate(cs, cv2.GaussianBlur(os_, (0, 0), 1.0).astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(r)
            if best is None or mv > best[0]: best = (mv, a0 * k, ml[0] / f, ml[1] / f)
        if best and best[0] > 0.3:
            for fid in t['floors']: out[fid] = best[1:]
    return out


def _to_orig_frame(res, frames):
    """스캔본 결과 층 변환을 원본 그림 화소 기준으로(채점용 사본)"""
    import copy
    r = copy.copy(res); r['floors'] = {}
    for f, F in res['floors'].items():
        F = dict(F)
        if f in frames:
            a, bx, by = frames[f]
            F['tx'] = F['tx'] + bx / F['s']; F['ty'] = F['ty'] - by / F['s']; F['s'] = F['s'] / a
        r['floors'][f] = F
    return r


def e2e(D, n=None):
    from evalr.score import score
    tag = os.path.basename(D.rstrip('/'))
    keys = ('card_floors_ok', 'fac_parse', 'floors_found', 'align_ok', 'stairs_found', 'stairs_correct', 'elevs_found', 'flags_planted_found', 'entrance_found', 'daejang_correct',
            'zone_flag_agree', 'site_facade_ok')
    fl = {m: dict(wall=[], door=[], sym={}) for m in ('text', 'ocr')}
    agg = {m: {k: [0, 0] for k in keys} for m in ('text', 'ocr')}
    same = dict(층목록=0, 계단수_카드=0, 승강기수_카드=0, 계단_도면=0, 승강기_도면=0, 진압작전도=[0, 0], 그림종류=0, 결과층=0)
    rows = []
    for d in _dirs(D, n):
        nm = os.path.basename(d)
        meta = json.load(open(os.path.join(d, 'scan.json')))
        gt = json.load(open(os.path.join(d, 'gt.json'), encoding='utf-8'))
        rt = _run(meta['src'], f'out_scan/{tag}_text/{nm}.json')
        ro = _run(os.path.join(d, 'card.pdf'), f'out_scan/{tag}_ocr/{nm}.json')
        frames = _plan_frames(meta['src'], os.path.join(d, 'card.pdf'), None if os.environ.get('SCAN_EVAL_OLDFRAMES') == '1' else (ro['card'].get('선택동') or 0))
        st, so = score(gt, rt), score(gt, _to_orig_frame(ro, frames))
        for m, s in (('text', st), ('ocr', so)):
            for k in keys:
                a, b = _frac(s.get(k)); agg[m][k][0] += a; agg[m][k][1] += b
            for k, kk in (('wall_recall', 'wall'), ('door_recall', 'door')):
                if s.get(k) is not None: fl[m][kk].append((s[k], s.get(k.replace('recall', 'precision'))))
            for t, v in (s.get('symbols') or {}).items(): fl[m]['sym'].setdefault(t, []).append((v['recall'], v['precision']))
        ct, co = rt['card'], ro['card']
        same['층목록'] += sorted(ct.get('floors') or {}) == sorted(co.get('floors') or {})
        same['계단수_카드'] += ct.get('stairs') == co.get('stairs')
        same['승강기수_카드'] += ct.get('elev') == co.get('elev')
        same['계단_도면'] += len(rt['stairs']) == len(ro['stairs'])
        same['승강기_도면'] += len(rt['elevs']) == len(ro['elevs'])
        same['결과층'] += sorted(rt['floors']) == sorted(ro['floors'])
        ft = [(f['kind'], tuple(f['floors']), f['dong']) for f in rt.get('figures') or []]; fo = [(f['kind'], tuple(f['floors']), f['dong']) for f in ro.get('figures') or []]
        same['그림종류'] += ft == fo
        has_sm = any(f['kind'] == '진압작전도' for f in rt.get('figures') or [])
        if has_sm or isinstance(rt.get('site'), dict) and rt['site'].get('src') == '진압작전도':
            same['진압작전도'][1] += 1
            same['진압작전도'][0] += isinstance(ro.get('site'), dict) and ro['site'].get('src') == '진압작전도'
        rows.append(dict(name=nm, dpi=meta['dpi'], sec_text=rt.get('_sec'), sec_ocr=ro.get('_sec'), ocr_sec=(ro['card'].get('ocr') or {}).get('seconds'),
                         text={k: st.get(k) for k in keys}, ocr={k: so.get(k) for k in keys},
                         site_text=(rt.get('site') or {}).get('src') if isinstance(rt.get('site'), dict) else None,
                         site_ocr=(ro.get('site') or {}).get('src') if isinstance(ro.get('site'), dict) else None,
                         ocr_flags=sum(1 for f in ro['flags'] if f['kind'] == 'card_ocr'), n_flags_text=len(rt['flags']), n_flags_ocr=len(ro['flags'])))
        print(nm, json.dumps({k: (st.get(k), so.get(k)) for k in keys if st.get(k) != so.get(k)}, ensure_ascii=False), 'OCR플래그', rows[-1]['ocr_flags'],
              f"시간 {rt.get('_sec')}→{ro.get('_sec')}s", flush=True)
    N = len(rows)
    import numpy as np
    avg = lambda L: f"R{np.mean([x[0] for x in L]):.3f}/P{np.mean([x[1] or 0 for x in L]):.3f}" if L else None
    geo = {m: dict(벽=avg(fl[m]['wall']), 문=avg(fl[m]['door']), **{f'기호_{t}': avg(v) for t, v in sorted(fl[m]['sym'].items())}) for m in fl}
    out = dict(n=N, 도면인식_원본=geo['text'], 도면인식_스캔본=geo['ocr'], 원본={k: f'{a}/{b}' for k, (a, b) in agg['text'].items()}, 스캔본={k: f'{a}/{b}' for k, (a, b) in agg['ocr'].items()},
               같음={k: (f'{v[0]}/{v[1]}' if isinstance(v, list) else f'{v}/{N}') for k, v in same.items()},
               시간_중앙={'원본': sorted(r['sec_text'] or 0 for r in rows)[N // 2] if N else None, '스캔본': sorted(r['sec_ocr'] or 0 for r in rows)[N // 2] if N else None,
                       'OCR만': sorted(r['ocr_sec'] or 0 for r in rows)[N // 2] if N else None},
               OCR플래그_평균=round(sum(r['ocr_flags'] for r in rows) / max(1, N), 1))
    return out, rows


if __name__ == '__main__':
    cmd, D = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    os.makedirs('out_scan', exist_ok=True)
    tag = os.path.basename(D.rstrip('/'))
    if cmd == 'fields':
        out, rows = fields(D, n)
        json.dump(dict(summary=out, rows=rows), open(f'out_scan/{tag}_fields.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1))
    elif cmd == 'calib':
        out = calib(D, n); print(json.dumps(out, ensure_ascii=False, indent=1))
        json.dump(out, open(f'out_scan/{tag}_calib.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    elif cmd == 'e2e':
        out, rows = e2e(D, n)
        json.dump(dict(summary=out, rows=rows), open(f'out_scan/{tag}_e2e.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1))
