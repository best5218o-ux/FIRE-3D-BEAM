# -*- coding: utf-8 -*-
"""표준양식 정보카드 항목 해석 정확도 — 파싱한 모든 필드 vs 가상 정답(gt['card_std'], gt['card_std_figs'])
python -m evalr.card_eval data_std [data_std_hold ...]      (run_all 도 건물마다 score_dir 를 불러 summary 'card_std' 에 합산)

비교 규칙: 글자 값은 공백 무시(한글→PDF 변환기가 글꼴 경계에 공백을 끼워 넣음 — 원문 공백은 보존하지 않음), 면적·연면적 ±0.005㎡, 거리 ±0.05 km,
층수·계단수·승강기 수 정확 일치, 소방시설 칸 = 「시설명:수량」 항목 사전 일치(층 단위)와 항목 단위 둘 다.
관계인 성명·연락처는 개인정보 — 해석은 채점하되(내부 PC 선택 기능용) 파이프라인 출력에는 기본적으로 남지 않음(privacy 항목으로 확인)."""
import glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fire3dbeam.card import parse_card, extract_figures, parse_fac

GROUPS = ['기본(①②③)', '소재지(④)', '거리(⑤)', '소방용수(⑥)', '건물구조(⑦)', '관계인(⑧)', '위험물(⑨)', '층별현황(⑩)', '소방시설항목(⑩)', '그림(⑪)', '동 수']


def _ns(v): return re.sub(r'\s+', '', str(v)) if v is not None else None


def _eq(a, b, tol=None):
    if a is None or a == '' or b is None or b == '': return (a in (None, '')) and (b in (None, ''))
    if tol is not None:
        try: return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError): return False
    return _ns(a) == _ns(b)


def score_std(gt, info, figs):
    """→ {그룹: [맞음, 전체]}, 틀린 필드 목록"""
    R = {g: [0, 0] for g in GROUPS}; bad = []
    def chk(g, name, a, b, tol=None):
        ok = _eq(a, b, tol); R[g][0] += ok; R[g][1] += 1
        if not ok: bad.append(dict(field=name, gt=a, got=b))
    G, P = gt['card_std'], info.get('동목록') or []
    R['동 수'] = [int(len(G) == len(P)), 1]
    for k, g in enumerate(G):
        p = P[k] if k < len(P) else {}
        pre = f'동{k + 1}.'
        for f in ('관리번호', '주용도', '상호'): chk('기본(①②③)', pre + f, g[f], p.get(f))
        for f in ('지번', '도로명'): chk('소재지(④)', pre + '소재지.' + f, g['소재지'][f], (p.get('소재지') or {}).get(f))
        for f in ('소방서', '관할센터', '최단센터'): chk('거리(⑤)', pre + '거리.' + f, g['거리'][f], (p.get('거리') or {}).get(f), 0.05)
        pw = p.get('소방용수') or []
        for i, w in enumerate(g['소방용수']):
            q = pw[i] if i < len(pw) else {}
            for f in ('종류', '관서', '동', '번호'): chk('소방용수(⑥)', f'{pre}소방용수{i + 1}.{f}', w[f], q.get(f))
            chk('소방용수(⑥)', f'{pre}소방용수{i + 1}.거리km', w['거리km'], q.get('거리km'), 0.0005)
        if len(pw) > len(g['소방용수']): R['소방용수(⑥)'][1] += len(pw) - len(g['소방용수']); bad.append(dict(field=pre + '소방용수 과검출', got=pw[len(g['소방용수']):]))
        S, PS = g['건물구조'], p.get('건물구조') or {}
        for f, v in S.items(): chk('건물구조(⑦)', pre + '건물구조.' + f, v, PS.get(f), 0.005 if f in ('연면적', '바닥면적') else (0 if isinstance(v, int) else None))
        prel = {r['구분']: r for r in p.get('관계인') or []}
        for r in g['관계인']:
            q = prel.get(r['구분'], {})
            for f in ('성명', '연락처'): chk('관계인(⑧)', f"{pre}관계인.{r['구분']}.{f}", r[f], q.get(f))
        for role, q in prel.items():                                     # 정답에 없는 역할에 값이 생기면 오답
            if role not in {r['구분'] for r in g['관계인']} and (q.get('성명') or q.get('연락처')):
                R['관계인(⑧)'][1] += 1; bad.append(dict(field=f'{pre}관계인.{role} 과검출', got=q))
        key = lambda h: (_ns(h['구분']), _ns(h['시설명']), _ns(h['품명']), _ns(h['수량']))
        gh, ph = [key(h) for h in g['위험물']], [key(h) for h in p.get('위험물') or []]
        for h in gh:
            ok = h in ph; R['위험물(⑨)'][0] += ok; R['위험물(⑨)'][1] += 1
            if ok: ph.remove(h)
            else: bad.append(dict(field=pre + '위험물', gt=h))
        if ph: R['위험물(⑨)'][1] += len(ph); bad.append(dict(field=pre + '위험물 과검출', got=ph))
        pf = {r['층']: r for r in p.get('층별현황') or []}
        for r in g['층별현황']:
            q = pf.get(r['층'], {})
            chk('층별현황(⑩)', f"{pre}{r['층']}.층", r['층_원문'], q.get('층_원문'))
            chk('층별현황(⑩)', f"{pre}{r['층']}.용도", r['용도'], q.get('용도'))
            chk('층별현황(⑩)', f"{pre}{r['층']}.면적", r['면적'], q.get('면적'), 0.005)
            chk('층별현황(⑩)', f"{pre}{r['층']}.기타", r['기타'], q.get('기타'))
            a, b = parse_fac(r['소방시설_원문']), parse_fac(q.get('소방시설_원문') or '')
            chk('층별현황(⑩)', f"{pre}{r['층']}.소방시설", json.dumps(a, ensure_ascii=False, sort_keys=True), json.dumps(b, ensure_ascii=False, sort_keys=True))
            for name, n in a.items():
                ok = b.get(name) == n; R['소방시설항목(⑩)'][0] += ok; R['소방시설항목(⑩)'][1] += 1
        extra = set(pf) - {r['층'] for r in g['층별현황']}
        if extra: R['층별현황(⑩)'][1] += len(extra); bad.append(dict(field=pre + '층 과검출', got=sorted(extra)))
    GF = gt.get('card_std_figs') or []
    for i, f in enumerate(GF):
        q = figs[i] if i < len(figs) else {}
        ok = q.get('dong') == f['동'] and q.get('kind') == f['종류'] and list(q.get('floors') or []) == list(f['floors'])
        R['그림(⑪)'][0] += ok; R['그림(⑪)'][1] += 1
        if not ok: bad.append(dict(field=f'그림{i + 1}', gt=[f['동'], f['종류'], f['floors']], got=[q.get('dong'), q.get('kind'), q.get('floors')]))
    if len(figs) > len(GF): R['그림(⑪)'][1] += len(figs) - len(GF)
    return R, bad


def score_dir(d, res=None):
    """건물 폴더(card.pdf + gt.json) → dict(card_std={그룹: 'a/b'}, card_std_bad=[...], privacy=...)"""
    if not os.path.exists(os.path.join(d, 'gt.json')) or not os.path.exists(os.path.join(d, 'card.pdf')): return {}
    gt = json.load(open(os.path.join(d, 'gt.json'), encoding='utf-8'))
    if 'card_std' not in gt: return {}
    info, doc = parse_card(os.path.join(d, 'card.pdf'), dong=0)
    figs = extract_figures(doc)
    R, bad = score_std(gt, info, figs)
    out = dict(card_std={g: f'{a}/{b}' for g, (a, b) in R.items()}, card_std_bad=bad[:30], card_variant=gt.get('card_form', {}).get('variant'))
    if res is not None:                                    # 개인정보: 파이프라인 결과(JSON)에 관계인 성명·연락처가 남으면 실패
        txt = json.dumps(res, ensure_ascii=False)
        pii = [r[k] for g in gt['card_std'] for r in g['관계인'] for k in ('성명', '연락처') if r.get(k)]
        leak = sorted({x for x in pii if x in txt})
        out['privacy_leak'] = leak
    return out


def summarize(rows):
    acc = {g: [0, 0] for g in GROUPS}
    for r in rows:
        for g, v in (r.get('card_std') or {}).items():
            a, b = map(int, v.split('/')); acc[g][0] += a; acc[g][1] += b
    if not any(b for a, b in acc.values()): return None
    tot = [sum(a for a, b in acc.values()), sum(b for a, b in acc.values())]
    out = {g: f'{a}/{b}' for g, (a, b) in acc.items()}
    out['전체'] = f'{tot[0]}/{tot[1]} ({tot[0] / max(1, tot[1]):.4f})'
    by_var = {}
    for r in rows:
        v = r.get('card_variant')
        if not v: continue
        t = by_var.setdefault(v, [0, 0])
        for g, s in (r.get('card_std') or {}).items():
            a, b = map(int, s.split('/')); t[0] += a; t[1] += b
    out['변형별'] = {k: f'{a}/{b}' for k, (a, b) in by_var.items()}
    pl = [r for r in rows if 'privacy_leak' in r]
    if pl: out['개인정보_유출_건물'] = f"{sum(1 for r in pl if r['privacy_leak'])}/{len(pl)}"
    return out


if __name__ == '__main__':
    for D in sys.argv[1:] or ['data_std']:
        rows = []
        for d in sorted(glob.glob(os.path.join(D, '*/'))):
            r = score_dir(d.rstrip('/'))
            if r: rows.append(dict(r, name=os.path.basename(d.rstrip('/'))))
        for r in rows:
            if r['card_std_bad']: print(r['name'], r['card_variant'], json.dumps(r['card_std_bad'][:6], ensure_ascii=False))
        print(D, json.dumps(summarize(rows), ensure_ascii=False))
