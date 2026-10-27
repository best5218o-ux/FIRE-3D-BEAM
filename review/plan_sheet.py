# -*- coding: utf-8 -*-
"""사전 활동계획서(인쇄용 HTML 1장) — 변환 결과(res) + 지휘화면 데이터(D) 에서 화점과 무관한 부분만 정리.
  위험요소 → 외부 부서(면별 접근·펌프차·송수구·소화전·고가사다리차·진입창) → 진입로 → 수직동선 → 층별(대피·소방시설) → 확인사항.
화점·경보를 넣어야 나오는 활동계획(화점 경로·호스 본수·공기호흡기 귀환 한계·화점 기준 인명검색 순서)은 3D 지휘화면에서 산출한다.
화점과 무관한 인명검색 우선 실(평시 대피 곤란·용도·계단 거리, fire3dbeam/search.static_order)은 여기 5-1 절에 싣는다.
인터넷 없이 열림(외부 글꼴·스크립트 없음). 관계인 성명·연락처는 결과 JSON 에 없으면(기본) 여기에도 없다."""
import datetime, html, os

CERT = {'located': '위치확인', 'estimated': '위치추정', 'present': '수량만', 'mixed': '일부위치'}
LV = {'high': '높음', 'mid': '보통', 'low': '낮음'}


def _e(x): return html.escape('' if x is None else str(x))


def _ko(fid):
    from fire3dbeam.export import ko_name
    try: return ko_name(fid)
    except Exception: return str(fid)


def _num(v, fmt='{:,.0f}'):
    try: return fmt.format(float(v))
    except (TypeError, ValueError): return '-'


def _table(head, rows, cls=''):
    if not rows: return '<p class="none">해당 없음(자료 없음)</p>'
    h = ''.join(f'<th>{_e(x)}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def build(res, D, html_name=None, src_note=None):
    from fire3dbeam.card import level_of
    card = res.get('card') or {}
    B = D.get('building') or {}
    S = card.get('건물구조') or {}
    fids = sorted(res.get('floors') or {}, key=level_of)
    out = []
    # ── 머리
    addr = (card.get('소재지') or {}).get('도로명') or (card.get('소재지') or {}).get('지번')
    dist = card.get('거리') or {}
    fr = card.get('내화구조')
    meta = [('대상물', B.get('name')), ('주용도', B.get('use')), ('층수', B.get('floors')), ('연면적', (_num(B.get('gfa')) + '㎡') if B.get('gfa') else None),
            ('구조', (S.get('구조') or '') + ((' (내화 추정)' if fr else ' (비내화 추정)') if fr is not None else '')), ('관리번호', card.get('관리번호')),
            ('소재지', addr), ('거리(관할센터)', f"{dist.get('관할센터')} km" if dist.get('관할센터') is not None else None),
            ('정보카드', f"스캔본 OCR — 확인 필요 {len((card.get('ocr') or {}).get('low') or [])}항목(확인사항 참고)" if card.get('source') == 'ocr' else None)]
    out.append('<div class="meta">' + ''.join(f'<div><b>{_e(k)}</b> {_e(v)}</div>' for k, v in meta if v) + '</div>')
    # ── 1. 위험요소
    iss = D.get('issues') or []
    haz = card.get('위험물') or []
    rows = [[_e(h.get('구분')), _e(h.get('시설명')), _e(h.get('품명')), _e(h.get('수량'))] for h in haz]
    out.append('<h2>1. 위험요소 (정보카드 ⑨)</h2>' + (_table(['구분', '시설명', '품명', '수량'], rows, 'haz') if rows else '<p class="none">정보카드에 기재된 위험물·가스·특수가연물 없음</p>'))
    # ── 2. 외부 부서
    site = res.get('site') if isinstance(res.get('site'), dict) and 'error' not in res.get('site', {}) else None
    out.append('<h2>2. 도착·외부 부서</h2>')
    if site:
        out.append(f"<p class='src'>부지 출처: {_e(site.get('file') or site.get('src'))}</p>")
        rows = [[_e(f.get('name')), '가능' if f.get('engine') else '<b class="bad">불가</b>', _e(f.get('road') or '-'),
                 (_num(f.get('setback'), '{:.1f}') + ' m') if f.get('setback') is not None else '-'] for f in site.get('facades') or []]
        out.append('<h3>면별 소방차 접근</h3>' + _table(['면', '펌프차 접근', '도로', '이격'], rows))
        pu = site.get('pumper') or {}
        if pu:
            out.append(f"<h3>펌프차 → 연결송수구</h3><p>{_e(pu.get('road'))} 에 부서 → {_e(pu.get('fdc'))} 까지 {_num(pu.get('fdc_d'), '{:.0f}')} m (호스 {_e(pu.get('fdc_hose'))}본)</p>")
            hy = sorted(pu.get('hydrants') or [], key=lambda h: h.get('road_d') or 1e9)[:4]
            rows = [[_e(h.get('id')), _e(h.get('type')), _num(h.get('road_d'), '{:.0f}') + ' m', _e(h.get('hose'))] for h in hy]
            out.append('<h3>가까운 소화전(도로 따라)</h3>' + _table(['번호', '종류', '거리', '호스(본)'], rows))
        rows = [[_e(a.get('id')), _e(a.get('road')), _e(', '.join(a.get('facades') or [])), _e(_ko(a['top']) if a.get('top') else '-')] for a in site.get('aerial') or []]
        out.append('<h3>고가사다리차 부서 후보</h3>' + _table(['후보', '도로', '닿는 면', '최고 도달층'], rows))
        rows = [[_e(_ko(w.get('floor'))), _e(w.get('facade')), _num(w.get('h'), '{:.1f}') + ' m', _e(w.get('by')), _num(w.get('dist'), '{:.1f}') + ' m'] for w in site.get('windows') or []]
        out.append('<h3>진입창 후보</h3>' + _table(['층', '면', '높이', '사다리차', '수평거리'], rows))
        if site.get('unreachable'): out.append(f"<p class='bad'>고가사다리차 도달 불가 층: {_e(', '.join(_ko(f) for f in site['unreachable']))} → 내부 진입·피난 유도 우선</p>")
    else:
        out.append('<p class="none">부지 자료(진압작전도·site.json) 없음 — 외부 부서 계산 생략</p>')
    rows = [[_e(w.get('종류')), _e('-'.join(str(x) for x in (w.get('관서'), w.get('동'), w.get('번호')) if x)), _num(w.get('거리km'), '{:.2f}') + ' km'] for w in card.get('소방용수') or []]
    if rows: out.append('<h3>소방용수 (정보카드 ⑥)</h3>' + _table(['종류', '번호', '거리'], rows))
    # ── 3. 진입로
    ev = res.get('evac') if isinstance(res.get('evac'), dict) and 'error' not in res.get('evac', {}) else None
    out.append('<h2>3. 진입로 (진입구 → 계단)</h2>')
    rows = []
    for en in (ev or {}).get('entry') or []:
        rt = sorted(en.get('routes') or [], key=lambda r: r.get('d') or 1e9)
        rows.append([_e(en.get('name')), _e(_ko(en.get('floor'))), '<br>'.join(f"{_e(r.get('name'))} {_num(r.get('d'), '{:.0f}')} m" + (' (벽 통과 가정)' if r.get('wall') else '') for r in rt[:3]) or '-'])
    out.append(_table(['진입구', '층', '계단까지 보행거리'], rows) if ev else '<p class="none">대피로·진입로 계산 없음</p>')
    # ── 4. 수직동선
    st = B.get('stairs') or {}
    out.append('<h2>4. 수직동선</h2>')
    rows = [[f'계단{i + 1}', _e(f"{_ko(s['floors'][0])} ~ {_ko(s['floors'][-1])}" if s.get('floors') else '-'), '검수자 추가' if s.get('review') == 'added' else '자동'] for i, s in enumerate(res.get('stairs') or [])]
    rows += [[f'승강기 {chr(65 + i)}', _e(f"{_ko(s['floors'][0])} ~ {_ko(s['floors'][-1])}" if s.get('floors') else '-'), '검수자 추가' if s.get('review') == 'added' else '자동'] for i, s in enumerate(res.get('elevs') or [])]
    out.append(_table(['구분', '연결층', '출처'], rows))
    cc = ' · '.join(f'{k} {v}' for k, v in st.items() if v)
    if cc: out.append(f'<p class="src">정보카드 ⑦: {_e(cc)}</p>')
    # ── 5. 층별
    out.append('<h2>5. 층별 대피·소방시설</h2>')
    rows = []
    for f in fids:
        c = (res['floors'][f].get('card') or {})
        es = (((ev or {}).get('floors') or {}).get(f) or {})
        sm = es.get('summary') or {}
        evt = (f"실 {sm.get('n', 0)} · 최대 {_num(sm.get('max'), '{:.0f}')} m / 기준 {_num(es.get('lim'), '{:.0f}')} m" + (f" · <b class='bad'>초과 {sm['over']}</b>" if sm.get('over') else '')) if sm else '-'
        fac = []
        for it in ((res.get('fac') or {}).get('floors') or {}).get(f) or []:
            n = it.get('card') if it.get('card') is not None else it.get('drawing')
            lab = it.get('name') if it.get('card') is None else (it.get('raw') or it.get('name'))
            fac.append(f"{_e(lab)} {_e(n if n is not None else '')}<span class='cert c-{_e(it.get('cert'))}'>{_e(CERT.get(it.get('cert'), it.get('cert') or ''))}</span>")
        rows.append([_e(_ko(f)), _e(c.get('use') or ''), (_num(c.get('area')) + '㎡') if c.get('area') else '-', evt, ' · '.join(fac) or '-'])
    out.append(_table(['층', '용도', '면적(카드)', '대피(보행거리)', '소방시설(정보카드 수량 · 위치 확실성)'], rows, 'floors'))
    out.append('<p class="src">위치확인 = 도면 기호에서 위치를 찾음 · 위치추정 = 법정 설치 기준으로 추정(현장 확인) · 수량만 = 정보카드 수량뿐(위치 모름) · 일부위치 = 일부만 도면에서 찾음</p>')
    # ── 5-1. 인명검색 우선 실(화점 미정) — fire3dbeam/search.py static_order(기준값 sop.py SEARCH, 임시값)
    from fire3dbeam.search import static_order, CLASS_KO, CAT_KO
    so = static_order(D, mode='night', top=3) if (D.get('evac') or {}).get('floors') else {}
    rows = []
    for f in fids:
        for i, z in enumerate(so.get(f) or []):
            rows.append([_e(_ko(f)) if i == 0 else '', str(i + 1), _e(CLASS_KO.get(z['cls'], z['cls'])), _e(CAT_KO.get(z['cat'], z['cat'])) + f" ({['높음', '보통', '낮음'][z['band']]})",
                         _num(z.get('area'), '{:.0f}') + '㎡', (_num(z['d'], '{:.0f}') + ' m') if z.get('d') is not None else '경로 없음', _e(z.get('why'))])
    out.append('<h2>5-1. 인명검색 우선 실 (화점 미정 · 야간 기준 · 층별 상위 3)</h2>'
               + (_table(['층', '순위', '평시 대피 곤란', '용도 범주(가중 등급)', '면적', '계단까지', '용도 근거'], rows, 'srch') if rows else '<p class="none">대피로 계산 없음 — 인명검색 우선 실 생략</p>')
               + '<p class="src">정렬: 평시 대피 곤란(경로 없음 → 기준 초과·한 방향 → 정상) → 용도·시간대 가중 등급 → 가까운 계단까지 보행거리. 화점을 지정하면 3D 지휘화면이 '
               '화재층 분류(고립·화재실·공격계단 공용·화점 인접·우회)와 공격계단 기준 거리로 다시 정렬하고 화재층 → 직상층 순으로 번호를 붙입니다. '
               '용도 범주는 정보카드 ⑩ 층 용도·실 모양(실명이 있으면 실명)으로 추정 · 가중치는 임시값(기관 SOP로 설정).</p>')
    # ── 6. 확인사항
    order = {'high': 0, 'mid': 1, 'low': 2}
    rows = [[f"<span class='lv lv-{_e(i.get('lv'))}'>{_e(LV.get(i.get('lv'), i.get('lv')))}</span>", _e(i.get('cat') or ''), f"<b>{_e(i.get('t'))}</b><br>{_e(i.get('d'))}"]
            for i in sorted(iss, key=lambda i: order.get(i.get('lv'), 3))]
    out.append(f'<h2>6. 확인사항 ({len(rows)}건 — 자동 교차검증, 현장·검수로 확인)</h2>' + _table(['중요도', '분류', '내용'], rows, 'iss'))
    # ── 7. 화점 입력 후
    out.append('<h2>7. 화점·경보 입력 후 산출 (3D 지휘화면)</h2><p>화점 경로(진입구 → 공격계단 → 화점), 필요 호스 본수(경로 위 1본 눈금), 공기호흡기 귀환 한계점, 인명검색 우선순위(근거 포함), 수신기 경보(층·회로) → 경계구역 화점 후보, 훈련 모드 채점은 '
               + (f'<b>{_e(html_name)}</b>' if html_name else '3D 지휘화면') + ' 에서 화점을 지정하면 계산됩니다.</p>')
    # ── 꼬리
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    foot = [f'생성 {now}', '자동 변환 결과(검수 전이면 오류가 있을 수 있음) — 현장 판단 보조자료', f"개인정보: {_e(card.get('개인정보') or '-')}"]
    if src_note: foot.append(_e(src_note))
    title = f"{B.get('name') or '대상물'} 사전 활동계획"
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>
body{{font-family:'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif;margin:16px auto;max-width:1000px;padding:0 16px;color:#111;background:#fff;font-size:13px;line-height:1.45}}
h1{{font-size:20px;margin:0 0 6px}} h2{{font-size:15px;margin:18px 0 6px;border-bottom:2px solid #333;padding-bottom:2px}} h3{{font-size:13px;margin:10px 0 4px}}
.meta{{display:flex;flex-wrap:wrap;gap:4px 16px;border:1px solid #999;padding:6px 8px}} table{{border-collapse:collapse;width:100%;margin:2px 0 6px}}
th,td{{border:1px solid #aaa;padding:3px 5px;vertical-align:top;text-align:left}} th{{background:#eee}} .none,.src{{color:#555;font-size:12px}}
.bad{{color:#b00}} table.haz td{{background:#fde8e8}} .lv{{font-weight:bold}} .lv-high{{color:#b00}} .lv-mid{{color:#a60}}
.cert{{font-size:10px;border:1px solid #888;border-radius:3px;padding:0 3px;margin-left:3px;color:#333}} .c-located{{border-color:#2a7}} .c-estimated{{border-style:dashed}}
.foot{{margin-top:16px;color:#555;font-size:11px;border-top:1px solid #aaa;padding-top:4px}} .warn{{background:#fff4d6;border:1px solid #d9a400;padding:4px 8px;margin:6px 0}}
@media print{{body{{margin:0;max-width:none}} h2{{page-break-after:avoid}} tr{{page-break-inside:avoid}}}}
</style></head><body><h1>{_e(title)}</h1>
<div class="warn">정보카드·도면 자동 변환으로 만든 사전 활동계획입니다. 확인사항과 '위치추정/수량만' 표시는 현장에서 확인하세요. 내부 PC 밖 반출 금지.</div>
{''.join(out)}<div class="foot">{' · '.join(foot)}</div></body></html>"""


def write(res, D, path, html_name=None, src_note=None):
    h = build(res, D, html_name=html_name, src_note=src_note)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: f.write(h)
    os.replace(tmp, path)
    return path
