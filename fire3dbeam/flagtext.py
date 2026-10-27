# -*- coding: utf-8 -*-
"""확인사항(플래그) 문구 — 현장 대원이 읽는 쉬운 한국어(msg) + 개발·검수용 기술 정보(detail)를 나눔 (2026-10-03 2차 강건화).
pipeline.run 끝에서 annotate(res) 가 모든 플래그에 msg·detail 을 붙이고, 층마다 scale_basis(축척 근거 쉬운 말)를 붙인다.
export(지휘화면 확인사항)·review/make(검수 화면)는 msg 를 본문으로, detail 을 따로(마우스 올림) 쓴다."""
import re

TITLE = dict(fac_count='소방시설 수량 불일치', area='층 면적 불일치', hose_count='옥내소화전 수량 불일치', stair_count='계단 수 불일치', align='층 위치 맞춤 신뢰도 낮음',
             elev_count='승강기 수 불일치', scale_disagree='도면 축척 확인 필요', stair_partial='계단 연결 층 불확실', elev_text='승강기(글자로만 찾음)',
             entrance_est='출입구 추정', emev_count='비상용 승강기 수 불일치', xcheck_wall='벽 인식 불일치(AI·규칙)', card_ocr='스캔본 항목 확인', hazmat='위험물')
SRC_KO = dict(sheet='도면 표제란의 축척 표기', dims='치수선 숫자', tags='창호 표기(폭×높이)와 개구부 폭', tread='계단 디딤판 간격',
              doors='표준 문 폭', area='정보카드 층 면적', grid='다른 층 도면과 그리드 간격 비교', default='기본값(근거 없음)')
STRONG = ('sheet', 'dims', 'tags')


def floor_ko(f):
    f = f or ''
    if not f: return ''
    if f == 'RF': return '옥탑'
    if f.startswith('B'): return f'지하{f[1:]}층'
    if f.endswith('F') and f[:-1].isdigit(): return f'{f[:-1]}층'
    return f


def scale_parts(src):
    """'real:sheet+grid:tags+tread' → [('sheet', False), ('tags', True), ('tread', False)] (True = 다른 층에서 그리드로 옮겨 온 근거)"""
    s = (src or '').replace('real:', '')
    out = []
    for p in s.split('+'):
        if not p: continue
        via = p.startswith('grid:')
        k = p.split(':')[-1]
        if (k, via) not in out: out.append((k, via))
    return out


def scale_basis(src):
    """축척 출처 문자열 → 쉬운 말 한 줄(근거 최대 3가지)"""
    P = scale_parts(src)
    if not P: return '축척 근거 없음(기본값) — 치수 확인 필요'
    own = [k for k, v in P if not v]
    via = [k for k, v in P if v and k not in own]
    order = ['sheet', 'dims', 'tags', 'tread', 'doors', 'area', 'grid', 'default']
    rk = lambda k: order.index(k) if k in order else 9
    words = [SRC_KO.get(k, k) for k in sorted(set(own), key=rk) if k in STRONG]
    vs = [k for k in sorted(set(via), key=rk) if k in STRONG]
    if vs: words.append('다른 층 도면의 ' + '·'.join(SRC_KO.get(k, k).split('(')[0] for k in vs[:2]) + '(그리드 간격으로 옮김)')
    words += [SRC_KO.get(k, k) for k in sorted(set(own), key=rk) if k not in STRONG]
    weak = not any(k in STRONG for k, _ in P)
    return '축척 근거: ' + ', '.join(words[:3]) + (' — 근거가 약함, 치수 한 곳 재어 확인' if weak else '')


def _pct(a, b):
    return abs(a / b - 1) * 100 if a and b else 0


def explain(f):
    """플래그 → (msg, detail)"""
    k = f.get('kind'); fl = floor_ko(f.get('floor'))
    det = ', '.join(f'{a}={b}' for a, b in f.items() if a not in ('kind', 'msg', 'detail') and not isinstance(b, (list, dict)))
    if k == 'scale_disagree':
        cs = f.get('cands') or []
        worst = max((_pct(c['s'], f['s']) for c in cs), default=0)
        msg = f"{fl} 도면의 축척을 정한 근거들이 서로 최대 {worst:.0f}% 달라, 3D 건물 크기·거리가 틀릴 수 있음 → 도면의 치수 한 곳을 재어 확인. {scale_basis(f.get('src'))}"
        det = f"s={f.get('s')} px/m · scale_src={f.get('src')} · 출처별 " + ', '.join(f"{c['src']} {c['s']}" for c in cs)
        return msg, det
    if k == 'area':
        r = f.get('ratio') or 0
        why = ('도면 외곽에 옥상정원·주차장·데크 같은 바깥 공간이 섞였거나 카드 면적이 작게 적혔을 수 있음' if r > 1 else
               '도면에서 일부 실이 빠졌거나 카드 면적이 다른 기준(전용·공용 합산 등)일 수 있음')
        return f"{fl}: 정보카드 면적 {f['card']:,.0f}㎡ 와 도면에서 잰 건물 면적 {f['drawing']:,.0f}㎡ 가 다름({why}) → 3D 외곽 확인", det
    if k == 'align':
        return f"{fl}: 다른 층과 위치를 맞출 기준(계단·승강기·그리드선)이 부족해 층이 어긋나 보일 수 있음 → 3D 에서 층 겹침 확인", det
    if k == 'stair_count':
        return f"정보카드에는 계단 {f['card']}개소, 도면에서 찾은 계단은 {f['drawing']}개소 → 빠진 계단(옥외·한 층만 그려진 계단)을 검수 화면에서 확인", det
    if k == 'elev_count':
        return f"정보카드에는 승강기 {f['card']}대, 도면에서 찾은 승강기는 {f['drawing']}대 → 승강기 위치를 검수 화면에서 확인", det
    if k == 'emev_count':
        return f"정보카드의 비상용 승강기 {f['card']}대 중 도면에서 {f['drawing']}대만 찾음 → 비상용 승강기 위치 확인", det
    if k == 'hose_count':
        return f"{fl}: 옥내소화전 정보카드 {f['card']}개 / 도면 표기 {f['drawing']}개 → 소화전 위치 확인", det
    if k == 'xcheck_wall':
        return f"{fl}: AI 모델과 규칙 인식이 본 벽이 여러 곳 다름 → 검수 화면에서 벽 확인", det
    if k == 'stair_partial':
        nm = f"{f.get('name', '')}{'(옥외)' if f.get('outdoor') else ''}"
        ex = [x for x in f.get('ext_floors') or []]
        sure = [x for x in f.get('floors') or [] if x not in ex]
        if ex:
            return (f"{nm}: {', '.join(floor_ko(x) for x in sure)} 도면에서는 확실하고, {', '.join(floor_ko(x) for x in ex)}은(는) 같은 자리의 점선·흐린 디딤판으로만 이어짐 "
                    f"→ 그 층 연결은 현장·도면으로 확인"), det
        return f"{nm}: {', '.join(floor_ko(x) for x in sure)} 도면에만 그려져 있음({'옥외 계단' if f.get('outdoor') else '위·아래 층 도면에 없는 계단 — 지하·옥상으로 이어질 수 있음'}) → 연결 층 확인", det
    if k == 'elev_text':
        return f"{f.get('name', '')}: 도면의 글자({f.get('word', '승강기 표기')})로만 위치를 찾음(승강로 X 표시 없음) → 위치·정차 층 확인", det
    if k == 'entrance_est':
        return f"{fl}: 출입구 {f.get('n', 1)}곳은 창호 기호·문 열림선·바깥과 닿음으로 추정한 것 → 현장 출입구와 대조", det
    if k == 'card_ocr':
        return f"스캔본 정보카드 '{f.get('field')}' 항목을 글자 인식으로 읽어 확실하지 않음 → 원본과 대조", det
    if k == 'fac_count':
        return f"{fl} {f.get('name')}: 정보카드 {f.get('card')}개 / 도면 기호 {f.get('drawing')}개 → 수량·위치 확인", det
    if k == 'hazmat':
        return f"위험물 {f.get('cls') or ''} {f.get('name') or f.get('item') or ''} {f.get('qty') or ''} — 진입 전 위치·종류 확인", det
    return None, det


def annotate(res):
    for f in res.get('flags') or []:
        if f.get('msg'): continue
        m, d = explain(f)
        if m: f['msg'] = m
        if d and not f.get('detail'): f['detail'] = d
    for fid, F in (res.get('floors') or {}).items():
        if F.get('scale_src'): F['scale_basis'] = scale_basis(F['scale_src'])
    for p in res.get('plans') or []:
        if p.get('scale_src'): p['scale_basis'] = scale_basis(p['scale_src'])
    return res
