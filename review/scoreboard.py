# -*- coding: utf-8 -*-
"""채점 화면(오프라인 HTML 1개): 가상 건물 자동 채점(out/score_all.json + out/summary.json) + 검수 기반 성능(학습데이터/review_metrics.json).
사용: python -m review.scoreboard [--score out/score_all.json] [--summary out/summary.json] [--review 학습데이터/review_metrics.json] [-o out/scoreboard.html]
목표값은 잠정(TARGETS) — 통과(✓)·주의(!)·미달(✗)은 색 + 기호 + 글자로 함께 표시."""
import argparse, datetime, html, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review.common import ROOT, KO, load_json

# (이름, 통과, 주의, 높을수록 좋음)
TARGETS = {
    'align': ('층정합(1m 이내)', 0.95, 0.85, True), 'align_m': ('정합 평균오차(m)', 0.5, 1.0, False),
    'wall_R': ('벽 재현율', 0.85, 0.75, True), 'wall_P': ('벽 정밀도', 0.80, 0.70, True),
    'door_R': ('문 재현율', 0.60, 0.40, True), 'door_P': ('문 정밀도', 0.60, 0.40, True),
    'stairs': ('계단 찾음', 0.95, 0.85, True), 'stairs_ok': ('계단 정확', 0.95, 0.85, True),
    'elevs': ('승강기 찾음', 0.95, 0.85, True), 'flags': ('교차검증(심은 오류)', 0.95, 0.85, True),
    'card': ('정보카드 해석', 1.0, 0.98, True), 'sym_R': ('기호 재현율', 0.90, 0.75, True), 'sym_P': ('기호 정밀도', 0.80, 0.60, True),
}
BT_KO = {'core': '코어형', 'corridor': '복도형', 'hall': '대공간형', 'ALL': '전체'}


def frac(v):
    if v is None: return None, ''
    if isinstance(v, (int, float)): return float(v), f'{v:.2f}'
    a, b = str(v).split('/')
    return (int(a) / int(b) if int(b) else None), str(v)


def status(key, val):
    if val is None: return 'na'
    _, ok, warn, hi = TARGETS[key]
    if hi: return 'good' if val >= ok - 1e-9 else ('warn' if val >= warn else 'bad')
    return 'good' if val <= ok else ('warn' if val <= warn else 'bad')


ICON = {'good': '✓', 'warn': '!', 'bad': '✗', 'na': '·'}
WORD = {'good': '통과', 'warn': '주의', 'bad': '미달', 'na': '없음'}


def cell(key, val, text=None, sub=''):
    st = status(key, val)
    t = text if text is not None else ('-' if val is None else f'{val:.2f}')
    nm, ok, warn, hi = TARGETS[key]
    tip = f"{nm}: 목표 {'≥' if hi else '≤'}{ok} (주의 {'≥' if hi else '≤'}{warn}) → {WORD[st]}"
    return f'<td class="c {st}" title="{html.escape(tip)}"><span class="ic" aria-label="{WORD[st]}">{ICON[st]}</span> {html.escape(t)}{sub}</td>'


def sym_rp(v):
    """'R0.98/P0.97' 또는 {'recall','precision'} → (R, P)"""
    if isinstance(v, dict): return v.get('recall'), v.get('precision')
    try: r, p = v[1:].split('/P'); return float(r), float(p)
    except Exception: return None, None


def row_cells(r, is_summary):
    g = (lambda k: r.get(k)) if is_summary else (lambda k: r.get({'align': 'align_ok', 'align_m': 'align_mean_m', 'wall_R': 'wall_recall', 'wall_P': 'wall_precision',
                                                                    'door_R': 'door_recall', 'door_P': 'door_precision', 'stairs': 'stairs_found', 'stairs_ok': 'stairs_correct',
                                                                    'elevs': 'elevs_found', 'flags': 'flags_planted_found', 'card': 'card_floors_ok'}[k]))
    keys = ['align', 'align_m', 'wall_R', 'wall_P', 'door_R', 'door_P', 'stairs', 'stairs_ok', 'elevs', 'flags', 'card']
    if is_summary: g2 = {'align': 'align_ok', 'align_m': 'align_mean_m', 'stairs': 'stairs', 'stairs_ok': 'stairs_ok', 'elevs': 'elevs', 'flags': 'flags', 'card': 'card'}
    out = []
    for k in keys:
        v = r.get(g2.get(k, k)) if is_summary else g(k)
        val, txt = frac(v)
        out.append(cell(k, val, txt if isinstance(v, str) else None))
    return out


def sym_cells(syms):
    out = []
    for t in ('exit', 'hose', 'panel', 'gas', 'ev'):
        if t not in (syms or {}): out.append('<td class="c na">·</td>'); continue
        R, P = sym_rp(syms[t])
        sr, sp = status('sym_R', R), status('sym_P', P)
        worst = 'bad' if 'bad' in (sr, sp) else ('warn' if 'warn' in (sr, sp) else 'good')
        n = f'<span class="n">n={syms[t]["n"]}</span>' if isinstance(syms[t], dict) and 'n' in syms[t] else ''
        out.append(f'<td class="c {worst}" title="{KO[t]}: 재현율 {R} (목표≥0.9) / 정밀도 {P} (목표≥0.8) → {WORD[worst]}"><span class="ic">{ICON[worst]}</span> R{R:.2f} P{P:.2f}{n}</td>')
    return out


HEAD = ['층정합', '평균오차m', '벽R', '벽P', '문R', '문P', '계단', '계단정확', '승강기', '교차검증', '카드']


def tiles(allr):
    T = []
    for k, key in (('align', 'align_ok'), ('wall_R', 'wall_R'), ('wall_P', 'wall_P'), ('door_R', 'door_R'), ('stairs', 'stairs'), ('elevs', 'elevs'), ('flags', 'flags'), ('card', 'card')):
        val, txt = frac(allr.get(key)); st = status(k, val)
        T.append(f'<div class="tile {st}"><div class="tl">{TARGETS[k][0]}</div><div class="tv">{"-" if val is None else f"{val * 100:.0f}%"}</div>'
                 f'<div class="ts"><span class="ic">{ICON[st]}</span> {WORD[st]}{" · " + html.escape(txt) if isinstance(allr.get(key), str) else ""} · 목표 {TARGETS[k][1] * 100:.0f}%</div></div>')
    return ''.join(T)


def review_section(mets):
    if not mets: return '<p class="mut">검수 기반 성능 없음 — 검수 결과를 <b>검수.bat</b>으로 반영하면 여기에 쌓입니다(정답 없는 실제 건물의 성능 지표).</p>'
    rows = []
    for m in mets:
        sy = m.get('symbols') or {}
        cells = []
        for t in ('exit', 'hose', 'panel', 'gas', 'ev'):
            v = sy.get(t)
            if not v or v.get('precision') is None and v.get('recall') is None: cells.append('<td class="c na">·</td>'); continue
            R, P = v.get('recall'), v.get('precision')
            sr, sp = status('sym_R', R), status('sym_P', P)
            worst = 'bad' if 'bad' in (sr, sp) else ('warn' if 'warn' in (sr, sp) else ('good' if R is not None and P is not None else 'na'))
            cells.append(f'<td class="c {worst}"><span class="ic">{ICON[worst]}</span> R{"-" if R is None else f"{R:.2f}"} P{"-" if P is None else f"{P:.2f}"}<span class="n">{v["n_final"]}</span></td>')
        d = m.get('doors') or {}
        cells.append(cell('door_R', d.get('recall'))); cells.append(cell('door_P', d.get('precision')))
        w = m.get('walls') or {}
        cells.append(cell('wall_P', w.get('precision'))); cells.append(cell('wall_R', w.get('recall')))
        rl = m.get('review_load') or {}
        load = f"{rl.get('items_done', '-')}/{rl.get('items_total', '-')} (자동 {rl.get('auto_total', '-')})"
        fl = m.get('flags') or {}
        rows.append(f"<tr><td class='nm'>{html.escape(str(m.get('building') or m.get('id')))}</td><td>{m.get('plans_used', 0)}</td>{''.join(cells)}"
                    f"<td>{load}</td><td>{fl.get('confirmed', 0)} / {fl.get('dismissed', 0)}</td><td class='mut'>{html.escape((m.get('date') or '')[:16])}</td></tr>")
    return ('<table><thead><tr><th>건물</th><th>완료 도면</th><th>비상구</th><th>소화전</th><th>수신기</th><th>가스</th><th>승강기</th><th>문R</th><th>문P</th><th>벽P</th><th>벽R</th>'
            '<th>검수 처리/필요</th><th>불일치 확인/기각</th><th>일시</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
            '<p class="mut">자동 인식 결과 대비 사람이 확정한 결과(검수 완료 도면만). 기호: 삭제·종류변경 = 오검출, 추가 = 누락. 벽: 삭제 길이/추가 길이 기준.</p>')


CSS = """
:root{--bg:#f7f7f5;--surf:#ffffff;--ink:#1a1a19;--ink2:#52524e;--mut:#8a8a84;--line:#e2e2de;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--goodbg:#0ca30c14;--warnbg:#fab21922;--badbg:#d03b3b17}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111110;--surf:#1a1a19;--ink:#ececea;--ink2:#b5b5b0;--mut:#85857f;--line:#2e2e2c;--goodbg:#0ca30c22;--warnbg:#fab21920;--badbg:#d03b3b26}}
:root[data-theme=dark]{--bg:#111110;--surf:#1a1a19;--ink:#ececea;--ink2:#b5b5b0;--mut:#85857f;--line:#2e2e2c;--goodbg:#0ca30c22;--warnbg:#fab21920;--badbg:#d03b3b26}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:1320px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px;color:var(--ink)}
.mut{color:var(--mut);font-size:12.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:14px}
.tile{background:var(--surf);border:1px solid var(--line);border-radius:10px;padding:10px 12px;border-top:4px solid var(--line)}
.tile.good{border-top-color:var(--good)}.tile.warn{border-top-color:var(--warn)}.tile.bad{border-top-color:var(--bad)}
.tl{font-size:12px;color:var(--ink2)}.tv{font-size:26px;font-weight:700;margin:2px 0}.ts{font-size:11.5px;color:var(--ink2)}
.wrap{overflow-x:auto;background:var(--surf);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{position:sticky;top:0;background:var(--surf);text-align:left;font-weight:600;color:var(--ink2);padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:hover td{background:color-mix(in srgb,var(--ink) 4%,transparent)}
td.nm{font-weight:600}td.c{font-variant-numeric:tabular-nums}
td.good{background:var(--goodbg)}td.warn{background:var(--warnbg)}td.bad{background:var(--badbg)}td.na{color:var(--mut)}
.ic{display:inline-block;width:14px;text-align:center;font-weight:700}
td.good .ic,.tile.good .ic{color:var(--good)}td.warn .ic,.tile.warn .ic{color:#b07a00}td.bad .ic,.tile.bad .ic{color:var(--bad)}
.n{color:var(--mut);font-size:11px;margin-left:4px}
.bt{display:inline-block;font-size:11px;padding:0 6px;border-radius:8px;border:1px solid var(--line);color:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin-top:8px}
.err{color:var(--bad)}
"""


def make(score='out/score_all.json', summary='out/summary.json', review=None, out='out/scoreboard.html'):
    rows = load_json(score) if score and os.path.exists(score) else []
    summ = load_json(summary) if summary and os.path.exists(summary) else {}
    mets = load_json(review) if review and os.path.exists(review) else []
    ok = [r for r in rows if 'error' not in r]; errs = [r for r in rows if 'error' in r]
    H = [f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>변환 채점 현황</title><style>{CSS}</style></head><body><main>',
         f'<h1>공통 변환프로그램 채점 현황</h1><div class="mut">생성 {datetime.datetime.now():%Y-%m-%d %H:%M} · 가상 건물 {len(ok)}동'
         f'{f" · 오류 {len(errs)}동" if errs else ""} · 목표값은 잠정 · 칸에 마우스를 올리면 목표가 보입니다</div>',
         '<div class="legend"><span><b style="color:var(--good)">✓</b> 통과</span><span><b style="color:#b07a00">!</b> 주의</span><span><b style="color:var(--bad)">✗</b> 미달</span><span>· 해당 없음</span></div>']
    if 'ALL' in summ:
        H.append('<div class="tiles">' + tiles(summ['ALL']) + '</div>')
    if summ:
        H.append('<h2>건물 유형별</h2><div class="wrap"><table><thead><tr><th>유형</th><th>동</th>' + ''.join(f'<th>{h}</th>' for h in HEAD) +
                 ''.join(f'<th>{KO[t]}</th>' for t in ('exit', 'hose', 'panel', 'gas', 'ev')) + '</tr></thead><tbody>')
        for bt, r in summ.items():
            H.append(f"<tr><td class='nm'>{BT_KO.get(bt, bt)}</td><td>{r.get('n', '')}</td>" + ''.join(row_cells(r, True)) + ''.join(sym_cells(r.get('symbols'))) + '</tr>')
        H.append('</tbody></table></div>')
    if ok:
        H.append('<h2>건물별</h2><div class="wrap"><table><thead><tr><th>건물</th><th>유형</th>' + ''.join(f'<th>{h}</th>' for h in HEAD) +
                 ''.join(f'<th>{KO[t]}</th>' for t in ('exit', 'hose', 'panel', 'gas', 'ev')) + '<th>플래그</th></tr></thead><tbody>')
        for r in sorted(ok, key=lambda r: (r['btype'], r['name'])):
            H.append(f"<tr><td class='nm'>{html.escape(r['name'])}</td><td><span class='bt'>{BT_KO.get(r['btype'], r['btype'])}</span></td>" + ''.join(row_cells(r, False)) +
                     ''.join(sym_cells(r.get('symbols'))) + f"<td>{r.get('flags_total', '')}</td></tr>")
        H.append('</tbody></table></div>')
    for e in errs:
        H.append(f"<p class='err'>오류: {html.escape(e['name'])} — {html.escape(e['error'])}</p>")
    H.append('<h2>검수 기반 성능 (실제 건물 · 사람이 확정한 결과 대비)</h2><div class="wrap" style="padding:0">' + review_section(mets) + '</div>')
    H.append('<p class="mut" style="margin-top:20px">※ 실제 정보카드에서 나온 결과·라벨은 내부 PC 밖으로 반출 금지.</p></main></body></html>')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f: f.write('\n'.join(H))
    print(out, f'가상 {len(ok)}동', f'검수 {len(mets)}동')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='채점 화면 HTML 생성')
    ap.add_argument('--score', default=os.path.join('out', 'score_all.json')); ap.add_argument('--summary', default=os.path.join('out', 'summary.json'))
    ap.add_argument('--review', default=None, help='review_metrics.json (검수 기반 성능)'); ap.add_argument('-o', '--out', default=os.path.join('out', 'scoreboard.html'))
    a = ap.parse_args()
    make(a.score, a.summary, a.review, a.out)
