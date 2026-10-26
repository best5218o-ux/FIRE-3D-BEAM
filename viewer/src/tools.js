// 도구 — 거리 재기(두 점: 직선·보행 거리) · 실 면적(누른 실) · 화면 저장(PNG, 3D + 라벨) · 한 장 브리핑(인쇄용 HTML: 진입구·계단·소화전·위험물·확인사항·화재 가정 요약)
// 모두 이 파일 안에서 끝남(외부 요청 없음). 화면 저장은 WebGL 그림 위에 화면 라벨(HTML)을 같은 자리에 다시 그려 합침.
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { CATNAME } from './build3d.js';

export function initTools(ctx) {
  const { scene, V, SLAB, zOf, getSel, D, BIM, FL, floors, esc, $, canvas, stage, render, gridFor, dijkstra, distAt, stairs, elevs, equips, getPlan, getSimSummary, getCrew, ACT, getBrief } = ctx;
  const G = new THREE.Group(); G.userData.own = true; scene.add(G);
  let mode = null, A = null, live = null;
  const meas = [];                                                   // [{fid,a,b,straight,walk}]
  let areaSel = null;
  const lab = (html, cls, p) => { const el = document.createElement('div'); el.className = 'lb ' + cls; el.innerHTML = html; const o = new CSS2DObject(el); o.position.copy(p); G.add(o); return o; };
  const dot = (p, col) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.22, 12, 8), new THREE.MeshBasicMaterial({ color: col, depthTest: false })); m.renderOrder = 50; m.position.copy(p); G.add(m); return m; };
  const seg = (a, b, col, dashed) => {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const l = new THREE.Line(g, dashed ? new THREE.LineDashedMaterial({ color: col, dashSize: 0.5, gapSize: 0.3, depthTest: false }) : new THREE.LineBasicMaterial({ color: col, depthTest: false }));
    if (dashed) l.computeLineDistances(); l.renderOrder = 50; G.add(l); return l;
  };
  function setMode(m) {
    mode = mode === m ? null : m; A = null; if (live) { live.forEach((o) => G.remove(o)); live = null; }
    canvas.style.cursor = mode ? 'crosshair' : '';
    return mode;
  }
  // 실 찾기: 대피 분석 실(용도 추정 있음) → 없으면 BIM 실
  const inPoly = (x, y, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  function roomAt(fid, x, y) {
    const ev = D.evac && D.evac.floors && D.evac.floors[fid];
    const L = ev ? ev.rooms.map((r) => ({ poly: r.poly, area: r.area, cat: r.cat, name: '' })) : [];
    const sp = ((BIM.floors[fid] || {}).spaces || []).map((s) => ({ poly: s.poly, area: s.area, cat: null, name: s.name }));
    const hit = L.filter((r) => inPoly(x, y, r.poly)).sort((a, b) => a.area - b.area)[0] || sp.filter((r) => inPoly(x, y, r.poly)).sort((a, b) => a.area - b.area)[0];
    if (!hit) return null;
    const nm = sp.find((s) => s.name && inPoly(x, y, s.poly));
    return { ...hit, name: (nm && nm.name) || hit.name };
  }
  const shoelace = (p) => Math.abs(p.reduce((s, [x, y], i) => { const [x2, y2] = p[(i + 1) % p.length]; return s + x * y2 - x2 * y; }, 0) / 2);
  const perim = (p) => p.reduce((s, [x, y], i) => { const [x2, y2] = p[(i + 1) % p.length]; return s + Math.hypot(x2 - x, y2 - y); }, 0);

  // 바닥 클릭(도면 좌표) — main.js 가 넘겨줌. 처리했으면 true
  function click(fid, x, y) {
    if (mode === 'measure') {
      const p = V(x, y, zOf(fid) + SLAB + 0.15);
      if (!A) { A = { fid, x, y, p }; live = [dot(p, 0xfde047)]; return true; }
      const B = { x, y }, straight = Math.hypot(x - A.x, y - A.y);
      let walk = null;
      try { const gr = gridFor(fid), R = dijkstra(gr, [[A.x, A.y]]), d = distAt(gr, R, x, y); walk = isFinite(d) ? d : null; } catch (e) { walk = null; }
      if (live) live.forEach((o) => G.remove(o)); live = null;
      dot(A.p, 0xfde047); dot(p, 0xfde047); seg(A.p, p, 0xfde047, true);
      lab(`📏 직선 <b>${straight.toFixed(1)}m</b>${walk != null ? ` · 보행 <b>${walk.toFixed(1)}m</b>` : ''}`, 'meas', A.p.clone().lerp(p, 0.5).add(new THREE.Vector3(0, 0.8, 0)));
      meas.push({ fid, a: [A.x, A.y], b: [x, y], straight, walk }); A = null;
      return true;
    }
    if (mode === 'area') {
      const r = roomAt(fid, x, y); if (!r) return true;
      if (areaSel) areaSel.forEach((o) => G.remove(o));
      const z = zOf(fid) + SLAB + 0.16, sh = new THREE.Shape(r.poly.map(([a, b]) => new THREE.Vector2(a, b))), g = new THREE.ShapeGeometry(sh); g.rotateX(-Math.PI / 2); g.translate(0, z, 0);
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xfde047, transparent: true, opacity: 0.32, depthTest: false, side: THREE.DoubleSide })); m.renderOrder = 48; G.add(m);
      const ln = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(r.poly.map(([a, b]) => V(a, b, z + 0.02))), new THREE.LineBasicMaterial({ color: 0xfde047, depthTest: false })); ln.renderOrder = 49; G.add(ln);
      // 면적 = 실 분석 값(벽 사이 빈 칸을 0.2 m 격자로 센 순 면적 — 실 이름·패널·인명검색과 같은 값). 다각형은 그 칸들의 '외곽 칸 중심'을 잇고 줄인 것이라
      // 둘레 × 0.1 m 만큼 작게 나옴(실제 도면 실처럼 둘레가 복잡하면 4 % 넘게) → 다각형 면적은 분석 값이 없을 때만 씀
      const Ap = shoelace(r.poly), A2 = r.area > 0 ? r.area : Ap, c = r.poly.reduce((s, q) => [s[0] + q[0] / r.poly.length, s[1] + q[1] / r.poly.length], [0, 0]);
      const o = lab(`▦ ${r.name ? esc(r.name) + ' · ' : ''}<b>${A2.toFixed(1)}㎡</b> · 둘레 ${perim(r.poly).toFixed(1)}m${r.cat ? `<br><span class="k">${esc(CATNAME[r.cat] || r.cat)}(추정)</span>` : ''}`, 'meas', V(c[0], c[1], z + 1.2));
      areaSel = [m, ln, o]; areaSel.info = { fid, area: A2, stored: r.area, polyArea: Ap, perim: perim(r.poly), cat: r.cat, name: r.name };
      return true;
    }
    return false;
  }
  function hover(fid, x, y) {
    if (mode !== 'measure' || !A || A.fid !== fid) return;
    if (live) { live.slice(1).forEach((o) => G.remove(o)); live = [live[0]]; }
    const p = V(x, y, zOf(fid) + SLAB + 0.15);
    live.push(seg(A.p, p, 0xfde047, true), lab(`${Math.hypot(x - A.x, y - A.y).toFixed(1)}m`, 'meas', A.p.clone().lerp(p, 0.5).add(new THREE.Vector3(0, 0.6, 0))));
  }
  function clear() { G.traverse((o) => { if (o.isCSS2DObject && o.element.parentNode) o.element.parentNode.removeChild(o.element); }); G.clear(); meas.length = 0; areaSel = null; A = null; live = null; }

  // ── 화면 저장: WebGL + 라벨(HTML) 합성
  function snapshot(type = 'image/png', q) {
    render();
    const w = canvas.width, h = canvas.height, rc = canvas.getBoundingClientRect(), k = w / rc.width;
    const out = document.createElement('canvas'); out.width = w; out.height = h; const c = out.getContext('2d');
    const g = c.createRadialGradient(w * 0.5, h * 0.3, 0, w * 0.5, h * 0.3, Math.max(w, h) * 0.72); g.addColorStop(0, '#13243f'); g.addColorStop(1, '#0b1220');
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    try { c.drawImage(canvas, 0, 0); } catch (e) { /* GL 없음(2D 대체 보기) */ }
    const fb = document.getElementById('fallback'); if (fb && !fb.hidden) c.drawImage(fb, 0, 0, w, h);
    const els = document.querySelectorAll('#labels .lb, #labels .co .bx, #labels .co .lg, #labels .co .pt, #labels .ftag, #labels .north');
    for (const el of els) {
      const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
      const st = getComputedStyle(el); if (st.visibility === 'hidden' || st.display === 'none' || (el.closest('.culled'))) continue;
      let hidden = false; for (let p = el; p && p.id !== 'labels'; p = p.parentElement) if (p.style && p.style.display === 'none') hidden = true; if (hidden) continue;
      const x = (r.left - rc.left) * k, y = (r.top - rc.top) * k, ww = r.width * k, hh = r.height * k;
      if (x > w || y > h || x + ww < 0 || y + hh < 0) continue;
      if (el.classList.contains('lg')) { c.fillStyle = getComputedStyle(el.parentElement.querySelector('.pt')).backgroundColor; c.fillRect(x, y, Math.max(1, ww), hh); continue; }
      const bg = st.backgroundColor, bc = st.borderTopColor, br = parseFloat(st.borderTopLeftRadius) * k || 0, bw = parseFloat(st.borderTopWidth) * k || 0;
      c.beginPath(); if (c.roundRect) c.roundRect(x, y, ww, hh, Math.min(br, hh / 2)); else c.rect(x, y, ww, hh);
      if (bg && bg !== 'rgba(0, 0, 0, 0)') { c.fillStyle = bg; c.fill(); }
      if (bw > 0 && bc) { c.strokeStyle = bc; c.lineWidth = bw; c.stroke(); }
      const txt = el.innerText || el.textContent || ''; if (!txt.trim()) continue;
      c.fillStyle = st.color; c.font = `${st.fontWeight} ${parseFloat(st.fontSize) * k}px ${st.fontFamily}`; c.textBaseline = 'middle'; c.textAlign = 'center';
      const lines = txt.split('\n'), lh = hh / lines.length;
      lines.forEach((ln, i) => c.fillText(ln, x + ww / 2, y + lh * (i + 0.5), ww + 4));
    }
    // 머리: 건물명 · 시각 · 상태
    const B = D.building, now = new Date(), ts = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    c.fillStyle = 'rgba(10,20,38,0.82)'; c.fillRect(0, h - 26 * k, w, 26 * k); c.fillStyle = '#e6edf7'; c.font = `600 ${12.5 * k}px system-ui,sans-serif`; c.textAlign = 'left'; c.textBaseline = 'middle';
    c.fillText(`${B.name} · ${getSel() ? FL[getSel()].name : ''} · ${ts} · ${document.getElementById('bstat').textContent} · AI 건축물 지휘정보 3D`, 10 * k, h - 13 * k);
    return out.toDataURL(type, q);
  }
  function save(name, url) { const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { a.remove(); if (url.startsWith('blob:')) URL.revokeObjectURL(url); }, 500); }
  const stamp = () => { const n = new Date(); return `${n.getFullYear()}${String(n.getMonth() + 1).padStart(2, '0')}${String(n.getDate()).padStart(2, '0')}_${String(n.getHours()).padStart(2, '0')}${String(n.getMinutes()).padStart(2, '0')}`; };
  const fname = (s) => String(s).replace(/[\\/:*?"<>|\s]+/g, '_');
  function savePng() { save(`${fname(D.building.name)}_3D_${stamp()}.png`, snapshot()); }

  // ── 한 장 브리핑 / A4 인쇄(Ctrl+P): 같은 내용(sheet) — 내려받는 HTML 은 문서 하나, 화면 인쇄는 #printSheet 에 넣음
  const SHEET_CSS = `.ps{max-width:192mm;margin:0 auto;font:11px/1.42 "Malgun Gothic","맑은 고딕","Apple SD Gothic Neo","Noto Sans KR",sans-serif;color:#0f172a;background:#fff}
.ps *{box-sizing:border-box}.ps h1{font-size:18px;margin:0}.ps h2{font-size:12.5px;margin:7px 0 3px;padding-bottom:2px;border-bottom:2px solid #0f172a;break-after:avoid}
.ps .sub{color:#475569;font-size:10.5px}.ps .top{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid #b91c1c;padding-bottom:4px}
.ps .st{font-size:10px;border:1px solid #94a3b8;border-radius:10px;padding:1px 8px;color:#334155}
.ps .shot{display:block;width:100%;max-height:78mm;object-fit:contain;margin-top:5px}
.ps .b5{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-top:5px}.ps .b5 div{border:1.5px solid #cbd5e1;border-radius:5px;padding:3px 5px;break-inside:avoid}
.ps .b5 div.bad{border-color:#b91c1c;background:#fef2f2}.ps .b5 div.warn{border-color:#d97706;background:#fffbeb}.ps .b5 small{display:block;font-size:9px;color:#475569;font-weight:700}.ps .b5 b{display:block;font-size:11px}.ps .b5 i{display:block;font-style:normal;font-size:9.5px;color:#334155}
.ps .cols{display:grid;grid-template-columns:1fr 1fr;gap:0 12px}.ps table{width:100%;border-collapse:collapse}.ps td{border-bottom:1px solid #e2e8f0;padding:1.5px 3px;vertical-align:top;text-align:left}.ps td:first-child{color:#475569;white-space:nowrap;width:30%}
.ps tr{break-inside:avoid}.ps .hz{background:#fef3c7;border:1px solid #f59e0b;border-radius:4px;padding:3px 6px;margin:2px 0}.ps .is{margin:1px 0;break-inside:avoid}.ps .is.high b{color:#b91c1c}.ps .is.mid b{color:#b45309}.ps .am{color:#b45309}
.ps .ft{margin-top:6px;color:#64748b;font-size:9px;border-top:1px solid #cbd5e1;padding-top:3px}`;
  function sheet() {
    const B = D.building, C = B.card || {}, L = C.소재지 || {}, plan = getPlan(), S = getSimSummary(), crew = getCrew();
    const img = snapshot('image/jpeg', 0.84);
    const ents = []; for (const f of [...floors].reverse()) for (const d of (BIM.floors[f.id] || {}).doors || []) if (d[0] === 'entrance') ents.push([f, d]);
    const hy = floors.map((f) => [f, equips.filter((q) => q.fid === f.id && q.type === 'hose').length, Number((f.fac || {})['옥내소화전']) || 0]).filter(([, a, b]) => a || b);
    const issues = [...D.issues].sort((a, b) => ({ high: 0, mid: 1, low: 2 }[a.lv] ?? 3) - ({ high: 0, mid: 1, low: 2 }[b.lv] ?? 3)).slice(0, 8);
    const rng = (a) => (a && a.length ? `${a[0]}~${a[a.length - 1]}` : '-');
    const sp = (s) => (s.special ? (s.special.kind === 'sp' ? ' <b class="am">특별피난계단' : ' <b class="am">비상용 승강기') + (s.special.cert === 'cand' ? '(후보)' : '') + '</b>' : '');
    const hz = C.위험물 || [];
    const b5 = getBrief ? getBrief() : [];
    let fire = '';
    if (plan) {
      fire += `<h2>🔥 화재 가정 — ${esc(FL[plan.fid].name)}</h2><table>`;
      fire += `<tr><td>진입구</td><td>${plan.entry ? esc(plan.entry.name) : '-'}</td></tr><tr><td>공격(진입)계단</td><td>${esc(plan.attackName || '없음(1층 화재 또는 계단 미검출)')}</td></tr>`;
      fire += `<tr><td>진입 예상</td><td>${(plan.tIn / 60).toFixed(1)}분 · 수평 ${plan.routeLen.toFixed(0)}m · 수직 ${plan.floorsUp}개 층</td></tr>`;
      if (plan.hoseN != null) fire += `<tr><td>호스</td><td>가장 가까운 소화전 ${plan.hoseL.toFixed(0)}m → ${plan.hoseN}본(${ACT.hose_len_m}m 기준)</td></tr>`;
      if (plan.scba) fire += `<tr><td>공기호흡기</td><td>${plan.scba.pt ? '왕복 불가 — 귀환 한계점이 화점 앞(중계·급기 필요)' : `화점 활동 가능 ${plan.scba.act.toFixed(1)}분`}</td></tr>`;
      if (plan.search && plan.search.length) fire += `<tr><td>인명검색 순서</td><td>${plan.search.slice(0, 5).map((z) => `${z.rank}. ${esc(FL[z.fid].name)} ${z.cat ? esc(CATNAME[z.cat] || '') : ''}`).join(' · ')}</td></tr>`;
      if (S) {
        fire += `<tr><td>확산 가정(예시 모델)</td><td>공격계단 연기 곤란 ${S.atkTTU != null ? S.fmt(S.atkTTU) : '-'} · 직상층 첫 곤란 ${S.fmt(S.firstUp)} · 화재층 대피로 모두 차단 ${S.fmt(S.allCut)}</td></tr>`;
        fire += `<tr><td>${S.fmt(S.t)} 시점</td><td>곤란 실 ${S.nHit} · 불 번진 실 ${S.nBurn} · 화재층 대피 가능 ${S.ev.nOk} / 고립 ${S.ev.nCut} / 연기 ${S.ev.nSmoke} · 영향 층 ${S.floorsHit.map((f) => esc(f)).join('·') || '-'}</td></tr>`;
      }
      if (crew && !crew.none) fire += `<tr><td>대원 도착(예시)</td><td>${crew.km != null ? `관할센터 ${crew.km}km → ` : ''}현장 진입 ${(crew.arrive / 60).toFixed(1)}분 · 화재층 도달 +${(crew.tFloor / 60).toFixed(1)}분 · 화점 +${(crew.tEnd / 60).toFixed(1)}분</td></tr>`;
      fire += '</table>';
    }
    const now = new Date();
    return `<div class="ps">
<div class="top"><div><h1>${esc(B.name)} — 출동 지휘 요약</h1><div class="sub">${esc([L.도로명 || L.지번, B.use, B.floors, B.gfa ? `연면적 ${Number(B.gfa).toLocaleString()}㎡` : '', C.관리번호 ? `관리번호 ${C.관리번호}` : ''].filter(Boolean).join(' · '))}</div></div>
<div style="text-align:right"><span class="st">${esc(document.getElementById('bstat').textContent)}</span><div class="sub">${now.toLocaleString('ko-KR')}</div></div></div>
${b5.length ? `<div class="b5">${b5.map((z) => `<div class="${esc(z.cls)}"><small>${esc(z.k)}</small><b>${esc(z.t)}</b><i>${esc(z.s)}</i></div>`).join('')}</div>` : ''}
<img class="shot" src="${img}" alt="3D 화면">
<div class="cols"><div>
<h2>진입구</h2><table>${ents.length ? ents.map(([f, d]) => `<tr><td>${esc(f.name)}</td><td>${esc(String(d[5] || '진입구').replace(/ · AI 모델 보충/, ''))}</td></tr>`).join('') : '<tr><td colspan="2">도면에서 진입구를 찾지 못함 — 진압작전도·현장 확인</td></tr>'}</table>
<h2>계단 · 승강기</h2><table>${stairs.map((s) => `<tr><td>${esc(s.name)}</td><td>${rng(s.floors)}${sp(s)}</td></tr>`).join('')}${elevs.map((e) => `<tr><td>${esc(e.name)}</td><td>${rng(e.floors)}${sp(e)}</td></tr>`).join('')}</table>
<h2>옥내소화전 (도면 표기 / 정보카드)</h2><table>${hy.map(([f, a, b]) => `<tr><td>${esc(f.name)}</td><td>${a} / ${b}</td></tr>`).join('') || '<tr><td colspan="2">-</td></tr>'}</table>
${equips.some((q) => q.type === 'panel') ? `<h2>수신기</h2><table>${equips.filter((q) => q.type === 'panel').map((q) => `<tr><td>${esc(FL[q.fid].name)}</td><td>${esc(q.name)}</td></tr>`).join('')}</table>` : ''}
</div><div>
<h2>위험물</h2>${hz.length ? hz.map((x) => `<div class="hz"><b>⚠ ${esc(x.구분)}</b> ${esc([x.시설명, x.품명, x.수량].filter(Boolean).join(' · '))}</div>`).join('') : '<div class="sub">정보카드 기재 없음</div>'}
<h2>확인사항 (상위 ${issues.length}건 / 전체 ${D.issues.length}건)</h2>${issues.map((i) => `<div class="is ${esc(i.lv)}"><b>${i.lv === 'high' ? '■' : '□'} ${esc(i.t)}</b> — ${esc(i.d)}</div>`).join('')}
${fire}
</div></div>
<div class="ft">정보카드·평면도 자동 변환 결과(검수 전이면 오차 가능). 경로·시간·확산은 기준값(가정값)으로 계산한 추정이며 현장 확인이 우선입니다. 화재 확산은 단순 구획 확산 예시 모델(연기 지수·문 전도도 등 임시값)입니다.</div>
</div>`;
  }
  function briefingHtml() {
    return `<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>${esc(D.building.name)} 출동 브리핑</title><style>
@page{size:A4;margin:9mm}body{margin:0;background:#fff}.wrap{padding:6mm 0}@media print{.wrap{padding:0}}
${SHEET_CSS}
</style></head><body><div class="wrap">${sheet()}</div></body></html>`;
  }
  // 화면 인쇄(Ctrl+P)용: 인쇄 때만 적용되는 스타일 + 같은 내용
  function printSheet() { return `<style media="print">@page{size:A4;margin:9mm}${SHEET_CSS}</style>${sheet()}`; }
  function saveBriefing() { const h = briefingHtml(); save(`${fname(D.building.name)}_출동브리핑_${stamp()}.html`, URL.createObjectURL(new Blob([h], { type: 'text/html' }))); return h.length; }
  return { setMode, mode: () => mode, click, hover, clear, snapshot, savePng, briefingHtml, saveBriefing, printSheet, measures: () => meas.slice(), area: () => (areaSel ? areaSel.info : null) };
}
