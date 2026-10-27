'use strict';
// 소방활동 정보카드 자동 변환 결과 검수 화면 (오프라인, 서버 없음). 데이터: 전역 D (review/make.py)
(function () {
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const TYPES = ['exit', 'hose', 'panel', 'gas', 'ev'];
const TCOL = { exit: '#3fb950', hose: '#ff4d4f', panel: '#a371f7', gas: '#f0883e', ev: '#58a6ff' };
const TABB = { exit: '비', hose: '소', panel: '수', gas: '가', ev: 'EV' };
const KO = D.types;
const PLANS = {}; D.plans.forEach((p) => { PLANS[p.i] = p; });
const ORDER = D.plans.map((p) => p.i);
const ITEM = {}; D.items.forEach((it) => { ITEM[it.id] = it; });
const BTN = {
  symbol: ['맞음', '틀림(삭제)'], flag: ['불일치 맞음', '문제 없음'], align: ['정합 정상', '정합 불량'], scale: ['축척 정상', '축척 불량'],
  missing: ['확인함', '해당 없음'], doors: ['확인함', '나중에'], elev_count: ['불일치 맞음', '문제 없음'],
};
// 교차검증(규칙 ↔ AI 모델) 항목 판정 버튼: [모델이 맞음, 규칙이 맞음, 나중에]
const XK = { xwall: 1, xdoor: 1 };
function xButtons(it) {
  if (it.kind === 'xdoor') return it.sub === 'added' ? ['모델이 맞음(문 유지)', '규칙이 맞음(문 삭제)', '나중에'] : ['모델이 맞음(문 추가)', '규칙이 맞음(문 아님)', '나중에'];
  if (it.sub === 'rule_only') return ['모델이 맞음(벽 삭제)', '규칙이 맞음(유지)', '나중에'];
  if (it.filled) return ['모델이 맞음(보충 벽 유지)', '규칙이 맞음(보충 벽 삭제)', '나중에'];
  if (it.bridge) return ['모델이 맞음(벽 추가·개구부 막음)', '규칙이 맞음(개구부 유지)', '나중에'];
  return ['모델이 맞음(벽 추가)', '규칙이 맞음(벽 아님)', '나중에'];
}
// 판정이 도면에 주는 효과: remove(자동 결과에서 삭제) · add(추가) · keep(자동 결과 유지·확정) · later
function xEffect(it, d) {
  if (!d || d === 'later') return d || null;
  if (it.kind === 'xdoor') return it.sub === 'added' ? (d === 'model' ? 'keep' : 'remove') : (d === 'model' ? 'add' : 'none');
  if (it.sub === 'rule_only') return d === 'model' ? 'remove' : 'keep';
  if (it.filled) return d === 'model' ? 'keep' : 'remove';
  return d === 'model' ? 'add' : 'none';
}
const VKO = { stairs: '계단', elevs: '승강기' };
const LSKEY = 'fire3dbeam-review:' + D.building.id + ':' + D.building.result_sha;
const clone = (o) => JSON.parse(JSON.stringify(o));
const now = () => new Date().toISOString().slice(0, 19);

// ───────── 상태 ─────────
function initialState() {
  const S = { plans: {}, flags: {}, vertical: {}, items: {}, xdec: {}, vadd: [], seq: 0, ops: [] };
  D.plans.forEach((p) => {
    S.plans[p.i] = { symbols: clone(p.symbols), doors: clone(p.doors), walls: clone(p.walls), status: { align: null, scale: null, done: false }, shift: [0, 0] };
  });
  return S;
}
let S = initialState();
const U = { plan: ORDER[0], mode: 'select', addType: 'exit', vType: 'stairs', vdrag: null, sel: null, ly: { img: 1, walls: 1, doors: 1, syms: 1, vert: 1, ghost: 0 }, views: {}, cur: null, hiType: null, wallStart: null, onlyPlan: false };
const undo = [], redo = [];
function snap() { return JSON.stringify(S); }
function commit(before, op) {
  undo.push(before); if (undo.length > 300) undo.shift(); redo.length = 0;
  if (op) S.ops.push(Object.assign({ t: now(), plan: U.plan }, op));
  autosave(); renderAll();
}
function doUndo() { if (!undo.length) return; redo.push(snap()); S = JSON.parse(undo.pop()); U.sel = null; autosave(); renderAll(); }
function doRedo() { if (!redo.length) return; undo.push(snap()); S = JSON.parse(redo.pop()); U.sel = null; autosave(); renderAll(); }
function autosave() { try { localStorage.setItem(LSKEY, JSON.stringify({ t: now(), S })); } catch (e) { /* 저장소 없음: 무시 */ } }

// ───────── 항목(검수 필요) 판정 ─────────
function itemState(it) {
  if (XK[it.kind]) return (S.xdec || {})[it.id] || null;
  const d = S.items[it.id]; if (d) return d;
  if (it.kind === 'flag') return S.flags[it.ref] ? (S.flags[it.ref] === 'confirm' ? 'yes' : 'no') : null;
  if (it.kind === 'symbol') {
    const q = findSym(it.plan, it.ref);
    if (!q) return 'no';
    if (q.ver) return 'yes';
    if (q.src && q.src !== 'auto') return 'fixed';
    return null;
  }
  if (it.kind === 'align') { const a = S.plans[it.plan].status.align; return a ? (a === 'ok' ? 'yes' : 'no') : null; }
  if (it.kind === 'scale') { const a = S.plans[it.plan].status.scale; return a ? (a === 'ok' ? 'yes' : 'no') : null; }
  return null;
}
function findSym(pi, id) { return S.plans[pi].symbols.find((q) => q.id === id); }
function decide(it, yes) {
  const b = snap();
  if (it.kind === 'symbol') {
    const P = S.plans[it.plan]; const q = findSym(it.plan, it.ref);
    if (yes && q) q.ver = true;
    if (!yes && q) { P.symbols = P.symbols.filter((x) => x !== q); if (U.sel && U.sel.id === q.id) U.sel = null; }
    S.items[it.id] = yes ? 'yes' : 'no';
  } else if (it.kind === 'flag') { S.flags[it.ref] = yes ? 'confirm' : 'reject'; }
  else if (XK[it.kind]) { S.xdec[it.id] = yes ? 'model' : 'rule'; }
  else if (it.kind === 'align') { S.plans[it.plan].status.align = yes ? 'ok' : 'bad'; }
  else if (it.kind === 'scale') { S.plans[it.plan].status.scale = yes ? 'ok' : 'bad'; }
  else S.items[it.id] = yes ? 'yes' : 'no';
  commit(b, { op: 'decide', item: it.id, kind: it.kind, ref: it.ref, decision: XK[it.kind] ? S.xdec[it.id] : (yes ? 'yes' : 'no') });
}
function decideX(it, d) {           // 교차검증 항목: 'model' | 'rule' | 'later'
  if (!it || !XK[it.kind]) return;
  const b = snap(); S.xdec[it.id] = d; commit(b, { op: 'decide', item: it.id, kind: it.kind, sub: it.sub, decision: d });
}
function visibleItems() {
  const L = D.items.filter((it) => !U.onlyPlan || it.plan === U.plan || it.plan == null);
  const open = L.filter((it) => !itemState(it)), done = L.filter((it) => itemState(it));
  return open.concat(done);
}

// ───────── SVG 뷰 ─────────
const svg = $('svg');
function view() { const p = PLANS[U.plan]; if (!U.views[U.plan]) U.views[U.plan] = fitView(p); return U.views[U.plan]; }
function fitView(p) { const m = 0.04 * Math.max(p.w, p.h); return { x: -m, y: -m, w: p.w + 2 * m, h: p.h + 2 * m }; }
function applyView() { const v = view(); svg.setAttribute('viewBox', `${v.x} ${v.y} ${v.w} ${v.h}`); }
function toImg(ev) { const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY; const m = svg.getScreenCTM(); return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 }; }
function zoomAt(f, cx, cy) {
  const v = view(); if (cx == null) { cx = v.x + v.w / 2; cy = v.y + v.h / 2; }
  const nw = Math.min(Math.max(v.w * f, 20), 20000);
  const k = nw / v.w; v.x = cx - (cx - v.x) * k; v.y = cy - (cy - v.y) * k; v.w = nw; v.h *= k; applyView(); hud();
}
function focusAt(x, y, span) { const v = view(); const r = v.h / v.w; v.w = span; v.h = span * r; v.x = x - v.w / 2; v.y = y - v.h / 2; applyView(); }
function pxPerUnit() { const r = svg.getBoundingClientRect(); const v = view(); return Math.min(r.width / v.w, r.height / v.h) || 1; }

function symBox(q) { return q.box || [q.px - 10, q.py - 10, q.px + 10, q.py + 10]; }
function render() {
  const p = PLANS[U.plan], P = S.plans[U.plan], s = p.s;
  const k = 1 / pxPerUnit();            // 화면 1px 이 도면 몇 px 인지
  const fs = 12 * k;
  svg.style.setProperty('--tsw', (2.5 * k) + 'px');
  const suspect = new Set(D.items.filter((it) => it.kind === 'symbol' && it.plan === U.plan && !itemState(it)).map((it) => it.ref));
  const sel = U.sel || {};
  let h = '';
  if (U.ly.img) h += `<image href="${p.img}" x="0" y="0" width="${p.w}" height="${p.h}" preserveAspectRatio="none" opacity="0.9"/>`;
  else h += `<rect x="0" y="0" width="${p.w}" height="${p.h}" fill="#fff" opacity="0.06"/>`;
  if (U.ly.ghost && p.ghost.length) {
    const dx = -P.shift[0] * s, dy = P.shift[1] * s;
    h += `<g transform="translate(${dx} ${dy})" opacity="0.85">` + p.ghost.map((w) => wline(w, '#ff3bd4', 2.2, '')).join('') + '</g>';
  }
  if (U.ly.vert) {
    for (const [kind, col] of [['stairs', '#3fb950'], ['elevs', '#58a6ff']]) {
      D[kind].forEach((V) => {
        const b = V.boxes[U.plan]; if (!b) return;
        const dec = S.vertical[kind + ':' + V.k]; const isSel = sel.kind === kind && sel.id === V.k;
        h += `<rect data-k="${kind}" data-id="${V.k}" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" fill="${col}" fill-opacity="${dec === 'reject' ? 0.02 : 0.12}" stroke="${isSel ? '#fff' : (dec === 'reject' ? '#888' : col)}" stroke-width="${isSel ? 3 : 2}" stroke-dasharray="${dec === 'reject' ? '2 4' : '8 5'}" vector-effect="non-scaling-stroke" style="cursor:pointer"/>`;
        h += `<text x="${b[0] + 2 * k}" y="${b[3] - 3 * k}" font-size="${fs * 0.85}" fill="${dec === 'reject' ? '#888' : col}">${esc(kind === 'stairs' ? V.name : 'EV' + V.name.slice(-1))}${dec === 'reject' ? ' ✗' : ''}</text>`;
      });
    }
  }
  // 교차검증(규칙 ↔ AI 모델) 불일치 조각: 규칙만=자홍 점선, 모델만=청록 점선, 모델 보충 문=주황 점선 — 현재 항목은 굵게
  // 검수자가 추가한 계단·승강기(실선 상자) — 그린 도면이 아니어도 연결층이 겹치면 전역 좌표로 옮겨 표시
  if (U.ly.vert) {
    const pf = new Set(p.floors.map((f) => f.id));
    (S.vadd || []).forEach((V) => {
      if (V.plan !== U.plan && !V.floors.some((f) => pf.has(f))) return;
      const b = vboxOn(V, U.plan); if (!b) return;
      const col = V.kind === 'stairs' ? '#3fb950' : '#58a6ff'; const isSel = sel.kind === 'vadd' && sel.id === V.id;
      h += `<rect data-k="vadd" data-id="${V.id}" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" fill="${col}" fill-opacity="0.22" stroke="${isSel ? '#fff' : col}" stroke-width="${isSel ? 3.5 : 2.5}" vector-effect="non-scaling-stroke" style="cursor:pointer"/>`;
      h += `<text x="${b[0] + 2 * k}" y="${b[1] + fs}" font-size="${fs * 0.9}" fill="${col}">${esc(V.name)} ${esc(V.floors.length ? V.floors[0] + (V.floors.length > 1 ? '~' + V.floors[V.floors.length - 1] : '') : '')}</text>`;
    });
  }
  if (U.mode === 'vert' && U.vdrag) {
    const v = U.vdrag, col = U.vType === 'stairs' ? '#3fb950' : '#58a6ff';
    h += `<rect id="vtmp" x="${Math.min(v.x0, v.x1)}" y="${Math.min(v.y0, v.y1)}" width="${Math.abs(v.x1 - v.x0)}" height="${Math.abs(v.y1 - v.y0)}" fill="${col}" fill-opacity="0.15" stroke="${col}" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>`;
  }
  let xs = '';
  //   판정 후: 삭제=빨강 가는 점선+✗, 추가=실선(벽 청록·문 주황), 유지(확정)=초록 가는 실선, 나중에=흐리게
  D.items.filter((it) => it.seg && it.plan === U.plan).forEach((it) => {
    const [x1, y1, x2, y2] = it.seg; const cur = it.id === U.cur; const st = itemState(it); const ef = XK[it.kind] ? xEffect(it, st) : (st ? 'done' : null);
    if (ef === 'done' || ef === 'none') { if (!cur) return; }
    const base = it.kind === 'xdoor' ? '#ff9f1c' : (it.sub === 'rule_only' ? '#ff3bd4' : '#00e5ff');
    let col = base, w = cur ? 9 : 5, dash = '6 4', op = cur ? 1 : 0.7;
    if (ef === 'remove') { col = '#ff4d4f'; w = cur ? 6 : 3; dash = '3 5'; op = 1; }
    else if (ef === 'add') { col = it.kind === 'xdoor' ? '#f0883e' : '#39c5cf'; w = cur ? 9 : 6; dash = ''; op = 1; }
    else if (ef === 'keep') { col = '#3fb950'; w = cur ? 6 : 3; dash = ''; op = 0.9; }
    else if (ef === 'later') { op = cur ? 0.8 : 0.35; }
    xs += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linecap="round" vector-effect="non-scaling-stroke" opacity="${op}" pointer-events="none"/>`;
    if (ef === 'remove') xs += `<text x="${(x1 + x2) / 2 + 3 * k}" y="${(y1 + y2) / 2 - 3 * k}" font-size="${fs * 1.1}" fill="#ff4d4f">✗</text>`;
    else if (ef === 'add') xs += `<text x="${(x1 + x2) / 2 + 3 * k}" y="${(y1 + y2) / 2 - 3 * k}" font-size="${fs}" fill="${col}">+</text>`;
  });
  if (U.ly.walls) h += P.walls.map((w) => wline(w, sel.kind === 'wall' && sel.id === w.id ? '#ffd33d' : (w.src === 'added' ? '#39c5cf' : '#2f81f7'), null, `data-k="wall" data-id="${w.id}"`, sel.kind === 'wall' && sel.id === w.id)).join('');
  if (U.ly.doors) {
    P.doors.forEach((d) => {
      const [x1, y1, x2, y2] = d.o === 'h' ? [d.px - d.w / 2, d.py, d.px + d.w / 2, d.py] : [d.px, d.py - d.w / 2, d.px, d.py + d.w / 2];
      const isSel = sel.kind === 'door' && sel.id === d.id; const col = isSel ? '#ffd33d' : (d.ext ? '#39c5cf' : '#f0883e');
      h += `<line data-k="door" data-id="${d.id}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${isSel ? 9 : 6}" stroke-linecap="round" vector-effect="non-scaling-stroke" style="cursor:move" opacity="0.9"/>`;
      h += `<circle data-k="door" data-id="${d.id}" cx="${d.px}" cy="${d.py}" r="${4 * k}" fill="${col}" stroke="#000" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
    });
  }
  h += xs;                               // 교차검증 조각은 벽·문 위에(판정 결과가 보이도록)
  if (U.ly.syms) {
    P.symbols.forEach((q) => {
      const b = symBox(q); const col = TCOL[q.type] || '#fff';
      const isSel = sel.kind === 'sym' && sel.id === q.id; const dim = U.hiType && U.hiType !== q.type;
      const pad = 4 * k;
      if (suspect.has(q.id)) h += `<rect x="${b[0] - pad * 2}" y="${b[1] - pad * 2}" width="${b[2] - b[0] + pad * 4}" height="${b[3] - b[1] + pad * 4}" rx="${pad}" fill="none" stroke="#ffd33d" stroke-width="2.5" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>`;
      h += `<rect data-k="sym" data-id="${q.id}" x="${b[0] - pad}" y="${b[1] - pad}" width="${b[2] - b[0] + 2 * pad}" height="${b[3] - b[1] + 2 * pad}" rx="${pad / 2}" fill="${col}" fill-opacity="${isSel ? 0.35 : 0.14}" stroke="${isSel ? '#fff' : col}" stroke-width="${isSel ? 3 : 2}" vector-effect="non-scaling-stroke" style="cursor:move" opacity="${dim ? 0.25 : 1}"/>`;
      const lab = TABB[q.type] + (q.src === 'added' ? '+' : (q.src && q.src !== 'auto' ? '*' : '')) + (q.ver ? '✓' : '');
      h += `<text x="${b[2] + pad}" y="${b[1] + fs * 0.2}" font-size="${fs * 0.9}" fill="${col}" opacity="${dim ? 0.3 : 1}">${esc(lab)}${q.conf != null && !q.ver ? `<tspan fill="#c9d1d9" font-size="${fs * 0.7}"> ${Math.round(q.conf * 100)}</tspan>` : ''}</text>`;
    });
  }
  if (U.mode === 'wall' && U.wallStart) h += `<circle cx="${U.wallStart.x}" cy="${U.wallStart.y}" r="${5 * k}" fill="#39c5cf"/><line id="wtmp" x1="${U.wallStart.x}" y1="${U.wallStart.y}" x2="${U.wallStart.x}" y2="${U.wallStart.y}" stroke="#39c5cf" stroke-width="3" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>`;
  svg.innerHTML = h;
  applyView();
}
function vboxOn(V, pi) {                // 추가한 계단·승강기 상자(그린 도면 px) → 도면 pi 의 px
  if (V.plan === pi) return V.box_px;
  const a = PLANS[V.plan], q = PLANS[pi]; if (!a || !q || a.tx == null || q.tx == null) return null;
  const g = [V.box_px[0] / a.s + a.tx, a.ty - V.box_px[1] / a.s, V.box_px[2] / a.s + a.tx, a.ty - V.box_px[3] / a.s];
  const x0 = (g[0] - q.tx) * q.s, x1 = (g[2] - q.tx) * q.s, y0 = (q.ty - g[1]) * q.s, y1 = (q.ty - g[3]) * q.s;
  return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
}
function wline(w, col, width, attr, isSel) {
  const [x1, y1, x2, y2] = w.o === 'h' ? [w.a, w.c, w.b, w.c] : [w.c, w.a, w.c, w.b];
  const sw = width ? `stroke-width="${width}" vector-effect="non-scaling-stroke"` : `stroke-width="${Math.max(w.t, 2)}"`;
  return `<line ${attr} x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" ${sw} stroke-opacity="${isSel ? 1 : 0.6}" stroke-linecap="square" ${attr ? 'style="cursor:pointer"' : ''}/>`;
}

// ───────── 패널 ─────────
function renderPlans() {
  $('plans').innerHTML = D.plans.map((p) => {
    const st = S.plans[p.i].status; const open = D.items.filter((it) => it.plan === p.i && !itemState(it)).length;
    return `<div class="pl ${p.i === U.plan ? 'sel' : ''}" data-plan="${p.i}"><span class="nm">${esc(p.label)}</span><span class="cp" title="${esc(p.caption)}">${esc(p.caption)}</span>` +
      (open ? `<span class="chip r" title="미처리 검수 항목">${open}</span>` : '') +
      (st.align === 'bad' ? '<span class="chip r">정합✗</span>' : '') + (st.done ? '<span class="chip g">완료</span>' : '') + '</div>';
  }).join('');
  const p = PLANS[U.plan], st = S.plans[U.plan].status, sh = S.plans[U.plan].shift;
  const fl = p.floors.map((f) => `<tr><td>${esc(f.name)}</td><td class="k">${esc(f.use)}</td><td style="text-align:right">${f.card_area ? Math.round(f.card_area).toLocaleString() + '㎡' : '-'}</td></tr>`).join('');
  $('pstat').innerHTML = `<h3>이 층 — ${esc(p.label)}</h3>
    <div class="k" style="font-size:12px">정합 신뢰도 <b style="color:#e6edf3">${esc(p.align || '-')}</b> · 축척 근거 <b style="color:#e6edf3"${p.scale_src ? ` title="${esc(p.scale_src)}"` : ''}>${esc(p.scale_basis ? p.scale_basis.replace(/^축척 근거: /, '') : ({ dims: '치수 글자', grid: '다른 층 그리드', area: '정보카드 면적', default: '기본값' }[p.scale_src] || p.scale_src || '-'))}</b></div>
    <table style="width:100%;font-size:12px;margin:4px 0">${fl}</table>
    <div class="row" style="margin:6px 0"><span class="k">층정합</span>
      <button class="ok ${st.align === 'ok' ? 'on' : ''}" id="aOk">정상 <kbd>O</kbd></button><button class="bad ${st.align === 'bad' ? 'on' : ''}" id="aBad">불량 <kbd>X</kbd></button></div>
    <div class="row" style="margin:6px 0"><span class="k">위치 보정</span><span>Δx ${sh[0].toFixed(1)}m Δy ${sh[1].toFixed(1)}m</span>
      <button id="sh0" title="보정 초기화">0</button></div>
    <div class="k" style="font-size:11px">Ctrl+화살표 0.1m · 아래층 겹쳐보기(G)${p.ghost_of ? ' = ' + esc(p.ghost_of) : ''}</div>
    <div class="row" style="margin-top:8px"><button class="ok ${st.done ? 'on' : ''}" id="pDone" style="flex:1">${st.done ? '✓ 이 층 검수 완료' : '이 층 검수 완료로 표시'} <kbd>Enter</kbd></button></div>`;
  $('aOk').onclick = () => setAlign('ok'); $('aBad').onclick = () => setAlign('bad');
  $('pDone').onclick = toggleDone; $('sh0').onclick = () => { const b = snap(); S.plans[U.plan].shift = [0, 0]; commit(b, { op: 'shift_reset' }); };
  $('flags').innerHTML = D.flags.length ? D.flags.map((f) => {
    const d = S.flags[f.key];
    return `<div class="flg ${f.lv}"${f.detail ? ` title="${esc('기술 정보: ' + f.detail)}"` : ''}><b>${esc(f.t)}</b> ${d ? `<span class="chip ${d === 'confirm' ? 'y' : 'g'}">${d === 'confirm' ? '불일치 확인' : '문제 없음'}</span>` : '<span class="chip r">미확인</span>'}<div class="k">${esc(f.d)}</div></div>`;
  }).join('') : '<div class="k">불일치 없음</div>';
}
function renderItems() {
  const L = visibleItems();
  $('items').innerHTML = L.map((it) => {
    const st = itemState(it); const b = XK[it.kind] ? xButtons(it) : (BTN[it.kind] || ['확인함', '무시']);
    const pc = it.p >= 0.7 ? 3 : (it.p >= 0.45 ? 2 : 1);
    const where = it.plan != null ? PLANS[it.plan].label : '건물 전체';
    const stTxt = st ? { yes: '✓ ' + b[0], no: '✗ ' + b[1], fixed: '✎ 수정함', model: '✓ ' + b[0], rule: '✓ ' + b[1], later: '… 나중에' }[st] : '';
    return `<div class="it p${pc} ${st ? 'done' : ''} ${U.cur === it.id ? 'cur' : ''}" data-it="${it.id}">
      <div class="h"><span class="chip">${esc(where)}</span><b>${esc(it.t)}</b><span class="pc" title="오류 가능성(의심도)">${Math.round(it.p * 100)}%</span></div>
      <div class="d">${esc(it.d)}</div>
      <div class="a">${st ? `<span class="chip ${st === 'no' ? 'r' : (st === 'model' ? 'y' : (st === 'later' ? '' : 'g'))}">${esc(stTxt)}</span><button data-act="reset">다시</button>` :
        (XK[it.kind] ? `<button class="mdl" data-act="y" title="Y">${esc(b[0])}</button><button class="rul" data-act="n" title="N">${esc(b[1])}</button><button data-act="l" title="L">${esc(b[2])}</button>` :
        `<button class="ok" data-act="y">${esc(b[0])}</button><button class="bad" data-act="n">${esc(b[1])}</button>`)}</div></div>`;
  }).join('') || '<div class="k" style="padding:10px">검수 필요 항목 없음</div>';
  const tot = D.items.length, dn = D.items.filter((it) => itemState(it)).length;
  $('progtxt').textContent = `검수 필요 ${tot}건 중 ${dn}건 처리 · 자동인식 ${D.stats.n_auto}건 중 ${tot ? (100 * tot / Math.max(1, D.stats.n_auto)).toFixed(1) : 0}%만 확인`;
  $('progbar').style.width = (tot ? 100 * dn / tot : 100) + '%';
  $('bUndo').disabled = !undo.length; $('bRedo').disabled = !redo.length;
}
function renderProps() {
  const sel = U.sel; const P = S.plans[U.plan]; let h = '<h3>선택 항목</h3>';
  if (!sel) h += '<div class="k">도면에서 기호·문·벽·계단을 클릭하세요.</div>';
  else if (sel.kind === 'sym') {
    const q = P.symbols.find((x) => x.id === sel.id); if (!q) { U.sel = null; return renderProps(); }
    h += `<div><b style="color:${TCOL[q.type]}">${esc(KO[q.type])}</b> <span class="k">신뢰도 ${q.conf != null ? q.conf : '-'} · ${esc({ auto: '자동 인식', added: '검수자 추가', moved: '위치 수정', retyped: '종류 수정', 'moved+retyped': '위치·종류 수정' }[q.src || 'auto'] || q.src)}${q.ver ? ' · ✓ 확인됨' : ''}</span></div>
      <div class="row" style="margin:6px 0">${TYPES.map((t, i) => `<button class="ty ${q.type === t ? 'on' : ''}" data-ty="${t}" style="border-color:${TCOL[t]}">${i + 1} ${esc(KO[t])}</button>`).join('')}</div>
      <div class="row"><button class="ok" id="pVer">맞음 표시</button><button class="bad" id="pDel">삭제 <kbd>Del</kbd></button></div>`;
  } else if (sel.kind === 'door') {
    const d = P.doors.find((x) => x.id === sel.id); if (!d) { U.sel = null; return renderProps(); }
    h += `<div><b>문</b> <span class="k">${d.src === 'added' ? '검수자 추가' : '자동 추정'} · 방향 ${d.o === 'h' ? '가로' : '세로'}</span></div>
      <div class="row" style="margin:6px 0">폭 <input type="number" id="dW" step="0.1" min="0.5" max="6" value="${(d.w / PLANS[U.plan].s).toFixed(1)}"> m
      <label class="tg"><input type="checkbox" id="dExt" ${d.ext ? 'checked' : ''}> 외부 출입구</label></div>
      <div class="row"><button id="dRot">방향 바꾸기 <kbd>R</kbd></button><button class="bad" id="pDel">삭제 <kbd>Del</kbd></button></div>`;
  } else if (sel.kind === 'wall') {
    const w = P.walls.find((x) => x.id === sel.id); if (!w) { U.sel = null; return renderProps(); }
    h += `<div><b>벽</b> <span class="k">${w.src === 'added' ? '검수자 추가' : '자동 벡터화'} · 길이 ${((w.b - w.a) / PLANS[U.plan].s).toFixed(1)}m</span></div>
      <div class="row" style="margin-top:6px"><button class="bad" id="pDel">삭제 <kbd>Del</kbd></button></div>`;
  } else if (sel.kind === 'vadd') {
    const V = S.vadd.find((x) => x.id === sel.id); if (!V) { U.sel = null; return renderProps(); }
    const p = PLANS[V.plan], bw = (V.box_px[2] - V.box_px[0]) / p.s, bh = (V.box_px[3] - V.box_px[1]) / p.s;
    h += `<div><b>${esc(V.name)}</b> <span class="k">검수자 추가 · ${esc(p.label)} 도면에 그림 · ${bw.toFixed(1)}×${bh.toFixed(1)}m</span></div>
      <div class="row" style="margin:6px 0"><span class="k">종류</span>${['stairs', 'elevs'].map((t) => `<button class="${V.kind === t ? 'on' : ''}" data-vk="${t}">${VKO[t]}</button>`).join('')}</div>
      <div class="k" style="font-size:12px">연결층 (사이 층은 자동으로 채움)</div>
      <div class="row" style="margin:4px 0">${D.floors_all.map((f) => `<label class="tg" style="margin:0 6px 0 0"><input type="checkbox" data-vf="${esc(f.id)}" ${V.floors.includes(f.id) ? 'checked' : ''}> ${esc(f.id)}</label>`).join('')}</div>
      <div class="row"><button id="vfAll">전체 층</button><button id="vfPlan">이 도면 층만</button><button class="bad" id="pDel">삭제 <kbd>Del</kbd></button></div>`;
  } else {
    const V = D[sel.kind][sel.id]; const dec = S.vertical[sel.kind + ':' + sel.id];
    h += `<div><b>${esc(V.name)}</b> <span class="k">연결층 ${esc(V.label)} · 층간 반복으로 자동 판정</span></div>
      <div class="row" style="margin-top:6px"><button class="ok ${dec === 'ok' ? 'on' : ''}" id="vOk">정상</button><button class="bad ${dec === 'reject' ? 'on' : ''}" id="vBad">오검출 <kbd>Del</kbd></button></div>`;
  }
  $('props').innerHTML = h;
  document.querySelectorAll('#props [data-ty]').forEach((b) => { b.onclick = () => setType(b.dataset.ty); });
  if ($('pDel')) $('pDel').onclick = delSel;
  if ($('pVer')) $('pVer').onclick = () => { const b = snap(); const q = P.symbols.find((x) => x.id === sel.id); q.ver = true; commit(b, { op: 'verify', id: q.id }); };
  if ($('dRot')) $('dRot').onclick = rotDoor;
  if ($('dW')) $('dW').onchange = (e) => { const b = snap(); const d = P.doors.find((x) => x.id === sel.id); d.w = Math.max(0.3, +e.target.value) * PLANS[U.plan].s; d.src = d.src === 'added' ? 'added' : 'edited'; commit(b, { op: 'door_width', id: d.id }); };
  if ($('dExt')) $('dExt').onchange = (e) => { const b = snap(); const d = P.doors.find((x) => x.id === sel.id); d.ext = e.target.checked; d.src = d.src === 'added' ? 'added' : 'edited'; commit(b, { op: 'door_ext', id: d.id }); };
  if ($('vOk')) $('vOk').onclick = () => setVert('ok'); if ($('vBad')) $('vBad').onclick = () => setVert('reject');
  document.querySelectorAll('#props [data-vk]').forEach((b) => { b.onclick = () => setVaddKind(sel.id, b.dataset.vk); });
  document.querySelectorAll('#props [data-vf]').forEach((c) => { c.onchange = () => {
    const V = S.vadd.find((x) => x.id === sel.id); const on = new Set(V.floors); if (c.checked) on.add(c.dataset.vf); else on.delete(c.dataset.vf);
    setVaddFloors(sel.id, D.floors_all.map((f) => f.id).filter((f) => on.has(f))); }; });
  if ($('vfAll')) $('vfAll').onclick = () => setVaddFloors(sel.id, D.floors_all.map((f) => f.id));
  if ($('vfPlan')) $('vfPlan').onclick = () => { const V = S.vadd.find((x) => x.id === sel.id); setVaddFloors(sel.id, PLANS[V.plan].floors.map((f) => f.id)); };
}
function renderTools() {
  const m = U.mode;
  $('tools').innerHTML = `<button data-mode="select" class="${m === 'select' ? 'on' : ''}">선택·이동 <kbd>V</kbd></button>
    <button data-mode="sym" class="${m === 'sym' ? 'on' : ''}">기호 추가 <kbd>A</kbd></button>
    <button data-mode="door" class="${m === 'door' ? 'on' : ''}">문 추가 <kbd>D</kbd></button>
    <button data-mode="wall" class="${m === 'wall' ? 'on' : ''}">벽 추가 <kbd>W</kbd></button>
    <button data-mode="vert" class="${m === 'vert' ? 'on' : ''}" title="계단·승강기 추가 — 도면에 상자를 끌어 그리기">계단 추가 <kbd>S</kbd></button><span class="sep"></span>` +
    (m === 'vert' ? ['stairs', 'elevs'].map((t) => `<button class="ty ${U.vType === t ? 'on' : ''}" data-vtype="${t}" style="border-color:${t === 'stairs' ? '#3fb950' : '#58a6ff'}">${VKO[t]}</button>`).join('') : TYPES.map((t, i) => `<button class="ty ${m === 'sym' && U.addType === t ? 'on' : ''}" data-add="${t}" style="border-color:${TCOL[t]}" title="${esc(KO[t])}">${i + 1} ${esc(KO[t])}</button>`).join('')) +
    `<span class="sep"></span><button data-z="fit">맞춤 <kbd>F</kbd></button><button data-z="in">+</button><button data-z="out">−</button>`;
  $('legend').innerHTML = TYPES.map((t) => `<span><i style="background:${TCOL[t]}"></i>${esc(KO[t])}</span>`).join('') +
    '<span><i style="background:#2f81f7"></i>벽</span><span><i style="background:#f0883e"></i>문</span><span><i style="background:#39c5cf"></i>외부 출입구·추가 벽</span><span><i style="background:#ff3bd4"></i>아래층 벽</span><span><i style="border:2px dashed #ffd33d;width:8px;height:8px"></i>검수 필요</span>' +
    '<span><i style="background:#ff3bd4"></i>규칙만 벽</span><span><i style="background:#00e5ff"></i>모델만 벽</span><span><i style="background:#ff9f1c"></i>모델 문</span><span><i style="background:#ff4d4f"></i>판정: 삭제</span><span><i style="background:#3fb950"></i>판정: 유지·계단</span>';
}
function hud() {
  const p = PLANS[U.plan], P = S.plans[U.plan];
  const mt = { select: '선택·이동: 기호/문을 끌어 옮기기 · 빈 곳 끌기 = 화면 이동', sym: `기호 추가: 클릭한 곳에 「${KO[U.addType]}」 추가`, door: '문 추가: 벽 근처 클릭', wall: U.wallStart ? '벽 추가: 끝점 클릭 (Esc 취소)' : '벽 추가: 시작점 클릭',
    vert: `${VKO[U.vType]} 추가: 도면에 상자를 끌어 그리기 → 오른쪽에서 연결층 선택` }[U.mode];
  $('hud').innerHTML = `${esc(mt)} · 기호 ${P.symbols.length} · 문 ${P.doors.length} · 벽 ${P.walls.length} · 축척 ${p.s.toFixed(2)}px/m · <kbd>?</kbd> 도움말`;
}
function renderAll() { render(); renderPlans(); renderItems(); renderProps(); renderTools(); hud(); }

// ───────── 편집 동작 ─────────
function newId(prefix) { S.seq += 1; return `${prefix}n${S.seq}`; }
function touch(q) { if (q && !q.orig && q.src !== 'added') q.orig = { type: q.type, px: q.px, py: q.py }; }
function markSym(q) {
  if (!q || q.src === 'added' || !q.orig) return;
  const o = q.orig; const moved = Math.hypot(q.px - o.px, q.py - o.py) > 0.5, ret = q.type !== o.type;
  q.src = moved && ret ? 'moved+retyped' : (moved ? 'moved' : (ret ? 'retyped' : 'auto'));
  if (q.src === 'auto') delete q.orig;
}
function setType(t) {
  if (U.sel && U.sel.kind === 'sym') {
    const q = S.plans[U.plan].symbols.find((x) => x.id === U.sel.id); if (!q || q.type === t) return;
    const b = snap(); const from = q.type; touch(q); q.type = t; markSym(q); commit(b, { op: 'retype', id: q.id, from, to: t });
  } else { U.addType = t; U.mode = 'sym'; renderAll(); }
}
function delSel() {
  const sel = U.sel; if (!sel) return; const P = S.plans[U.plan]; const b = snap();
  if (sel.kind === 'sym') P.symbols = P.symbols.filter((x) => x.id !== sel.id);
  else if (sel.kind === 'door') P.doors = P.doors.filter((x) => x.id !== sel.id);
  else if (sel.kind === 'wall') P.walls = P.walls.filter((x) => x.id !== sel.id);
  else if (sel.kind === 'vadd') S.vadd = S.vadd.filter((x) => x.id !== sel.id);
  else { S.vertical[sel.kind + ':' + sel.id] = 'reject'; commit(b, { op: 'vertical', kind: sel.kind, k: sel.id, decision: 'reject' }); return; }
  U.sel = null; commit(b, { op: 'delete', kind: sel.kind, id: sel.id });
}
function setVert(d) { const sel = U.sel; const b = snap(); S.vertical[sel.kind + ':' + sel.id] = d; commit(b, { op: 'vertical', kind: sel.kind, k: sel.id, decision: d }); }
function rotDoor() { const P = S.plans[U.plan]; const d = U.sel && U.sel.kind === 'door' && P.doors.find((x) => x.id === U.sel.id); if (!d) return; const b = snap(); d.o = d.o === 'h' ? 'v' : 'h'; if (d.src !== 'added') d.src = 'edited'; commit(b, { op: 'door_rotate', id: d.id }); }
function setAlign(v) { const b = snap(); const st = S.plans[U.plan].status; st.align = st.align === v ? null : v; commit(b, { op: 'align', value: st.align }); }
function toggleDone() { const b = snap(); const st = S.plans[U.plan].status; st.done = !st.done; commit(b, { op: 'done', value: st.done }); }
function shiftPlan(dx, dy) { const b = snap(); const sh = S.plans[U.plan].shift; sh[0] = Math.round((sh[0] + dx) * 100) / 100; sh[1] = Math.round((sh[1] + dy) * 100) / 100; if (!U.ly.ghost) U.ly.ghost = 1, syncLayers(); commit(b, { op: 'shift', dx, dy }); }
function nudgeSel(dx, dy) {
  const P = S.plans[U.plan]; const sel = U.sel; if (!sel) return; const b = snap();
  if (sel.kind === 'sym') { const q = P.symbols.find((x) => x.id === sel.id); touch(q); moveSym(q, q.px + dx, q.py + dy); markSym(q); }
  else if (sel.kind === 'door') { const d = P.doors.find((x) => x.id === sel.id); d.px += dx; d.py += dy; if (d.src !== 'added') d.src = 'edited'; }
  else return;
  commit(b, { op: 'nudge', kind: sel.kind, id: sel.id });
}
function moveSym(q, x, y) { const dx = x - q.px, dy = y - q.py; q.px = Math.round(x * 10) / 10; q.py = Math.round(y * 10) / 10; if (q.box) q.box = q.box.map((v, i) => Math.round((v + (i % 2 ? dy : dx)) * 10) / 10); }
function medianSize(t) {
  const P = S.plans[U.plan]; const L = P.symbols.filter((q) => q.type === t && q.box).map((q) => [q.box[2] - q.box[0], q.box[3] - q.box[1]]);
  if (!L.length) { const s = PLANS[U.plan].s; return [s, s]; }
  const med = (a) => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
  return [med(L.map((v) => v[0])), med(L.map((v) => v[1]))];
}
function addSym(x, y) {
  const b = snap(); const [w, h] = medianSize(U.addType); const id = newId(`p${U.plan}s`);
  S.plans[U.plan].symbols.push({ id, type: U.addType, px: Math.round(x * 10) / 10, py: Math.round(y * 10) / 10, box: [x - w / 2, y - h / 2, x + w / 2, y + h / 2].map((v) => Math.round(v * 10) / 10), box_est: true, conf: null, src: 'added' });
  U.sel = { kind: 'sym', id }; commit(b, { op: 'add_symbol', id, type: U.addType });
}
function nearestWall(x, y) {
  let best = null; const P = S.plans[U.plan];
  for (const w of P.walls) {
    const [u, v] = w.o === 'h' ? [x, y] : [y, x];
    const du = u < w.a ? w.a - u : (u > w.b ? u - w.b : 0), dv = Math.abs(v - w.c); const d = Math.hypot(du, dv);
    if (!best || d < best.d) best = { d, w };
  }
  return best;
}
function addDoor(x, y) {
  const b = snap(); const s = PLANS[U.plan].s; const nw = nearestWall(x, y); let o = 'h';
  if (nw && nw.d < 1.2 * s) { o = nw.w.o; if (o === 'h') y = nw.w.c; else x = nw.w.c; }
  const id = newId(`p${U.plan}d`);
  S.plans[U.plan].doors.push({ id, o, px: Math.round(x * 10) / 10, py: Math.round(y * 10) / 10, w: Math.round(0.9 * s * 10) / 10, ext: false, src: 'added' });
  U.sel = { kind: 'door', id }; commit(b, { op: 'add_door', id });
}
function addWall(p0, p1) {
  const b = snap(); const s = PLANS[U.plan].s; const hz = Math.abs(p1.x - p0.x) >= Math.abs(p1.y - p0.y);
  const w = hz ? { o: 'h', c: (p0.y + p1.y) / 2, a: Math.min(p0.x, p1.x), b: Math.max(p0.x, p1.x) } : { o: 'v', c: (p0.x + p1.x) / 2, a: Math.min(p0.y, p1.y), b: Math.max(p0.y, p1.y) };
  if (w.b - w.a < 0.3 * s) return;
  const id = newId(`p${U.plan}w`);
  S.plans[U.plan].walls.push(Object.assign(w, { id, t: Math.max(1.5, 0.15 * s), src: 'added' }, { c: Math.round(w.c * 10) / 10, a: Math.round(w.a * 10) / 10, b: Math.round(w.b * 10) / 10 }));
  U.sel = { kind: 'wall', id }; commit(b, { op: 'add_wall', id });
}
function addVert(b, kind, floors) {
  const p = PLANS[U.plan]; const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1), y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
  if ((x1 - x0) < 0.8 * p.s || (y1 - y0) < 0.8 * p.s) { banner('상자가 너무 작습니다(한 변 0.8m 이상)'); return null; }
  const bf = snap(); kind = kind || U.vType; const id = newId('v');
  const n = S.vadd.filter((v) => v.kind === kind).length + 1;
  const V = { id, kind, name: `${VKO[kind]}+${n}`, plan: U.plan, box_px: [x0, y0, x1, y1].map((v) => Math.round(v * 10) / 10), floors: floors || D.floors_all.map((f) => f.id) };
  S.vadd.push(V); U.sel = { kind: 'vadd', id }; U.mode = 'select';
  commit(bf, { op: 'add_vertical', id, kind, floors: V.floors }); return id;
}
function setVaddKind(id, kind) { const V = S.vadd.find((x) => x.id === id); if (!V || V.kind === kind) return; const b = snap(); V.kind = kind; V.name = `${VKO[kind]}+${S.vadd.filter((v) => v.kind === kind).length}`; commit(b, { op: 'vertical_kind', id, kind }); }
function setVaddFloors(id, floors) { const V = S.vadd.find((x) => x.id === id); if (!V) return; const b = snap(); V.floors = floors; commit(b, { op: 'vertical_floors', id, floors }); }
function gotoPlan(i) { if (i == null || !PLANS[i]) return; U.plan = i; U.sel = null; U.wallStart = null; U.hiType = null; renderAll(); }
function gotoItem(it) {
  U.cur = it.id;
  if (it.plan != null && it.plan !== U.plan) gotoPlan(it.plan);
  U.hiType = it.kind === 'flag' && it.fk === 'hose_count' ? 'hose' : null;
  if (it.kind === 'symbol') { U.sel = findSym(it.plan, it.ref) ? { kind: 'sym', id: it.ref } : null; if (it.loc) focusAt(it.loc[0], it.loc[1], Math.max(PLANS[U.plan].s * 18, 160)); }
  else if (XK[it.kind] && it.loc) {
    U.sel = it.ref && S.plans[it.plan].walls.some((w) => w.id === it.ref) ? { kind: 'wall', id: it.ref } : null;
    const L = Math.hypot(it.seg[2] - it.seg[0], it.seg[3] - it.seg[1]); focusAt(it.loc[0], it.loc[1], Math.max(PLANS[U.plan].s * 14, L * 2.2, 140));
  }
  else if (it.kind === 'align') { U.ly.ghost = 1; syncLayers(); U.views[U.plan] = fitView(PLANS[U.plan]); }
  else if (it.plan != null) U.views[U.plan] = fitView(PLANS[U.plan]);
  renderAll();
  const el = document.querySelector(`[data-it="${it.id}"]`); if (el) el.scrollIntoView({ block: 'nearest' });
}
function stepItem(d) {
  const L = visibleItems(); if (!L.length) return;
  let j = L.findIndex((it) => it.id === U.cur);
  if (d > 0) { const open = L.filter((it) => !itemState(it)); const nx = open.find((it) => L.indexOf(it) > j) || open[0] || L[(j + 1) % L.length]; gotoItem(nx); }
  else gotoItem(L[(j - 1 + L.length) % L.length] || L[0]);
}
function syncLayers() { document.querySelectorAll('[data-ly]').forEach((c) => { c.checked = !!U.ly[c.dataset.ly]; }); }

// ───────── 마우스 ─────────
let drag = null;
svg.addEventListener('pointerdown', (ev) => {
  const t = ev.target; const pt = toImg(ev); const k = t.dataset && t.dataset.k;
  if (ev.button === 1 || ev.button === 2 || (ev.button === 0 && spaceDown)) { drag = { pan: true, sx: ev.clientX, sy: ev.clientY, v: Object.assign({}, view()) }; svg.setPointerCapture(ev.pointerId); ev.preventDefault(); return; }
  if (U.mode === 'sym') { addSym(pt.x, pt.y); return; }
  if (U.mode === 'door') { addDoor(pt.x, pt.y); return; }
  if (U.mode === 'vert') { U.vdrag = { x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y }; svg.setPointerCapture(ev.pointerId); render(); return; }
  if (U.mode === 'wall') { if (!U.wallStart) { U.wallStart = { x: pt.x, y: pt.y }; render(); hud(); } else { const p0 = U.wallStart; U.wallStart = null; addWall(p0, pt); } return; }
  if (k) {
    const id = k === 'stairs' || k === 'elevs' ? +t.dataset.id : t.dataset.id;   // vadd: 문자열 id
    U.sel = { kind: k, id }; renderProps();
    if (k === 'sym' || k === 'door') drag = { item: true, before: snap(), x0: pt.x, y0: pt.y, moved: false, kind: k, id };
    render(); svg.setPointerCapture(ev.pointerId);
    return;
  }
  U.sel = null; renderProps(); render();
  drag = { pan: true, sx: ev.clientX, sy: ev.clientY, v: Object.assign({}, view()) }; svg.setPointerCapture(ev.pointerId);
});
svg.addEventListener('pointermove', (ev) => {
  if (U.mode === 'wall' && U.wallStart) { const pt = toImg(ev); const l = $('wtmp'); if (l) { const hz = Math.abs(pt.x - U.wallStart.x) >= Math.abs(pt.y - U.wallStart.y); l.setAttribute('x2', hz ? pt.x : U.wallStart.x); l.setAttribute('y2', hz ? U.wallStart.y : pt.y); } }
  if (U.mode === 'vert' && U.vdrag) { const pt = toImg(ev); U.vdrag.x1 = pt.x; U.vdrag.y1 = pt.y; const r = $('vtmp'); if (r) { const v = U.vdrag; r.setAttribute('x', Math.min(v.x0, v.x1)); r.setAttribute('y', Math.min(v.y0, v.y1)); r.setAttribute('width', Math.abs(v.x1 - v.x0)); r.setAttribute('height', Math.abs(v.y1 - v.y0)); } return; }
  if (!drag) return;
  if (drag.pan) { const k = 1 / pxPerUnit(); const v = view(); v.x = drag.v.x - (ev.clientX - drag.sx) * k; v.y = drag.v.y - (ev.clientY - drag.sy) * k; applyView(); return; }
  const pt = toImg(ev); const dx = pt.x - drag.x0, dy = pt.y - drag.y0;
  if (!drag.moved && Math.hypot(dx, dy) * pxPerUnit() < 3) return;
  drag.moved = true; drag.x0 = pt.x; drag.y0 = pt.y;
  const P = S.plans[U.plan];
  if (drag.kind === 'sym') { const q = P.symbols.find((x) => x.id === drag.id); touch(q); moveSym(q, q.px + dx, q.py + dy); }
  else { const d = P.doors.find((x) => x.id === drag.id); d.px += dx; d.py += dy; }
  render();
});
svg.addEventListener('pointerup', (ev) => {
  if (U.mode === 'vert' && U.vdrag) { const pt = toImg(ev); const v = U.vdrag; v.x1 = pt.x; v.y1 = pt.y; U.vdrag = null; if (!addVert(v)) render(); return; }
  if (drag && drag.item && drag.moved) {
    const P = S.plans[U.plan];
    if (drag.kind === 'sym') markSym(P.symbols.find((x) => x.id === drag.id));
    else { const d = P.doors.find((x) => x.id === drag.id); d.px = Math.round(d.px * 10) / 10; d.py = Math.round(d.py * 10) / 10; if (d.src !== 'added') d.src = 'edited'; }
    const b = drag.before; drag = null; commit(b, { op: 'move', kind: 'sym', id: U.sel && U.sel.id });
    return;
  }
  drag = null;
});
svg.addEventListener('contextmenu', (e) => e.preventDefault());
svg.addEventListener('wheel', (ev) => { ev.preventDefault(); const pt = toImg(ev); zoomAt(ev.deltaY > 0 ? 1.15 : 1 / 1.15, pt.x, pt.y); render(); }, { passive: false });
window.addEventListener('resize', () => render());

// ───────── 키보드 ─────────
let spaceDown = false;
window.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceDown = false; });
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox') return;
  // 한글 입력(IME) 상태에서도 단축키가 되도록 물리 키(e.code) 기준으로 정규화
  const ctrl = e.ctrlKey || e.metaKey, code = e.code || '';
  let key = e.key;
  if (/^Key[A-Z]$/.test(code)) key = code.slice(3).toLowerCase();
  else if (/^(Digit|Numpad)[0-9]$/.test(code)) key = code.slice(-1);
  else if (code === 'NumpadAdd') key = '+'; else if (code === 'NumpadSubtract') key = '-';
  else if (code === 'Slash' && e.shiftKey) key = '?';
  if (e.code === 'Space') { spaceDown = true; e.preventDefault(); return; }
  if (ctrl && (key === 'z' || key === 'Z')) { e.preventDefault(); if (e.shiftKey) doRedo(); else doUndo(); return; }
  if (ctrl && (key === 'y' || key === 'Y')) { e.preventDefault(); doRedo(); return; }
  if (ctrl && (key === 's' || key === 'S')) { e.preventDefault(); exportCorrections(); return; }
  if (key.startsWith('Arrow')) {
    e.preventDefault();
    const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[key];
    if (ctrl) { const st = e.shiftKey ? 0.5 : 0.1; shiftPlan(dir[0] * st, -dir[1] * st); }     // 전역 y 는 위쪽 +
    else nudgeSel(dir[0] * (e.shiftKey ? 10 : 1), dir[1] * (e.shiftKey ? 10 : 1));
    return;
  }
  if (ctrl || e.altKey) return;
  const cur = D.items.find((it) => it.id === U.cur);
  switch (key) {
    case 'v': case 'V': U.mode = 'select'; U.wallStart = null; renderAll(); break;
    case 'a': case 'A': U.mode = 'sym'; renderAll(); break;
    case 'd': case 'D': U.mode = 'door'; renderAll(); break;
    case 'w': case 'W': U.mode = 'wall'; U.wallStart = null; renderAll(); break;
    case 's': case 'S': U.mode = 'vert'; U.wallStart = null; renderAll(); break;
    case 'l': case 'L': if (cur && XK[cur.kind] && !itemState(cur)) { decideX(cur, 'later'); stepItem(1); } break;
    case '1': case '2': case '3': case '4': case '5': setType(TYPES[+key - 1]); break;
    case 'Delete': case 'Backspace': e.preventDefault(); delSel(); break;
    case 'r': case 'R': rotDoor(); break;
    case 'o': case 'O': setAlign('ok'); break;
    case 'x': case 'X': setAlign('bad'); break;
    case 'Enter': e.preventDefault(); toggleDone(); break;
    case 'g': case 'G': U.ly.ghost = U.ly.ghost ? 0 : 1; syncLayers(); render(); break;
    case 'f': case 'F': U.views[U.plan] = fitView(PLANS[U.plan]); render(); break;
    case '+': case '=': zoomAt(1 / 1.3); render(); break;
    case '-': case '_': zoomAt(1.3); render(); break;
    case 'j': case 'J': stepItem(1); break;
    case 'k': case 'K': stepItem(-1); break;
    case 'y': case 'Y': if (cur && !itemState(cur)) { decide(cur, true); stepItem(1); } break;
    case 'n': case 'N': if (cur && !itemState(cur)) { decide(cur, false); stepItem(1); } break;
    case 'PageDown': e.preventDefault(); gotoPlan(ORDER[Math.min(ORDER.length - 1, ORDER.indexOf(U.plan) + 1)]); break;
    case 'PageUp': e.preventDefault(); gotoPlan(ORDER[Math.max(0, ORDER.indexOf(U.plan) - 1)]); break;
    case 'Escape': U.sel = null; U.mode = 'select'; U.wallStart = null; U.vdrag = null; U.hiType = null; $('help').style.display = 'none'; renderAll(); break;
    case '?': $('help').style.display = 'flex'; break;
    default: return;
  }
});

// ───────── 패널 클릭 ─────────
$('plans').addEventListener('click', (e) => { const el = e.target.closest('[data-plan]'); if (el) gotoPlan(+el.dataset.plan); });
$('items').addEventListener('click', (e) => {
  const el = e.target.closest('[data-it]'); if (!el) return; const it = ITEM[el.dataset.it];
  const act = e.target.dataset && e.target.dataset.act;
  if (act === 'y' || act === 'n') { U.cur = it.id; decide(it, act === 'y'); return; }
  if (act === 'l') { U.cur = it.id; decideX(it, 'later'); return; }
  if (act === 'reset') { resetItem(it); return; }
  gotoItem(it);
});
function resetItem(it) {
  const b = snap(); delete S.items[it.id]; if (S.xdec) delete S.xdec[it.id];
  if (it.kind === 'flag') delete S.flags[it.ref];
  if (it.kind === 'align') S.plans[it.plan].status.align = null;
  if (it.kind === 'scale') S.plans[it.plan].status.scale = null;
  if (it.kind === 'symbol') {
    const q = findSym(it.plan, it.ref);
    if (q) delete q.ver;
    else { const o = PLANS[it.plan].symbols.find((x) => x.id === it.ref); if (o) S.plans[it.plan].symbols.push(clone(o)); }
  }
  commit(b, { op: 'reset_item', item: it.id });
}
$('tools').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.mode) { U.mode = b.dataset.mode; U.wallStart = null; renderAll(); }
  if (b.dataset.add) { U.addType = b.dataset.add; U.mode = 'sym'; renderAll(); }
  if (b.dataset.vtype) { U.vType = b.dataset.vtype; U.mode = 'vert'; renderAll(); }
  if (b.dataset.z === 'fit') { U.views[U.plan] = fitView(PLANS[U.plan]); render(); }
  if (b.dataset.z === 'in') { zoomAt(1 / 1.3); render(); }
  if (b.dataset.z === 'out') { zoomAt(1.3); render(); }
});
document.querySelectorAll('[data-ly]').forEach((c) => { c.onchange = () => { U.ly[c.dataset.ly] = c.checked ? 1 : 0; render(); }; });
$('onlyPlan').onchange = (e) => { U.onlyPlan = e.target.checked; renderItems(); };
$('bUndo').onclick = doUndo; $('bRedo').onclick = doRedo; $('bSave').onclick = exportCorrections;
$('bHelp').onclick = () => { $('help').style.display = 'flex'; };
$('bLoad').onclick = () => $('file').click();
$('file').onchange = (e) => {
  const f = e.target.files[0]; if (!f) return; const r = new FileReader();
  r.onload = () => { try { const b = snap(); loadCorrections(JSON.parse(r.result)); commit(b, { op: 'load', file: f.name }); banner(`불러옴: ${f.name}`); } catch (err) { banner('불러오기 실패: ' + err.message); } };
  r.readAsText(f, 'utf-8'); e.target.value = '';
};

// ───────── 저장(교정 JSON) ─────────
function buildCorrections() {
  const out = {
    schema: D.schema, tool: 'fire3dbeam-review-html/1', created: S.created || now(), updated: now(),
    building: D.building, source: D.source,
    summary: { items_total: D.items.length, items_done: D.items.filter((it) => itemState(it)).length, plans_total: D.plans.length, plans_done: D.plans.filter((p) => S.plans[p.i].status.done).length, auto_total: D.stats.n_auto,
      xcheck_total: D.items.filter((it) => XK[it.kind]).length, xcheck_done: D.items.filter((it) => XK[it.kind] && itemState(it)).length, vertical_added: S.vadd.length },
    plans: {}, flags: [], vertical: [], vertical_added: [], xcheck: [], items: [], ops: S.ops,
  };
  D.plans.forEach((p) => {
    const P = S.plans[p.i]; const ids = (L) => new Set(L.map((x) => x.id));
    const sI = ids(P.symbols), dI = ids(P.doors), wI = ids(P.walls);
    out.plans[p.i] = {
      floors: p.floors.map((f) => f.id), caption: p.caption, img_w: p.w, img_h: p.h, s: p.s, status: P.status, shift_m: P.shift,
      symbols: P.symbols.map((q) => ({ id: q.id, type: q.type, px: q.px, py: q.py, box: q.box, box_est: !!q.box_est, conf: q.conf, src: q.src || 'auto', verified: !!q.ver, orig: q.orig })),
      symbols_deleted: p.symbols.filter((q) => !sI.has(q.id)),
      doors: P.doors.map((d) => ({ id: d.id, o: d.o, px: d.px, py: d.py, w: d.w, ext: !!d.ext, src: d.src || 'auto' })),
      doors_deleted: p.doors.filter((d) => !dI.has(d.id)),
      walls_deleted: p.walls.filter((w) => !wI.has(w.id)),
      walls_added: P.walls.filter((w) => w.src === 'added'),
    };
  });
  D.flags.forEach((f) => { if (S.flags[f.key]) out.flags.push({ key: f.key, kind: f.kind, floor: f.floor, decision: S.flags[f.key] }); });
  Object.keys(S.vertical).forEach((k) => { const [kind, i] = k.split(':'); const V = D[kind][+i]; out.vertical.push({ kind, k: +i, name: V.name, box_m: V.box_m, floors: V.floors, decision: S.vertical[k] }); });
  D.items.forEach((it) => { const st = itemState(it); out.items.push({ id: it.id, kind: it.kind, plan: it.plan, ref: it.ref, p: it.p, decision: st }); });
  // 교차검증 판정(도면 px 기하 포함 → 결과 JSON 이 바뀌어도 다시 적용 가능)
  D.items.filter((it) => XK[it.kind] && S.xdec[it.id]).forEach((it) => {
    const e = { id: it.id, kind: it.kind, sub: it.sub, plan: it.plan, p: it.p, seg: it.seg, loc: it.loc, o: it.o, ref: it.ref || null, decision: S.xdec[it.id] };
    if (it.kind === 'xwall') Object.assign(e, { tpx: it.tpx, filled: !!it.filled, bridge: !!it.bridge }); else Object.assign(e, { wpx: it.wpx, ext: !!it.ext });
    out.xcheck.push(e);
  });
  S.vadd.forEach((V) => {
    const p = PLANS[V.plan]; const b = V.box_px;
    const bm = p.tx != null ? [b[0] / p.s + p.tx, p.ty - b[3] / p.s, b[2] / p.s + p.tx, p.ty - b[1] / p.s].map((v) => Math.round(v * 100) / 100) : null;
    out.vertical_added.push({ id: V.id, kind: V.kind, name: V.name, plan: V.plan, box_px: b, box_m: bm, floors: V.floors });
  });
  return out;
}
function exportCorrections() {
  if (!S.created) S.created = now();
  const c = buildCorrections();
  const blob = new Blob([JSON.stringify(c, null, 1)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `${D.building.id}_corrections.json`; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  window.__lastCorrections = c;
  banner(`저장: ${a.download} (다운로드 폴더) → 검수.bat 에 끌어다 놓으세요`);
  return c;
}
function loadCorrections(c) {
  if (!c || !c.plans) throw new Error('검수 파일 형식이 아님');
  if (c.building && c.building.result_sha && c.building.result_sha !== D.building.result_sha) banner('주의: 다른 변환 결과에서 만든 검수 파일입니다(위치로 맞춰 불러옴)');
  S = initialState(); S.created = c.created;
  Object.keys(c.plans).forEach((i) => {
    const C = c.plans[i], P = S.plans[i]; if (!P) return;
    P.symbols = C.symbols.map((q) => { const o = { id: q.id, type: q.type, px: q.px, py: q.py, box: q.box, box_est: q.box_est, conf: q.conf, src: q.src }; if (q.verified) o.ver = true; if (q.orig) o.orig = q.orig; return o; });
    P.doors = C.doors.map((d) => Object.assign({}, d));
    const del = new Set((C.walls_deleted || []).map((w) => w.id));
    P.walls = P.walls.filter((w) => !del.has(w.id)).concat((C.walls_added || []).map((w) => Object.assign({}, w)));
    P.status = Object.assign({ align: null, scale: null, done: false }, C.status || {}); P.shift = C.shift_m || [0, 0];
    [P.symbols, P.doors, P.walls].forEach((L) => L.forEach((x) => { const m = /n(\d+)$/.exec(x.id); if (m) S.seq = Math.max(S.seq, +m[1]); }));
  });
  (c.flags || []).forEach((f) => { S.flags[f.key] = f.decision; });
  (c.vertical || []).forEach((v) => { S.vertical[v.kind + ':' + v.k] = v.decision; });
  (c.xcheck || []).forEach((e) => {       // 같은 id·같은 기하 → 아니면 같은 종류·도면에서 기하가 맞는 항목
    const same = (it) => it.kind === e.kind && it.plan === e.plan && it.seg && e.seg && it.seg.every((v, j) => Math.abs(v - e.seg[j]) < 3);
    const it = (ITEM[e.id] && same(ITEM[e.id])) ? ITEM[e.id] : D.items.find(same);
    if (it && ['model', 'rule', 'later'].includes(e.decision)) S.xdec[it.id] = e.decision;
  });
  S.vadd = (c.vertical_added || []).filter((v) => PLANS[v.plan] && v.box_px).map((v) => ({ id: v.id, kind: v.kind, name: v.name, plan: v.plan, box_px: v.box_px, floors: v.floors || [] }));
  S.vadd.forEach((v) => { const m = /v?n(\d+)$/.exec(v.id || ''); if (m) S.seq = Math.max(S.seq, +m[1]); });
  (c.items || []).forEach((it) => { if (it.decision && ITEM[it.id] && !['flag', 'align', 'scale'].includes(it.kind) && (it.decision === 'yes' || it.decision === 'no')) S.items[it.id] = it.decision; });
  S.ops = c.ops || [];
}
let bt = null;
function banner(msg, btn) {
  const b = $('banner'); b.innerHTML = esc(msg) + (btn ? ` <button id="bnBtn">${esc(btn[0])}</button>` : ''); b.style.display = 'block';
  if (btn) $('bnBtn').onclick = () => { btn[1](); b.style.display = 'none'; };
  clearTimeout(bt); bt = setTimeout(() => { b.style.display = 'none'; }, btn ? 9000 : 4500);
}

// ───────── 시작 ─────────
$('title').textContent = `${D.building.name} — 자동 변환 검수`;
document.title = `${D.building.name} 검수`;
let restored = false;
if (D.preload) { try { loadCorrections(D.preload); restored = 'file'; } catch (e) { console.error(e); } }
else {
  try { const a = JSON.parse(localStorage.getItem(LSKEY) || 'null'); if (a && a.S && a.S.plans) { S = a.S; S.xdec = S.xdec || {}; S.vadd = S.vadd || []; restored = a.t; } } catch (e) { /* 없음 */ }
}
renderAll();
if (restored === 'file') banner('이전 검수 파일을 반영해 열었습니다');
else if (restored) banner(`자동 저장된 작업(${restored.replace('T', ' ')})을 불러왔습니다`, ['처음 상태로', () => { const b = snap(); S = initialState(); commit(b, { op: 'reset_all' }); }]);
const first = D.items.find((it) => !itemState(it)); if (first) { U.cur = first.id; renderItems(); }
// 자동 시험(헤드리스)용
window.__review = { get S() { return S; }, U, D, buildCorrections, exportCorrections, loadCorrections,  decide: (id, y) => decide(ITEM[id], y), decideX: (id, d) => decideX(ITEM[id], d), addVert: (x0, y0, x1, y1, kind, floors) => addVert({ x0, y0, x1, y1 }, kind, floors), setVaddKind, setVaddFloors, gotoPlan, gotoItem: (id) => gotoItem(ITEM[id]), addSym: (x, y, t) => { if (t) U.addType = t; addSym(x, y); }, addDoor, addWall: (x0, y0, x1, y1) => addWall({ x: x0, y: y0 }, { x: x1, y: y1 }), setType, delSel, select: (kind, id) => { U.sel = { kind, id }; renderAll(); }, setAlign, toggleDone, shiftPlan, doUndo, doRedo, render: renderAll };
})();
