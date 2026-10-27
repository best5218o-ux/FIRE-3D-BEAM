// 훈련 모드 채점 — 시나리오(무작위·현재 화점, 주간/야간)에서 훈련자가 진입구·공격계단·호스 본수·인명검색 순서를 정하면
// 같은 모델의 산출안(main.js drawFire → lastPlan)과 항목별 비교·채점. 산출안은 제출 전까지 숨김. 결과는 JSON·HTML 로 저장(오프라인, 서버 없음).
// 배점·허용 범위 = D.sop_train(fire3dbeam/sop.py TRAIN, 임시값).
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { CLASS_KO, CAT_KO } from './search.js';

const DEF = { pts_stair: 25, pts_route: 25, pts_hose: 15, pts_search: 25, pts_time: 10, route_ratio_zero: 1.5, hose_tol: 1, search_k: 6, time_full_s: 120, time_zero_s: 300 };

// Kendall 순위 상관(tau-a): 훈련자 순서 a(실 번호 배열)와 산출안 순위 rk(실 번호 → 순위). 동순위 없음.
export function kendallTau(a, rk) {
  let c = 0, d = 0;
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) {
    const s = Math.sign(rk[a[j]] - rk[a[i]]); if (s > 0) c++; else if (s < 0) d++;
  }
  return c + d ? (c - d) / (c + d) : null;
}

export function initTrain(ctx) {
  const { D, BIM, FL, V, SLAB, esc, $, zOf, dijkstra, pathTo, scene } = ctx;
  const T = Object.assign({}, DEF, D.sop_train || {});
  const EV = D.evac || null;
  const G = new THREE.Group(); scene.add(G);                 // 훈련자 표시(T1, T2 …)
  let S = null;                                              // 진행 중 시나리오
  const ctr = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  const inPoly = (x, y, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  const lab = (t, cls, x, y, z) => { const el = document.createElement('div'); el.className = 'lb ' + cls; el.textContent = t; const o = new CSS2DObject(el); o.position.copy(V(x, y, z)); G.add(o); };
  const roomsOf = (fid) => (EV && EV.floors[fid] ? EV.floors[fid].rooms : ((BIM.floors[fid] || {}).spaces || []).map((s) => ({ poly: s.poly, area: s.area, c: s.poly.reduce((a, p) => [a[0] + p[0] / s.poly.length, a[1] + p[1] / s.poly.length], [0, 0]) })));
  const ents = () => { const g = ctx.ground(); return (BIM.floors[g]?.doors || []).filter((d) => d[0] === 'entrance' || (d[0] === 'exit' && d[6])); };

  // ── 화면
  const box = $('trainPanel');
  function menu() {
    box.innerHTML = `<div class="trn-box"><b>훈련 모드(채점)</b> — 산출안은 제출 전까지 숨깁니다.<div class="row" style="margin-top:5px">
      <select id="trMode"><option value="night">야간(취침)</option><option value="day">주간</option></select>
      <button id="trRnd">무작위 화점으로 시작</button><button id="trCur" title="화점이 없으면 무작위">현재 화점으로 시작</button></div>
      <div class="note">순서: 진입구 → 공격계단 → 호스 본수 → 화재층 실을 클릭해 인명검색 순서(${T.search_k}곳) → 제출. 시간도 채점(${T.time_full_s}초 이내 만점).</div></div>`;
  }
  function form() {
    const P = S.plan; if (!P) { box.innerHTML = '<div class="trn-box">산출 중…</div>'; return; }
    const E = ents(), cand = P.cand.map((id) => BIM.stairs.find((s) => s.id === id)).filter(Boolean);
    let h = `<div class="trn-box"><b>훈련 중</b> — ${esc(FL[P.fid].name)} 화재 · ${S.mode === 'night' ? '야간' : '주간'} <span class="k" id="trClock"></span>`;
    h += `<table class="trf"><tr><td class="k">① 진입구</td><td><select id="trEnt"><option value="">선택</option>${E.map((d, i) => `<option value="${i}" ${S.ans.ent === i ? 'selected' : ''}>${esc(d[5])}</option>`).join('')}</select></td></tr>`;
    h += P.fid === P.ground ? `<tr><td class="k">② 공격계단</td><td class="k">해당 없음(진입층 화재)</td></tr>`
      : `<tr><td class="k">② 공격계단</td><td><select id="trSt"><option value="">선택</option>${cand.map((s) => `<option value="${s.id}" ${S.ans.st === s.id ? 'selected' : ''}>${esc(s.name)}</option>`).join('')}</select></td></tr>`;
    h += `<tr><td class="k">③ 호스 본수</td><td><input id="trHose" type="number" min="0" max="30" value="${S.ans.hose ?? ''}"></td></tr>`;
    h += `<tr><td class="k">④ 인명검색 순서</td><td><button id="trPick" class="${S.picking ? 'on' : ''}">${S.picking ? '실 클릭 중…' : '화재층 실 클릭'}</button> <button id="trUndo">되돌리기</button></td></tr></table>`;
    h += `<div class="note">고른 실: ${S.ans.order.length ? S.ans.order.map((k, i) => `T${i + 1}`).join(' → ') : '없음'} (${S.ans.order.length}/${T.search_k})</div>`;
    h += `<div class="row" style="margin-top:5px"><button class="fire" id="trGo">제출·채점</button><button id="trQuit">그만두기</button></div></div>`;
    box.innerHTML = h;
  }
  function tick() { const el = $('trClock'); if (el && S && !S.result) el.textContent = `${((performance.now() - S.t0) / 1000).toFixed(0)}초`; if (S && !S.result) setTimeout(tick, 1000); }

  function start(kind) {
    G.clear();
    S = { kind, mode: $('trMode') ? $('trMode').value : 'night', ans: { ent: null, st: null, hose: null, order: [] }, picking: false, plan: null, result: null, t0: performance.now(), started: new Date().toISOString() };
    if (ctx.setMode) ctx.setMode(S.mode);
    if (kind === 'cur' && !ctx.getFire()) S.kind = kind = 'rnd';
    if (kind === 'rnd') ctx.randomFire(); else ctx.redraw();
    form(); tick();
  }
  function quit() { S = null; G.clear(); ctx.redraw(); menu(); }

  // ── 채점
  function score() {
    const P = S.plan, A = S.ans, items = [];
    const E = ents(), e = A.ent != null ? E[A.ent] : null;
    const grF = P.grF, RF = P.RF;
    // 경로 길이(훈련자 진입구·계단 → 화점, 산출안과 같은 격자·pathTo 길이)
    let Lt = null, stair = null;
    if (e) {
      if (P.fid === P.ground) Lt = pathTo(grF, RF, e[2], e[3]).len;
      else if (A.st) {
        stair = BIM.stairs.find((s) => s.id === A.st); const [sx, sy] = ctr(stair.box);
        const RE = dijkstra(P.grG, [[e[2], e[3]]]);
        Lt = pathTo(P.grG, RE, sx, sy).len + pathTo(grF, RF, sx, sy).len;
      }
    }
    const ratio = Lt != null && P.routeLen > 0 ? Lt / P.routeLen : Lt != null ? 1 : null;
    const fr = ratio == null ? 0 : Math.max(0, Math.min(1, 1 - (Math.max(1, ratio) - 1) / (T.route_ratio_zero - 1)));
    items.push({ key: 'route', label: '진입 경로 길이 비(훈련자/산출안)', value: Lt != null ? `${Lt.toFixed(0)}m` : '미입력', ref: `${P.routeLen.toFixed(0)}m${P.entry ? ' · ' + P.entry.name : ''}`, metric: ratio, pts: T.pts_route * fr, max: T.pts_route });
    if (P.fid !== P.ground) {
      const ok = A.st === P.attack;
      items.push({ key: 'stair', label: '공격계단 선택', value: stair ? stair.name : '미입력', ref: P.attackName || '-', metric: ok ? 1 : 0, pts: ok ? T.pts_stair : 0, max: T.pts_stair });
    }
    if (P.hoseN != null) {
      const dh = A.hose == null ? null : A.hose - P.hoseN, ok = dh != null && Math.abs(dh) <= T.hose_tol;
      items.push({ key: 'hose', label: `호스 본수(±${T.hose_tol} 허용)`, value: A.hose ?? '미입력', ref: `${P.hoseN}본(소화전 ${P.hoseL.toFixed(0)}m)`, metric: dh, pts: ok ? T.pts_hose : 0, max: T.pts_hose });
    }
    const refF = P.search.filter((z) => z.fid === P.fid), rk = {}; refF.forEach((z, i) => (rk[z.k] = i));
    const tau = kendallTau(A.order, rk), k = Math.min(T.search_k, refF.length);
    const top = new Set(refF.slice(0, k).map((z) => z.k)), hit = A.order.slice(0, k).filter((x) => top.has(x)).length, rec = k ? hit / k : 0;
    items.push({ key: 'search', label: `인명검색 순서(Kendall τ · 상위 ${k} 포함률)`, value: `τ=${tau == null ? '-' : tau.toFixed(2)} · ${hit}/${k}`, ref: refF.slice(0, k).map((z) => `#${z.rank} ${CLASS_KO[z.cls]}`).join(', '),
      metric: { tau, recall: rec }, pts: T.pts_search * (0.5 * (tau != null ? (tau + 1) / 2 : k <= 1 ? rec : 0) + 0.5 * rec), max: T.pts_search });   // 화재층 실이 1곳뿐이면 순위 상관 대신 포함 여부
    const sec = (performance.now() - S.t0) / 1000;
    const ft = Math.max(0, Math.min(1, (T.time_zero_s - sec) / (T.time_zero_s - T.time_full_s)));
    items.push({ key: 'time', label: '결정 시간', value: `${sec.toFixed(0)}초`, ref: `${T.time_full_s}초 이내 만점`, metric: sec, pts: T.pts_time * ft, max: T.pts_time });
    // 공기호흡기 귀환 한계 위반(정보 — 점수 없음): 훈련자 경로로 왕복 + 안전여유 > 사용 가능 시간
    let scba = null;
    if (Lt != null && P.scba) {
      const tIn = (Lt / ctx.ACT.walk_mps + P.floorsUp * ctx.ACT.stair_s_per_floor) / 60;
      scba = { t_in_min: +tIn.toFixed(2), available_min: +P.scba.TA.toFixed(1), violation: 2 * tIn + ctx.ACT.scba_margin_min > P.scba.TA };
    }
    const tot = items.reduce((a, z) => a + z.pts, 0), mx = items.reduce((a, z) => a + z.max, 0);
    const rooms = roomsOf(P.fid);
    return {
      tool: 'FIRE-3D BEAM 훈련 모드 채점', version: 1, created: new Date().toISOString(), building: D.building.name,
      scenario: { kind: S.kind, floor: P.fid, floor_name: FL[P.fid].name, fire: [+P.x.toFixed(2), +P.y.toFixed(2)], mode: S.mode, started: S.started },
      answers: { entrance: e ? e[5] : null, stair: stair ? stair.name : null, hose: A.hose, search: A.order.map((k) => ({ room: k, area: rooms[k] ? rooms[k].area : null, c: rooms[k] ? rooms[k].c : null })) },
      reference: { entrance: P.entry ? P.entry.name : null, stair: P.attackName, route_m: +P.routeLen.toFixed(1), hose: P.hoseN, hose_m: P.hoseL != null ? +P.hoseL.toFixed(1) : null,
        search: refF.slice(0, Math.max(k, 1) + 4).map((z) => ({ rank: z.rank, room: z.k, cls: z.cls, cat: z.cat, d: z.d != null ? +z.d.toFixed(1) : null, reasons: z.reasons })) },
      items: items.map((z) => ({ ...z, pts: +z.pts.toFixed(1) })), scba, total: +tot.toFixed(1), max: mx, score100: mx ? +((100 * tot) / mx).toFixed(1) : 0, elapsed_s: +sec.toFixed(1), cfg: T,
      note: '산출안은 자동 변환 모델 위 계산값(임시 기준값) — 정답이 아니라 비교 기준입니다.',
    };
  }
  function resultHtml(R, full) {
    let h = `<table class="trf"><tr><td class="k">항목</td><td class="k">훈련자</td><td class="k">산출안</td><td class="k">점수</td></tr>`;
    for (const z of R.items) h += `<tr><td style="text-align:left">${esc(z.label)}</td><td>${esc(z.value)}</td><td style="text-align:left">${esc(z.ref)}</td><td><b>${z.pts}</b>/${z.max}</td></tr>`;
    h += `</table>`;
    if (R.scba) h += `<div class="note">공기호흡기(훈련자 경로): 진입 ${R.scba.t_in_min}분 × 2 + 안전여유 vs 사용 가능 ${R.scba.available_min}분 → ${R.scba.violation ? '<b style="color:#ff7b72">귀환 한계 위반</b>' : '여유 있음'}</div>`;
    if (full) h += `<div class="note">${esc(R.note)}</div>`;
    return h;
  }
  function submit() {
    readForm();
    S.picking = false;
    const R = score(); S.result = R; window.__train = Object.assign(window.__train || {}, { last: R });
    ctx.redraw();                                                // 숨김 해제 → 산출안 표시
    box.innerHTML = `<div class="trn-box"><div class="big">훈련 점수 <b>${R.score100}</b>/100</div><div class="k">${esc(R.scenario.floor_name)} · ${R.scenario.mode === 'night' ? '야간' : '주간'} · ${R.elapsed_s}초</div>${resultHtml(R, true)}
      <div class="row" style="margin-top:5px"><button id="trJson">결과 저장(JSON)</button><button id="trHtml">보고서 저장(HTML)</button><button id="trAgain">새 시나리오</button><button id="trQuit">종료</button></div></div>`;
  }
  function save(name, text, type) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  const stamp = () => new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  function report(R) {
    const css = 'body{font-family:"Malgun Gothic","Noto Sans KR",sans-serif;max-width:860px;margin:16px auto;padding:0 16px;font-size:13px}table{border-collapse:collapse;width:100%}td{border:1px solid #aaa;padding:3px 6px}.k{color:#555}.note{color:#555;font-size:12px}.big{font-size:20px}';
    const ref = R.reference.search.map((z) => `<tr><td>${z.rank}</td><td>${esc(CLASS_KO[z.cls] || z.cls)}</td><td>${esc(CAT_KO[z.cat] || z.cat)}</td><td>${z.d ?? '-'}m</td><td>${esc((z.reasons || []).join(' · '))}</td></tr>`).join('');
    return `<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>훈련 채점 보고서</title><style>${css}</style></head><body><h1>훈련 채점 보고서</h1>
<p>${esc(R.building)} · ${esc(R.scenario.floor_name)} 화재(${R.scenario.fire.join(', ')}) · ${R.scenario.mode === 'night' ? '야간' : '주간'} · ${esc(R.created)}</p><p class="big">점수 <b>${R.score100}</b>/100 (${R.total}/${R.max}) · 결정 시간 ${R.elapsed_s}초</p>
${resultHtml(R, true)}<h3>산출안 인명검색 순서(화재층)</h3><table><tr><td class="k">순위</td><td class="k">분류</td><td class="k">용도</td><td class="k">거리</td><td class="k">근거</td></tr>${ref}</table>
<p class="note">FIRE-3D BEAM 3D 지휘화면 훈련 모드에서 내보냄 — 배점·기준값은 임시값(교육 지침으로 설정). 개인 식별 정보 없음.</p></body></html>`;
  }
  function readForm() {
    if (!S || S.result) return;
    const g = (id) => $(id);
    if (g('trEnt')) S.ans.ent = g('trEnt').value === '' ? null : Number(g('trEnt').value);
    if (g('trSt')) S.ans.st = g('trSt').value || null;
    if (g('trHose')) S.ans.hose = g('trHose').value === '' ? null : Number(g('trHose').value);
  }
  box.addEventListener('change', readForm);
  box.addEventListener('click', (ev) => {
    const id = ev.target.id;
    if (id === 'trRnd') start('rnd'); else if (id === 'trCur') start('cur');
    else if (id === 'trPick') { readForm(); S.picking = !S.picking; if (S.picking && ctx.getSel() !== S.plan.fid) ctx.selectFloor(S.plan.fid); form(); }
    else if (id === 'trUndo') { readForm(); S.ans.order.pop(); redrawPicks(); form(); }
    else if (id === 'trGo') submit();
    else if (id === 'trQuit') quit();
    else if (id === 'trAgain') { const m = S.mode; quit(); $('trMode').value = m; start('rnd'); }
    else if (id === 'trJson') save(`훈련채점_${stamp()}.json`, JSON.stringify(S.result, null, 1), 'application/json');
    else if (id === 'trHtml') save(`훈련채점_${stamp()}.html`, report(S.result), 'text/html');
  });
  function redrawPicks() {
    G.clear(); if (!S || !S.plan) return;
    const R = roomsOf(S.plan.fid);
    S.ans.order.forEach((k, i) => { const r = R[k]; if (r) lab(`T${i + 1}`, 'trn', r.c[0], r.c[1], zOf(S.plan.fid) + SLAB + 2.0); });
  }
  // 캔버스 클릭(main.js) → 화재층 실 고르기. 처리했으면 true
  function pickAt(fid, x, y) {
    if (!S || !S.picking || S.result || !S.plan || fid !== S.plan.fid) return false;
    const R = roomsOf(fid);
    let k = R.findIndex((r) => inPoly(x, y, r.poly));
    if (k < 0) { let b = 1e9; R.forEach((r, i) => { const d = Math.hypot(r.c[0] - x, r.c[1] - y); if (d < b && d < 3) { b = d; k = i; } }); }
    if (k < 0 || S.ans.order.includes(k) || S.ans.order.length >= T.search_k) return true;
    S.ans.order.push(k); readForm(); redrawPicks(); form();
    if (S.ans.order.length >= T.search_k) { S.picking = false; form(); }
    return true;
  }
  menu();
  return {
    hidden: () => !!(S && !S.result), active: () => !!S, picking: () => !!(S && S.picking),
    maskHtml: () => `<div class="res"><div class="big">🔥 훈련 화재</div><div>산출안은 제출 후 공개됩니다. 아래 '훈련 모드' 칸에서 답을 정하세요.</div></div>`,
    onPlan: (P) => { if (!S) return; const first = !S.plan; S.plan = P; if (first && !S.result) form(); redrawPicks(); },
    pickAt, start, quit, submit, score: () => S && S.result, answers: () => (S ? JSON.parse(JSON.stringify(S.ans)) : null),
    // 시험용: 답 직접 넣기 (entIdx, stairId, hose, roomOrder[])
    answer: (ent, st, hose, order) => { if (!S) return; S.ans = { ent, st, hose, order: order.slice(0, T.search_k) }; redrawPicks(); form(); },
  };
}
