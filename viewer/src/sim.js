// 화재 확산 가정 시뮬레이션(예시 모델 — 임시값) — 화점 지정 후 0~15분 재생
// 모델(단순 구획 확산, 문서화된 가정):
//  · 마디 = 실(대피 분석의 자동 분할 실, 없으면 BIM 실) + 계단실·승강로(층마다 1마디) + 바깥.  가지 = 문(도면 검출) · 문 미검출 경계(맞닿은 실 — 닫힌 문 누설 가정) · 외벽 누설 · 계단실 문 · 승강로 위아래.
//  · 연기 지수 s(0~1) = 실 부피 대비 연기량. 화원 실은 성장(선형 → 최대) 연기 발생, 가지마다 전도도 C(㎥/s)로 진한 쪽 → 옅은 쪽 이동(계단·승강로는 위로 C_up, 아래로 C_dn = 굴뚝 효과).
//  · s ≥ 1 이면 넘친 양을 이웃으로 밀어냄. 불(화염)은 같은 층 이웃 실로 문 300초·문 미검출 경계 720초 뒤 번짐(가정).
//  · 진입·대피 곤란(untenable) = s ≥ 0.5 또는 불 붙은 지 30초.  실마다 '곤란 도달 시각(TTU)'으로 색칠.
//  · 시간대별 대피로 재평가: 곤란해진 실·계단을 막고, 실 연결망에서 피난계단(공격계단 제외 — 없으면 포함)·옥외 출구까지 닿는지 다시 찾음.
// 값은 fire3dbeam/sop.py SIM(→ D.sop_sim)에서 바꿈. 실제 연기 거동(CFAST·FDS 등 공학 모델)과 다르며 현장 판단을 대신하지 않음.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { flameTexture } from './hud.js';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export const SIM_DEFAULT = {
  _note: '화재 확산 가정 시뮬레이션 — 모두 예시·임시값(기관 SOP·공학 검토로 설정 필요)',
  t_max_min: 15, dt_s: 2, frame_s: 5,
  smoke_max_m3s: 6.0, grow_s: 240, untenable: 0.5, burn_untenable_s: 30,
  c_door: 0.5, c_wall: 0.12, c_ext: 0.04, c_extdoor: 0.3, c_stair_door: 0.15, c_stair_door_sp: 0.04, c_ev_door: 0.06,
  c_stack_up: 2.5, c_stack_dn: 0.25, c_ev_up: 1.5, c_ev_dn: 0.15,
  spread_door_s: 300, spread_wall_s: 720,
  dispatch_s: 60, drive_kmh: 30, default_arrive_s: 360, setup_s: 60, team2_delay_s: 90, rescue_delay_s: 30, search_dwell_s: 40,
};
export const TTU_BANDS = [[120, '#991b1b', '2분 이내'], [300, '#dc2626', '2~5분'], [600, '#f97316', '5~10분'], [Infinity, '#facc15', '10분 이후']];
const ttuCol = (t) => TTU_BANDS.find((b) => t <= b[0])[1];

export function initSim(ctx) {
  const { D, BIM, floors, FL, V, SLAB, H, WH, G, esc, $ } = ctx;
  const P = Object.assign({}, SIM_DEFAULT, D.sop_sim || {});
  const inPoly = (x, y, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  const ctrOf = (p) => p.reduce((a, q) => [a[0] + q[0] / p.length, a[1] + q[1] / p.length], [0, 0]);
  const areaOf = (p) => Math.abs(p.reduce((s, [x, y], i) => { const [x2, y2] = p[(i + 1) % p.length]; return s + x * y2 - x2 * y; }, 0) / 2);
  const inFloor = (f, x, y) => f.poly.some((r) => inPoly(x, y, r));

  // ═════════ 연결망(한 번 만듦)
  let N = null;
  function graph() {
    if (N) return N;
    const nodes = [], edges = [], byFloor = {}, shaftNode = {};
    const OUT = 0; nodes.push({ id: 'OUT', type: 'out', vol: Infinity });
    const addE = (a, b, C, Cba, kind, p) => edges.push({ a, b, C, Cba: Cba ?? C, kind, p });
    for (const f of floors) {
      const ev = D.evac && D.evac.floors && D.evac.floors[f.id];
      const src = ev ? ev.rooms.map((r) => ({ poly: r.poly, c: r.c, area: r.area, cat: r.cat || 'unk' }))
        : ((BIM.floors[f.id] || {}).spaces || []).map((s) => ({ poly: s.poly, c: ctrOf(s.poly), area: s.area, cat: 'unk' }));
      const ids = [];
      src.forEach((r, k) => {
        if (!r.poly || r.poly.length < 3) return;
        const xs = r.poly.map((q) => q[0]), ys = r.poly.map((q) => q[1]);
        const n = { id: `${f.id}:${k}`, type: 'room', fid: f.id, k, poly: r.poly, c: r.c || ctrOf(r.poly), area: r.area || areaOf(r.poly), cat: r.cat, bb: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] };
        n.vol = Math.max(8, n.area) * WH; ids.push(nodes.length); nodes.push(n);
      });
      byFloor[f.id] = ids;
    }
    const roomAt = (fid, x, y) => { for (const i of byFloor[fid] || []) { const n = nodes[i], b = n.bb; if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] && inPoly(x, y, n.poly)) return i; } return -1; };
    // 계단·승강로 마디(층마다)
    for (const [kind, S] of [...BIM.stairs.map((s) => ['stair', s]), ...BIM.elevs.map((e) => ['ev', e])]) {
      const fl = S.floors.filter((f) => FL[f]).sort((a, b) => FL[a].level - FL[b].level);
      const [x0, y0, x1, y1] = S.box, a = Math.max(2, (x1 - x0) * (y1 - y0));
      let prev = -1;
      for (const fid of fl) {
        const i = nodes.length; nodes.push({ id: `${S.id}@${fid}`, type: 'shaft', kind, S, fid, c: [(x0 + x1) / 2, (y0 + y1) / 2], area: a, vol: a * H, poly: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], cat: 'shaft' });
        shaftNode[`${S.id}@${fid}`] = i;
        if (prev >= 0) addE(prev, i, kind === 'ev' ? P.c_ev_up : P.c_stack_up, kind === 'ev' ? P.c_ev_dn : P.c_stack_dn, 'vert');
        prev = i;
        // 계단실에 닿은 실: 상자 둘레 바깥 0.5m 표본점 → 실 / 상자 중심을 품은 실(계단 홀)
        const sp = S.special && S.special.kind === 'sp', Cd = kind === 'ev' ? P.c_ev_door : sp ? P.c_stair_door_sp : P.c_stair_door;
        const touch = new Map();
        const hall = roomAt(fid, (x0 + x1) / 2, (y0 + y1) / 2); if (hall >= 0) touch.set(hall, [(x0 + x1) / 2, (y0 + y1) / 2]);
        const per = 2 * (x1 - x0 + y1 - y0), ns = Math.max(8, Math.round(per / 0.8));
        for (let k = 0; k < ns; k++) {
          let u = (k / ns) * per, px, py;
          if (u < x1 - x0) { px = x0 + u; py = y0 - 0.5; } else if ((u -= x1 - x0) < y1 - y0) { px = x1 + 0.5; py = y0 + u; } else if ((u -= y1 - y0) < x1 - x0) { px = x1 - u; py = y1 + 0.5; } else { u -= x1 - x0; px = x0 - 0.5; py = y1 - u; }
          const r = roomAt(fid, px, py); if (r >= 0 && !touch.has(r)) touch.set(r, [px, py]);
        }
        for (const [r, p] of touch) addE(r, i, r === hall ? Cd * 3 : Cd, null, kind === 'ev' ? 'evdoor' : 'stairdoor', p);
      }
    }
    // 실끼리: 문 → 문 가지 / 경계 표본 → 문 미검출 경계 · 외벽
    for (const f of floors) {
      const ids = byFloor[f.id]; if (!ids || !ids.length) continue;
      const pair = new Map(), key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
      for (const d of (BIM.floors[f.id] || {}).doors || []) {
        const [kind, o, x, y, w, , ext] = d, off = 0.6;
        const p1 = o === 'h' ? [x, y + off] : [x + off, y], p2 = o === 'h' ? [x, y - off] : [x - off, y];
        const a = roomAt(f.id, ...p1), b = roomAt(f.id, ...p2);
        if (a >= 0 && b >= 0 && a !== b) { const k = key(a, b); if (!pair.has(k)) { pair.set(k, 1); addE(a, b, P.c_door * Math.max(0.6, Math.min(2, w)), null, 'door', [x, y]); } }
        else if ((a >= 0) !== (b >= 0) && (ext || kind !== 'door')) addE(a >= 0 ? a : b, OUT, P.c_extdoor, P.c_extdoor, 'extdoor', [x, y]);
      }
      const wallCnt = new Map(), extCnt = new Map();
      for (const i of ids) {
        const n = nodes[i], ring = n.poly;
        for (let e = 0; e < ring.length; e++) {
          const [ax, ay] = ring[e], [bx, by] = ring[(e + 1) % ring.length], L = Math.hypot(bx - ax, by - ay); if (L < 0.4) continue;
          const nx = -(by - ay) / L, ny = (bx - ax) / L, m = Math.max(1, Math.round(L / 1.0));
          for (let s = 0; s < m; s++) {
            const t = (s + 0.5) / m, px = ax + (bx - ax) * t, py = ay + (by - ay) * t;
            for (const sg of [1, -1]) {
              const qx = px + sg * nx * 0.45, qy = py + sg * ny * 0.45;
              if (inPoly(qx, qy, ring)) continue;
              const r = roomAt(f.id, qx, qy);
              if (r >= 0 && r !== i) { const k = key(i, r); const z = wallCnt.get(k) || { n: 0, p: [px, py], a: i, b: r }; z.n++; wallCnt.set(k, z); }
              else if (r < 0 && !inFloor(f, qx, qy)) extCnt.set(i, (extCnt.get(i) || 0) + 1);
            }
          }
        }
      }
      for (const [k, z] of wallCnt) if (!pair.has(k) && z.n >= 2) addE(z.a, z.b, P.c_wall, null, 'wall', z.p);
      for (const [i, c] of extCnt) addE(i, OUT, P.c_ext * Math.min(4, c / 4), P.c_ext * Math.min(4, c / 4), 'ext');
    }
    const adj = nodes.map(() => []);
    edges.forEach((e, k) => { adj[e.a].push(k); adj[e.b].push(k); });
    N = { nodes, edges, adj, byFloor, roomAt, shaftNode };
    return N;
  }

  // ═════════ 계산(0 ~ t_max, dt 마다, frame_s 마다 저장)
  let R = null;
  function run(fire) {
    const g = graph(), { nodes, edges, adj } = g, n = nodes.length;
    let src = g.roomAt(fire.fid, fire.x, fire.y);
    if (src < 0) { let bd = Infinity; for (const i of g.byFloor[fire.fid] || []) { const c = nodes[i].c, d = Math.hypot(c[0] - fire.x, c[1] - fire.y); if (d < bd) { bd = d; src = i; } } }
    const T = P.t_max_min * 60, dt = P.dt_s, nf = Math.floor(T / P.frame_s) + 1;
    const s = new Float64Array(n), sv = new Float64Array(n), tIgn = new Float64Array(n).fill(Infinity), ttu = new Float64Array(n).fill(Infinity);
    const frames = new Float32Array(nf * n);
    if (src >= 0) tIgn[src] = 0;
    const vol = nodes.map((z) => z.vol);
    let fi = 0;
    for (let t = 0; t <= T + 1e-6; t += dt) {
      // 저장
      if (t >= fi * P.frame_s - 1e-6 && fi < nf) { frames.set(Float32Array.from(s), fi * n); fi++; }
      // 화원 연기
      for (let i = 1; i < n; i++) if (tIgn[i] <= t) sv[i] += P.smoke_max_m3s * Math.min(1, (t - tIgn[i] + dt) / P.grow_s) * dt;
      // 이동
      const d = new Float64Array(n);
      for (const e of edges) {
        const sa = e.a === 0 ? 0 : sv[e.a] / vol[e.a], sb = e.b === 0 ? 0 : sv[e.b] / vol[e.b], df = sa - sb;
        let q = df > 0 ? e.C * df * dt : e.Cba * df * dt;
        if (q > 0) q = Math.min(q, 0.45 * sv[e.a]); else q = Math.max(q, -0.45 * (e.b === 0 ? 0 : sv[e.b]));
        d[e.a] -= q; d[e.b] += q;
      }
      for (let i = 1; i < n; i++) sv[i] = Math.max(0, sv[i] + d[i]);
      // 넘침 → 이웃(전도도 비례)
      for (let i = 1; i < n; i++) {
        const ex = sv[i] - vol[i]; if (ex <= 0) continue;
        let cs = 0; for (const k of adj[i]) { const e = edges[k]; cs += e.a === i ? e.C : e.Cba; }
        if (cs <= 0) { sv[i] = vol[i]; continue; }
        for (const k of adj[i]) { const e = edges[k], j = e.a === i ? e.b : e.a, c = e.a === i ? e.C : e.Cba; if (j !== 0) sv[j] += (ex * c) / cs; }
        sv[i] = vol[i];
      }
      for (let i = 1; i < n; i++) s[i] = Math.min(1, sv[i] / vol[i]);
      // 불 번짐(같은 층 실끼리)
      for (let i = 1; i < n; i++) {
        if (!(tIgn[i] <= t) || nodes[i].type !== 'room') continue;
        for (const k of adj[i]) {
          const e = edges[k], j = e.a === i ? e.b : e.a; if (j === 0 || nodes[j].type !== 'room' || tIgn[j] < Infinity) continue;
          const tt = e.kind === 'door' ? P.spread_door_s : e.kind === 'wall' ? P.spread_wall_s : Infinity;
          if (t - tIgn[i] >= tt) tIgn[j] = t + dt;
        }
      }
      for (let i = 1; i < n; i++) if (ttu[i] === Infinity && (s[i] >= P.untenable || t - tIgn[i] >= P.burn_untenable_s)) ttu[i] = t;
    }
    R = { fire, src, frames, nf, n, ttu, tIgn, T };
    evacCache = new Map();
    return R;
  }
  const sAt = (t) => {                                              // 시각 t 의 연기 지수(프레임 사이 보간)
    const f = Math.min(R.nf - 1, t / P.frame_s), a = Math.floor(f), b = Math.min(R.nf - 1, a + 1), w = f - a, out = new Float32Array(R.n);
    for (let i = 0; i < R.n; i++) out[i] = R.frames[a * R.n + i] * (1 - w) + R.frames[b * R.n + i] * w;
    return out;
  };
  const bad = (i, t) => R.ttu[i] <= t;

  // ═════════ 시각 t 대피로 재평가(실 연결망, 곤란 마디 차단)
  let evacCache = new Map();
  function evacAt(t, fid, attackId) {
    const g = graph(), { nodes, edges, adj } = g;
    const blk = (g.byFloor[fid] || []).map((i) => (bad(i, t) ? 1 : 0)).join('') + '|' + Object.keys(g.shaftNode).filter((k) => k.endsWith('@' + fid)).map((k) => (bad(g.shaftNode[k], t) ? 1 : 0)).join('');
    const ck = fid + '|' + blk + '|' + attackId;
    if (evacCache.has(ck)) return evacCache.get(ck);
    const ground = (floors.find((f) => f.level === 0) || floors[0]).id;
    const shafts = Object.entries(g.shaftNode).filter(([k]) => k.endsWith('@' + fid)).map(([, i]) => i).filter((i) => nodes[i].kind === 'stair');
    let tg = shafts.filter((i) => !bad(i, t) && nodes[i].S.id !== attackId && nodes[i].S.floors.includes(ground));
    let shared = false;
    if (!tg.length) { tg = shafts.filter((i) => !bad(i, t) && nodes[i].S.floors.includes(ground)); shared = tg.length > 0; }
    const goals = new Set(tg); if (fid === ground) goals.add(0);
    // 목표에서 거꾸로 BFS(가지 길이 = 실 중심 사이 거리)
    const dist = new Map(), via = new Map(), Q = [];
    for (const i of goals) { dist.set(i, 0); Q.push(i); }
    while (Q.length) {
      Q.sort((a, b) => dist.get(a) - dist.get(b)); const u = Q.shift();
      for (const k of adj[u]) {
        const e = edges[k], v = e.a === u ? e.b : e.a;
        if (u === 0 && e.kind !== 'extdoor') continue;                 // 바깥으로는 외부 출입문으로만(외벽 누설은 대피로 아님)
        if (v === 0 || nodes[v].type !== 'room' || nodes[v].fid !== fid || bad(v, t)) continue;
        if (nodes[u].type === 'room' && nodes[u].fid !== fid) continue;
        const cu = u === 0 ? (e.p || nodes[v].c) : nodes[u].c, w = Math.hypot(nodes[v].c[0] - cu[0], nodes[v].c[1] - cu[1]) + (e.kind === 'wall' ? 3 : 0);
        const nd = dist.get(u) + w;
        if (!dist.has(v) || nd < dist.get(v)) { dist.set(v, nd); via.set(v, [u, e]); Q.push(v); }
      }
    }
    const rooms = (g.byFloor[fid] || []).map((i) => {
      const st = bad(i, t) ? 'smoke' : dist.has(i) ? 'ok' : 'cut';
      let path = null;
      if (st === 'ok') { path = [nodes[i].c]; let u = i, guard = 0; while (via.has(u) && guard++ < 200) { const [p, e] = via.get(u); if (e.p) path.push(e.p); if (p !== 0) path.push(nodes[p].c); u = p; } }
      return { i, st, path, d: dist.get(i) };
    });
    const out = { t, fid, rooms, shared, nOk: rooms.filter((z) => z.st === 'ok').length, nCut: rooms.filter((z) => z.st === 'cut').length, nSmoke: rooms.filter((z) => z.st === 'smoke').length, exits: tg.map((i) => nodes[i].S.name) };
    evacCache.set(ck, out);
    return out;
  }

  // ═════════ 그리기(층마다 바닥 색 1장 + 연기 부피 1장, 꼭짓점 속성만 바꿈)
  const VIS = {};
  function floorVis(fid) {
    if (VIS[fid]) return VIS[fid];
    const g = graph(), ids = (g.byFloor[fid] || []).concat(Object.entries(g.shaftNode).filter(([k]) => k.endsWith('@' + fid)).map(([, i]) => i));
    if (!ids.length || !G[fid]) return null;
    const pg = [], vg = [], pr = [], vr = [];
    for (const i of ids) {
      const n = g.nodes[i], sh = new THREE.Shape(n.poly.map(([x, y]) => new THREE.Vector2(x, y)));
      const a = new THREE.ShapeGeometry(sh); a.rotateX(-Math.PI / 2); a.translate(0, SLAB + 0.13, 0); const an = a.index ? a.toNonIndexed() : a; pg.push(an); pr.push([i, an.attributes.position.count]);
      const b = new THREE.ExtrudeGeometry(sh, { depth: 1, bevelEnabled: false, curveSegments: 1 }); b.rotateX(-Math.PI / 2); const bn = b.index ? b.toNonIndexed() : b; bn.deleteAttribute('uv'); vg.push(bn); vr.push([i, bn.attributes.position.count]);
    }
    const mk = (gs, rr, mat) => {
      const geo = mergeGeometries(gs.map((q) => { q.deleteAttribute('uv'); return q; }));
      const cnt = geo.attributes.position.count, col = new Float32Array(cnt * 4), idx = new Int32Array(cnt);
      let o = 0; for (const [i, c] of rr) { idx.fill(i, o, o + c); o += c; }
      geo.setAttribute('color', new THREE.BufferAttribute(col, 4)); geo.userData.idx = idx;
      const m = new THREE.Mesh(geo, mat); m.userData.own = true; m.frustumCulled = false; return m;
    };
    const patch = mk(pg, pr, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    patch.renderOrder = 7;
    const smat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    const top = SLAB + WH - 0.05;
    smat.onBeforeCompile = (sh) => {   // 연기층: 천장에서 아래로 두께 = s × 층 높이(꼭짓점 색 알파 = 표시 여부, 색 a 채널 대신 color.a 사용)
      sh.vertexShader = sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>\n float th = clamp(color.a * 4.0, 0.0, 1.0) * ${(WH * 0.92).toFixed(3)};\n transformed.y = ${top.toFixed(3)} - (1.0 - position.y) * max(th, 0.02);`);
      sh.fragmentShader = sh.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\n diffuseColor.a = vColor.a > 0.004 ? 0.1 + vColor.a * 1.25 : 0.0;');
    };
    smat.customProgramCacheKey = () => 'smokeV3' + top.toFixed(2);
    const smoke = mk(vg, vr, smat); smoke.renderOrder = 8;
    const grp = new THREE.Group(); grp.userData.own = true; grp.add(patch, smoke); G[fid].g.add(grp);
    const flames = new THREE.Group(); grp.add(flames);
    VIS[fid] = { grp, patch, smoke, flames, ids, fl: {} };
    return VIS[fid];
  }
  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  let shownT = -1, active = false;
  function show(t, opt = {}) {
    shownT = t;
    if (!R) return;
    const S = sAt(t), g = graph();
    for (const f of floors) {
      const anyS = (g.byFloor[f.id] || []).some((i) => S[i] > 0.02 || bad(i, t));
      if (!anyS && !VIS[f.id]) continue;
      const v = floorVis(f.id); if (!v) continue;
      v.grp.visible = active && opt.smoke !== false;
      // 바닥: 곤란 도달 시각 색(도달한 실만) · 연기 조금 = 회색
      const pc = v.patch.geometry.attributes.color, pi = v.patch.geometry.userData.idx;
      for (let k = 0; k < pi.length; k++) {
        const i = pi[k], tt = R.ttu[i];
        if (tt <= t) { const [r, gg, b] = hex(ttuCol(tt)); pc.setXYZW(k, r, gg, b, g.nodes[i].type === 'shaft' ? 0.75 : 0.5); }
        else if (S[i] > 0.04) pc.setXYZW(k, 0.6, 0.62, 0.66, Math.min(0.35, S[i]));
        else pc.setXYZW(k, 0, 0, 0, 0);
      }
      pc.needsUpdate = true;
      const sc = v.smoke.geometry.attributes.color, si = v.smoke.geometry.userData.idx;
      for (let k = 0; k < si.length; k++) { const i = si[k], q = S[i] > 0.03 ? S[i] : 0, dk = 0.46 - 0.24 * q; sc.setXYZW(k, dk, dk, dk + 0.02, q * 0.25); }   // 진할수록 어둡고 두꺼운 연기층
      sc.needsUpdate = true;
      // 불꽃: 불 붙은 실
      for (const i of v.ids) {
        const on = R.tIgn[i] <= t;
        if (on && !v.fl[i]) {
          const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTexture(), transparent: true, depthWrite: false })); const c = g.nodes[i].c;
          sp.center.set(0.5, 0.05); sp.position.copy(V(c[0], c[1], SLAB + 0.1)); sp.renderOrder = 30; sp.userData.flame = true; v.flames.add(sp); v.fl[i] = sp;
        }
        if (v.fl[i]) { v.fl[i].visible = on; if (on) { const a = Math.min(1, (t - R.tIgn[i]) / 120 + 0.3); v.fl[i].scale.set(2.6 * a, 3.4 * a, 1); } }
      }
    }
  }
  function setActive(on) { active = on; for (const v of Object.values(VIS)) v.grp.visible = on; }
  function clear() { for (const v of Object.values(VIS)) { v.grp.parent && v.grp.parent.remove(v.grp); } for (const k of Object.keys(VIS)) delete VIS[k]; R = null; active = false; shownT = -1; }

  // ═════════ 요약
  function summary(t, attackId) {
    if (!R) return null;
    const g = graph(), fid = R.fire.fid, fmt = (s) => (isFinite(s) ? `${Math.floor(s / 60)}분 ${String(Math.round(s % 60)).padStart(2, '0')}초` : '15분 내 없음');
    const rooms = g.nodes.map((n, i) => ({ n, i })).filter((z) => z.n.type === 'room');
    const hit = rooms.filter((z) => R.ttu[z.i] <= t), burn = rooms.filter((z) => R.tIgn[z.i] <= t);
    const floorsHit = [...new Set(hit.map((z) => z.n.fid))].sort((a, b) => FL[a].level - FL[b].level);
    const atk = attackId ? g.shaftNode[`${attackId}@${fid}`] : null;
    const firstUp = rooms.filter((z) => FL[z.n.fid].level > FL[fid].level).reduce((m, z) => Math.min(m, R.ttu[z.i]), Infinity);
    const ev = evacAt(t, fid, attackId);
    const allCut = (() => { for (let tt = 0; tt <= R.T; tt += 30) { const e = evacAt(tt, fid, attackId); if (e.nOk === 0 && e.rooms.length) return tt; } return Infinity; })();
    const top = rooms.filter((z) => isFinite(R.ttu[z.i])).sort((a, b) => R.ttu[a.i] - R.ttu[b.i]).slice(0, 10);
    return { t, fid, nHit: hit.length, nBurn: burn.length, floorsHit, atkTTU: atk != null ? R.ttu[atk] : null, firstUp, allCut, ev, top: top.map((z) => ({ fid: z.n.fid, cat: z.n.cat, area: z.n.area, c: z.n.c, ttu: R.ttu[z.i], burn: R.tIgn[z.i] })), fmt,
      shafts: Object.entries(g.shaftNode).filter(([k]) => k.endsWith('@' + fid)).map(([k, i]) => ({ name: g.nodes[i].S.name, kind: g.nodes[i].kind, ttu: R.ttu[i] })) };
  }
  return { run, show, setActive, clear, summary, evacAt, graph, P, last: () => R, nodes: () => graph().nodes, isActive: () => active, shownT: () => shownT, ttuCol };
}
