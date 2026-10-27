// 대피로(초록) · 진입로(빨강) — fire3dbeam/evac.py 결과(D.evac) 표시 + 화점 지정 시 재계산(화점 구역·공격계단 회피)
// main.js 에서 initEvac(ctx) 로 연결. 경로 계산은 main.js 의 0.25 m 격자 Dijkstra(gridFor/dijkstra/pathTo)를 그대로 씀.
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const C = { ok: 0x2ea043, mid: 0xd29922, high: 0xf0883e, none: 0x8b949e, path: 0x3fb950, entry: 0xff4d4f, entry2: 0xff9e9e, iso: 0xa371f7, near: 0xf0883e, fire: 0xff3b30 };

export function initEvac(ctx) {
  const { D, BIM, scene, G, floors, FL, V, SLAB, esc, $, zOf, getSel, getViewMode, camera, canvas, gridFor, dijkstra, distAt, pathTo } = ctx;
  const EV = D.evac;
  if (!EV || !EV.floors) { const s = $('evSec'); if (s) s.style.display = 'none'; return null; }
  const ground = EV.ground;
  const ko = (fid) => (FL[fid] ? FL[fid].name : fid);
  const stName = (id) => { const k = Number(String(id).slice(2)) - 1; return BIM.stairs[k] ? BIM.stairs[k].name : id; };
  const tgName = (fid, id) => { if (!id) return '-'; if (id.startsWith('ST')) return stName(id); const t = (EV.floors[fid].targets || []).find((x) => x.id === id); return t ? t.name : id; };
  const lab = (html, cls, x, y, z, parent) => {
    const el = document.createElement('div'); el.className = 'lb ' + cls; el.innerHTML = html;
    const o = new CSS2DObject(el); o.position.copy(V(x, y, z)); parent.add(o); return o;
  };
  // 경로 → 관(tube) + 화살촉(원뿔) 형상 (같은 칸·같은 방향 화살촉은 한 번만)
  function arrowGeoms(pts, z, r, seen, step = 4.5) {
    const out = [];
    if (pts.length < 2) return out;
    const v3 = pts.map(([x, y]) => V(x, y, z));
    const path = new THREE.CurvePath(); for (let k = 0; k < v3.length - 1; k++) path.add(new THREE.LineCurve3(v3[k], v3[k + 1]));
    out.push(new THREE.TubeGeometry(path, Math.max(4, v3.length * 3), r, 5, false).toNonIndexed());
    let acc = step * 0.5;
    for (let k = 0; k < pts.length - 1; k++) {
      const [x0, y0] = pts[k], [x1, y1] = pts[k + 1], L = Math.hypot(x1 - x0, y1 - y0); if (L < 1e-3) continue;
      const ux = (x1 - x0) / L, uy = (y1 - y0) / L;
      for (; acc < L; acc += step) {
        const x = x0 + ux * acc, y = y0 + uy * acc;
        const key = `${Math.round(x / 2.2)},${Math.round(y / 2.2)},${Math.round(Math.atan2(uy, ux) / (Math.PI / 4))}`;
        if (seen && seen.has(key)) continue; if (seen) seen.add(key);
        const cg = new THREE.ConeGeometry(r * 3.2, r * 7, 8); cg.rotateZ(-Math.PI / 2);           // +x 방향
        cg.rotateY(Math.atan2(uy, ux)); cg.translate(x, z, -y);
        out.push(cg.toNonIndexed());
      }
      acc -= L;
    }
    return out;
  }
  function mesh(geoms, color, op = 0.95) {
    if (!geoms.length) return null;
    const g = mergeGeometries(geoms.map((q) => { const a = q.index ? q.toNonIndexed() : q; ['uv', 'normal'].forEach((n) => a.deleteAttribute(n)); return a; }));
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false })); m.renderOrder = 6; return m;
  }
  const roomCol = (r) => (r.over === 'high' ? C.high : r.over === 'mid' ? C.mid : r.over === 'none' ? C.none : C.ok);
  function patch(r, color, op, z, parent, info) {
    const sh = new THREE.Shape(r.poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ShapeGeometry(sh); g.rotateX(-Math.PI / 2); g.translate(0, z, 0);
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false, side: THREE.DoubleSide }));
    m.renderOrder = 4; m.userData = info; parent.add(m); return m;
  }

  // ───────── 정적 층(화점 없음): 실 색 + 대피 화살표, 피난층 계단→옥외 출구, 진입로
  const SG = {}, EG = {}, hov = [];
  for (const f of floors) {
    const rec = EV.floors[f.id]; if (!rec || !G[f.id]) continue;
    const g = new THREE.Group(); G[f.id].g.add(g); SG[f.id] = g;
    const seen = new Set(), geo = [], legs = [];
    for (const r of rec.rooms) {
      hov.push(patch(r, roomCol(r), r.over ? 0.42 : 0.2, SLAB + 0.1, g, { fid: f.id, r }));
      if (r.path && r.path.length > 1) geo.push(...arrowGeoms(r.path, SLAB + 0.35, 0.09, seen));
      if (r.over === 'high') lab(`${r.dmax.toFixed(0)}m`, 'evo', r.c[0], r.c[1], SLAB + 0.9, g);
    }
    for (const [sid, L] of Object.entries(rec.stair_exit || {})) legs.push(...arrowGeoms(L.path, SLAB + 0.4, 0.14, seen));
    const m1 = mesh(geo, C.path); if (m1) g.add(m1);
    const m2 = mesh(legs, 0x56d364); if (m2) g.add(m2);
    for (const t of rec.targets) if (t.kind === 'exit') lab('대피 출구', 'evx', t.x, t.y, SLAB + 2.4, g);
  }
  // 진입로: 진입구마다 가까운 계단 1순위(굵게)·2순위(가늘게)
  if (G[ground]) {
    const g = new THREE.Group(); G[ground].g.add(g); EG[ground] = g;
    const a = [], b = [];
    for (const e of EV.entry || []) {
      const [r1, r2] = e.routes;
      if (r1) { a.push(...arrowGeoms(r1.path, SLAB + 0.6, 0.13, null, 5)); lab(`진입 ${e.id.slice(1)} → ${esc(r1.name)} ${r1.d.toFixed(0)}m`, 'ent', e.x, e.y, SLAB + 1.6, g); }
      if (r2) b.push(...arrowGeoms(r2.path, SLAB + 0.55, 0.06, null, 6));
    }
    const m1 = mesh(a, C.entry); if (m1) g.add(m1);
    const m2 = mesh(b, C.entry2, 0.7); if (m2) g.add(m2);
  }

  // ───────── 화점 지정 시 재계산
  const FG = new THREE.Group(); scene.add(FG);
  let fireOn = false, lastSt = null, simOn = false;            // simOn: 확산 시뮬레이션 재생 중(시간대별 대피로가 대신 표시)
  const near = (pts, x, y) => { let d = Infinity; for (let k = 0; k < pts.length - 1; k++) d = Math.min(d, segD(x, y, pts[k], pts[k + 1])); return d; };
  function segD(x, y, [ax, ay], [bx, by]) { const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy; const t = L ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L)) : 0; return Math.hypot(x - ax - t * dx, y - ay - t * dy); }
  function overlap(path, ref, tol = 1.2) {      // path 가 ref(진입 경로) 1.2 m 안을 지나는 길이
    if (!ref || ref.length < 2) return 0; let s = 0;
    for (let k = 0; k < path.length - 1; k++) {
      const [ax, ay] = path[k], [bx, by] = path[k + 1], L = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.ceil(L / 0.5));
      for (let i = 0; i < n; i++) { const t = (i + 0.5) / n; if (near(ref, ax + (bx - ax) * t, ay + (by - ay) * t) < tol) s += L / n; }
    }
    return s;
  }
  function penalize(gr, ref, fac, rad) {           // 진입(공격) 경로 주변 칸 비용 ↑ → 대피 경로가 피해 감
    if (!ref || ref.length < 2) return;
    const RES = 0.25;
    for (let k = 0; k < ref.length - 1; k++) {
      const [ax, ay] = ref[k], [bx, by] = ref[k + 1];
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - rad - gr.x0) / RES)), i1 = Math.min(gr.W - 1, Math.ceil((Math.max(ax, bx) + rad - gr.x0) / RES));
      const j0 = Math.max(0, Math.floor((gr.y1 - Math.max(ay, by) - rad) / RES)), j1 = Math.min(gr.H - 1, Math.ceil((gr.y1 - Math.min(ay, by) + rad) / RES));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const v = j * gr.W + i; if (gr.cost[v] >= 80 || !isFinite(gr.cost[v]) || gr.mark[v]) continue;
        if (segD(gr.x0 + i * RES, gr.y1 - j * RES, ref[k], ref[k + 1]) < rad) { gr.cost[v] *= fac; gr.mark[v] = 1; }
      }
    }
  }
  // 대피 격자: 뷰어 격자(계단·승강기 둘레 벽 통과 6)를 복사한 뒤 승강로 둘레는 벽으로 되돌리고, 계단은 문 쪽(비상구 표지 방향) 변만 통과 허용
  const clone = (gr, fid) => {
    const g2 = { ...gr, cost: Float32Array.from(gr.cost), mark: new Uint8Array(gr.W * gr.H) };
    const sides = (EV.floors[fid] || {}).stair_sides || {};
    for (const S of [...BIM.stairs.map((s) => [s, sides[s.id]]), ...BIM.elevs.map((e) => [e, 'X'])]) {
      const [o, sd] = S; if (!o.floors.includes(fid) || sd == null) continue;
      const [x0, y0, x1, y1] = o.box, p = 0.6;
      const keep = { W: [x0 - p, y0 - p, x0 + p, y1 + p], E: [x1 - p, y0 - p, x1 + p, y1 + p], S: [x0 - p, y0 - p, x1 + p, y0 + p], N: [x0 - p, y1 - p, x1 + p, y1 + p] }[sd];
      const i0 = Math.max(0, Math.floor((x0 - p - g2.x0) / 0.25)), i1 = Math.min(g2.W - 1, Math.ceil((x1 + p - g2.x0) / 0.25));
      const j0 = Math.max(0, Math.floor((g2.y1 - y1 - p) / 0.25)), j1 = Math.min(g2.H - 1, Math.ceil((g2.y1 - y0 + p) / 0.25));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const v = j * g2.W + i; if (g2.cost[v] !== 6) continue;
        const px = g2.x0 + i * 0.25, py = g2.y1 - j * 0.25;
        if (!keep || !(px >= keep[0] && px <= keep[2] && py >= keep[1] && py <= keep[3])) g2.cost[v] = 80;
      }
    }
    return g2;
  };
  const inPoly = (x, y, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  const area = (p) => Math.abs(p.reduce((s, [x, y], i) => { const [x2, y2] = p[(i + 1) % p.length]; return s + x * y2 - x2 * y; }, 0) / 2);

  function onFire(F) {
    FG.clear(); fireOn = !!F; lastSt = null; applyVis();
    if (!F) return '';
    const { fid, x, y, attack, atkFloor, atkGround, ents } = F;
    const rec = EV.floors[fid]; if (!rec) return '';
    const fz = zOf(fid) + SLAB, gz = zOf(ground) + SLAB;
    const RF = 6.0;                                             // 화점 구역 반경(가정값 — 기관 SOP 로 조정)
    const elong = (p) => { const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]); const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys); return Math.max(w, h) / Math.max(0.1, Math.min(w, h)); };
    const froom = rec.rooms.find((r) => inPoly(x, y, r.poly) && elong(r.poly) < 4);    // 복도(가늘고 긴 실)는 통째로 막지 않음
    const gr = clone(gridFor(fid), fid);
    const blockFire = (g2) => {
      for (let j = 0; j < g2.H; j++) for (let i = 0; i < g2.W; i++) {
        const px = g2.x0 + i * 0.25, py = g2.y1 - j * 0.25;
        if (Math.hypot(px - x, py - y) < RF || (froom && froom.area <= 400 && inPoly(px, py, froom.poly))) g2.cost[j * g2.W + i] = Infinity;
      }
    };
    blockFire(gr);
    penalize(gr, atkFloor, 3, 1.5);
    const cand = BIM.stairs.filter((s) => s.floors.includes(fid) && s.floors.includes(ground));
    let tg = cand.filter((s) => !attack || s.id !== attack.id);
    const shared = !tg.length && cand.length;
    if (shared) tg = cand;
    const ctr = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
    const srcs = fid === ground ? (BIM.floors[ground].doors || []).filter((d) => d[6]).map((d) => [d[2], d[3]]) : tg.map((s) => ctr(s.box));
    const R = srcs.length ? dijkstra(gr, srcs) : null;
    // 피난계단으로 못 가는 실: 공격계단까지는 갈 수 있나(공용 → 진입대와 충돌) / 어디로도 못 감(고립)
    const RA = attack && fid !== ground && !shared ? dijkstra(gr, cand.map((s) => ctr(s.box))) : null;
    const res = [];
    const geo = [], seen = new Set();
    for (const r of rec.rooms) {
      const inFire = (froom && r === froom) || Math.hypot(r.c[0] - x, r.c[1] - y) < RF;
      let st = 'ok', P = null, d = Infinity;
      if (inFire) st = 'fire';
      else {
        d = R ? distAt(gr, R, r.c[0], r.c[1]) : Infinity;
        if (!isFinite(d)) {
          const da = RA ? distAt(gr, RA, r.c[0], r.c[1]) : Infinity;
          if (isFinite(da)) { P = pathTo(gr, RA, r.c[0], r.c[1]); st = P.wall && !r.wall ? 'iso' : 'shared'; if (st === 'iso') P = null; }
          else st = 'iso';
        } else {
          P = pathTo(gr, R, r.c[0], r.c[1]);
          const oldNear = r.path && r.path.length > 1 ? near(r.path, x, y) : 99;
          if (P.wall && !r.wall) {                     // 벽을 뚫어야만 피난계단에 닿음 → 공격계단 공용 또는 고립
            const PA = RA && isFinite(distAt(gr, RA, r.c[0], r.c[1])) ? pathTo(gr, RA, r.c[0], r.c[1]) : null;
            if (PA && !PA.wall) { P = PA; st = 'shared'; } else { P = null; st = 'iso'; }
          } else if (near(P, x, y) < RF + 4 || oldNear < RF + 1) st = 'near';
          else if (r.d != null && P.len - r.d > 10) st = 'detour';
        }
      }
      res.push({ r, st, P, d: P ? P.len : null });
    }
    // 그리기
    let ov = 0, ovN = 0;
    for (const z of res) {
      const col = z.st === 'fire' ? C.fire : z.st === 'iso' ? C.iso : z.st === 'near' || z.st === 'shared' ? C.near : z.st === 'detour' ? C.mid : C.ok;
      patch(z.r, col, z.st === 'ok' ? 0.2 : 0.5, fz + 0.1, FG, { fid, r: z.r, fire: z });
      if (z.P && z.P.length > 1) {
        geo.push(...arrowGeoms(z.P, fz + 0.35, 0.1, seen));
        const o = overlap(z.P, atkFloor); if (o > 1) { ov += o; ovN++; }
      }
    }
    const m = mesh(geo, C.path); if (m) FG.add(m);
    // 피난층: 피난계단 → 옥외 출구 (진입 경로 회피)
    let gOv = 0, gHtml = '';
    if (fid !== ground && G[ground]) {
      const gg = clone(gridFor(ground), ground); penalize(gg, atkGround, 3, 1.5);
      const exits = (BIM.floors[ground].doors || []).filter((d) => d[6]).map((d) => [d[2], d[3]]);
      if (exits.length) {
        const RG = dijkstra(gg, exits), lg = [];
        for (const s of tg) {
          const P = pathTo(gg, RG, ...ctr(s.box)); lg.push(...arrowGeoms(P, gz + 0.4, 0.14, null)); gOv += overlap(P, atkGround);
          gHtml += `${esc(s.name)} → 옥외 ${P.len.toFixed(0)}m · `;
          const vv = [V(...ctr(s.box), gz + 0.4), V(...ctr(s.box), fz + 0.4)];
          const tb = new THREE.Mesh(new THREE.TubeGeometry(new THREE.LineCurve3(vv[0], vv[1]), 4, 0.2, 6), new THREE.MeshBasicMaterial({ color: C.path, transparent: true, opacity: 0.8 }));
          FG.add(tb);
          lab(`대피 ↓ ${esc(s.name)}`, 'evac', ...ctr(s.box), fz + 5.2, FG);
        }
        const mg = mesh(lg, 0x56d364); if (mg) FG.add(mg);
      }
    }
    // 진입 대안: 진입구 × 계단 (1층 이동 + 화재층 계단→화점)
    let alt = '';
    if (ents && ents.length && fid !== ground) {
      const grG = gridFor(ground), grF = gridFor(fid), RFd = dijkstra(grF, [[x, y]]);
      const rows = [];
      for (const [k, e] of ents.entries()) {
        const RE = dijkstra(grG, [[e[2], e[3]]]);
        for (const s of cand) {
          const c = ctr(s.box), a = distAt(grG, RE, ...c), b = distAt(grF, RFd, ...c);
          if (isFinite(a) && isFinite(b)) rows.push({ k, e, s, a, b, t: a + b, RE });
        }
      }
      rows.sort((p, q) => p.t - q.t);
      const byEnt = new Map(); for (const z of rows) if (!byEnt.has(z.k)) byEnt.set(z.k, z);
      const lst = [...byEnt.values()].sort((p, q) => p.t - q.t);
      const ag = [];
      lst.forEach((z, i) => {
        if (i === 0) return;                         // 1순위는 main.js 공격 경로(노랑·주황)
        const P = pathTo(grG, z.RE, ...ctr(z.s.box)); ag.push(...arrowGeoms(P.slice().reverse(), gz + 0.7, 0.07, null, 6));
        lab(`대안 진입 ${i + 1}`, 'ent', z.e[2], z.e[3], gz + 2.6, FG);
      });
      const ma = mesh(ag, C.entry2, 0.8); if (ma) FG.add(ma);
      alt = `<div style="margin-top:4px"><b>진입 대안</b> (진입구별 최선 계단, 1층 이동 + 화재층 계단→화점)</div><table>` +
        lst.slice(0, 4).map((z, i) => `<tr><td class="k">${i + 1}. ${esc(z.e[5]).replace(/ \(.*/, '')} ${i + 1 === 1 ? '(주)' : ''}</td><td>${esc(z.s.name)} · ${z.a.toFixed(0)}+${z.b.toFixed(0)}=<b>${z.t.toFixed(0)}m</b>${attack && z.s.id !== attack.id ? ' <span style="color:#ffd33d">피난계단 사용</span>' : ''}</td></tr>`).join('') + '</table>';
    }
    // 요약
    const cnt = (s) => res.filter((z) => z.st === s).length;
    const RK = { iso: 0, shared: 1, near: 2 }, TX = { iso: '고립', shared: '공격계단 공용', near: '화점 인접' };
    const pri = res.filter((z) => z.st in RK).sort((a, b) => RK[a.st] - RK[b.st] || Math.hypot(a.r.c[0] - x, a.r.c[1] - y) - Math.hypot(b.r.c[0] - x, b.r.c[1] - y));
    pri.slice(0, 8).forEach((z) => lab(TX[z.st], z.st === 'iso' ? 'evi' : 'evo', z.r.c[0], z.r.c[1], fz + 2.6, FG));     // 번호는 인명검색 우선순위(search.js)가 붙임
    lastSt = { fid, st: res.map((z) => z.st) };
    window.__evacFire = { res: res.map((z) => ({ st: z.st, d: z.d, c: z.r.c, area: z.r.area })), ov, gOv };
    let h = `<div class="evres"><div><b style="color:#7ee787">대피로 재계산</b> — 화점 반경 ${RF}m·화재실 차단, ${attack ? `공격계단 <b>${esc(attack.name)}</b> 제외` : '진입 경로 회피'}</div>`;
    if (shared) h += `<div style="color:#ffd33d">⚠ 이 층의 피난계단이 공격계단뿐 → 진입대와 대피자가 같은 계단 사용(유도 통제 필요)</div>`;
    h += `<div>실 ${res.length}: 정상 ${cnt('ok')} · 우회 증가 ${cnt('detour')} · <span style="color:#ffa657">화점 인접 경로 ${cnt('near')}</span> · <span style="color:#ffa657">공격계단 공용 ${cnt('shared')}</span> · <span style="color:#d2a8ff">고립(대피로 없음) ${cnt('iso')}</span> · 화재실 ${cnt('fire')}</div>`;
    if (pri.length) h += `<div>인명검색 우선순위 입력(대피 곤란, 고립→공용→인접 — 아래 '인명검색 우선순위' 표에 반영): ${pri.slice(0, 6).map((z) => `${TX[z.st]} ${z.r.area.toFixed(0)}㎡`).join(', ')}${pri.length > 6 ? ` 외 ${pri.length - 6}` : ''}</div>`;
    h += `<div>진입·대피 동선 겹침: 화재층 ${ov.toFixed(0)}m(${ovN}실 경로)${fid !== ground ? ` · 피난층 ${gOv.toFixed(0)}m` : ''} ${ov + gOv > 5 ? '<span style="color:#ffd33d">→ 교차 구간 유도 통제</span>' : ''}</div>`;
    if (gHtml) h += `<div class="note">피난층: ${gHtml.replace(/ · $/, '')}</div>`;
    h += alt + `<div class="note">초록 = 대피 방향, 빨강 계열 = 진입. 화점 반경·가중치는 가정값입니다.</div></div>`;
    return h;
  }

  // ───────── 보이기
  function applyVis() {
    const sel = getSel(), exp = getViewMode() === 'exp';
    const on = $('tEvR').checked, onE = $('tEntR').checked;
    for (const [fid, g] of Object.entries(SG)) g.visible = on && !fireOn && (fid === sel || exp);
    for (const [fid, g] of Object.entries(EG)) {
      g.visible = onE && !fireOn && (fid === sel || exp || sel == null);     // 전체 요약 화면: 진입로 화살표는 보임
      g.traverse((o) => { if (o.isCSS2DObject) o.visible = sel != null || exp; });   // 진입로 글자는 층을 골랐을 때만(전체 화면 글자 겹침 줄임)
    }
    FG.visible = (on || onE) && !(ctx.isHidden && ctx.isHidden()) && !simOn;
    panel();
  }
  $('tEvR').onchange = applyVis; $('tEntR').onchange = applyVis;

  // ───────── 옆 패널
  function panel() {
    const sel = getSel(), rec = EV.floors[sel];
    let h = `<div class="note" style="margin-top:0">${esc(EV.law.basis)} · ${esc(EV.law.note || '')}</div>`;
    if (rec) {
      const S = rec.summary;
      h += `<div style="margin:5px 0"><b>${esc(ko(sel))}</b> (${esc(rec.kind)}) — ${rec.kind === '피난층' ? `거실→옥외 출구 기준 ${rec.lim_room}m` : `거실→직통계단 기준 ${rec.lim}m`}</div><table>`;
      h += `<tr><td class="k">실(자동 분할) · 경로 있음</td><td>${S.n} · ${S.reach}</td></tr><tr><td class="k">최원점 보행거리 최대</td><td><b>${S.max != null ? S.max.toFixed(0) + 'm' : '-'}</b></td></tr>`;
      h += `<tr><td class="k">기준 초과 실</td><td>${S.over ? `<b style="color:#ffa657">${S.over}</b>` : 0}${S.mid ? ` (+30m 초과 ${S.mid})` : ''}</td></tr>`;
      h += `<tr><td class="k">한 방향 피난 · 공통경로 ${EV.law.common_path_m}m↑</td><td>${S.single} · ${S.common}</td></tr><tr><td class="k">벽 통과 가정(문 미검출)</td><td>${S.wall}</td></tr></table>`;
      if (rec.kind === '피난층') for (const [sid, L] of Object.entries(rec.stair_exit || {})) h += `<div class="mk">${esc(stName(sid))} → 옥외 출구 <b>${L.d.toFixed(0)}m</b>${L.over ? ' <span style="color:#ffa657">초과</span>' : ''}</div>`;
    }
    h += '<table style="margin-top:6px"><tr><td class="k">층</td><td class="k">실</td><td class="k">최대</td><td class="k">초과</td><td class="k">한방향</td></tr>';
    for (const f of [...floors].reverse()) {
      const r = EV.floors[f.id]; if (!r) continue; const S = r.summary;
      h += `<tr class="mk" data-evf="${f.id}" style="${f.id === sel ? 'background:#1f6feb33' : ''}"><td style="text-align:left">${esc(f.name)}</td><td>${S.n}</td><td>${S.max != null ? S.max.toFixed(0) : '-'}</td><td>${S.over ? `<b style="color:#ffa657">${S.over}</b>` : 0}</td><td>${S.single}</td></tr>`;
    }
    h += '</table>';
    if ((EV.entry || []).length) {
      h += `<div style="margin-top:6px"><b>진입로</b> (${esc(ko(ground))} 진입구 → 계단)</div>`;
      for (const e of EV.entry) h += `<div class="mk" data-fly="${e.x},${e.y}" data-fl="${ground}"><span class="dot" style="background:#ff4d4f"></span>진입 ${e.id.slice(1)} ${esc(e.name.replace(/^진입구 \d+ /, ''))} → ${e.routes.slice(0, 2).map((z) => `${esc(z.name)} ${z.d.toFixed(0)}m`).join(' / ') || '계단 없음'}</div>`;
    }
    $('evPanel').innerHTML = h;
  }
  $('evPanel').addEventListener('click', (e) => { const r = e.target.closest('[data-evf]'); if (r) ctx.selectFloor(r.dataset.evf, true); });

  // ───────── 마우스 올리면 실별 보행거리
  const tip = document.createElement('div'); tip.id = 'evTip'; document.body.appendChild(tip);
  const ray = new THREE.Raycaster(), mv = new THREE.Vector2(); let last = 0;
  canvas.addEventListener('pointermove', (e) => {
    const t = performance.now(); if (t - last < 60) return; last = t;
    const rc = canvas.getBoundingClientRect(); mv.set(((e.clientX - rc.left) / rc.width) * 2 - 1, -((e.clientY - rc.top) / rc.height) * 2 + 1);
    ray.setFromCamera(mv, camera);
    const objs = [...hov.filter((o) => o.parent.visible && o.parent.parent.visible && o.parent.parent.parent?.visible !== false), ...FG.children.filter((o) => o.userData && o.userData.r)];
    const hit = FG.visible || Object.values(SG).some((g) => g.visible) ? ray.intersectObjects(objs, false)[0] : null;
    if (!hit) { tip.style.display = 'none'; return; }
    const { fid, r, fire } = hit.object.userData, rec = EV.floors[fid];
    let s = `<b>실 ${r.area.toFixed(0)}㎡</b> · ${esc(ko(fid))}<br>`;
    if (fire) s += fire.st === 'fire' ? '<b style="color:#ff7b72">화재실(화점 구역)</b>' : fire.st === 'shared' ? `<b style="color:#ffa657">피난계단 경로 없음 — 공격계단으로만 대피 ${fire.d != null ? fire.d.toFixed(0) + 'm' : ''} (진입대와 충돌) → 인명검색 우선</b>` : fire.st === 'iso' ? '<b style="color:#d2a8ff">고립 — 화점·공격계단을 피한 대피로 없음 → 인명검색 1순위</b>' : `대피 ${fire.d != null ? fire.d.toFixed(0) + 'm' : '-'}${fire.st === 'near' ? ' <b style="color:#ffa657">· 화점 인접 경로</b>' : ''}${r.d != null ? ` (평시 ${r.d.toFixed(0)}m)` : ''}`;
    else if (r.d == null) s += '대피 목표(계단·출구)까지 경로 없음';
    else {
      s += `${esc(tgName(fid, r.tgt))}까지 <b>${r.d.toFixed(0)}m</b> · 최원점 <b style="color:${r.over === 'high' ? '#ffa657' : '#e6edf3'}">${r.dmax.toFixed(0)}m</b> / 기준 ${rec.lim_room}m`;
      if (r.total != null) s += `<br>피난층 옥외까지 ≈ ${r.total.toFixed(0)}m + 계단 ${r.nfl}개 층`;
      if (r.n_reach === 1 && rec.targets.length > 1) s += '<br><span style="color:#ffa657">한 방향 피난(다른 계단 경로 없음)</span>';
      else if (r.common != null && r.common > EV.law.common_path_m) s += `<br><span style="color:#ffa657">두 방향 공통 경로 ${r.common.toFixed(0)}m</span>`;
      if (r.wall) s += '<br><span style="color:#ffd33d">문 미검출 — 벽 통과 가정</span>';
    }
    tip.innerHTML = s; tip.style.display = 'block'; tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 12 + 'px';
  });

  applyVis();
  window.__evac = { onFire, applyVis };
  return { onFire, onView: applyVis, lastFire: () => lastSt, setSim: (v) => { simOn = !!v; applyVis(); } };
}
