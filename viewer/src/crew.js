// 대원 이동 재생 — 화재 가정 산출안(진입구 → 공격계단 → 화재층 → 화점)을 따라 팀 표식이 움직임(시뮬레이션 시간축과 같은 시계)
// 도착 = 출동 준비(dispatch_s) + 관할센터 거리 ÷ 주행 속도(drive_kmh) (정보카드 거리 없으면 default_arrive_s) + 전개(setup_s).  수평 보행 ACT.walk_mps · 계단 층당 ACT.stair_s_per_floor.
// 진압1(호스 전개: 화재층 도착부터 가장 가까운 옥내소화전 → 화점, 1본 눈금) · 진압2(뒤따름) · 구조(인명검색 순서대로 실 방문, 실마다 search_dwell_s) — 모두 예시값.
// 공기호흡기: 진입구 통과부터 소모(ACT.scba_lpm), 잔압 표시 · 귀환 한계 시각(산출안 tLim)을 넘으면 경고.
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

const TEAMS = [['atk1', '진압1', '#f97316'], ['atk2', '진압2', '#fb7185'], ['res', '구조', '#a78bfa']];

export function initCrew(ctx) {
  const { scene, V, SLAB, zOf, FL, floors, ACT, D, esc, gridFor, dijkstra, pathTo, pxScale } = ctx;
  const root = new THREE.Group(); root.userData.own = true; scene.add(root);
  let plan = null, teams = [], hose = null, arrive = 0, info = null;
  const pathCache = { key: '', p: [] };

  function arriveTime(P) {
    const G = (D.building.card || {}).거리 || {}, km = G.관할센터 ?? G.최단센터;
    const drive = km != null ? (km / P.drive_kmh) * 3600 : null;
    return { t: (drive != null ? P.dispatch_s + drive : P.default_arrive_s) + P.setup_s, km, drive };
  }
  // 경로(구간 목록) → 시간이 붙은 3D 점열 [{p:Vector3, t(초, 출발 기준), fid, note}]
  function timed(legs, fromT = 0) {
    const out = []; let t = fromT;
    for (const L of legs) {
      if (L.vert) {
        const n = Math.abs(L.n), dir = FL[L.f1].idx > FL[L.f0].idx ? 1 : -1;
        for (let k = 0; k <= n; k++) {
          const f = floors[FL[L.f0].idx + dir * k]; if (!f) continue;
          out.push({ p: V(L.x, L.y, zOf(f.id) + SLAB + 0.05), t, fid: f.id, note: `${f.name} 계단` }); if (k < n) t += ACT.stair_s_per_floor;
        }
        continue;
      }
      for (let i = 0; i < L.pts.length; i++) {
        const [x, y] = L.pts[i];
        if (i) { const [a, b] = L.pts[i - 1]; t += Math.hypot(x - a, y - b) / ACT.walk_mps; }
        out.push({ p: V(x, y, zOf(L.fid) + SLAB + 0.05), t, fid: L.fid, note: L.note || FL[L.fid].name });
      }
    }
    return out;
  }
  function at(seq, t) {                                              // 시각 t 의 위치(선형 보간)
    if (!seq.length) return null;
    if (t <= seq[0].t) return { p: seq[0].p.clone(), k: 0, s: seq[0] };
    for (let k = 1; k < seq.length; k++) if (t <= seq[k].t) { const a = seq[k - 1], b = seq[k], w = (t - a.t) / Math.max(1e-6, b.t - a.t); return { p: a.p.clone().lerp(b.p, w), k, s: b }; }
    const z = seq[seq.length - 1]; return { p: z.p.clone(), k: seq.length, s: z, done: true };
  }
  // 팀 표식: 화면 크기 고정 아이콘(사람 모양, 깊이 무시) + 바닥 고리 + 이름·시간·잔압 라벨
  const iconCache = {};
  function icon(color) {
    if (iconCache[color]) return iconCache[color];
    const cv = document.createElement('canvas'); cv.width = cv.height = 96; const c = cv.getContext('2d');
    c.beginPath(); c.arc(48, 48, 40, 0, 7); c.fillStyle = color; c.fill(); c.lineWidth = 7; c.strokeStyle = '#ffffff'; c.stroke();
    c.fillStyle = '#ffffff'; c.beginPath(); c.arc(48, 33, 11, 0, 7); c.fill();
    c.beginPath(); c.moveTo(28, 72); c.quadraticCurveTo(30, 46, 48, 46); c.quadraticCurveTo(66, 46, 68, 72); c.closePath(); c.fill();
    c.fillStyle = color; c.fillRect(36, 25, 24, 5);                                       // 헬멧 챙
    const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return (iconCache[color] = t);
  }
  function marker(color, name) {
    const g = new THREE.Group();
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: icon(color), depthTest: false, transparent: true, sizeAttenuation: false }));
    sp.center.set(0.5, 0.1); sp.renderOrder = 46; sp.userData.px = 30; g.add(sp);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.6, 0.9, 28), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, depthTest: false, transparent: true, opacity: 0.85 }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.06; ring.renderOrder = 44; g.add(ring);
    const el = document.createElement('div'); el.className = 'lb crew'; el.style.borderColor = color; el.innerHTML = `<i style="background:${color}"></i><b>${esc(name)}</b> <span></span>`;
    const lab = new CSS2DObject(el); lab.position.y = 0.2; lab.center.set(0.5, 2.2); g.add(lab);
    return { g, el, sp };
  }

  // 산출안 → 팀 경로
  function build(P, plan0, sim) {
    clear(); plan = plan0;
    if (!plan || !plan.legs || !plan.legs.length) { info = { none: true }; return info; }
    const A = arriveTime(sim); arrive = A.t;
    const base = timed(plan.legs);
    const stairEnd = plan.legs.findIndex((L) => L.vert);
    // 화재층 도착(계단 끝) 시각 — 수직 구간 끝 점
    let tFloor = 0; { let acc = 0; for (const s of base) if (s.fid === plan.fid) { acc = s.t; break; } tFloor = acc; }
    const tEnd = base[base.length - 1].t;
    // 구조팀: 화재층 계단(공격계단) 도착 후 인명검색 순서(화재층 상위 실)대로 격자 최단경로 + 실마다 머무름
    let resSeq = base;
    if (plan.search && plan.search.length && plan.attackXY) {
      const fid = plan.fid, gr = gridFor(fid), rooms = plan.search.filter((z) => z.fid === fid && z.c).slice(0, 5);
      const pre = base.filter((s) => s.t <= tFloor);
      let cur = plan.attackXY, t = tFloor; const seq = pre.slice();
      const ck = `${plan.fid}|${plan.x}|${plan.y}|${plan.attack}`;
      if (pathCache.key !== ck) { pathCache.key = ck; pathCache.p = []; }
      for (const [ri, z] of rooms.entries()) {
        if (!pathCache.p[ri]) { const R = dijkstra(gr, [cur]); pathCache.p[ri] = pathTo(gr, R, z.c[0], z.c[1]).slice().reverse(); }   // 격자 최단경로는 화점이 같으면 다시 계산하지 않음(보기 전환마다 호출됨)
        const pth = pathCache.p[ri];
        for (let i = 1; i < pth.length; i++) { t += Math.hypot(pth[i][0] - pth[i - 1][0], pth[i][1] - pth[i - 1][1]) / ACT.walk_mps; seq.push({ p: V(pth[i][0], pth[i][1], zOf(fid) + SLAB + 0.05), t, fid, note: `검색 ${z.rank}번 실로` }); }
        t += sim.search_dwell_s; seq.push({ p: V(z.c[0], z.c[1], zOf(fid) + SLAB + 0.05), t, fid, note: `검색 ${z.rank}번 실 확인` });
        cur = z.c;
      }
      resSeq = seq;
    }
    const st = { atk1: arrive, atk2: arrive + sim.team2_delay_s, res: arrive + sim.rescue_delay_s };
    teams = TEAMS.map(([id, name, col]) => {
      const seq = id === 'res' ? resSeq : base, M = marker(col, name);
      root.add(M.g);
      const pos = new Float32Array(seq.length * 3); seq.forEach((s, i) => { pos[i * 3] = s.p.x; pos[i * 3 + 1] = s.p.y + 0.15; pos[i * 3 + 2] = s.p.z; });
      const tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); tg.setDrawRange(0, 0);
      const trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: col, depthTest: false, transparent: true, opacity: 0.95 })); trail.renderOrder = 43; root.add(trail);
      return { id, name, col, seq, t0: st[id], M, trail, dur: seq[seq.length - 1].t };
    });
    // 호스 전개(진압1 화재층 도착 → 소화전 → 화점): 관 형상의 그리기 범위를 늘림
    if (plan.hosePath && plan.hosePath.length > 1) {
      const z = zOf(plan.fid) + SLAB + 0.3, pts = plan.hosePath.map(([x, y]) => V(x, y, z));
      const path = new THREE.CurvePath(); for (let k = 0; k < pts.length - 1; k++) if (pts[k].distanceTo(pts[k + 1]) > 0.01) path.add(new THREE.LineCurve3(pts[k], pts[k + 1]));
      if (path.curves.length) {
        const seg = Math.max(16, pts.length * 6), tg = new THREE.TubeGeometry(path, seg, 0.17, 6, false);
        const m = new THREE.Mesh(tg, new THREE.MeshBasicMaterial({ color: 0xff3b3b, depthTest: false, transparent: true })); m.renderOrder = 42; root.add(m);
        hose = { m, n: tg.index.count, seg, len: path.getLength(), t0: st.atk1 + tFloor };
        tg.setDrawRange(0, 0);
      }
    }
    info = { arrive, km: A.km, drive: A.drive, tFloor, tEnd, scbaLim: plan.scba ? plan.scba.tLim * 60 : null, TA: plan.scba ? plan.scba.TA * 60 : null };
    return info;
  }
  const mmss = (s) => `${s < 0 ? '-' : ''}${Math.floor(Math.abs(s) / 60)}:${String(Math.floor(Math.abs(s) % 60)).padStart(2, '0')}`;
  // 시각 t(화재 경과 초)로 갱신 → 팀 상태 목록
  function update(t) {
    const out = [];
    for (const T of teams) {
      const rel = t - T.t0;
      if (rel < 0) { T.M.g.visible = false; T.trail.geometry.setDrawRange(0, 0); out.push({ id: T.id, name: T.name, col: T.col, state: 'wait', txt: `도착 전 (${mmss(-rel)} 후 진입)` }); continue; }
      const q = at(T.seq, rel); T.M.g.visible = true; T.M.g.position.copy(q.p); const k = pxScale() * T.M.sp.userData.px; T.M.sp.scale.set(k, k, 1);
      T.trail.geometry.setDrawRange(0, Math.min(T.seq.length, q.k + 1));
      const usedL = (rel / 60) * ACT.scba_lpm, bar = Math.max(0, ACT.scba_fill_bar - usedL / ACT.scba_cyl_l);
      const lim = info.scbaLim != null && rel >= info.scbaLim && T.id !== 'res';
      const low = bar <= ACT.scba_return_bar;
      T.M.el.querySelector('span').textContent = `${mmss(rel)} · ${Math.round(bar)}bar`;
      T.M.el.classList.toggle('warn', lim || low);
      out.push({ id: T.id, name: T.name, col: T.col, state: q.done ? 'done' : 'move', rel, bar, lim, low, at: q.s.note, fid: q.s.fid,
        txt: `${q.done ? (T.id === 'res' ? '검색 완료' : '화점 도착') : esc(q.s.note) + ' 이동'} · 진입 ${mmss(rel)} · 잔압 ${Math.round(bar)}bar${lim ? ' · ⚠ 귀환 한계 넘음' : low ? ' · ⚠ 귀환 잔압' : ''}`, p: q.p });
    }
    if (hose) {
      const d = Math.max(0, Math.min(hose.len, (t - hose.t0) * ACT.walk_mps)), k = Math.floor((d / hose.len) * hose.seg);
      hose.m.geometry.setDrawRange(0, Math.min(hose.n, k * 6 * 6)); hose.m.visible = d > 0;
      out.hose = { d, len: hose.len, n: Math.ceil(d / ACT.hose_len_m) };
    }
    return out;
  }
  function clear() { root.traverse((o) => { if (o.isCSS2DObject && o.element.parentNode) o.element.parentNode.removeChild(o.element); }); root.clear(); teams = []; hose = null; info = null; }   // 묶음 안 HTML 라벨은 three 가 지워 주지 않음
  function setVisible(v) { root.visible = v; }
  return { build, update, clear, setVisible, info: () => info, teams: () => teams, mmss };
}
