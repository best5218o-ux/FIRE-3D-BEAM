// 2D 미니맵 — 고른 층(화재 가정이면 화재층) 평면을 구석에 그리고, 카메라 시야(바닥면과 만나는 사다리꼴)·위치를 겹쳐 표시. 누르거나 끌면 그 지점으로 카메라 이동.
// 고정 그림(벽·문·계단·승강기·진입구·소화전)은 층마다 한 번 그려 두고, 움직이는 것(시야·화점·경로·대원·연기)만 다시 그림. 큰 화면(2D 대체 보기)에도 같은 그리기를 씀.
import * as THREE from 'three';

export function initMinimap(ctx) {
  const { floors, FL, BIM, D, camera, getFid, getZ, bounds, onPick } = ctx;
  const pad = 3;
  const cache = {};
  // 화면 맞춤: 기본 = 건물 전체 범위(미니맵 — 층이 바뀌어도 같은 축척), bb·inset 을 주면 그 층 범위를 안쪽 여백 안에(2D 대체 보기)
  function fit(W, Hh, bb, ins) {
    const B = bb ? { bx0: bb[0], by0: bb[1], bx1: bb[2], by1: bb[3] } : bounds, I = ins || { l: 0, t: 0, r: 0, b: 0 };
    const w2 = W - I.l - I.r, h2 = Hh - I.t - I.b;
    const bw = B.bx1 - B.bx0 + 2 * pad, bh = B.by1 - B.by0 + 2 * pad, k = Math.min(w2 / bw, h2 / bh);
    const ox = I.l + (w2 - bw * k) / 2, oy = I.t + (h2 - bh * k) / 2;
    return { k, X: (x) => ox + (x - B.bx0 + pad) * k, Y: (y) => oy + (B.by1 + pad - y) * k, inv: (px, py) => [(px - ox) / k + B.bx0 - pad, B.by1 + pad - (py - oy) / k] };
  }
  function staticLayer(fid, W, Hh, T, tag) {
    const key = `${fid}|${W}|${Hh}|${tag}`; if (cache[key]) return cache[key];
    const cv = document.createElement('canvas'); cv.width = W; cv.height = Hh; const c = cv.getContext('2d'), f = FL[fid], fb = BIM.floors[fid] || { walls: [], doors: [] };
    c.fillStyle = 'rgba(30,58,95,0.55)'; c.strokeStyle = '#38bdf8'; c.lineWidth = 1.2;
    for (const ring of f.poly) { c.beginPath(); ring.forEach(([x, y], i) => (i ? c.lineTo(T.X(x), T.Y(y)) : c.moveTo(T.X(x), T.Y(y)))); c.closePath(); c.fill(); c.stroke(); }
    c.strokeStyle = '#cfe7ff'; c.lineCap = 'butt';
    for (const [o, cc, a, b, t] of fb.walls) { c.lineWidth = Math.max(1, Math.max(0.15, t) * T.k); c.beginPath(); if (o === 'h') { c.moveTo(T.X(a), T.Y(cc)); c.lineTo(T.X(b), T.Y(cc)); } else { c.moveTo(T.X(cc), T.Y(a)); c.lineTo(T.X(cc), T.Y(b)); } c.stroke(); }
    for (const [kind, , x, y] of fb.doors) { c.fillStyle = kind === 'entrance' ? '#22d3ee' : kind === 'exit' ? '#22c55e' : '#c8a27a'; const r = kind === 'door' ? 1.6 : 3.2; c.beginPath(); c.arc(T.X(x), T.Y(y), r, 0, 7); c.fill(); }
    for (const [kind, S] of [...BIM.stairs.map((s) => ['st', s]), ...BIM.elevs.map((e) => ['ev', e])]) {
      if (!S.floors.includes(fid)) continue; const [x0, y0, x1, y1] = S.box;
      c.fillStyle = kind === 'st' ? 'rgba(74,222,128,0.45)' : 'rgba(96,165,250,0.5)'; c.strokeStyle = kind === 'st' ? '#4ade80' : '#60a5fa'; c.lineWidth = 1;
      c.fillRect(T.X(x0), T.Y(y1), (x1 - x0) * T.k, (y1 - y0) * T.k); c.strokeRect(T.X(x0), T.Y(y1), (x1 - x0) * T.k, (y1 - y0) * T.k);
    }
    for (const [type, f2, x, y] of BIM.equip) { if (f2 !== fid) continue; c.fillStyle = type === 'hose' ? '#ef4444' : type === 'panel' ? '#a78bfa' : '#f59e0b'; c.fillRect(T.X(x) - 2, T.Y(y) - 2, 4, 4); }
    return (cache[key] = cv);
  }
  const ray = new THREE.Raycaster(), pl = new THREE.Plane(), hit = new THREE.Vector3();
  function frustum(z) {                                              // 화면 네 모서리 광선 ∩ 바닥면(지평선 위는 아래로 눌러서)
    pl.set(new THREE.Vector3(0, 1, 0), -z); const out = [];
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      ray.setFromCamera(new THREE.Vector2(a, b), camera);
      const d = ray.ray.direction.clone(); if (d.y > -0.08) { d.y = -0.08; d.normalize(); ray.ray.direction.copy(d); }
      if (ray.ray.intersectPlane(pl, hit)) out.push([hit.x, -hit.z]);
    }
    return out;
  }
  // dyn: { fire, routes:[[pts,color,width]], crew:[{p:[x,y],col,name}], rooms:[{poly,col,a}], measure:[[x,y],[x,y]] }
  function draw(cv, dyn = {}, opt = {}) {
    const dpr = Math.min(2, devicePixelRatio || 1), W = Math.round(cv.clientWidth * dpr), Hh = Math.round(cv.clientHeight * dpr);
    if (!W || !Hh) return;
    if (cv.width !== W || cv.height !== Hh) { cv.width = W; cv.height = Hh; }
    const c = cv.getContext('2d'), fid = dyn.fid || getFid();
    const ins = opt.inset ? { l: opt.inset.l * dpr, t: opt.inset.t * dpr, r: opt.inset.r * dpr, b: opt.inset.b * dpr } : null, T = opt.fitFloor ? fit(W, Hh, FL[fid].bb, ins) : fit(W, Hh);
    c.clearRect(0, 0, W, Hh); c.fillStyle = 'rgba(8,14,26,0.92)'; c.fillRect(0, 0, W, Hh);
    c.drawImage(staticLayer(fid, W, Hh, T, opt.fitFloor ? 'f' : 'g'), 0, 0);
    for (const r of dyn.rooms || []) { c.fillStyle = r.col; c.globalAlpha = r.a ?? 0.5; c.beginPath(); r.poly.forEach(([x, y], i) => (i ? c.lineTo(T.X(x), T.Y(y)) : c.moveTo(T.X(x), T.Y(y)))); c.closePath(); c.fill(); c.globalAlpha = 1; }
    for (const [pts, col, w] of dyn.routes || []) { if (pts.length < 2) continue; c.strokeStyle = col; c.lineWidth = (w || 2) * dpr; c.beginPath(); pts.forEach(([x, y], i) => (i ? c.lineTo(T.X(x), T.Y(y)) : c.moveTo(T.X(x), T.Y(y)))); c.stroke(); }
    if (dyn.measure) { c.strokeStyle = '#fde047'; c.lineWidth = 2 * dpr; c.setLineDash([5 * dpr, 4 * dpr]); c.beginPath(); c.moveTo(T.X(dyn.measure[0][0]), T.Y(dyn.measure[0][1])); c.lineTo(T.X(dyn.measure[1][0]), T.Y(dyn.measure[1][1])); c.stroke(); c.setLineDash([]); }
    if (dyn.fire) { const t = performance.now() / 300; c.fillStyle = '#ff4d2e'; c.beginPath(); c.arc(T.X(dyn.fire[0]), T.Y(dyn.fire[1]), (4 + Math.sin(t) * 1.2) * dpr, 0, 7); c.fill(); c.strokeStyle = '#fff3'; c.lineWidth = 2 * dpr; c.beginPath(); c.arc(T.X(dyn.fire[0]), T.Y(dyn.fire[1]), 9 * dpr, 0, 7); c.stroke(); }
    for (const m of dyn.crew || []) { c.fillStyle = m.col; c.strokeStyle = '#fff'; c.lineWidth = 1.5 * dpr; c.beginPath(); c.arc(T.X(m.p[0]), T.Y(m.p[1]), 4.5 * dpr, 0, 7); c.fill(); c.stroke(); }
    if (!opt.noFrustum) {
      const fr = frustum(getZ(fid));
      if (fr.length === 4) {
        c.fillStyle = 'rgba(251,191,36,0.13)'; c.strokeStyle = '#fbbf24'; c.lineWidth = 1.3 * dpr; c.beginPath();
        fr.forEach(([x, y], i) => (i ? c.lineTo(T.X(x), T.Y(y)) : c.moveTo(T.X(x), T.Y(y)))); c.closePath(); c.fill(); c.stroke();
      }
      const cx = camera.position.x, cy = -camera.position.z; c.fillStyle = '#fbbf24'; c.beginPath(); c.arc(T.X(cx), T.Y(cy), 3.5 * dpr, 0, 7); c.fill();
    }
    // 북쪽·층 이름
    const ty = (opt.inset ? opt.inset.t - 6 : 14) * dpr, tx = (opt.inset ? opt.inset.l : 6) * dpr, tr = (opt.inset ? opt.inset.r : 6) * dpr;
    c.fillStyle = '#93c5fd'; c.font = `bold ${(opt.inset ? 14 : 11) * dpr}px system-ui,sans-serif`; c.textAlign = 'right'; c.fillText('N ↑', W - tr, ty);
    c.textAlign = 'left'; c.fillStyle = '#e6edf7'; c.fillText(FL[fid].name + (opt.inset ? ' — 2D 평면 보기' : ''), tx, ty);
    cv.__T = T; cv.__dpr = dpr;
  }
  function attach(cv) {
    let drag = false;
    const pick = (e) => { const r = cv.getBoundingClientRect(), T = cv.__T; if (!T) return; const [x, y] = T.inv((e.clientX - r.left) * cv.__dpr, (e.clientY - r.top) * cv.__dpr); onPick(x, y, drag); };
    cv.addEventListener('pointerdown', (e) => { e.stopPropagation(); drag = false; pick(e); drag = true; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', (e) => { if (drag && e.buttons) pick(e); });
    cv.addEventListener('pointerup', () => { drag = false; });
  }
  return { draw, attach, fit, toPx: (cv, x, y) => { const T = cv.__T, r = cv.getBoundingClientRect(); return [r.left + T.X(x) / cv.__dpr, r.top + T.Y(y) / cv.__dpr]; } };
}
