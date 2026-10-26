// v3 건물 형상 — 벽(두께·문 개구부 절단·문틀 위 인방)·문짝 기호·외벽 유리 띠·바닥판 테두리 빛·가짜 그림자(접촉 음영)·실 용도 입체
// 모두 BIM JSON(벽 중심선 [o,c,a,b,t,ext], 문 [kind,o,x,y,w,…], 실 다각형)에서 만든다. 층마다 재질 수를 고정해 그리기 호출(draw call)을 줄임.
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export const CATCOL = { crit: 0xf43f5e, sleep: 0xa78bfa, occupied: 0x38bdf8, hall: 0x2dd4bf, corr: 0x94a3b8, svc: 0xf59e0b, unk: 0x64748b };
export const CATNAME = { crit: '거동 불가 실', sleep: '취침 실', occupied: '재실(사무·교실 등)', hall: '대공간', corr: '복도·홀', svc: '설비·창고', unk: '용도 미상' };

export const ni = (g) => (g.index ? g.toNonIndexed() : g);
export function boxGeom(x0, y0, x1, y1, z0, z1) {
  const g = new THREE.BoxGeometry(Math.max(0.03, x1 - x0), Math.max(0.03, z1 - z0), Math.max(0.03, y1 - y0));
  g.translate((x0 + x1) / 2, (z0 + z1) / 2, -(y0 + y1) / 2); return g;
}
// 벽 방향 기준 상자(along: 벽 길이 방향 구간, perp: 두께 방향 중심)
function wallBox(o, c, p, q, t, z0, z1) { return o === 'h' ? boxGeom(p, c - t / 2, q, c + t / 2, z0, z1) : boxGeom(c - t / 2, p, c + t / 2, q, z0, z1); }

// ── 유리·홀로그램 재질: 표준 재질 + 프레넬 테두리 빛 + 아래쪽 어둡게(가짜 AO) + 비스듬한 면일수록 덜 투명
export function glassMat({ color, opacity, rim = 0x7dd3fc, rimK = 0.8, emissive = 0x0b3b5a, ei = 0.5, aoY = null, side = THREE.DoubleSide }) {
  const m = new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: ei, transparent: true, opacity, roughness: 0.25, metalness: 0.1, depthWrite: false, side });
  m.userData.op = opacity;
  const uni = { uRim: { value: new THREE.Color(rim) }, uRimK: { value: rimK }, uAO: { value: new THREE.Vector2(aoY ? aoY[0] : -1e5, aoY ? aoY[1] : 1) } };
  m.userData.uni = uni;
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uni);
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying float vLocY;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocY = position.y;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform vec3 uRim; uniform float uRimK; uniform vec2 uAO; varying float vLocY;')
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float fres = pow(1.0 - abs(dot(normalize(vViewPosition), normal)), 2.0);
        totalEmissiveRadiance += uRim * fres * uRimK;
        float aoK = smoothstep(uAO.x, uAO.x + uAO.y, vLocY);
        diffuseColor.rgb *= mix(0.3, 1.0, aoK); totalEmissiveRadiance *= mix(0.45, 1.0, aoK);
        diffuseColor.a = clamp(diffuseColor.a * (0.55 + 1.7 * fres), 0.0, 1.0);`);
  };
  m.customProgramCacheKey = () => 'glassV3';
  return m;
}

// ── 문 위치로 벽 중심선을 자름(도면 벡터화가 이미 틈을 남긴 경우는 그대로) → 벽 조각
export function cutWalls(walls, doors) {
  const out = [];
  for (const w of walls) {
    const [o, c, a, b, t, ext] = w, tt = Math.max(0.15, t);
    let iv = [[Math.min(a, b), Math.max(a, b)]];
    for (const d of doors) {
      if (d[1] !== o) continue;
      const along = o === 'h' ? d[2] : d[3], perp = o === 'h' ? d[3] : d[2];
      if (Math.abs(perp - c) > tt / 2 + 0.25) continue;
      const g0 = along - d[4] / 2, g1 = along + d[4] / 2;
      iv = iv.flatMap(([p, q]) => (g1 <= p || g0 >= q ? [[p, q]] : [[p, g0], [g1, q]].filter(([u, v]) => v - u > 0.05)));
    }
    for (const [p, q] of iv) out.push({ o, c, a: p, b: q, t: tt, ext: !!ext, gid: w[6] });
  }
  return out;
}
// 문이 놓인 벽 두께(가장 가까운 같은 방향 벽)
function doorWallT(walls, d) {
  let best = null;
  for (const w of walls) {
    if (w[0] !== d[1]) continue;
    const [o, c, a, b, t] = w, along = o === 'h' ? d[2] : d[3], perp = o === 'h' ? d[3] : d[2];
    const dd = Math.abs(perp - c) + Math.max(0, Math.min(a, b) - along - d[4], along - d[4] - Math.max(a, b));
    if (dd < 1.0 && (!best || dd < best[0])) best = [dd, Math.max(0.15, t), c];
  }
  return best ? { t: best[1], c: best[2] } : { t: d[6] ? 0.3 : 0.15, c: d[1] === 'h' ? d[3] : d[2] };
}

// ── 층 벽·문·창 형상
// 반환: { wallE, wallI, glass, frame, lintel, leaves, leafDoors(면 → 문), arcs(선), entFrames[{geo,door}], footRects(단면 채움용 [o,c,a,b,t,z0,z1]) }
export function floorShell(fb, dims) {
  const { SLAB, WALLH } = dims;
  const segs = cutWalls(fb.walls, fb.doors);
  const gE = [], gI = [], gGl = [], gFr = [], gLi = [], foot = [];
  const SILL = 0.9, HEAD = Math.min(WALLH - 0.25, 2.3), MUL = 1.6;
  for (const s of segs) {
    const L = s.b - s.a;
    foot.push([s.o, s.c, s.a, s.b, s.t, SLAB, SLAB + WALLH]);
    if (s.ext && L >= 1.4 && HEAD > SILL + 0.5) {
      // 외벽: 아래 벽(허리벽) + 유리 띠 + 위 벽 + 끝 기둥 + 멀리언(창 띠는 표현용 — 실제 창 위치가 아님)
      const e0 = s.a + 0.25, e1 = s.b - 0.25;
      gE.push(wallBox(s.o, s.c, s.a, s.b, s.t, SLAB, SLAB + SILL), wallBox(s.o, s.c, s.a, s.b, s.t, SLAB + HEAD, SLAB + WALLH));
      gE.push(wallBox(s.o, s.c, s.a, e0, s.t, SLAB + SILL, SLAB + HEAD), wallBox(s.o, s.c, e1, s.b, s.t, SLAB + SILL, SLAB + HEAD));
      gGl.push(wallBox(s.o, s.c, e0, e1, s.t * 0.35, SLAB + SILL, SLAB + HEAD));
      const n = Math.max(1, Math.round((e1 - e0) / MUL));
      for (let k = 1; k < n; k++) { const u = e0 + ((e1 - e0) * k) / n; gFr.push(wallBox(s.o, s.c, u - 0.04, u + 0.04, s.t * 0.6, SLAB + SILL, SLAB + HEAD)); }
      gFr.push(wallBox(s.o, s.c, e0, e1, s.t * 0.7, SLAB + SILL - 0.04, SLAB + SILL), wallBox(s.o, s.c, e0, e1, s.t * 0.7, SLAB + HEAD, SLAB + HEAD + 0.04));
    } else (s.ext ? gE : gI).push(wallBox(s.o, s.c, s.a, s.b, s.t, SLAB, SLAB + WALLH));
  }
  // 문: 인방(문 위 벽) · 문짝(열린 상태 70°) · 바닥 회전 호 · 진입구/비상구 문틀(빛)
  const leaves = [], leafDoors = [], arcPts = [], ent = [];
  let tri = 0;
  for (let i = 0; i < fb.doors.length; i++) {
    const d = fb.doors[i], [kind, o, x, y, w] = d;
    const hh = kind === 'door' ? Math.min(2.1, WALLH - 0.2) : Math.min(2.6, WALLH - 0.15);
    const { t, c } = doorWallT(fb.walls, d), along = o === 'h' ? x : y;
    if (WALLH - hh > 0.05) gLi.push(wallBox(o, c, along - w / 2, along + w / 2, t, SLAB + hh, SLAB + WALLH));
    foot.push([o, c, along - w / 2, along + w / 2, t, SLAB + hh, SLAB + WALLH]);
    // 문짝: 폭 1.3 m 넘으면 양여닫이
    const parts = w > 1.3 ? [[along - w / 2, w / 2, 1], [along + w / 2, w / 2, -1]] : [[along - w / 2, w, 1]];
    const ang = (kind === 'door' ? 70 : 80) * Math.PI / 180;
    for (const [h0, lw, sgn] of parts) {
      const g = new THREE.BoxGeometry(lw, hh - 0.05, 0.05); g.translate((sgn * lw) / 2, (hh - 0.05) / 2, 0);
      g.rotateY(sgn * ang);                                         // 경첩 축 기준으로 열림(평면 +수직 방향 쪽으로)
      if (o === 'h') g.translate(h0, SLAB + 0.02, -c); else { g.rotateY(Math.PI / 2); g.translate(c, SLAB + 0.02, -h0); }
      const ng = ni(g); leaves.push(ng); const nt = ng.attributes.position.count / 3; leafDoors.push([tri, tri + nt, i]); tri += nt;
      // 바닥 회전 호(점선) — 닫힌 위치 → 열린 위치
      const pts = [];
      for (let k = 0; k <= 14; k++) {
        const a = (sgn * ang * k) / 14, u = sgn * lw * Math.cos(a), v = lw * Math.sin(Math.abs(a));
        const px = o === 'h' ? h0 + u : c - v, py = o === 'h' ? c + v : h0 + u;
        pts.push(new THREE.Vector3(px, SLAB + 0.04, -py));
      }
      for (let k = 0; k < pts.length - 1; k += 1) arcPts.push(pts[k], pts[k + 1]);
    }
    if (kind !== 'door') {                                          // 진입구·비상구: 문틀(양 옆 + 머리) 빛나는 막대
      const fr = [wallBox(o, c, along - w / 2 - 0.07, along - w / 2, t + 0.08, SLAB, SLAB + hh), wallBox(o, c, along + w / 2, along + w / 2 + 0.07, t + 0.08, SLAB, SLAB + hh),
        wallBox(o, c, along - w / 2 - 0.07, along + w / 2 + 0.07, t + 0.08, SLAB + hh, SLAB + hh + 0.08)];
      ent.push({ geo: mergeGeometries(fr), door: d, i });
    }
  }
  const M = (a) => (a.length ? mergeGeometries(a.map((g) => (g.index ? g.toNonIndexed() : g))) : null);
  return { wallE: M(gE), wallI: M(gI), glass: M(gGl), frame: M(gFr), lintel: M(gLi), leaves: leaves.length ? mergeGeometries(leaves) : null, leafDoors,
    arcs: arcPts.length ? new THREE.BufferGeometry().setFromPoints(arcPts) : null, ent, foot, nSeg: segs.length };
}

// ── 가짜 그림자(접촉 음영): 벽 둘레를 흐리게 칠한 투명 텍스처 1장(층마다) — 바닥과 벽이 닿는 곳이 어두워짐
export function aoPlane(f, fb, shafts, dims) {
  // 저해상도(≤ 6 px/m) 캔버스에 벽·계단실을 한 번에 그린 뒤, 흐림(blur)은 전체에 딱 한 번만 — 획마다 흐림을 걸면 소프트웨어 그리기에서 수십 초 걸림
  const [x0, y0, x1, y1] = f.bb, pad = 1.5, W = x1 - x0 + 2 * pad, Hh = y1 - y0 + 2 * pad;
  const ppm = Math.max(2, Math.min(6, 640 / Math.max(W, Hh)));
  const w = Math.ceil(W * ppm), h = Math.ceil(Hh * ppm);
  const a = document.createElement('canvas'); a.width = w; a.height = h;
  const c = a.getContext('2d'); const X = (x) => (x - x0 + pad) * ppm, Y = (y) => (y1 + pad - y) * ppm;
  c.strokeStyle = 'rgba(0,0,0,0.9)'; c.lineCap = 'butt';
  for (const [o, cc, p, q, t] of fb.walls) {
    c.lineWidth = (Math.max(0.15, t) + 0.55) * ppm; c.beginPath();
    if (o === 'h') { c.moveTo(X(p), Y(cc)); c.lineTo(X(q), Y(cc)); } else { c.moveTo(X(cc), Y(p)); c.lineTo(X(cc), Y(q)); }
    c.stroke();
  }
  c.fillStyle = 'rgba(0,0,0,0.55)';
  for (const S of shafts) { const [p, b, q, r] = S.box; c.fillRect(X(p) - 0.3 * ppm, Y(r) - 0.3 * ppm, (q - p + 0.6) * ppm, (r - b + 0.6) * ppm); }
  c.lineWidth = 0.9 * ppm; c.strokeStyle = 'rgba(0,0,0,0.6)';
  for (const ring of f.poly) { c.beginPath(); ring.forEach(([x, y], i) => (i ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)))); c.closePath(); c.stroke(); }
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h; const c2 = cv.getContext('2d');
  c2.filter = `blur(${Math.max(1, 0.4 * ppm).toFixed(1)}px)`; c2.drawImage(a, 0, 0);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(W, Hh), new THREE.MeshBasicMaterial({ map: t, transparent: true, opacity: 0.5, depthWrite: false }));   // 검은 획(알파) → 바닥이 어두워짐
  m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, dims.SLAB + 0.035, -(y0 + y1) / 2); m.renderOrder = 2;
  return m;
}

// ── 바닥판 테두리 빛: 윗면 윤곽선 + 바깥쪽으로 퍼지는 빛 띠(가산 혼합, 꼭짓점 색으로 흐려짐)
export function slabGlow(poly, z, color = 0x38bdf8, width = 0.9) {
  const col = new THREE.Color(color), pos = [], cols = [], line = [];
  for (const ring of poly) {
    if (ring.length < 3) continue;
    let A = 0; for (let i = 0; i < ring.length; i++) { const [x, y] = ring[i], [x2, y2] = ring[(i + 1) % ring.length]; A += x * y2 - x2 * y; }
    const s = A > 0 ? 1 : -1;                                       // 반시계(+) → 바깥쪽 법선 = (dy,-dx)
    for (let i = 0; i < ring.length; i++) {
      const [ax, ay] = ring[i], [bx, by] = ring[(i + 1) % ring.length], L = Math.hypot(bx - ax, by - ay); if (L < 1e-3) continue;
      const nx = (s * (by - ay)) / L, ny = (s * -(bx - ax)) / L;
      const P = [[ax, ay, 1], [bx, by, 1], [bx + nx * width, by + ny * width, 0], [ax + nx * width, ay + ny * width, 0]];
      for (const k of [0, 1, 2, 0, 2, 3]) { const [x, y, a] = P[k]; pos.push(x, z, -y); cols.push(col.r * a * 0.55, col.g * a * 0.55, col.b * a * 0.55); }
      line.push(new THREE.Vector3(ax, z + 0.01, -ay), new THREE.Vector3(bx, z + 0.01, -by));
    }
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  const band = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  const ln = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(line), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  band.renderOrder = 5; ln.renderOrder = 5;
  return { band, line: ln };
}

// ── 실 용도 입체(반투명 부피) — 범주별로 합쳐 층마다 범주 수만큼만 그림
export function roomVolumes(rooms, dims) {
  const by = {};
  for (const r of rooms) {
    if (!r.poly || r.poly.length < 3 || r.area < 2) continue;
    const sh = new THREE.Shape(r.poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ExtrudeGeometry(sh, { depth: dims.WALLH * 0.9, bevelEnabled: false, curveSegments: 1 });
    g.rotateX(-Math.PI / 2); g.translate(0, dims.SLAB + 0.04, 0);
    (by[r.cat || 'unk'] = by[r.cat || 'unk'] || []).push(ni(g));
  }
  const out = {};
  for (const [cat, gs] of Object.entries(by)) {
    const col = CATCOL[cat] ?? CATCOL.unk;
    const m = new THREE.Mesh(mergeGeometries(gs), glassMat({ color: col, opacity: 0.07, rim: col, rimK: 0.35, emissive: col, ei: 0.25 }));
    m.renderOrder = 2; m.userData = { own: true, kind: 'roomvol', cat }; out[cat] = m;
  }
  return out;
}

// ── 단면 채움(가로 단면): 벽 발자국(층 기준 높이 0) 한 장 — 높이는 단면 위치에 맞춰 옮김
export function footprintGeom(foot, shafts) {
  const gs = [];
  for (const [o, c, a, b, t] of foot) gs.push(wallBox(o, c, a, b, t, -0.004, 0.004));
  for (const S of shafts) { const [x0, y0, x1, y1] = S.box, w = 0.18; gs.push(boxGeom(x0, y0, x1, y0 + w, -0.004, 0.004), boxGeom(x0, y1 - w, x1, y1, -0.004, 0.004), boxGeom(x0, y0, x0 + w, y1, -0.004, 0.004), boxGeom(x1 - w, y0, x1, y1, -0.004, 0.004)); }
  return gs.length ? mergeGeometries(gs.map(ni)) : null;
}
// 세로 단면 채움: 축('x' = 동서 위치 X 에서 자른 면 / 'y' = 남북 위치 Y) — 벽·문 위 인방·바닥판·계단실 단면 사각형
export function vsectionGeom(foot, poly, shafts, axis, P, dims) {
  const quads = [], { SLAB, WALLH } = dims;
  const quad = (u0, u1, z0, z1) => quads.push([u0, u1, z0, z1]);
  for (const [o, c, a, b, t, z0, z1] of foot) {
    if ((axis === 'x') === (o === 'h')) { if (P >= Math.min(a, b) && P <= Math.max(a, b)) quad(c - t / 2, c + t / 2, z0, z1); }   // 벽을 가로질러 자름 → 두께 단면
    else if (Math.abs(P - c) <= t / 2) quad(Math.min(a, b), Math.max(a, b), z0, z1);                                         // 벽 속을 따라 자름 → 벽 전체 면
  }
  for (const ring of poly) {                                         // 바닥판: 다각형 ∩ 직선
    const xs = [];
    for (let i = 0; i < ring.length; i++) {
      const p = ring[i], q = ring[(i + 1) % ring.length], k = axis === 'x' ? 0 : 1, m = 1 - k;
      if ((p[k] <= P) !== (q[k] <= P)) xs.push(p[m] + ((P - p[k]) * (q[m] - p[m])) / (q[k] - p[k]));
    }
    xs.sort((a, b) => a - b); for (let i = 0; i + 1 < xs.length; i += 2) quad(xs[i], xs[i + 1], 0, SLAB);
  }
  for (const S of shafts) {
    const [x0, y0, x1, y1] = S.box, lo = axis === 'x' ? x0 : y0, hi = axis === 'x' ? x1 : y1, u0 = axis === 'x' ? y0 : x0, u1 = axis === 'x' ? y1 : x1;
    if (P > lo && P < hi) { quad(u0, u0 + 0.2, SLAB, SLAB + WALLH); quad(u1 - 0.2, u1, SLAB, SLAB + WALLH); }
  }
  if (!quads.length) return null;
  const pos = [], uv = [], eps = 0.012;
  for (const [u0, u1, z0, z1] of quads) {
    const P3 = (u, z) => (axis === 'x' ? [P - eps, z, -u] : [u, z, -(P - eps)]);
    const v = [P3(u0, z0), P3(u1, z0), P3(u1, z1), P3(u0, z1)];
    for (const k of [0, 1, 2, 0, 2, 3]) { pos.push(...v[k]); uv.push([u0, u1, u1, u0][k] / 0.6, [z0, z0, z1, z1][k] / 0.6); }
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return g;
}
// 단면 해치 텍스처(사선)
let hatchTex = null;
export function hatchTexture() {
  if (hatchTex) return hatchTex;
  const cv = document.createElement('canvas'); cv.width = cv.height = 64; const c = cv.getContext('2d');
  c.fillStyle = '#fde2e2'; c.fillRect(0, 0, 64, 64); c.strokeStyle = '#ef4444'; c.lineWidth = 7;
  for (let k = -64; k <= 128; k += 21) { c.beginPath(); c.moveTo(k, 64); c.lineTo(k + 64, 0); c.stroke(); }
  hatchTex = new THREE.CanvasTexture(cv); hatchTex.wrapS = hatchTex.wrapT = THREE.RepeatWrapping; hatchTex.colorSpace = THREE.SRGBColorSpace; hatchTex.anisotropy = 4;
  return hatchTex;
}
