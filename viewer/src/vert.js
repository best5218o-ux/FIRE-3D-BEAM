// 수직 동선 — 계단·승강기를 연결층 전체를 꿰는 빛나는 승강로(층 들림·층 분리를 따라 늘어남)로, 층마다 정지 고리, 고른 층은 계단 디딤판(반 꺾임 2경간·참)
// 꼭짓점마다 '몇 번째 층' 속성(aFl)을 두고 정점 셰이더에서 그 층 그룹의 현재 높이(uFy[])를 더함 → 층이 움직여도 CPU 갱신 없이 이어진 기둥(승강로 1개 = 그리기 3회)
// 특별피난계단·비상용승강기: 정보카드 수량·소방시설(EMEV) 추정 위치와 맞춰 강조(확정 = 정보카드 수와 도면 계단 수 일치 / 후보 = 일부만 해당)
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { boxGeom } from './build3d.js';

export function initVert(ctx) {
  const { scene, G, floors, FL, stairs, elevs, D, SLAB, H, WH, COL, pickables } = ctx;
  const NF = floors.length;
  const uFy = { value: new Array(NF).fill(0) };
  const uK = { value: 1 };                                          // 전체 밝기(흐림·X-ray)
  const B = D.building || {}, card = B.stairs || {};

  // ── 특별피난계단·비상용승강기 판정
  const nSp = card['특별피난계단'] || 0;
  if (nSp > 0 && stairs.length) {
    const ord = [...stairs].sort((a, b) => b.floors.length - a.floors.length);
    ord.forEach((s, i) => { if (i < nSp) s.special = { kind: 'sp', cert: nSp >= stairs.length ? 'card' : 'cand', why: nSp >= stairs.length ? `정보카드 특별피난계단 ${nSp}개소 = 도면 계단 ${stairs.length}개소` : `정보카드 특별피난계단 ${nSp}개소 / 도면 계단 ${stairs.length}개소 — 연결층이 긴 계단을 후보로 표시` }; });
  }
  for (const s of stairs) if (/특별/.test(s.kind || '') || /특별/.test(s.name || '')) s.special = { kind: 'sp', cert: 'card', why: '도면·검수 표기' };
  const em = [];
  if (D.fac && D.fac.floors) for (const items of Object.values(D.fac.floors)) for (const it of items) if (it.code === 'EMEV') for (const q of it.est || []) if (q.box) em.push(q);
  const ov = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  for (const e of elevs) {
    const q = em.find((z) => ov(z.box, e.box) > 0.3 * (e.box[2] - e.box[0]) * (e.box[3] - e.box[1]));
    if (q) e.special = { kind: 'emev', cert: q.alt ? 'cand' : 'card', why: q.why || '소방시설 비상용승강기 추정 위치' };
    else if (/비상/.test(e.note || '') || /비상/.test(e.name || '')) e.special = { kind: 'emev', cert: 'card', why: '도면 표기' };
  }

  // ── 재질: 층 높이 이동(aFl → uFy) + 프레넬 테두리
  const addFy = (sh) => {
    sh.uniforms.uFy = uFy; sh.uniforms.uK = uK;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', `#include <common>\nattribute float aFl; uniform float uFy[${NF}];`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\ntransformed.y += uFy[int(aFl + 0.5)];');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uK;').replace('#include <dithering_fragment>', '#include <dithering_fragment>\ngl_FragColor.a *= uK;');
  };
  const glass = (color, op) => {
    const m = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.28, transparent: true, opacity: op, depthWrite: false, side: THREE.DoubleSide, roughness: 0.3 });
    m.onBeforeCompile = (sh) => {
      addFy(sh);
      sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float fres = pow(1.0 - abs(dot(normalize(vViewPosition), normal)), 2.0); totalEmissiveRadiance += emissive * fres * 1.4; diffuseColor.a = clamp(diffuseColor.a * (0.5 + 1.8 * fres), 0.0, 1.0);`);
    };
    m.customProgramCacheKey = () => 'shaftG' + NF; m.userData.op = op; return m;
  };
  const basic = (color, op, add = true) => { const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false, blending: add ? THREE.AdditiveBlending : THREE.NormalBlending, side: THREE.DoubleSide }); m.onBeforeCompile = addFy; m.customProgramCacheKey = () => 'shaftB' + NF; m.userData.op = op; return m; };
  const lineM = (color, op) => { const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: op, depthWrite: false, blending: THREE.AdditiveBlending }); m.onBeforeCompile = addFy; m.customProgramCacheKey = () => 'shaftL' + NF; m.userData.op = op; return m; };

  // 사각 기둥(옆면만) — 층마다 고리 2개(바닥판 위·층 꼭대기), 고리 사이를 옆면으로 이음. 들림·분리 시 고리 사이가 늘어나 이어짐
  function prism(box, idx, shrink = 0) {
    const [x0, y0, x1, y1] = box, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, k = 1 - shrink;
    const C = [[cx + (x0 - cx) * k, cy + (y0 - cy) * k], [cx + (x1 - cx) * k, cy + (y0 - cy) * k], [cx + (x1 - cx) * k, cy + (y1 - cy) * k], [cx + (x0 - cx) * k, cy + (y1 - cy) * k]];
    const rings = []; for (const i of idx) rings.push([i, SLAB * 0.5], [i, H]);
    const pos = [], fl = [], nor = [];
    for (let r = 0; r < rings.length - 1; r++) {
      const [fa, za] = rings[r], [fb, zb] = rings[r + 1];
      for (let s = 0; s < 4; s++) {
        const [ax, ay] = C[s], [bx, by] = C[(s + 1) % 4], nx = by - ay, ny = -(bx - ax), nl = Math.hypot(nx, ny) || 1;
        const v = [[ax, ay, za, fa], [bx, by, za, fa], [bx, by, zb, fb], [ax, ay, zb, fb]];
        for (const q of [0, 1, 2, 0, 2, 3]) { const [x, y, z, f] = v[q]; pos.push(x, z, -y); fl.push(f); nor.push(nx / nl, 0, -ny / nl); }
      }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3)); g.setAttribute('aFl', new THREE.Float32BufferAttribute(fl, 1));
    return g;
  }
  function rails(box, idx) {                                       // 네 모서리 세로선 + 층마다 정지 고리(바닥판 높이)
    const [x0, y0, x1, y1] = box, C = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], pos = [], fl = [];
    const seg = (a, fa, b, fb) => { pos.push(a[0], a[2], -a[1], b[0], b[2], -b[1]); fl.push(fa, fb); };
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k];
      for (let s = 0; s < 4; s++) {
        const [ax, ay] = C[s], [bx, by] = C[(s + 1) % 4];
        seg([ax, ay, SLAB + 0.03], i, [bx, by, SLAB + 0.03], i);                       // 정지 고리
        seg([ax, ay, SLAB * 0.5], i, [ax, ay, H], i);                                  // 층 안 세로선
        if (k + 1 < idx.length) seg([ax, ay, H], i, [ax, ay, SLAB * 0.5], idx[k + 1]); // 다음 층까지(늘어나는 구간)
      }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('aFl', new THREE.Float32BufferAttribute(fl, 1)); return g;
  }

  // 반 꺾임 계단(2경간 + 중간참 + 손스침) — 층 그룹 기준 높이
  function flights(box) {
    const [x0, y0, x1, y1] = box, wx = x1 - x0, wy = y1 - y0, along = wx >= wy, L = along ? wx : wy, Wd = (along ? wy : wx) / 2;
    const n = Math.max(8, Math.round(WH / 0.18)), m = Math.ceil(n / 2), rise = WH / (2 * m);
    const dn = Math.min(1.0, L * 0.16), df = Math.min(1.3, L * 0.24), run = Math.max(0.5, L - dn - df), tr = run / m, gs = [];
    const B = (u0, u1, v0, v1, z0, z1) => gs.push(along ? boxGeom(x0 + u0, y0 + v0, x0 + u1, y0 + v1, z0, z1) : boxGeom(x0 + v0, y0 + u0, x0 + v1, y0 + u1, z0, z1));
    for (let k = 0; k < m; k++) {
      const z = SLAB + (k + 1) * rise; B(dn + k * tr, dn + (k + 1) * tr, 0.05, Wd - 0.05, z - 0.07, z);                  // 1경간(올라감)
      const z2 = SLAB + (m + k + 1) * rise; B(L - df - (k + 1) * tr, L - df - k * tr, Wd + 0.05, 2 * Wd - 0.05, z2 - 0.07, z2);   // 2경간(되돌아 올라감)
    }
    B(L - df, L, 0.05, 2 * Wd - 0.05, SLAB + m * rise - 0.12, SLAB + m * rise);                                         // 중간참
    B(dn, L - df, Wd - 0.04, Wd + 0.04, SLAB, SLAB + 0.9);                                                              // 가운데 벽(난간)
    return mergeGeometries(gs.map((g) => (g.index ? g.toNonIndexed() : g)));
  }
  function flightRails(box) {                                      // 손스침(기울어진 선)
    const [x0, y0, x1, y1] = box, wx = x1 - x0, wy = y1 - y0, along = wx >= wy, L = along ? wx : wy, Wd = (along ? wy : wx) / 2;
    const n = Math.max(8, Math.round(WH / 0.18)), m = Math.ceil(n / 2), rise = WH / (2 * m), dn = Math.min(1.0, L * 0.16), df = Math.min(1.3, L * 0.24);
    const P = (u, v, z) => (along ? new THREE.Vector3(x0 + u, z, -(y0 + v)) : new THREE.Vector3(x0 + v, z, -(y0 + u)));
    const pts = [P(dn, Wd - 0.08, SLAB + 0.9), P(L - df, Wd - 0.08, SLAB + m * rise + 0.9), P(L - df, Wd + 0.08, SLAB + m * rise + 0.9), P(dn, Wd + 0.08, SLAB + 2 * m * rise + 0.9)];
    return new THREE.BufferGeometry().setFromPoints([pts[0], pts[1], pts[2], pts[3]]);
  }

  const shafts = [], root = new THREE.Group(); root.userData.own = true; scene.add(root);
  for (const [kind, S] of [...stairs.map((s) => ['stair', s]), ...elevs.map((e) => ['ev', e])]) {
    const fl = S.floors.filter((f) => FL[f]); if (!fl.length) continue;
    const idx = fl.map((f) => FL[f].idx).sort((a, b) => a - b);
    const sp = S.special, col = sp ? (sp.kind === 'sp' ? 0xfacc15 : 0xfb923c) : COL[kind];
    const gGlass = new THREE.Mesh(prism(S.box, idx), glass(col, kind === 'ev' ? 0.2 : 0.12)); gGlass.renderOrder = 4;
    const gBeam = new THREE.Mesh(prism(S.box, idx, 0.45), basic(col, sp ? 0.16 : 0.1)); gBeam.renderOrder = 4;
    const gRail = new THREE.LineSegments(rails(S.box, idx), lineM(col, sp ? 1 : 0.85)); gRail.renderOrder = 6;
    for (const o of [gGlass, gBeam, gRail]) { o.frustumCulled = false; o.userData.own = true; root.add(o); }
    const Z = { kind, S, idx, glass: gGlass, beam: gBeam, rail: gRail, per: {} };
    // 층마다: 고르기용 상자(평소 안 그림, 단층 평면 보기에서는 표시) + 계단 디딤판
    for (const fid of fl) {
      const P = G[fid]; if (!P) continue;
      const pm = new THREE.MeshStandardMaterial({ color: col, transparent: true, opacity: kind === 'ev' ? 0.3 : 0.16, emissive: col, emissiveIntensity: 0.35, depthWrite: false });
      pm.visible = false;
      const m = new THREE.Mesh(boxGeom(S.box[0], S.box[1], S.box[2], S.box[3], SLAB, SLAB + WH - 0.05), pm);
      m.userData = { own: true, kind, name: S.name, S, fid, layer: kind, pickBox: true }; P.vert.add(m); pickables.push(m);
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
      e.userData = { own: true, layer: kind, edge: true, oneOnly: true }; e.visible = false; P.vert.add(e);
      Z.per[fid] = { pick: m, edge: e };
      if (kind === 'stair') {
        const st = new THREE.Mesh(flights(S.box), new THREE.MeshStandardMaterial({ color: sp ? 0xfde68a : 0x86efac, emissive: sp ? 0x713f12 : 0x14532d, emissiveIntensity: 0.7, roughness: 0.6 }));
        st.userData = { own: true, layer: 'stair', steps: true }; P.vert.add(st); P.steps.push(st);
        const hr = new THREE.Line(flightRails(S.box), new THREE.LineBasicMaterial({ color: sp ? 0xfacc15 : 0x4ade80 }));
        hr.userData = { own: true, layer: 'stair', steps: true }; P.vert.add(hr); P.steps.push(hr);
        Z.per[fid].steps = st;
      }
    }
    shafts.push(Z);
  }

  // 매 화면: 층 그룹 높이 → uFy
  function sync() { for (const f of floors) uFy.value[f.idx] = G[f.id].g.position.y; }
  // 보기 상태: one(단층 평면)에서는 기둥 대신 그 층 상자, xray 는 더 밝게, dimK 흐림
  function style({ one, xray, dimAll, layer }) {
    uK.value = dimAll ? 0.45 : 1;
    for (const Z of shafts) {
      const on = layer[Z.kind] !== false;
      for (const o of [Z.glass, Z.beam, Z.rail]) o.visible = on && !one;
      Z.glass.material.opacity = Z.glass.material.userData.op * (xray ? 1.6 : 1);
      Z.beam.material.opacity = Z.beam.material.userData.op * (xray ? 2.2 : 1);
      for (const p of Object.values(Z.per)) { p.pick.visible = on; p.pick.material.visible = on && one; p.edge.visible = on && one; }
    }
  }
  const list = () => shafts.map((Z) => ({ id: Z.S.id, kind: Z.kind, name: Z.S.name, floors: Z.S.floors, special: Z.S.special || null }));
  return { sync, style, shafts, root, list, uFy };
}
