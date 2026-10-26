// 단면 보기 — 가로(높이) 단면 · 세로(동서/남북) 단면 + 단면 채움(해치)
// 잘림은 재질마다 같은 평면 2장(localClipping)을 공유 — 꺼짐 = 아주 먼 평면(셰이더 재컴파일 없음).
// 채움은 스텐실 대신 벽 중심선·두께·바닥판 다각형에서 직접 계산(어떤 GPU 에서도 같은 결과): 가로 = 벽 발자국 한 장을 단면 높이로 옮김, 세로 = 평면과 만나는 벽·인방·바닥판 사각형.
import * as THREE from 'three';
import { footprintGeom, vsectionGeom, hatchTexture } from './build3d.js';

export function initSection(ctx) {
  const { scene, floors, G, SLAB, WALLH, H, bounds, shells, shaftsOf, invalidate, focusFid } = ctx;
  const FAR = 1e6;
  const hPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), FAR), vPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), FAR);
  const PL = [hPlane, vPlane], onlyV = [vPlane], onlyH = [hPlane];
  const S = { h: false, hv: 0.38, v: false, axis: 'x', vv: 0.5, flip: false };
  const capMat = (planes, col = 0xffffff) => { const m = new THREE.MeshBasicMaterial({ map: hatchTexture(), color: col, side: THREE.DoubleSide, clippingPlanes: planes, polygonOffset: true, polygonOffsetFactor: -1 }); m.userData.noClip = true; return m; };
  const hMat = capMat(onlyV), hSlab = capMat(onlyV, 0x93a4bd), vMat = capMat(onlyH);
  const caps = {};
  for (const f of floors) {
    const sh = shells[f.id]; if (!sh) continue;
    const fg = footprintGeom(sh.foot, shaftsOf(f.id));
    const hw = fg ? new THREE.Mesh(fg, hMat) : null;
    let hs = null;
    if (f.poly.length) {
      const shapes = f.poly.map((ring) => new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y))));
      const g = new THREE.ShapeGeometry(shapes); g.rotateX(-Math.PI / 2);
      const uv = g.attributes.uv; for (let i = 0; i < uv.count; i++) uv.setXY(i, g.attributes.position.getX(i) / 0.6, g.attributes.position.getZ(i) / 0.6);
      hs = new THREE.Mesh(g, hSlab);
    }
    for (const m of [hw, hs]) if (m) { m.visible = false; m.renderOrder = 9; m.userData.own = true; m.userData.cap = true; G[f.id].g.add(m); }
    caps[f.id] = { hw, hs, v: null };
  }
  // 세로 단면 위치(도면 좌표)
  const vPos = () => (S.axis === 'x' ? bounds.bx0 + S.vv * (bounds.bx1 - bounds.bx0) : bounds.by0 + S.vv * (bounds.by1 - bounds.by0));
  let vKey = '';
  function rebuildV() {
    const P = vPos(), key = `${S.v}|${S.axis}|${P.toFixed(3)}|${S.flip}`;
    if (key === vKey) return; vKey = key;
    for (const f of floors) {
      const c = caps[f.id]; if (!c) continue;
      if (c.v) { c.v.parent.remove(c.v); c.v.geometry.dispose(); c.v = null; }
      if (!S.v) continue;
      const g = vsectionGeom(shells[f.id].foot, f.poly, shaftsOf(f.id), S.axis, P, { SLAB, WALLH });
      if (!g) continue;
      if (S.flip) { const p = g.attributes.position; for (let i = 0; i < p.count; i++) { if (S.axis === 'x') p.setX(i, p.getX(i) + 0.024); else p.setZ(i, p.getZ(i) - 0.024); } }
      const m = new THREE.Mesh(g, vMat); m.renderOrder = 9; m.userData.own = true; m.userData.cap = true; G[f.id].g.add(m); c.v = m;
    }
  }
  // 높이 범위(현재 층 그룹 위치)
  function yRange() { let a = Infinity, b = -Infinity; for (const f of floors) { const g = G[f.id].g; if (!g.visible) continue; a = Math.min(a, g.position.y); b = Math.max(b, g.position.y + H); } return isFinite(a) ? [a, b] : [0, H]; }
  // 층을 골랐으면 그 층 바닥 위 높이(0 ~ 벽 높이의 1.15배), 아니면 건물 전체 높이 비율
  function hWorld() { const f = focusFid(); if (f && G[f]) return G[f].g.position.y + SLAB + S.hv * WALLH * 1.15; const [a, b] = yRange(); return a + S.hv * (b - a); }
  function hText() { const f = focusFid(); return f ? `${f} 바닥에서 ${(S.hv * WALLH * 1.15).toFixed(1)}m` : `건물 높이 ${(S.hv * 100).toFixed(0)}%`; }
  // 매 화면(단면 켜짐일 때)
  function update() {
    const h = hWorld();
    hPlane.constant = S.h ? h : FAR;
    if (S.v) { const P = vPos(); if (S.axis === 'x') vPlane.normal.set(S.flip ? 1 : -1, 0, 0), (vPlane.constant = S.flip ? -P : P); else vPlane.normal.set(0, 0, S.flip ? -1 : 1), (vPlane.constant = S.flip ? -P : P); }
    else vPlane.constant = FAR;
    for (const f of floors) {
      const c = caps[f.id]; if (!c) continue;
      const g = G[f.id].g, loc = h - g.position.y;
      if (c.hw) { c.hw.visible = S.h && g.visible && loc >= SLAB + 0.02 && loc <= SLAB + WALLH; c.hw.position.y = loc - 0.015; }
      if (c.hs) { c.hs.visible = S.h && g.visible && loc > 0 && loc < SLAB + 0.02; c.hs.position.y = Math.min(loc, SLAB) - 0.015; }
    }
    rebuildV();
  }
  // 재질에 평면 연결(새로 만든 객체 포함 — 단면을 바꿀 때·주기적으로 호출)
  function attach() {
    scene.traverse((o) => {
      const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
      for (const m of ms) { if (m.userData.noClip || m.clippingPlanes === PL) continue; m.clippingPlanes = PL; m.needsUpdate = true; }
    });
  }
  // 잘린 쪽 라벨 숨김(CSS 라벨은 GL 잘림이 안 닿음)
  const wp = new THREE.Vector3();
  function labels() {
    const on = S.h || S.v;
    scene.traverse((o) => {
      if (!o.isCSS2DObject) return;
      if (!on) { if (o.layers.mask !== 1) o.layers.set(0); return; }
      o.getWorldPosition(wp); const cut = hPlane.distanceToPoint(wp) < -0.05 || vPlane.distanceToPoint(wp) < -0.05;
      o.layers.set(cut ? 31 : 0);
    });
  }
  function set(o) { Object.assign(S, o); attach(); update(); labels(); invalidate(); return { ...S }; }
  return { S, set, update, attach, labels, active: () => S.h || S.v, planes: PL, hWorld, hText, vPos, caps };
}
