// 소방활동 3D 지휘화면 — 정보카드 자동 변환 결과(D, BIM) 표시 · 화점 경로 · 호스/공기호흡기 · 인명검색 순서
// 화면 구성(v2): 머리줄 · 왼쪽 탭(확인사항·개요·위험물·설비·진입·원리) · 가운데 3D(범례 칩·층 버튼·보기 전환·나침반) · 오른쪽(층 요약/상세·화재 대응) · 바닥줄(원천)
// v3: 두께 있는 유리 벽(문 개구부·문짝·외벽 유리 띠)·접촉 음영·실 용도 입체 · 빛나는 승강로(vert.js) · 단면/투시/층 격리(section.js) · 확산 시뮬레이션(sim.js)·대원 이동(crew.js)
//     · 진입로 따라가기(follow.js) · 미니맵(minimap.js) · 거리·면적·화면 저장·브리핑(tools.js) · 단축키·터치·화면 속도 조절 · WebGL 없으면 2D 대체 보기
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { initExt } from './ext.js';
import { initFac } from './fac.js';                         // 소방시설 레이어(분류·위치 확실성)
import { initEvac } from './evac.js';
import { initSearch, CAT_KO } from './search.js';        // 인명검색 우선순위(용도·시간대·대피 곤란·공격계단 거리)
import { initTrain } from './train.js';                  // 훈련 모드 채점
import { callout, declutter, pin, flameTexture } from './hud.js';
import { initPanels } from './panels.js';
import { floorShell, aoPlane, slabGlow, roomVolumes, glassMat, CATCOL, CATNAME } from './build3d.js';
import { initVert } from './vert.js';
import { initSection } from './section.js';
import { initSim, TTU_BANDS } from './sim.js';
import { initCrew } from './crew.js';
import { initFollow } from './follow.js';
import { initMinimap } from './minimap.js';
import { initTools } from './tools.js';
import { svg } from './icons.js';                          // v3.1: 인라인 SVG 아이콘 1벌(오프라인)

const { D, BIM } = window.__DATA__;
const H = BIM.H || 4.6, SLAB = 0.3, WH = H - SLAB, WALLH = Math.min(WH, Math.max(2.6, WH * 0.8));
const $ = (id) => document.getElementById(id);
const LAY = { leftX: 70, topY: 53, XR: 0, stackTop: 0, W: 0, Hs: 0, topBand: 53 };   // 고정 UI 배치 결과(layoutUI)
const UI_SEL = ['.lay:not(.ovf)', '#layMore', '#dock', '.views', '#fbar', '#compass', '#mini', '#tl', '#volLeg', '#secPanel', '#follow', '#hint', '#info', '#glmsg'];
const vis = (el) => el && !el.hidden && el.offsetParent !== null && getComputedStyle(el).display !== 'none' && getComputedStyle(el).visibility !== 'hidden';
const px = (v) => Math.round(v) + 'px';
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const COL = { exit: 0x22c55e, entrance: 0x22d3ee, door: 0xc8a27a, hose: 0xef4444, gas: 0xf59e0b, panel: 0xa78bfa, stair: 0x4ade80, ev: 0x60a5fa };
const EQNAME = { hose: '옥내소화전', gas: '가스 차단밸브', panel: '수신기' };
const EQLAYER = { hose: 'hose', gas: 'key', panel: 'key' };
const centroid = (p) => p.reduce((a, q) => [a[0] + q[0] / p.length, a[1] + q[1] / p.length], [0, 0]);
// 대피 계산에서 경로가 없어 중심점이 빠진 실(계단 미검출 층 등) → 다각형 중심으로 보충(화점·인명검색 계산이 멈추지 않게)
if (D.evac && D.evac.floors) for (const r of Object.values(D.evac.floors)) for (const q of r.rooms || []) if (!q.c && q.poly && q.poly.length) q.c = centroid(q.poly).map((v) => Math.round(v * 100) / 100);

// ── 층 순서(지하→지상), 표시 높이
const floors = [...D.floors].sort((a, b) => a.level - b.level);
const groundIdx = Math.max(0, floors.findIndex((f) => f.level === 0));
floors.forEach((f, i) => { f.idx = i; f.ord = i - groundIdx; });
const FL = Object.fromEntries(floors.map((f) => [f.id, f]));
let explode = 0, viewMode = 'all', selId = floors[groundIdx].id;
let focus = false;                                         // 사용자가 층을 골랐는가(아니면 건물 전체 요약)
let fire = null, wallWarn = false, picking = false;
let fireMode = false, rTab = 'info';                       // 오른쪽 패널: 화재 대응 보기
let ext = null;                                            // 기능 12·18·24 (ext.js)
let fac = null;                                            // 소방시설 레이어 (fac.js)
let evac = null;                                           // 대피로·진입로 (evac.js)
let search = null, train = null;                           // 인명검색 우선순위(search.js) · 훈련 채점(train.js)
let lastPlan = null;                                       // 마지막 화점 산출안(훈련 채점 기준)
let xray = false, isolate = false, lowQ = false, shadows = false, farView = false;   // v3 보기 상태(farView: 카메라가 멀어 작은 글자 숨김)
// 활동계획 기준값(firebim/sop.py ACT — 임시값). 옛 데이터(D.sop_act 없음)는 이전 동작과 같은 값
const ACT = Object.assign({ atk_w_standpipe_m: 0, standpipe_r_m: 5, atk_w_demand_m: 0, hose_len_m: 15, hose_slack_m: 0, hose_carry: 2, walk_mps: 0.8, stair_s_per_floor: 25, scba_cyl_l: 6.8, scba_fill_bar: 300, scba_return_bar: 55, scba_lpm: 55, scba_margin_min: 5 }, D.sop_act || {});
const LIFT = Math.max(9, H * 2.2);                         // 층을 고르면 위층을 이만큼 들어 그 층을 드러냄
const fireActive = () => !!(fire || picking);
const liftOn = () => focus && viewMode === 'all' && !fireActive() && !isolate;
const zOf = (fid) => FL[fid].ord * (H + explode) + (liftOn() && FL[fid].level > FL[selId].level ? LIFT : 0);

// ── three 기본 (WebGL 을 못 쓰면 2D 대체 보기: 그리기 없는 대역 렌더러로 나머지 계산·패널은 그대로)
const stage = $('stage'), canvas = $('c');
let GL = true, renderer;
const NOGL = /[?&]nogl\b/.test(location.search) || window.__FORCE_NOGL;
try {
  if (NOGL) throw new Error('2D 대체 보기 강제');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
} catch (e) {
  GL = false;
  renderer = { domElement: canvas, shadowMap: { enabled: false }, info: { render: { calls: 0, triangles: 0 } }, localClippingEnabled: false, setPixelRatio() {}, getPixelRatio: () => 1, setClearColor() {}, setSize(w, h) { canvas.width = w; canvas.height = h; }, render() {} };
  console.info('[FireBIM] WebGL 사용 불가 → 2D 대체 보기', e && e.message);
}
const PR0 = Math.min(2, devicePixelRatio || 1);
renderer.setPixelRatio(PR0); renderer.setClearColor(0x000000, 0); renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 5000);
const labelR = new CSS2DRenderer({ element: $('labels') });
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.dampingFactor = 0.12; controls.maxPolarAngle = Math.PI * 0.495;
controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x0b1220, 1.25));
const sun = new THREE.DirectionalLight(0xffffff, 1.1); sun.position.set(-80, 160, 120); scene.add(sun); scene.add(sun.target);
let dirty = true; const invalidate = () => (dirty = true);

// 전체 범위
let bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
for (const f of floors) { bx0 = Math.min(bx0, f.bb[0]); by0 = Math.min(by0, f.bb[1]); bx1 = Math.max(bx1, f.bb[2]); by1 = Math.max(by1, f.bb[3]); }
const CX = (bx0 + bx1) / 2, CY = (by0 + by1) / 2, SPAN = Math.max(bx1 - bx0, by1 - by0, 20);
const V = (x, y, z) => new THREE.Vector3(x, z, -y);     // 도면(x,y 북쪽+) → three(X, 높이, -Z)
scene.fog = new THREE.Fog(0x0b1220, SPAN * 3.2, SPAN * 8);
controls.maxDistance = SPAN * 6; controls.minDistance = 4;
// 그림자(선택, 기본 끔): 해 그림자 범위 = 건물
sun.position.set(CX - SPAN * 0.85, SPAN * 0.95, -CY + SPAN * 0.75); sun.target.position.set(CX, 0, -CY);
{ const s = sun.shadow, R = SPAN * 1.05; s.mapSize.set(2048, 2048); Object.assign(s.camera, { left: -R, right: R, top: R, bottom: -R, near: 1, far: SPAN * 6 }); s.camera.updateProjectionMatrix(); s.bias = -0.0006; s.normalBias = 0.04; if ('intensity' in s) s.intensity = 0.85; }

function label(text, cls, x, y, z, parent) {
  const el = document.createElement('div'); el.className = 'lb ' + cls; el.textContent = text;
  const o = new CSS2DObject(el); o.position.copy(V(x, y, z)); parent.add(o); return o;
}
function boxGeom(x0, y0, x1, y1, z0, z1) {
  const g = new THREE.BoxGeometry(Math.max(0.05, x1 - x0), Math.max(0.05, z1 - z0), Math.max(0.05, y1 - y0));
  g.translate((x0 + x1) / 2, (z0 + z1) / 2, -(y0 + y1) / 2); return g;
}
const texLoader = new THREE.TextureLoader();
const pickables = [];
const own = (o) => { o.userData.own = true; return o; };    // 이 파일이 직접 스타일을 정하는 객체(일괄 흐림 대상 아님)
const lineMat = (color, opacity) => new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
const noClip = (o) => { o.traverse((q) => { const ms = q.material ? (Array.isArray(q.material) ? q.material : [q.material]) : []; ms.forEach((m) => (m.userData.noClip = true)); }); return o; };

// ── 바닥 격자 · 북쪽 화살표 · 그림자 받는 바닥
const ground = new THREE.Group(); scene.add(ground);
let shadowCatch = null;
{
  const cell = SPAN > 140 ? 10 : 5, size = Math.ceil(Math.max(SPAN * 3.2, 160) / cell / 2) * cell * 2;
  const grid = new THREE.GridHelper(size, size / cell, 0x1e3a5f, 0x13233b);
  grid.material.transparent = true; grid.material.opacity = 0.55; grid.material.depthWrite = false; grid.position.set(CX, 0, -CY); ground.add(grid);
  const disc = new THREE.Mesh(new THREE.CircleGeometry(SPAN * 1.15, 64), new THREE.MeshBasicMaterial({ color: 0x12305a, transparent: true, opacity: 0.18, depthWrite: false }));
  disc.rotation.x = -Math.PI / 2; disc.position.set(CX, 0.02, -CY); ground.add(disc);
  shadowCatch = new THREE.Mesh(new THREE.PlaneGeometry(SPAN * 3, SPAN * 3), new THREE.ShadowMaterial({ opacity: 0.38 }));
  shadowCatch.rotation.x = -Math.PI / 2; shadowCatch.position.set(CX, 0.03, -CY); shadowCatch.receiveShadow = true; shadowCatch.visible = false; ground.add(shadowCatch);
  // 북쪽 화살표: 건물 오른쪽 앞(층 꼬리표 기둥과 겹치지 않게)
  const L = Math.max(9, SPAN * 0.16), ox = bx1 + Math.max(6, SPAN * 0.12), oy = by0;
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), V(ox, oy, 0.1), L, 0x60a5fa, L * 0.3, L * 0.16); ground.add(arrow);
  const el = document.createElement('div'); el.className = 'north'; el.textContent = 'N';
  const n = new CSS2DObject(el); n.position.copy(V(ox, oy + L + 2.5, 0.1)); ground.add(n);
  noClip(ground);
}

// ── 층별 그룹
const G = {};
const callouts = [];                                        // {o, fid, kind, home}
const SHELL = {};                                           // 층별 벽 형상(단면 채움용 발자국 포함)
const stairs = BIM.stairs.map((s) => ({ ...s })), elevs = BIM.elevs.map((e) => ({ ...e }));
const evRooms = (fid) => (D.evac && D.evac.floors && D.evac.floors[fid] ? D.evac.floors[fid].rooms : null);
let nEnt = 0;
// 층 꼬리표 기둥 자리: 지상층 외곽의 남서 모서리(지하층도 같은 기둥 — 건물 속·아래에 묻히지 않게)
const upper = floors.filter((f) => f.level >= 0 && f.poly.length), tagSrc = upper.length ? upper : floors;
const TAGX = Math.min(...tagSrc.map((f) => f.bb[0])) - Math.max(2.5, SPAN * 0.03), TAGY = Math.min(...tagSrc.map((f) => f.bb[1])) - Math.max(1.5, SPAN * 0.02);
for (const f of floors) {
  const fb = BIM.floors[f.id] || { walls: [], doors: [], spaces: [] };
  const g = new THREE.Group(); g.userData.fid = f.id; scene.add(g);
  const parts = G[f.id] = { g, walls: new THREE.Group(), doors: new THREE.Group(), tex: null, evac: null, spaces: new THREE.Group(), eq: new THREE.Group(), labels: new THREE.Group(), slab: null,
    pins: new THREE.Group(), rooms: new THREE.Group(), vert: new THREE.Group(), vols: new THREE.Group(), steps: [], shell: null, shellEdge: null, wallMat: [], wallEdge: null, tag: null, glow: null, ao: null, glassM: null, frameM: null, leafM: null };
  Object.values(parts).forEach((p) => p && p.isObject3D && p !== g && g.add(p));
  const [x0, y0, x1, y1] = f.bb;
  // 바닥 슬래브(옆면 빛) + 유리 외피(층 높이) + 빛나는 윤곽선 + 테두리 빛 띠
  const shapes = f.poly.map((ring) => new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y))));
  if (shapes.length) {
    const sg = new THREE.ExtrudeGeometry(shapes, { depth: SLAB, bevelEnabled: false }); sg.rotateX(-Math.PI / 2);
    const slabTop = new THREE.MeshStandardMaterial({ color: 0x1c2b44, transparent: true, opacity: 0.85, roughness: 0.9, metalness: 0.05, depthWrite: false });
    const slabSide = new THREE.MeshStandardMaterial({ color: 0x0e2a46, emissive: 0x1d6fa8, emissiveIntensity: 0.9, transparent: true, opacity: 0.9, depthWrite: false });
    const slab = new THREE.Mesh(sg, [slabTop, slabSide]); slab.receiveShadow = true;
    slab.userData = { own: true, kind: 'slab', fid: f.id, name: f.name + ' 바닥' }; g.add(slab); parts.slab = slab; pickables.push(slab);
    const hg = new THREE.ExtrudeGeometry(shapes, { depth: WH - 0.05, bevelEnabled: false }); hg.rotateX(-Math.PI / 2); hg.translate(0, SLAB, 0);
    const shell = new THREE.Mesh(hg, glassMat({ color: 0x7dd3fc, opacity: 0.045, rim: 0x38bdf8, rimK: 0.22, emissive: 0x0b3b5a, ei: 0.2 }));
    shell.userData = { own: true, kind: 'shell', fid: f.id, name: f.name }; shell.renderOrder = 1; g.add(shell); parts.shell = shell; pickables.push(shell);
    parts.shellEdge = own(new THREE.LineSegments(new THREE.EdgesGeometry(hg, 25), lineMat(0x38bdf8, 0.45))); g.add(parts.shellEdge);
    const gl = slabGlow(f.poly, SLAB, 0x38bdf8, Math.max(0.6, Math.min(1.6, SPAN / 90))); own(gl.band); own(gl.line); g.add(gl.band, gl.line); parts.glow = gl;
  }
  // 도면 바닥 텍스처
  const plane = (url, zo, op) => {
    const t = texLoader.load(url, invalidate); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
    const m = own(new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), new THREE.MeshBasicMaterial({ map: t, transparent: true, opacity: op, depthWrite: false })));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, SLAB + zo, -(y0 + y1) / 2); m.renderOrder = 2; g.add(m); return m;
  };
  if (f.tex) parts.tex = plane(f.tex, 0.02, 0.95);
  if (f.evac) { parts.evac = plane(f.evac, 0.05, 1); parts.evac.material.blending = THREE.AdditiveBlending; parts.evac.visible = false; }
  // 벽: 두께 있는 유리(외벽·내벽, 아래쪽 접촉 음영) · 외벽 유리 띠·멀리언 · 문 위 인방 · 빛나는 윤곽선
  const SH = SHELL[f.id] = floorShell(fb, { SLAB, WALLH });
  const wallMesh = (geo, mat, name) => { const m = own(new THREE.Mesh(geo, mat)); m.userData = { own: true, kind: 'walls', fid: f.id, name }; m.renderOrder = 3; m.castShadow = true; parts.walls.add(m); return m; };
  const wName = `${f.name} 벽체 ${fb.walls.length}개`;
  const allW = [];
  if (SH.wallE) { const m = glassMat({ color: 0x5fb8e6, opacity: 0.22, rim: 0x7dd3fc, rimK: 0.55, aoY: [SLAB, 1.0] }); parts.wallMat.push(m); wallMesh(SH.wallE, m, wName); allW.push(SH.wallE); }
  if (SH.wallI) { const m = glassMat({ color: 0xcfeaff, opacity: 0.13, rim: 0xbae6fd, rimK: 0.4, aoY: [SLAB, 1.0] }); parts.wallMat.push(m); wallMesh(SH.wallI, m, wName); allW.push(SH.wallI); }
  if (SH.lintel) { const m = glassMat({ color: 0xcfeaff, opacity: 0.16, rim: 0xbae6fd, rimK: 0.4 }); parts.wallMat.push(m); wallMesh(SH.lintel, m, wName); allW.push(SH.lintel); }
  if (SH.glass) { const m = glassMat({ color: 0x7dd3fc, opacity: 0.1, rim: 0xa5f3fc, rimK: 1.1, emissive: 0x0e4a6e, ei: 0.7 }); m.userData.op = 0.1; parts.glassM = m; wallMesh(SH.glass, m, `${f.name} 외벽 유리 띠(표현)`); }
  if (SH.frame) { const m = new THREE.MeshStandardMaterial({ color: 0x9ddcff, emissive: 0x2a86c4, emissiveIntensity: 0.8, transparent: true, opacity: 0.55, depthWrite: false }); m.userData.op = 0.55; parts.frameM = m; wallMesh(SH.frame, m, `${f.name} 창틀(표현)`); }
  if (allW.length) { parts.wallEdge = own(new THREE.LineSegments(new THREE.EdgesGeometry(allW.length > 1 ? mergeGeometries(allW) : allW[0], 20), lineMat(0x8fdcff, 0.5))); parts.walls.add(parts.wallEdge); }
  // 문짝(열린 상태)·회전 호 · 진입구(하늘색)·비상구(초록) 문틀 빛
  if (SH.leaves) {
    const lm = new THREE.MeshStandardMaterial({ color: COL.door, emissive: 0x5a3d1e, emissiveIntensity: 0.5, transparent: true, opacity: 0.75, roughness: 0.6 }); parts.leafM = lm;
    const m = own(new THREE.Mesh(SH.leaves, lm)); m.userData = { own: true, kind: 'door', fid: f.id, layer: 'walls', leafDoors: SH.leafDoors, doors: fb.doors }; m.castShadow = true; parts.doors.add(m); pickables.push(m);
  }
  if (SH.arcs) { const a = own(new THREE.LineSegments(SH.arcs, new THREE.LineBasicMaterial({ color: 0xe8c79a, transparent: true, opacity: 0.55, depthWrite: false }))); a.userData.layer = 'walls'; parts.doors.add(a); }
  let ek = 0;
  for (const E of SH.ent) {
    const [kind, , x, y, w, name, extd, gid] = E.door;
    const m = own(new THREE.Mesh(E.geo, new THREE.MeshStandardMaterial({ color: COL[kind], emissive: COL[kind], emissiveIntensity: 0.9, transparent: true, opacity: 0.95 })));
    m.userData = { own: true, kind: 'door', fid: f.id, name, gid, sub: kind, layer: kind === 'entrance' ? 'entrance' : 'exit' }; parts.doors.add(m); pickables.push(m);
    if (kind === 'entrance') {
      ek++; nEnt++;
      const p = pin('entrance', V(x, y, SLAB + 0.3), 30); p.userData = { ...m.userData, pin: 'entrance', size: 30 }; parts.pins.add(p); pickables.push(p);
      const wd = (String(name || '').match(/폭 ([\d.]+m)/) || [])[1];
      const co = callout(`진입 ${ek}${wd ? ' · ' + wd : ''}`, 'ent', V(x, y, SLAB + 2.4), parts.labels, 0, () => flyToPt(f.id, x, y)); co.element.title = String(name || '진입구');
      callouts.push({ o: co, fid: f.id, kind: 'entrance' });
    } else if (kind === 'exit' && extd) {
      const p = pin('exit', V(x, y, SLAB + 0.3), 22); p.userData = { ...m.userData, pin: 'exit', size: 22 }; parts.pins.add(p); pickables.push(p);
    }
  }
  // 그림자 받는 판(그림자 켤 때만): 평면도 그림(조명 없음) 위에 그림자만 그림
  parts.shadowRx = own(new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0 + 2, y1 - y0 + 2), new THREE.ShadowMaterial({ opacity: 0.5, depthWrite: false })));
  parts.shadowRx.rotation.x = -Math.PI / 2; parts.shadowRx.position.set((x0 + x1) / 2, SLAB + 0.045, -(y0 + y1) / 2); parts.shadowRx.receiveShadow = true; parts.shadowRx.visible = false; parts.shadowRx.renderOrder = 3; g.add(parts.shadowRx);
  // 접촉 음영(가짜 AO) — 벽 둘레 흐린 그림자 텍스처 1장
  parts.ao = own(aoPlane(f, fb, [...stairs, ...elevs].filter((s) => s.floors.includes(f.id)), { SLAB })); g.add(parts.ao);
  // 실 구획(벽망 자동 분할)
  const SPCOL = [0x60a5fa, 0x38bdf8, 0x818cf8, 0x22d3ee, 0x93c5fd, 0x67e8f9];
  for (const sp of fb.spaces) {
    const sh = new THREE.Shape(sp.poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const sg = new THREE.ShapeGeometry(sh); sg.rotateX(-Math.PI / 2); sg.translate(0, SLAB + 0.08, 0);
    const k = Math.abs(Math.round(sp.poly[0][0] * 7 + sp.poly[0][1] * 13)) % SPCOL.length;
    const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: sp.crit ? 0xf43f5e : SPCOL[k], transparent: true, opacity: sp.crit ? 0.3 : 0.16, depthWrite: false, side: THREE.DoubleSide }));
    m.userData = { kind: 'space', fid: f.id, name: sp.name || `실(면적 ${sp.area}㎡)`, gid: sp.gid, sp }; parts.spaces.add(m);
  }
  parts.spaces.visible = false;
  // 실 용도 입체(대피 분석의 실 용도 추정 → 없으면 BIM 실 '용도 미상')
  const EVr = evRooms(f.id);
  const vr = EVr ? EVr.map((r) => ({ poly: r.poly, area: r.area, cat: r.cat === 'corr' ? 'corr' : r.cat })) : fb.spaces.map((s) => ({ poly: s.poly, area: s.area, cat: 'unk' }));
  parts.volCats = new Set();
  for (const [cat, m] of Object.entries(roomVolumes(vr.filter((r) => r.area < 0.9 * (f.area || 1e9) || vr.length > 1), { SLAB, WALLH }))) { parts.vols.add(m); parts.volCats.add(cat); }
  // 실 이름(도면 실명 → 없으면 대피 분석의 실 용도 추정)
  const SHORTCAT = { crit: '거동불가', sleep: '취침', occupied: '재실', hall: '대공간', svc: '설비', unk: '' };
  if (fb.spaces.some((s) => s.name)) for (const sp of fb.spaces) { if (sp.name) { const c = centroid(sp.poly); label(sp.name, 'rname', c[0], c[1], SLAB + 0.5, parts.rooms); } }
  else if (EVr) for (const r of EVr) {
    if (!r.c || r.area < 8 || r.cat === 'corr') continue;
    const t = SHORTCAT[r.cat] ?? (CAT_KO[r.cat] || ''); label(`${t ? t + ' ' : ''}${Math.round(r.area)}㎡`, 'rname', r.c[0], r.c[1], SLAB + 0.5, parts.rooms);
  }
  // 층 꼬리표 — 건물 전체의 남서 모서리 바깥에 한 줄 기둥으로(층마다 외곽이 달라도 같은 자리, 지하층이 건물 속에 묻히지 않게)
  {
    const el = document.createElement('div'); el.className = 'ftag'; el.textContent = f.id; el.title = `${f.name} · ${f.use || ''}`;
    el.addEventListener('click', (e) => { e.stopPropagation(); selectFloor(f.id, true); });
    const o = new CSS2DObject(el); o.position.copy(V(TAGX, TAGY, WH * 0.45)); g.add(o); parts.tag = o;
  }
}

// ── 소방시설(도면 기호)
const eqGeo = { hose: [0.9, 0.35, 1.6], panel: [1.0, 0.5, 1.8], gas: [0.5, 0.3, 0.5] };
const equips = BIM.equip.map(([type, fid, x, y, name, gid]) => ({ type, fid, x, y, name, gid }));
for (const q of equips) {
  if (!G[q.fid]) continue;
  const [L, T, Hh] = eqGeo[q.type] || [0.6, 0.6, 0.6];
  const z0 = SLAB + (q.type === 'gas' ? 1.1 : 0);
  const m = new THREE.Mesh(boxGeom(q.x - L / 2, q.y - T / 2, q.x + L / 2, q.y + T / 2, z0, z0 + Hh), new THREE.MeshStandardMaterial({ color: COL[q.type] || 0x94a3b8, emissive: COL[q.type] || 0x94a3b8, emissiveIntensity: 0.5 }));
  m.userData = { kind: 'equip', fid: q.fid, name: q.name, gid: q.gid, sub: q.type, layer: EQLAYER[q.type] || 'key' }; G[q.fid].eq.add(m); pickables.push(m);
  const ps = q.type === 'panel' ? 26 : 22;
  const p = pin(q.type, V(q.x, q.y, z0 + Hh + 0.1), ps); p.userData = { ...m.userData, pin: q.type, size: ps }; G[q.fid].pins.add(p); pickables.push(p);
  if (q.type === 'panel') callouts.push({ o: callout(q.name || '수신기', 'panel', V(q.x, q.y, z0 + Hh + 2.2), G[q.fid].labels, 1, () => flyToPt(q.fid, q.x, q.y)), fid: q.fid, kind: 'key' });
}

// ── 계단·승강기: 연결층 전체를 꿰는 빛나는 승강로(vert.js) + 층마다 말풍선
const vert = initVert({ scene, G, floors, FL, stairs, elevs, D, SLAB, H, WH, COL, pickables });
for (const [kind, S] of [...stairs.map((s) => ['stair', s]), ...elevs.map((e) => ['ev', e])]) {
  const fl = S.floors.filter((f) => FL[f]); if (!fl.length) continue;
  const [x0, y0, x1, y1] = S.box, c = [(x0 + x1) / 2, (y0 + y1) / 2], top = fl[fl.length - 1];
  const spx = S.special ? (S.special.kind === 'sp' ? ' · 특별피난' : ' · 비상용') + (S.special.cert === 'cand' ? '?' : '') : '';
  for (const fid of fl) {
    const P = G[fid];
    const nm = (kind === 'ev' ? `${S.name}${S.cars > 1 ? ` (${S.cars}대)` : ''}` : S.name) + spx;
    callouts.push({ o: callout(nm, S.special ? kind + ' sp' : kind, V(c[0], c[1], SLAB + WH * 0.75), P.labels, 1, () => flyToPt(fid, c[0], c[1])), fid, kind, floorOnly: true });
    if (fid === top) callouts.push({ o: callout(`${nm} · ${fl[0]}~${top}`, S.special ? kind + ' sp' : kind, V(c[0], c[1], SLAB + WH + 0.4), P.labels, 1, () => flyToPt(fl.includes(floors[groundIdx].id) ? floors[groundIdx].id : fid, c[0], c[1])), fid, kind, home: true });
  }
}

// ── 경로 계산(0.25m 격자 Dijkstra)
const RES = 0.25, WALL = 80;            // 벽은 막힘에 가깝게(벽 사이 문을 못 찾아도 경로는 나오도록 무한대 대신 큰 값)
const gridCache = {};
function gridFor(fid) {
  if (gridCache[fid]) return gridCache[fid];
  const f = FL[fid], fb = BIM.floors[fid];
  const [x0, y0, x1, y1] = f.bb, W = Math.ceil((x1 - x0) / RES) + 1, Hh = Math.ceil((y1 - y0) / RES) + 1;
  const cost = new Float32Array(W * Hh).fill(1);
  // 건물 밖은 비싸게(외부 동선 억제)
  const cv = document.createElement('canvas'); cv.width = W; cv.height = Hh; const cx = cv.getContext('2d');
  cx.fillStyle = '#000'; cx.fillRect(0, 0, W, Hh); cx.fillStyle = '#fff';
  for (const ring of f.poly) { cx.beginPath(); ring.forEach(([x, y], i) => (i ? cx.lineTo : cx.moveTo).call(cx, (x - x0) / RES, (y1 - y) / RES)); cx.closePath(); cx.fill(); }
  const inside = cx.getImageData(0, 0, W, Hh).data;
  for (let i = 0; i < W * Hh; i++) if (inside[i * 4] < 128) cost[i] = 4;
  for (const [o, c, a, b, t] of fb.walls) {
    const tt = Math.max(0.2, t) / 2 + 0.1;
    const [ax0, ay0, ax1, ay1] = o === 'h' ? [a, c - tt, b, c + tt] : [c - tt, a, c + tt, b];
    const i0 = Math.max(0, Math.floor((ax0 - x0) / RES)), i1 = Math.min(W - 1, Math.ceil((ax1 - x0) / RES));
    const j0 = Math.max(0, Math.floor((y1 - ay1) / RES)), j1 = Math.min(Hh - 1, Math.ceil((y1 - ay0) / RES));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const px = x0 + i * RES, py = y1 - j * RES;
      if (px >= ax0 && px <= ax1 && py >= ay0 && py <= ay1) cost[j * W + i] = WALL;
    }
  }
  // 계단실·승강기 둘레: 문을 못 찾은 경우 대비해 통과 가능(약한 벌점)
  for (const S of [...BIM.stairs, ...BIM.elevs]) {
    if (!S.floors.includes(fid)) continue;
    const [sx0, sy0, sx1, sy1] = S.box, pad = 0.6;
    const i0 = Math.max(0, Math.floor((sx0 - pad - x0) / RES)), i1 = Math.min(W - 1, Math.ceil((sx1 + pad - x0) / RES));
    const j0 = Math.max(0, Math.floor((y1 - sy1 - pad) / RES)), j1 = Math.min(Hh - 1, Math.ceil((y1 - sy0 + pad) / RES));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (cost[j * W + i] === WALL) cost[j * W + i] = 6;
  }
  return (gridCache[fid] = { x0, y1, W, H: Hh, cost });
}
const cellOf = (gr, x, y) => [Math.round((x - gr.x0) / RES), Math.round((gr.y1 - y) / RES)];
function nearestFree(gr, i, j) {
  for (let r = 0; r < 12; r++) for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
    const a = i + di, b = j + dj;
    if (a >= 0 && b >= 0 && a < gr.W && b < gr.H && gr.cost[b * gr.W + a] < WALL) return [a, b];
  }
  return [i, j];
}
function dijkstra(gr, sources) {           // sources: [[x,y]] → dist(m), prev
  const N = gr.W * gr.H, dist = new Float64Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1);
  let hk = new Float64Array(1024), hv = new Int32Array(1024), n = 0;
  const push = (k, v) => {
    if (n >= hk.length) { const a = new Float64Array(n * 2); a.set(hk); hk = a; const b = new Int32Array(n * 2); b.set(hv); hv = b; }
    let i = n++; while (i > 0) { const p = (i - 1) >> 1; if (hk[p] <= k) break; hk[i] = hk[p]; hv[i] = hv[p]; i = p; } hk[i] = k; hv[i] = v;
  };
  const pop = () => {
    const v = hv[0], k = hk[--n], kv = hv[n]; let i = 0;
    for (;;) { let c = 2 * i + 1; if (c >= n) break; if (c + 1 < n && hk[c + 1] < hk[c]) c++; if (hk[c] >= k) break; hk[i] = hk[c]; hv[i] = hv[c]; i = c; }
    hk[i] = k; hv[i] = kv; return v;
  };
  for (const [x, y] of sources) { let [i, j] = cellOf(gr, x, y); [i, j] = nearestFree(gr, i, j); if (i < 0 || j < 0 || i >= gr.W || j >= gr.H) continue; dist[j * gr.W + i] = 0; push(0, j * gr.W + i); }
  const D8 = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
  while (n) {
    const u = pop(), du = dist[u], ui = u % gr.W, uj = (u / gr.W) | 0;
    for (const [di, dj, w] of D8) {
      const a = ui + di, b = uj + dj; if (a < 0 || b < 0 || a >= gr.W || b >= gr.H) continue;
      const v = b * gr.W + a, c = gr.cost[v];
      if (di && dj && (gr.cost[uj * gr.W + a] >= WALL || gr.cost[b * gr.W + ui] >= WALL || c >= WALL)) continue;   // 벽 모서리 대각 통과 금지
      const nd = du + w * RES * c; if (nd < dist[v]) { dist[v] = nd; prev[v] = u; push(nd, v); }
    }
  }
  return { dist, prev };
}
function distAt(gr, R, x, y) { let [i, j] = cellOf(gr, x, y); [i, j] = nearestFree(gr, i, j); return i >= 0 && j >= 0 && i < gr.W && j < gr.H ? R.dist[j * gr.W + i] : Infinity; }
function pathTo(gr, R, x, y) {
  let [i, j] = cellOf(gr, x, y); [i, j] = nearestFree(gr, i, j);
  let u = j * gr.W + i; const pts = []; let wall = 0;
  const runs = []; let inW = false;                      // 벽 칸이 이어진 구간 = 문 미검출 '불확실 구간'
  while (u >= 0 && pts.length < 100000) {
    const isW = gr.cost[u] >= WALL, p = [gr.x0 + (u % gr.W) * RES, gr.y1 - ((u / gr.W) | 0) * RES];
    if (isW) { wall++; if (!inW) runs.push([p, p]); else runs[runs.length - 1][1] = p; }
    inW = isW; pts.push(p); u = R.prev[u];
  }
  // 직선 구간 단순화
  const out = [pts[0]];
  for (let k = 1; k < pts.length - 1; k++) {
    const a = out[out.length - 1], b = pts[k], c = pts[k + 1];
    if (Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-6) out.push(b);
  }
  if (pts.length > 1) out.push(pts[pts.length - 1]);
  let len = 0; for (let k = 1; k < out.length; k++) len += Math.hypot(out[k][0] - out[k - 1][0], out[k][1] - out[k - 1][1]);
  out.len = len; out.wall = wall; out.wallSeg = runs.map(([a, b]) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]); if (wall) wallWarn = true;
  return out;
}
const center = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];


// ── 화재 대응 결과 그리기
const fireG = new THREE.Group(); scene.add(fireG);
const fireCone = new THREE.Group(); scene.add(fireCone);      // 훈련 중에도 보이는 화점 표시(산출안 fireG 는 숨김)
let routeAnim = null;                                        // 진입 경로 위를 흐르는 점(화재 가정)
function tube(pts3, color, r = 0.18) {
  if (pts3.length < 2) return;
  const path = new THREE.CurvePath(); for (let k = 0; k < pts3.length - 1; k++) path.add(new THREE.LineCurve3(pts3[k], pts3[k + 1]));
  const m = new THREE.Mesh(new THREE.TubeGeometry(path, Math.max(8, pts3.length * 4), r, 6, false), new THREE.MeshBasicMaterial({ color }));
  m.userData.route = true; fireG.add(m); return m;
}
function smallLabel(text, cls, x, y, z) { return label(text, cls, x, y, z, fireG); }
// 경로 위 '문 미검출 불확실 구간' 표시(벽 칸이 이어진 구간마다 1개)
function markWall(P, z) { for (const [a, b] of P.wallSeg || []) smallLabel('문?', 'wseg', a, b, z + 1.0); return (P.wallSeg || []).length; }
// 호스 경로 위 호스 1본 길이 눈금(소화전 쪽에서부터 L_h, 2L_h …) — 눈금 점은 한 번에(InstancedMesh)
function hoseTicks(P, z) {
  const Lh = ACT.hose_len_m; let acc = 0, k = 1; const pts = [];
  for (let i = 0; i < P.length - 1; i++) {
    const [ax, ay] = P[i], [bx, by] = P[i + 1], L = Math.hypot(bx - ax, by - ay);
    while (acc + L >= k * Lh && k * Lh <= P.len) {
      const t = (k * Lh - acc) / L, x = ax + (bx - ax) * t, y = ay + (by - ay) * t;
      pts.push(V(x, y, z)); smallLabel(`${k}본 ${k * Lh}m`, 'tick', x, y, z + 0.6); k++;
    }
    acc += L;
  }
  if (pts.length) {
    const im = new THREE.InstancedMesh(new THREE.SphereGeometry(0.28, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }), pts.length);
    const M = new THREE.Matrix4(); pts.forEach((p, i) => im.setMatrixAt(i, M.makeTranslation(p.x, p.y, p.z))); fireG.add(im);
  }
  return k - 1;
}
// 공기호흡기 귀환 한계(완전식): T_A = V(P_f − P_r)/q, 경로 위 점 p 의 진입 시간 t(p) — 같은 길로 귀환(t_back = t) → 2t(p) + t_m ≤ T_A 인 가장 먼 점
function scbaLimit(legs) {
  const TA = (ACT.scba_cyl_l * (ACT.scba_fill_bar - ACT.scba_return_bar)) / ACT.scba_lpm, tLim = (TA - ACT.scba_margin_min) / 2;   // 분
  const v = ACT.walk_mps * 60;                                                                                                    // m/분
  let t = 0, pt = null;
  for (const L of legs) {
    if (L.vert) {
      const dt = (L.n * ACT.stair_s_per_floor) / 60;
      if (!pt && t + dt > tLim) { const k = Math.floor(((tLim - t) * 60) / ACT.stair_s_per_floor); pt = { fid: floors[FL[L.f0].idx + k] ? floors[FL[L.f0].idx + k].id : L.f0, x: L.x, y: L.y, z: zOf(floors[FL[L.f0].idx + k] ? floors[FL[L.f0].idx + k].id : L.f0) + SLAB }; }
      t += dt; continue;
    }
    for (let i = 0; i < L.pts.length - 1; i++) {
      const [ax, ay] = L.pts[i], [bx, by] = L.pts[i + 1], d = Math.hypot(bx - ax, by - ay), dt = d / v;
      if (!pt && t + dt > tLim) { const r = (tLim - t) / dt; pt = { fid: L.fid, x: ax + (bx - ax) * r, y: ay + (by - ay) * r, z: zOf(L.fid) + SLAB }; }
      t += dt;
    }
  }
  return { TA, tLim, tFire: t, act: TA - 2 * t - ACT.scba_margin_min, pt, beyond_m: pt ? null : (tLim - t) * v };
}
function drawFire() {
  fireG.clear(); fireCone.clear(); $('fireRes').innerHTML = ''; wallWarn = false; lastPlan = null; routeAnim = null;
  if (!fire) { if (evac) evac.onFire(null); if (search) search.run(null); fireG.visible = true; scenarioClear(); return; }
  const { fid, x, y } = fire, fz = zOf(fid) + SLAB;
  const fl = FL[fid];
  // 화점 표시
  const cone = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTexture(), depthTest: false, transparent: true }));
  cone.scale.set(4.2, 5.4, 1); cone.center.set(0.5, 0.06); cone.renderOrder = 40; cone.position.copy(V(x, y, fz + 0.15)); fireG.add(cone); fire.flame = cone;
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.5, 3.1, 40), new THREE.MeshBasicMaterial({ color: 0xff6b35, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }));
  ring.rotation.x = -Math.PI / 2; ring.position.copy(V(x, y, fz + 0.15)); fireG.add(ring); fire.ring = ring;
  label('🔥 화점 ' + fl.name, 'fire', x, y, fz + 4, fireG);
  const hid = !!(train && train.hidden());
  if (hid) { fireCone.add(cone.clone(), ring.clone()); label('🔥 훈련 화점 ' + fl.name, 'fire', x, y, fz + 4, fireCone); }   // 훈련 중: 화점만 보이고 산출안은 숨김

  const grF = gridFor(fid), RF = dijkstra(grF, [[x, y]]);
  // 이 층에 닿는 계단 → 공격계단(화점까지 가깝고 진입구와 이어짐) / 피난계단(나머지)
  const ground = floors[groundIdx].id;
  const cand = stairs.filter((s) => s.floors.includes(fid) && s.floors.includes(ground));
  const ents = (BIM.floors[ground]?.doors || []).filter((d) => d[0] === 'entrance' || (d[0] === 'exit' && d[6]));
  const grG = gridFor(ground), RG = ents.length ? dijkstra(grG, ents.map((d) => [d[2], d[3]])) : null;
  // 공격계단 점수 = d_G + d_F − 방수구 가점 + 피난 수요 벌점(sop.py ACT, 임시값) — 방수구는 소방시설 레이어(도면 위치 또는 NFPC 502 규칙 추정)
  const sps = ((D.fac && D.fac.floors && D.fac.floors[fid]) || []).filter((it) => it.code === 'STANDPIPE')
    .flatMap((it) => [...(it.loc || []).map((q) => ({ x: q.x, y: q.y, w: 1 })), ...(it.est || []).map((q) => ({ x: q.x, y: q.y, w: q.alt ? 0.5 : 1 }))]);
  const boxD = (b, x0, y0) => Math.hypot(Math.max(b[0] - x0, 0, x0 - b[2]), Math.max(b[1] - y0, 0, y0 - b[3]));
  const evR = (D.evac && D.evac.floors[fid] && D.evac.floors[fid].rooms) || [];
  const scored = cand.map((s) => {
    const [sx, sy] = center(s.box);
    const dF = distAt(grF, RF, sx, sy), dG = fid === ground ? 0 : RG ? distAt(grG, RG, sx, sy) : 0;
    const sp = Math.max(0, ...sps.filter((q) => boxD(s.box, q.x, q.y) <= ACT.standpipe_r_m).map((q) => q.w));
    const dem = evR.filter((r) => r.tgt === s.id && r.cw && Math.max(...r.cw) >= 0.8).length;
    return { s, dF, dG, tot: dF + dG, sp, dem, cost: dF + dG - (ACT.atk_w_standpipe_m || 0) * sp + (ACT.atk_w_demand_m || 0) * dem };
  }).filter((z) => isFinite(z.tot)).sort((a, b) => a.cost - b.cost || a.tot - b.tot);
  const X = ext ? ext.fireExtra(fire, fireG, fz, { distAt: (a, b) => distAt(grF, RF, a, b) }) : { top: '', bottom: '' };
  let html = X.top;                                          // v3.1: 계산 근거(접힘) — 핵심 값은 아래 요약 카드(fk)로
  const floorsUp = Math.abs(fl.ord - FL[ground].ord);
  let horiz = 0, evA = null, evPF = null, evPG = null, entryPt = null, entryDoor = null, nSeg = 0;   // 대피로 재계산용(공격계단·진입 경로)
  const legs = [];                                           // 공기호흡기 귀환 한계: 진입구 → (계단 → 수직) → 화점 순서의 구간
  if (fid === ground || !scored.length) {
    // 지상1층 화재 또는 계단 없음: 진입구에서 바로
    if (RG && fid === ground) {
      const RE = dijkstra(grF, [[x, y]]);
      let best = null; for (const d of ents) { const dd = distAt(grF, RE, d[2], d[3]); if (!best || dd < best[0]) best = [dd, d]; }
      if (best && isFinite(best[0])) {
        const P = pathTo(grF, RE, best[1][2], best[1][3]); horiz = P.len; evPF = P; entryPt = [best[1][2], best[1][3]]; entryDoor = best[1];
        tube(P.map(([a, b]) => V(a, b, fz + 0.4)), 0xffd33d); nSeg += markWall(P, fz); routeDots(P.map(([a, b]) => V(a, b, fz + 0.4)));
        legs.push({ fid, pts: P, note: '진입구 → 화점' });
        html += `<div>진입: <b>${esc(best[1][5])}</b> → 화점 <b>${P.len.toFixed(0)}m</b></div>`;
      }
    } else html += `<div>이 층과 1층을 잇는 계단을 찾지 못했습니다 → 도면 확인</div>`;
  } else {
    const A = scored[0]; const [sx, sy] = center(A.s.box);
    // 1층: 진입구 → 계단
    const PG = RG ? pathTo(grG, RG, sx, sy) : Object.assign([], { len: 0 });
    tube(PG.map(([a, b]) => V(a, b, zOf(ground) + SLAB + 0.4)), 0xffd33d); nSeg += markWall(PG, zOf(ground) + SLAB);
    // 수직 이동
    tube([V(sx, sy, zOf(ground) + SLAB + 0.4), V(sx, sy, fz + 0.4)], 0xffd33d, 0.25);
    // 화재층: 계단 → 화점
    const PF = pathTo(grF, RF, sx, sy);
    tube(PF.map(([a, b]) => V(a, b, fz + 0.4)), 0xff6b35); nSeg += markWall(PF, fz);
    A.dG = PG.len; A.dF = PF.len; horiz = A.dF + A.dG; evA = A.s; evPF = PF; evPG = PG;
    if (PG.length) { const e0 = PG[PG.length - 1]; entryDoor = ents.reduce((b, d) => (!b || Math.hypot(d[2] - e0[0], d[3] - e0[1]) < Math.hypot(b[2] - e0[0], b[3] - e0[1]) ? d : b), null); }
    const PFr = PF.slice().reverse();                       // 계단 → 화점 순서(pathTo 는 화점에서 거꾸로 걸어 나옴)
    legs.push({ fid: ground, pts: PG.slice().reverse(), note: `${FL[ground].name} 진입구 → ${A.s.name}` }, { vert: true, f0: ground, f1: fid, n: floorsUp, x: sx, y: sy }, { fid, pts: PFr, note: `${fl.name} ${A.s.name} → 화점` });
    routeDots([...PG.slice().reverse().map(([a, b]) => V(a, b, zOf(ground) + SLAB + 0.4)), V(sx, sy, zOf(ground) + SLAB + 0.4), V(sx, sy, fz + 0.4), ...PFr.map(([a, b]) => V(a, b, fz + 0.4))]);
    label('공격계단 ' + A.s.name, 'atk', sx, sy, fz + 3.2, fireG);
    for (const E of scored.slice(1)) { const [ex, ey] = center(E.s.box); label('피난계단 ' + E.s.name, 'evac', ex, ey, fz + 3.2, fireG); }
    html += `<div>공격(진입)계단: <b>${esc(A.s.name)}</b> — 1층 이동 ${A.dG.toFixed(0)}m + ${floorsUp}개 층 + 화재층 ${A.dF.toFixed(0)}m</div>`;
    if (scored.length > 1) html += `<div>피난(대피)계단: <b>${scored.slice(1).map((z) => esc(z.s.name)).join(', ')}</b> — 진압대와 동선 분리</div>`;
    else html += `<div class="note">이 층에 닿는 계단이 1개뿐 → 공격·피난 동선이 겹침(주의)</div>`;
    if ((ACT.atk_w_standpipe_m && sps.length) || ACT.atk_w_demand_m) html += `<div class="note">공격계단 점수(작을수록 우선) = 이동 비용 합(m) − 방수구 가점 ${ACT.atk_w_standpipe_m}m${ACT.atk_w_demand_m ? ` + 피난 수요 ${ACT.atk_w_demand_m}m/실` : ''}: ${scored.map((z) => `${esc(z.s.name)} ${z.tot.toFixed(0)}${z.sp ? `−${(ACT.atk_w_standpipe_m * z.sp).toFixed(0)}(방수구${z.sp < 1 ? ' 후보' : ''})` : ''}${ACT.atk_w_demand_m && z.dem ? `+${(ACT.atk_w_demand_m * z.dem).toFixed(0)}` : ''}=${z.cost.toFixed(0)}`).join(' · ')}</div>`;
  }
  // 가장 가까운 옥내소화전 + 호스 본수 N = ceil((L + α) / L_h), 경로 위 L_h 눈금
  const hs = equips.filter((q) => q.fid === fid && q.type === 'hose').map((q) => ({ q, d: distAt(grF, RF, q.x, q.y) })).filter((z) => isFinite(z.d)).sort((a, b) => a.d - b.d);
  let hoseN = null, hoseL = null, nTick = 0, hosePath = null;
  const nHose = (L) => Math.ceil((L + ACT.hose_slack_m) / ACT.hose_len_m);
  if (hs.length) {
    hs.slice(0, 2).forEach((z, k) => {
      const P = pathTo(grF, RF, z.q.x, z.q.y); z.d = P.len; tube(P.map(([a, b]) => V(a, b, fz + 0.25)), k ? 0xff9e9e : 0xff4d4f, 0.12);
      if (!k) { nTick = hoseTicks(P, fz + 0.25); nSeg += markWall(P, fz); hosePath = P.map((p) => [p[0], p[1]]); }
    });
    const n = nHose(hs[0].d); hoseN = n; hoseL = hs[0].d;
    html += `<div>가장 가까운 옥내소화전: <b>${hs[0].d.toFixed(0)}m</b> → 호스 <b>${n}본</b>(${ACT.hose_len_m}m 기준${ACT.hose_slack_m ? `, 여유 ${ACT.hose_slack_m}m` : ''} · 경로 위 흰 점 = 1본 눈금)${n > ACT.hose_carry ? ` <b>· 기본 ${ACT.hose_carry}본 초과, 추가 호스 필요</b>` : ''}</div>`;
    if (hs[1]) html += `<div class="note">두 번째 소화전 ${hs[1].d.toFixed(0)}m (${nHose(hs[1].d)}본)</div>`;
  } else html += `<div>이 층에서 옥내소화전을 찾지 못했습니다 → 도면 확인</div>`;
  // 진입 소요시간 + 공기호흡기 귀환 한계(완전식, 기준값 sop.py ACT — 임시값)
  const tIn = horiz / ACT.walk_mps + floorsUp * ACT.stair_s_per_floor;
  html += `<div>진입 예상 <b>${(tIn / 60).toFixed(1)}분</b> (수평 ${horiz.toFixed(0)}m · 수직 ${floorsUp}개 층)</div>`;
  let scba = null;
  if (legs.length) {
    scba = scbaLimit(legs);
    if (scba.pt) {
      const m = new THREE.Mesh(new THREE.TorusGeometry(1.1, 0.18, 8, 28), new THREE.MeshBasicMaterial({ color: 0xd2a8ff })); m.rotation.x = Math.PI / 2; m.position.copy(V(scba.pt.x, scba.pt.y, scba.pt.z + 0.5)); fireG.add(m);
      smallLabel('귀환 한계점', 'scba', scba.pt.x, scba.pt.y, scba.pt.z + 2.2);
      html += `<div><b>공기호흡기: 왕복 불가</b> — 귀환 한계점(보라 고리)이 화점 앞(${esc(FL[scba.pt.fid].name)}) → 중계 대원·급기 필요</div>`;
    } else html += `<div>공기호흡기: 사용 가능 ${scba.TA.toFixed(1)}분 − 진입 ${scba.tFire.toFixed(1)}분 × 2 − 안전여유 ${ACT.scba_margin_min}분 → 화점 활동 가능 <b>${scba.act.toFixed(1)}분</b> (귀환 한계점은 화점 너머 약 ${scba.beyond_m.toFixed(0)}m)</div>`;
    html += `<div class="note">귀환 한계 = 용기 ${ACT.scba_cyl_l}L × (충전 ${ACT.scba_fill_bar} − 귀환 잔압 ${ACT.scba_return_bar}bar) ÷ ${ACT.scba_lpm}L/분, 같은 길로 귀환 가정 — 보행 ${ACT.walk_mps}m/s·층당 ${ACT.stair_s_per_floor}초 포함 모두 가정값(SOP 설정)</div>`;
  }
  const evHtml = evac ? evac.onFire({ fid, x, y, attack: evA, atkFloor: evPF, atkGround: evPG, ents }) : '';
  // 인명검색 우선순위(search.js): 화재층 대피 분류 + 용도·시간대 + 공격계단 기준 보행거리 → 화재층·위층 번호
  let SR = null;
  if (search) {
    const ev = evac ? evac.lastFire() : null;
    SR = search.run({ fid, x, y, attack: evA, entry: entryPt, st: ev && ev.fid === fid ? ev.st : null }, fireG);
  }
  html += X.bottom;
  // ── 요약 카드(가장 먼저 볼 값) → 경고 → 접힌 상세(계산 근거·대피로 재계산) → 인명검색 표
  const s1 = SR && SR.items && SR.items[0], scbaBad = !!(scba && scba.pt);
  const fk = [
    ['진입구', entryDoor ? String(entryDoor[5] || '진입구').replace(/\s*\(폭 ([\d.]+m)[^)]*\)/, ' · $1') : '-', !entryDoor, 'wide'],
    ['공격계단', evA ? evA.name : fid === ground ? '해당 없음(1층 화재)' : '없음 — 계단 미검출', !evA && fid !== ground],
    ['진입 예상', `${(tIn / 60).toFixed(1)}분 · ${horiz.toFixed(0)}m`, false],
    ['호스', hoseN != null ? `${hoseN}본 · 소화전 ${hoseL.toFixed(0)}m` : '소화전 없음', hoseN == null || hoseN > ACT.hose_carry],
    ['공기호흡기', scba ? (scbaBad ? '왕복 불가 — 중계 필요' : `활동 ${scba.act.toFixed(1)}분`) : '-', scbaBad],
    ['인명검색 1순위', s1 ? `${FL[s1.fid].name} · ${CAT_KO[s1.cat] || s1.cat || ''}${s1.area != null ? ' ' + Math.round(s1.area) + '㎡' : ''}` : '-', false],
  ];
  let out = `<div class="res"><div class="big">🔥 ${esc(fl.name)} 화재</div><div class="fk">${fk.map(([k, v, bad, w]) => `<div class="${bad ? 'bad' : ''} ${w || ''}"><small>${k}</small><b title="${esc(v)}">${esc(v)}</b></div>`).join('')}</div>`;
  if (/피난계단이 공격계단뿐/.test(evHtml)) out += '<div style="color:#ffd33d">⚠ 이 층의 피난계단이 공격계단뿐 → 진입대·대피자 같은 계단(유도 통제)</div>';
  if (wallWarn) out += `<div class="note" style="color:#ffd33d">⚠ 일부 구간은 문을 찾지 못해 벽을 가로지르는 것으로 계산됨${nSeg ? ` — 진입·호스 경로 위 불확실 구간 <b>${nSeg}곳</b>('문?' 표시)` : ''} → 도면으로 문 위치 확인</div>`;
  out += `<details class="more"><summary>계산 근거 · 상세 <span class="cnt k">공격계단·호스·공기호흡기</span></summary>${html}</details>`;
  if (evHtml) out += `<details class="more"><summary>대피로 재계산 · 동선 분리</summary>${evHtml}</details>`;
  if (search) out += search.panel();
  out += `<div class="note">경로는 도면에서 뽑은 벽·문으로 계산한 추정값입니다. 현장 확인이 우선입니다.</div></div>`;
  html = out;
  lastPlan = { fid, x, y, ground, attack: evA ? evA.id : null, attackName: evA ? evA.name : null, cand: cand.map((s) => s.id), routeLen: horiz, dG: evA ? horiz - evPF.len : 0, dF: evPF ? evPF.len : null,
    entry: entryDoor ? { x: entryDoor[2], y: entryDoor[3], name: entryDoor[5] } : null, floorsUp, hoseN, hoseL, hoseTicks: nTick, tIn, scba, wallSeg: nSeg,
    search: SR ? SR.items.map((z) => ({ rank: z.rank, fid: z.fid, k: z.k, cls: z.cls, cat: z.cat, band: z.band, d: z.d, reasons: z.reasons, c: z.c })) : [], mode: SR ? SR.mode : null, grF, RF, grG,
    legs: legs.map((L) => (L.vert ? { ...L } : { fid: L.fid, note: L.note, pts: L.pts.map((p) => [p[0], p[1]]) })), hosePath, attackXY: evA ? center(evA.s ? evA.s.box : evA.box) : null };
  fireG.visible = !hid;
  $('fireRes').innerHTML = hid ? train.maskHtml() : html;
  if (train) train.onPlan(lastPlan);
  fireHead();
  scenarioBuild();
  if (xray) setDepth(fireG, false);
}
// 진입 경로 위를 따라 흐르는 점(방향 표시) — 점은 InstancedMesh 1개
function routeDots(pts3) {
  const path = new THREE.CurvePath();
  for (let k = 1; k < pts3.length; k++) if (pts3[k].distanceTo(pts3[k - 1]) > 0.01) path.add(new THREE.LineCurve3(pts3[k - 1], pts3[k]));
  if (!path.curves.length) return;
  const n = Math.min(18, Math.max(6, Math.round(path.getLength() / 6)));
  const im = new THREE.InstancedMesh(new THREE.SphereGeometry(0.32, 10, 8), new THREE.MeshBasicMaterial({ color: 0xfff3b0, depthTest: false, transparent: true }), n);
  im.renderOrder = 32; im.frustumCulled = false; fireG.add(im);
  routeAnim = { path, im, n, len: path.getLength() };
}
const setDepth = (root, on) => root.traverse((o) => { if (o.material && !o.isSprite && !Array.isArray(o.material)) { o.material.depthTest = on; o.renderOrder = on ? o.renderOrder : Math.max(o.renderOrder, 20); } });

// ═════════════════════════ 화재 확산 시뮬레이션 · 대원 이동 · 시간축
const sim = initSim({ D, BIM, floors, FL, V, SLAB, H, WH, G, esc, $ });
const SIMP = sim.P;
const crew = initCrew({ scene, V, SLAB, zOf, FL, floors, ACT, D, esc, gridFor, dijkstra, pathTo, pxScale: () => 1 / (camera.projectionMatrix.elements[5] * Math.max(50, stage.clientHeight) / 2) });
const TL = { t: 0, playing: false, speed: 30, T: SIMP.t_max_min * 60, on: false, smoke: true, crew: true, key: '' };
const simEvG = new THREE.Group(); scene.add(simEvG);
let simEv = null, crewInfo = null, crewState = [], simSum = null, lastPanel = 0;
function scenarioClear() { sim.clear(); crew.clear(); simEvG.clear(); TL.t = 0; TL.playing = false; TL.on = false; TL.key = ''; simEv = null; crewInfo = null; simSum = null; if (evac && evac.setSim) evac.setSim(false); syncTL(); }
function scenarioBuild() {
  if (!fire || !lastPlan) return;
  const key = `${fire.fid}|${fire.x.toFixed(2)}|${fire.y.toFixed(2)}`;
  if (key !== TL.key) { sim.run(fire); TL.key = key; TL.t = 0; TL.playing = false; }
  crewInfo = crew.build(ACT, lastPlan, SIMP);
  simEv = null; simEvG.clear();
  applyTime(true);
}
// 시각 t 반영: 연기·불꽃(sim) · 대원(crew) · 대피로 재평가 · 시간축·패널
function applyTime(force) {
  const hid = !!(train && train.hidden());
  const on = !!fire && !hid && (TL.t > 0 || TL.playing);
  if (on !== TL.on || force) { const ch = on !== TL.on; TL.on = on; sim.setActive(on && TL.smoke); if (evac && evac.setSim) evac.setSim(on); stage.classList.toggle('simOn', on); if (ch) styleAll(); }   // 재생 중엔 실 용도 입체·범례를 숨김(연기 색과 겹침)
  sim.show(TL.t, { smoke: TL.smoke });
  crew.setVisible(!!fire && !hid && TL.crew && TL.t > 0);
  crewState = crew.update(TL.t);
  // 대피로 재평가(화재층) — 막힌 실 조합이 바뀔 때만 다시 그림
  if (on && sim.last()) {
    const ev = sim.evacAt(TL.t, fire.fid, lastPlan ? lastPlan.attack : null);
    if (ev !== simEv) { simEv = ev; drawSimEvac(ev); }
  } else if (simEv) { simEv = null; simEvG.clear(); }
  simEvG.visible = on;
  syncTL();
  const now = performance.now();
  if (force || now - lastPanel > 250) { lastPanel = now; simPanel(); }
  invalidate();
}
function drawSimEvac(ev) {
  simEvG.clear();
  const z = zOf(ev.fid) + SLAB + 0.45, gs = [], nodes = sim.nodes(), seen = new Set();
  for (const r of ev.rooms) {
    if (r.st === 'ok' && r.path && r.path.length > 1) {
      for (let k = 0; k < r.path.length - 1; k++) {
        const a = r.path[k], b = r.path[k + 1], key = `${a[0].toFixed(1)},${a[1].toFixed(1)}|${b[0].toFixed(1)},${b[1].toFixed(1)}`; if (seen.has(key)) continue; seen.add(key);
        const A = V(a[0], a[1], z), B = V(b[0], b[1], z), L = A.distanceTo(B); if (L < 0.05) continue;
        const cg = new THREE.CylinderGeometry(0.09, 0.09, L, 6); cg.rotateX(Math.PI / 2); cg.lookAt(B.clone().sub(A)); cg.translate((A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2); gs.push(cg.index ? cg.toNonIndexed() : cg);
        const hd = new THREE.ConeGeometry(0.28, 0.7, 8); hd.rotateX(Math.PI / 2); hd.lookAt(B.clone().sub(A)); const m = A.clone().lerp(B, 0.6); hd.translate(m.x, m.y, m.z); gs.push(hd.index ? hd.toNonIndexed() : hd);
      }
    } else if (r.st === 'cut') { const c = nodes[r.i].c; label('고립', 'evi', c[0], c[1], zOf(ev.fid) + SLAB + 2.0, simEvG); }
  }
  if (gs.length) { const m = new THREE.Mesh(mergeGeometries(gs.map((g) => { g.deleteAttribute('uv'); return g; })), new THREE.MeshBasicMaterial({ color: 0x3fb950, transparent: true, opacity: 0.95, depthWrite: false })); m.renderOrder = 9; simEvG.add(m); }
}
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
function syncTL() {
  const show = !!fire && !(train && train.hidden());
  $('tl').hidden = !show; stage.classList.toggle('tlOn', show);
  if (!show) return;
  $('tlR').max = TL.T; $('tlR').value = TL.t; $('tlT').textContent = mmss(TL.t); if ($('tlPlay').dataset.st !== String(TL.playing)) { $('tlPlay').dataset.st = TL.playing; $('tlPlay').innerHTML = svg(TL.playing ? 'pause' : 'play'); $('tlPlay').setAttribute('aria-label', TL.playing ? '멈춤' : '재생'); }
  const S = sim.last();
  if (S && !$('tlTicks').dataset.k) {
    const arr = crewInfo && !crewInfo.none ? crewInfo.arrive : null;
    $('tlTicks').innerHTML = (arr != null && arr < TL.T ? `<i class="arr" style="left:${(100 * arr) / TL.T}%" title="대원 진입 시작"></i>` : '') + [300, 600].map((s) => `<i style="left:${(100 * s) / TL.T}%"></i>`).join('');
    $('tlTicks').dataset.k = '1';
  }
  const teams = crewState.filter((z) => z && z.name).map((z) => `<span style="color:${z.col}">● ${esc(z.name)}</span> ${z.txt}${z.id === 'atk1' && crewState.hose && crewState.hose.d > 0 ? ` · 호스 전개 ${crewState.hose.d.toFixed(0)}/${crewState.hose.len.toFixed(0)}m(${crewState.hose.n}본)` : ''}`).join('<br>');
  $('tlInfo').innerHTML = (TL.on && simEv ? `<b>${mmss(TL.t)}</b> 화재층 대피 가능 <b style="color:#4ade80">${simEv.nOk}</b> · 고립 <b style="color:#c4b5fd">${simEv.nCut}</b> · 연기 곤란 <b style="color:#f87171">${simEv.nSmoke}</b>${simEv.shared ? ' · <span style="color:#fbbf24">피난계단이 공격계단뿐</span>' : ''}<br>` : '<span class="k">▶ 를 누르면 0~15분 확산·대원 이동을 재생합니다(예시 모델)</span><br>') + teams;
}
function simPanel() {
  const el = $('simPanel'); if (!el) return;
  if (!fire || (train && train.hidden())) { el.innerHTML = ''; $('simSec').hidden = true; return; }
  $('simSec').hidden = false;
  const S = simSum = sim.summary(TL.t, lastPlan ? lastPlan.attack : null); if (!S) { el.innerHTML = ''; return; }
  let h = `<div class="kpis kp3"><div><b>${S.nHit}</b>곤란 실</div><div><b>${S.nBurn}</b>불 번진 실</div><div><b>${S.floorsHit.length}</b>영향 층</div></div>`;
  h += `<table style="margin-top:6px"><tr><td class="k">공격계단(화재층) 연기 곤란</td><td><b>${S.atkTTU != null ? S.fmt(S.atkTTU) : '-'}</b></td></tr><tr><td class="k">직상층 이상 첫 곤란</td><td><b>${S.fmt(S.firstUp)}</b></td></tr>`;
  h += `<tr><td class="k">화재층 대피로 모두 차단</td><td><b>${S.fmt(S.allCut)}</b></td></tr>`;
  for (const z of S.shafts) h += `<tr><td class="k">${esc(z.name)}(화재층) 곤란</td><td>${S.fmt(z.ttu)}</td></tr>`;
  h += `</table><div style="margin-top:6px"><b>곤란 도달 빠른 실</b> <span class="k">(누르면 이동)</span></div><table>` + S.top.map((z) => `<tr class="mk" data-fly="${z.c[0]},${z.c[1]}" data-fl="${esc(z.fid)}"><td style="text-align:left"><span class="sq" style="display:inline-block;background:${sim.ttuCol(z.ttu)}"></span> ${esc(FL[z.fid].name)} · ${esc(CATNAME[z.cat] || '계단·승강로')} ${Math.round(z.area)}㎡</td><td>${S.fmt(z.ttu)}${isFinite(z.burn) ? ' 🔥' : ''}</td></tr>`).join('') + '</table>';
  if (crewInfo && !crewInfo.none) h += `<div style="margin-top:6px"><b>🚒 대원(예시)</b> ${crewInfo.km != null ? `관할센터 ${crewInfo.km}km → ` : '거리 없음 → '}진입 시작 ${mmss(crewInfo.arrive)} · 화재층 +${mmss(crewInfo.tFloor)} · 화점 +${mmss(crewInfo.tEnd)}${crewInfo.scbaLim != null ? ` · 귀환 한계 진입 후 ${mmss(crewInfo.scbaLim)}` : ''}</div>`;
  else if (crewInfo && crewInfo.none) h += '<div class="note">진입 경로가 없어(계단 미검출) 대원 이동은 재생하지 않습니다.</div>';
  h += `<div class="note">${esc(SIMP._note || '예시·임시값')} — 연기 지수 ${SIMP.untenable} 이상 또는 불 붙은 지 ${SIMP.burn_untenable_s}초면 '곤란'. 문 전도도 ${SIMP.c_door}㎥/s·문 미검출 경계 ${SIMP.c_wall}·계단 위로 ${SIMP.c_stack_up} 등(firebim/sop.py SIM). 공학 연기 모델(CFAST·FDS)과 다르며 현장 판단을 대신하지 않습니다.</div>`;
  el.innerHTML = h;
}
$('tlPlay').onclick = () => { if (TL.t >= TL.T) TL.t = 0; TL.playing = !TL.playing; applyTime(); };
$('tlR').addEventListener('input', () => { TL.t = Number($('tlR').value); applyTime(); });
$('tlSp').onchange = () => (TL.speed = Number($('tlSp').value));
$('tlSmoke').onchange = () => { TL.smoke = $('tlSmoke').checked; applyTime(true); };
$('tlCrew').onchange = () => { TL.crew = $('tlCrew').checked; applyTime(true); };
$('tlFollow').onclick = () => startFollow();

// ═════════════════════════ 화면 상태 → 3D 스타일
const userHidden = {};
const L = { entrance: true, stair: true, ev: true, hose: true, exit: true, key: true, fac: true, room: true, walls: true, vol: true, spaces: false, route: true, plan: true, evimg: false, zone: false, site: false };
const LAYERS = [['entrance', '진입구·출입구', '#22d3ee'], ['stair', '계단', '#4ade80'], ['ev', '승강기', '#60a5fa'], ['hose', '소화전·송수구', '#ef4444'], ['exit', '비상구', '#22c55e'],
  ['key', '수신기·설비실', '#a78bfa'], ['fac', '소방시설(확실성)', '#fbbf24', !!D.fac], ['room', '실 이름', '#fb923c'], ['walls', '벽·문 (BIM)', '#cbd5e1'], ['vol', '실 용도(입체)', '#2dd4bf'], ['spaces', '실 영역 (BIM)', '#93c5fd'],
  ['route', '대피경로/진입로', '#f43f5e', !!(D.evac && D.evac.floors)], ['plan', '평면도', '#e2e8f0', floors.some((f) => f.tex)], ['evimg', '도면 피난표시', '#f87171', floors.some((f) => f.evac)],
  ['zone', '경계구역', '#58a6ff', !!D.zones], ['site', '외부 부서', '#39c5cf', !!D.site]].filter((r) => r[3] !== false);
const tw = new Map();                                      // 층 그룹 높이 전환(부드럽게)
let hoverFid = null;
const groundId = floors[groundIdx].id;
const modSel = () => (focus || viewMode === 'one' || fireActive() ? selId : null);   // 모듈(소방시설·대피로·경계구역)의 '선택 층' — 전체 요약 화면이면 없음

function stateOf(f) {
  if (viewMode === 'one') return f.id === selId ? 'sel' : 'off';
  if (f.id === selId && (focus || fireActive())) return 'sel';
  if (isolate && (focus || fireActive())) return 'ghost';
  if (fireActive()) return f.id === groundId ? 'norm' : 'dim';
  return focus ? 'dim' : 'norm';
}
function styleFloor(f) {
  const P = G[f.id], st = P.state, sel = st === 'sel', ghost = st === 'ghost', dim = st === 'dim' || ghost, one = viewMode === 'one';
  const detail = sel || (st === 'norm' && (viewMode === 'exp' || xray));
  const fireFl = fire && fire.fid === f.id;
  const xk = xray ? 0.22 : 1;                                 // 투시: 벽·외피를 거의 투명하게
  if (P.slab) { const op = ghost ? 0.05 : sel ? 0.95 : dim ? 0.2 : 0.62; P.slab.material[0].opacity = op * (xray ? 0.35 : 1); P.slab.material[1].opacity = Math.min(0.9, op + 0.1) * (xray ? 0.5 : 1); P.slab.material.forEach((m) => (m.depthWrite = sel && !xray)); }   // 고른 층 바닥만 깊이 기록(아래층을 가림) — 나머지는 승강로·경로가 비쳐 보이게
  if (P.shell) { P.shell.visible = !one && !xray && !ghost; P.shell.material.opacity = sel ? 0.035 : dim ? 0.015 : 0.045; }
  if (P.shellEdge) {
    const m = P.shellEdge.material; m.color.setHex(fireFl ? 0xff6b6b : sel ? 0xfbbf24 : 0x38bdf8);
    m.opacity = hoverFid === f.id && !sel ? 0.95 : sel ? 0.9 : ghost ? 0.08 : dim ? 0.12 : 0.45;
  }
  if (P.glow) { P.glow.band.visible = !lowQ && !ghost; P.glow.line.material.color.setHex(fireFl ? 0xff6b6b : sel ? 0xfbbf24 : 0x38bdf8); P.glow.line.material.opacity = ghost ? 0.1 : dim ? 0.3 : 0.9; P.glow.band.material.opacity = dim ? 0.35 : 1; }
  const wOn = L.walls && !dim;
  P.walls.visible = wOn; P.wallMat.forEach((m) => (m.opacity = m.userData.op * (sel ? 1.25 : 0.75) * xk));
  if (P.glassM) P.glassM.opacity = P.glassM.userData.op * xk; if (P.frameM) P.frameM.opacity = P.frameM.userData.op * (xray ? 0.3 : 1);
  if (P.wallEdge) { P.wallEdge.material.opacity = (sel ? 0.62 : 0.26) * (xray ? 0.45 : 1); P.wallEdge.visible = !lowQ || sel; }
  if (P.leafM) P.leafM.opacity = xray ? 0.25 : 0.75;
  P.doors.children.forEach((m) => { const ly = m.userData.layer; m.visible = ly === 'walls' ? wOn : L[ly] && !dim; });
  if (P.ao) P.ao.visible = !lowQ && !dim && L.walls && !xray;
  if (P.shadowRx) P.shadowRx.visible = shadows && !dim && !xray;
  if (P.tex) { P.tex.visible = L.plan && !dim; P.tex.material.opacity = (one ? (L.walls ? 0.8 : 1) : sel ? (L.walls ? 0.55 : 0.9) : L.walls ? 0.26 : 0.55) * (xray ? 0.6 : 1); }
  if (P.evac) P.evac.visible = L.evimg && !dim;
  P.vols.visible = L.vol && !lowQ && !xray && (sel || (st === 'norm' && viewMode === 'exp')) && !(fire && TL.on);
  P.spaces.visible = $('tSpace').checked && !dim;
  P.eq.visible = $('tEq').checked; P.eq.children.forEach((m) => (m.visible = L[m.userData.layer] && detail));
  P.pins.children.forEach((p) => { const k = p.userData.pin, ly = k === 'entrance' ? 'entrance' : k === 'exit' ? 'exit' : EQLAYER[k] || 'key'; p.visible = L[ly] && (k === 'entrance' ? !dim : detail); });
  P.rooms.visible = L.room && $('tLab').checked && sel && !farView;   // 멀리서 보면 실 이름은 숨김(글자 겹침 줄임)
  P.vert.children.forEach((o) => { const ly = o.userData.layer; if (o.userData.steps) o.visible = L[ly] && (sel || (xray && !dim)); });
  P.labels.visible = $('tLab').checked && !ghost;
  if (P.tag) { P.tag.visible = !one && (st === 'norm' || sel || viewMode === 'exp' || ghost); P.tag.element.classList.toggle('on', sel); P.tag.element.classList.toggle('fire', !!fireFl); }
}
function styleAll() {
  for (const f of floors) styleFloor(f);
  vert.style({ one: viewMode === 'one', xray, dimAll: fireActive() && !xray, layer: L });
  for (const c of callouts) {
    const st = G[c.fid].state, sel = st === 'sel';
    let v = L[c.kind] && $('tLab').checked && st !== 'off' && st !== 'ghost' && !(followOn && c.kind !== 'entrance');
    if (c.kind === 'entrance') v = v && st !== 'dim';
    else if (c.home) v = v && st === 'norm' && !fireActive();
    else v = v && sel;
    c.o.visible = v;
  }
  volLegend();
  syncChips(); invalidate();
}
function volLegend() {
  const P = G[selId], show = GL && L.vol && !lowQ && !xray && (focus || viewMode === 'one') && !(fire && TL.on) && P && P.volCats && P.volCats.size;
  $('volLeg').hidden = !show;
  if (show) $('volLeg').innerHTML = '<b>실 용도(추정)</b>' + [...P.volCats].map((c) => `<span><i style="background:#${CATCOL[c].toString(16).padStart(6, '0')}"></i>${CATNAME[c]}</span>`).join('');
}
function applyView(opt = {}) {
  for (const f of floors) {
    const P = G[f.id], g = P.g; P.state = stateOf(f);
    const to = zOf(f.id);
    if (opt.instant) { g.position.y = to; tw.delete(g); }
    else if (Math.abs(g.position.y - to) > 1e-3 && (!tw.has(g) || tw.get(g).to !== to)) tw.set(g, { from: g.position.y, to, t0: performance.now() });
    else if (Math.abs(g.position.y - to) <= 1e-3) tw.delete(g);
    g.visible = !userHidden[f.id] && P.state !== 'off';
    const k = P.state === 'dim' ? 0.35 : P.state === 'ghost' ? 0.12 : 1;      // 모듈이 붙인 객체(소방시설·대피로·경계구역 등)는 일괄 흐림
    g.traverse((o) => { const m = o.material; if (!m || o.userData.own || Array.isArray(m)) return; if (m.userData.base === undefined) m.userData.base = m.opacity; m.opacity = m.userData.base * k; });
  }
  ground.position.y = zOf(floors[0].id) - 0.6;
  styleAll();
  if (fire && !opt.noFire) drawFire();
  if (ext) ext.onView();
  if (fac) fac.onView();
  if (evac) evac.onView();
  invalidate();
}

// ═════════════════════════ 패널
const panels = initPanels({ D, BIM, floors, FL, stairs, elevs, equips, ACT, esc, $, groundId, SIMP, TTU_BANDS });
function renderRight() { if (focus || viewMode === 'one' || fireActive()) panels.floor(selId); else panels.home(); }
function fireHead() {
  $('fireSub').innerHTML = fire ? `<b>${esc(FL[fire.fid].name)}</b> 화재 가정 — 아래 요약 · 시간축 ▶ 재생` : picking ? `<b>${esc(FL[selId].name)}</b> 바닥에서 화재 위치를 누르세요 · Esc 취소` : '층을 고른 뒤 「화점 다시 지정」으로 화점을 지정하세요';
}
function syncChips() {
  L.zone = $('tZone').checked; L.site = $('tSite').checked; L.fac = $('tFac').checked; L.route = $('tEvR').checked || $('tEntR').checked;    // 모듈이 직접 켠 경우(경보 → 경계구역 등)
  document.querySelectorAll('.lay').forEach((b) => { b.classList.toggle('on', !!L[b.dataset.l]); b.setAttribute('aria-pressed', !!L[b.dataset.l]); });
}
function syncUI() {
  $('vAll').classList.toggle('on', viewMode === 'all' && !focus && !fireActive());
  $('vExp').classList.toggle('on', viewMode === 'exp'); $('vOne').classList.toggle('on', viewMode === 'one');
  $('bFire').classList.toggle('on', fireActive());
  const selOn = focus || viewMode === 'one' || fireActive();
  document.querySelectorAll('.fbtn').forEach((b) => {
    const id = b.dataset.f; b.classList.toggle('on', selOn && id === selId); b.classList.toggle('off', !!userHidden[id]);
    const fx = b.querySelector('.fx'); if (fx) fx.textContent = fire && fire.fid === id ? '🔥' : '';
  });
  stage.classList.toggle('picking', !!(picking || tools.mode() || (train && train.picking && train.picking())));   // 바닥을 눌러야 하는 동안은 말풍선이 클릭을 가로채지 않게
  $('hint').style.display = picking || tools.mode() ? 'block' : 'none';
  if (picking) $('hint').innerHTML = `🔥 <b>${esc(FL[selId].name)}</b> 바닥에서 화재 위치를 누르세요 · <u>Esc 취소</u>`;
  else if (tools.mode() === 'measure') $('hint').innerHTML = `📏 <b>${esc(FL[selId].name)}</b> 바닥에서 두 점을 누르세요(직선·보행 거리) · <u>Esc 끝</u> · Delete 지우기`;
  else if (tools.mode() === 'area') $('hint').innerHTML = `▦ <b>${esc(FL[selId].name)}</b> 실을 누르면 면적 · <u>Esc 끝</u> · Delete 지우기`;
  $('rseg').hidden = !fireMode;
  document.querySelectorAll('#rseg button').forEach((b) => b.classList.toggle('on', b.dataset.r === rTab));
  const fv = fireMode && rTab === 'fire';
  $('rpFire').hidden = !fv; $('rpInfo').hidden = fv;
  document.querySelectorAll('#dock [data-tool]').forEach((b) => {
    const k = b.dataset.tool;
    const on = (k === 'section' && section && section.active()) || (k === 'xray' && xray) || (k === 'isolate' && isolate) || (k === 'measure' && tools.mode() === 'measure') || (k === 'area' && tools.mode() === 'area') ||
      (k === 'minimap' && miniOn) || (k === 'shadow' && shadows) || (k === 'lowq' && lowQ) || (k === 'follow' && followOn);
    b.classList.toggle('on', on); if (!['shot', 'brief', 'help'].includes(k)) b.setAttribute('aria-pressed', on);
  });
  for (const [id, on] of [['vAll', viewMode === 'all' && !focus && !fireActive()], ['vExp', viewMode === 'exp'], ['vOne', viewMode === 'one'], ['bFire', fireActive()]]) $(id).setAttribute('aria-pressed', !!on);
  document.querySelectorAll('.fbtn[data-f]').forEach((b) => { if (b.classList.contains('on')) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  document.querySelectorAll('#rseg button').forEach((b) => b.setAttribute('aria-selected', b.classList.contains('on')));
  fireHead(); brief30(); layoutUI(); invalidate();
}

// ═════════════════════════ 층 선택 · 보기 전환 · 카메라
function selectFloor(fid, flyNow) {
  if (!FL[fid]) return;
  selId = fid; focus = true;
  applyView(); renderRight(); syncUI();
  if (flyNow) flyFloor();
}
function goHome() {
  focus = false; if (viewMode !== 'exp') viewMode = 'all';
  if (viewMode === 'all') explode = 0;
  applyView(); renderRight(); syncUI(); flyHome();
}
function setMode(m) {
  viewMode = m; explode = m === 'exp' ? H * 1.6 : 0;
  if (m === 'one') focus = true;
  applyView(); renderRight(); syncUI();
  if (m === 'one' || (m === 'all' && focus)) flyFloor(); else flyHome();
}
let fly = null;
function flyRaw(toT, toP, ms = 700) { fly = { t0: performance.now(), ms, fromT: controls.target.clone(), fromP: camera.position.clone(), toT, toP }; invalidate(); }
function flyTo(x, y, z, dist) {
  const tgt = V(x, y, z);
  const dir = camera.position.clone().sub(controls.target).normalize();
  const dd = dist || camera.position.distanceTo(controls.target);
  flyRaw(tgt, tgt.clone().add(dir.multiplyScalar(dd)));
}
function flyToPt(fid, x, y) {
  if (fid !== selId || !focus) selectFloor(fid, false);
  flyTo(x, y, zOf(fid) + SLAB, Math.max(26, SPAN * 0.35)); ping(x, y, zOf(fid) + SLAB);
}
function floorSpan(f) { return Math.max(f.bb[2] - f.bb[0], f.bb[3] - f.bb[1], 16); }
function flyFloor() {
  const f = FL[selId], c = [(f.bb[0] + f.bb[2]) / 2, (f.bb[1] + f.bb[3]) / 2], s = floorSpan(f), t = V(c[0], c[1], zOf(selId) + SLAB);
  const asp = Math.max(0.6, camera.aspect || 1.5), fit = s / (2 * Math.tan((camera.fov * Math.PI) / 360)) * (asp < 1.2 ? 1.35 / asp * 1.2 : 1.12);
  if (viewMode === 'one') { flyRaw(t, t.clone().add(new THREE.Vector3(0, fit * 1.02, fit * 0.1))); return; }
  const h = camera.position.clone().sub(controls.target); h.y = 0; if (h.lengthSq() < 1e-6) h.set(0.6, 0, 0.8); h.normalize();
  const d = fit * 1.15 + 6;
  flyRaw(t, t.clone().add(h.multiplyScalar(d * 0.72)).add(new THREE.Vector3(0, d * 0.7, 0)));
}
function homePose() {
  const topZ = zOf(floors[floors.length - 1].id) + H, botZ = zOf(floors[0].id);
  const t = V(CX, CY, (topZ + botZ) / 2 - H * 0.3), h = Math.max(SPAN, (topZ - botZ) * 0.9);
  const d = h / (2 * Math.tan((camera.fov * Math.PI) / 360)) * 1.25 + 8;
  const dir = new THREE.Vector3(0.62, 0.52, 0.78).normalize();
  return [t, t.clone().add(dir.multiplyScalar(d))];
}
function flyHome() { const [t, p] = homePose(); flyRaw(t, p); }
function setView(kind) {
  const t = V(CX, CY, zOf(selId));
  const pos = kind === 'top' ? t.clone().add(new THREE.Vector3(0, SPAN * 0.95, 0.01)) : t.clone().add(new THREE.Vector3(SPAN * 0.55, SPAN * 0.6, SPAN * 0.75));
  flyRaw(t, pos);
}
const pings = [];
function ping(x, y, z) {
  const r = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.7, 40), new THREE.MeshBasicMaterial({ color: 0xfbbf24, side: THREE.DoubleSide, transparent: true, depthTest: false }));
  r.rotation.x = -Math.PI / 2; r.position.copy(V(x, y, z + 0.3)); r.renderOrder = 20; scene.add(r); pings.push({ r, t0: performance.now() });
}

// ═════════════════════════ 머리줄·범례 칩·층 버튼·탭
function setLayer(k, v) {
  L[k] = v;
  const chk = (id, on) => { const e = $(id); if (e) e.checked = on; };
  if (k === 'walls') { chk('tWall', v); chk('tDoor', v); }
  if (k === 'spaces') chk('tSpace', v);
  if (k === 'plan') chk('tTex', v);
  if (k === 'evimg') chk('tEvac', v);
  if (k === 'room') chk('tRoom', v);
  if (k === 'fac') { chk('tFac', v); if (fac) fac.onView(); }
  if (k === 'route') { chk('tEvR', v); chk('tEntR', v); if ($('tEvR').onchange) $('tEvR').onchange(); }
  if (k === 'zone') { chk('tZone', v); if (ext) ext.onView(); }
  if (k === 'site') { chk('tSite', v); $('tSite').onchange(); }
  styleAll();
}
const LAYICON = { entrance: 'entrance', stair: 'stair', ev: 'ev', hose: 'hose', exit: 'exit', key: 'panel', fac: 'fac', room: 'info', walls: 'iso', vol: 'isolate', spaces: 'area', route: 'follow', plan: 'plan', evimg: 'exit', zone: 'alert', site: 'pin' };
$('layers').innerHTML = LAYERS.map(([k, t, c]) => `<button class="lay" data-l="${k}" style="--c:${c}" aria-pressed="false" title="${t} 켜고 끄기">${svg(LAYICON[k] || 'layers')}${t}</button>`).join('');
$('layers').addEventListener('click', (e) => { const b = e.target.closest('.lay'); if (b) setLayer(b.dataset.l, !L[b.dataset.l]); });
// 범례 칩이 한 줄을 넘으면 나머지는 '+N' 단추 안으로(누르면 펼침) — 위쪽 3D 화면을 가리지 않게
$('layMore').onclick = (e) => { e.stopPropagation(); const o = !$('layers').classList.contains('open'); $('layers').classList.toggle('open', o); $('layMore').setAttribute('aria-expanded', o); layoutUI(); };
document.addEventListener('pointerdown', (e) => { if ($('layers').classList.contains('open') && !e.target.closest('#layers,#layMore')) { $('layers').classList.remove('open'); $('layMore').setAttribute('aria-expanded', false); layoutUI(); } });
function floorBar() {
  const C = D.building.card || {}, DL = C.동목록 || [], files = D.building.dong_files || [];
  let h = '';
  if (DL.length > 1) {
    h += '<div class="fgrp">동</div>' + DL.map((d, i) => d.선택 ? `<button class="fbtn on" title="${esc(d.동명 || '')}" disabled>${esc((d.동명 || i + 1 + '동').slice(0, 4))}</button>`
      : files[i] ? `<a class="fbtn" style="text-align:center;color:inherit;text-decoration:none" href="${esc(files[i])}" title="${esc(d.동명 || '')} 3D 열기">${esc((d.동명 || i + 1 + '동').slice(0, 4))}</a>` : '').join('');
  }
  const groups = [...new Set(floors.map((f) => f.group || 'main'))];
  for (const gname of groups) {
    if (groups.length > 1 || DL.length > 1) h += `<div class="fgrp">${esc(gname === 'main' ? '본관' : gname)}</div>`;
    h += [...floors].reverse().filter((f) => (f.group || 'main') === gname).map((f) => `<button class="fbtn ${panels.alignOf[f.id] === 'low' ? 'lo' : ''}" data-f="${esc(f.id)}" title="${esc(f.name)} · ${esc(f.use || '')}${panels.alignOf[f.id] === 'low' ? ' · 층 위치 맞춤 신뢰도 낮음' : ''}${keyOf(f) ? ` · 단축키 ${keyOf(f)}` : ''}">${esc(f.id.length <= 4 ? f.id : f.name.replace('지상', '').replace('지하', 'B'))}<span class="fx"></span></button>`).join('');
  }
  $('fbar').innerHTML = h;
}
const keyOf = (f) => (f.level >= 1 && f.level <= 9 ? String(f.level) : f.level === -1 ? '0' : f.level < -1 && f.level >= -9 ? `Shift+${-f.level}` : '');
$('fbar').addEventListener('click', (e) => {
  const b = e.target.closest('.fbtn[data-f]'); if (!b) return;
  if (b.dataset.f === selId && focus && viewMode === 'all' && !fireActive()) goHome(); else selectFloor(b.dataset.f, true);
});
document.querySelectorAll('.tab').forEach((b) => (b.onclick = () => showTab(b.dataset.t)));
function showTab(t) {
  document.querySelectorAll('.tab').forEach((x) => { x.classList.toggle('on', x.dataset.t === t); x.setAttribute('aria-selected', x.dataset.t === t); });
  document.querySelectorAll('.tabp').forEach((p) => (p.hidden = p.id !== 'tp-' + t));
}
const narrow = () => innerWidth <= 1100;
$('bissue').onclick = () => { showTab('issues'); if (narrow()) { $('app').classList.add('showL'); $('app').classList.remove('showR'); } else $('app').classList.remove('noL'); };
$('tgL').onclick = () => { if (narrow()) { $('app').classList.toggle('showL'); $('app').classList.remove('showR'); } else $('app').classList.toggle('noL'); };
$('tgR').onclick = () => { if (narrow()) { $('app').classList.toggle('showR'); $('app').classList.remove('showL'); } else $('app').classList.toggle('noR'); };
const panelAria = () => { const A = $('app').classList, n = narrow(); $('tgL').setAttribute('aria-expanded', n ? A.contains('showL') : !A.contains('noL')); $('tgR').setAttribute('aria-expanded', n ? A.contains('showR') : !A.contains('noR')); };
for (const id of ['tgL', 'tgR', 'bissue']) $(id).addEventListener('click', panelAria);
$('rseg').addEventListener('click', (e) => { const b = e.target.closest('[data-r]'); if (b) { rTab = b.dataset.r; syncUI(); } });

// 공통 클릭: data-fly(위치로 이동) · data-go(층 선택) · data-act · 평면도
document.addEventListener('click', (e) => {
  const v = e.target.closest('[data-v]');
  if (v && v.tagName === 'INPUT') { userHidden[v.dataset.v] = !v.checked; applyView(); syncUI(); e.stopPropagation(); return; }
  const m = e.target.closest('[data-fly]');
  if (m) {
    const [x, y] = m.dataset.fly.split(',').map(Number);
    flyToPt(m.dataset.fl && FL[m.dataset.fl] ? m.dataset.fl : selId, x, y);
    if (narrow()) $('app').classList.remove('showL', 'showR');
    return;
  }
  const g = e.target.closest('[data-go]'); if (g) { selectFloor(g.dataset.go, true); return; }
  const a = e.target.closest('[data-act]'); if (a && a.dataset.act === 'home') { goHome(); return; }
  if (e.target.id === 'vPlan') openPlan();
});

// ── 원본 평면도 창(확대·이동)
const mz = { k: 1, x: 0, y: 0, drag: null };
const mzApply = () => ($('mimg').style.transform = `translate(${mz.x}px,${mz.y}px) scale(${mz.k})`);
function openPlan() {
  const f = FL[selId]; if (!f.plan) return;
  $('mimg').src = f.plan; $('mtitle').textContent = `${f.name} — 원본 평면도`; $('mcap').textContent = '  휠: 확대 · 끌기: 이동 · Esc: 닫기';
  Object.assign(mz, { k: 1, x: 0, y: 0 }); mzApply(); $('modal').style.display = 'flex';
}
const closePlan = () => ($('modal').style.display = 'none');
$('mclose').onclick = closePlan;
$('modal').addEventListener('click', (e) => { if (e.target === $('modal')) closePlan(); });
$('mview').addEventListener('wheel', (e) => { e.preventDefault(); mz.k = Math.min(6, Math.max(1, mz.k * (e.deltaY < 0 ? 1.15 : 0.87))); mzApply(); }, { passive: false });
$('mview').addEventListener('pointerdown', (e) => (mz.drag = [e.clientX - mz.x, e.clientY - mz.y]));
addEventListener('pointerup', () => (mz.drag = null));
addEventListener('pointermove', (e) => { if (mz.drag) { mz.x = e.clientX - mz.drag[0]; mz.y = e.clientY - mz.drag[1]; mzApply(); } });
$('zin').onclick = () => { mz.k = Math.min(6, mz.k * 1.4); mzApply(); };
$('zout').onclick = () => { mz.k = Math.max(1, mz.k / 1.4); if (mz.k === 1) mz.x = mz.y = 0; mzApply(); };

// ── 보기 버튼
$('vAll').onclick = () => goHome();
$('vExp').onclick = () => setMode(viewMode === 'exp' ? 'all' : 'exp');
$('vOne').onclick = () => setMode(viewMode === 'one' ? 'all' : 'one');
$('vTop').onclick = () => setView('top'); $('vIso').onclick = () => setView('iso');
$('compass').onclick = () => { const t = controls.target.clone(), d = camera.position.distanceTo(t), el = Math.max(0.35, Math.min(1.35, Math.acos(Math.min(1, (camera.position.y - t.y) / d)))); flyRaw(t, t.clone().add(new THREE.Vector3(0, Math.cos(el) * d, Math.sin(el) * d))); };
for (const id of ['tWall', 'tDoor', 'tTex', 'tEvac', 'tSpace', 'tEq', 'tRoom']) $(id).onchange = () => { L.walls = $('tWall').checked; L.spaces = $('tSpace').checked; L.plan = $('tTex').checked; L.evimg = $('tEvac').checked; L.room = $('tRoom').checked; styleAll(); };
$('tLab').onchange = () => styleAll();

// ═════════════════════════ 화점 지정 · 훈련
function startPick() { picking = true; fireMode = true; rTab = 'fire'; tools.setMode(null); if (viewMode !== 'one') setMode('one'); else { applyView({ noFire: true }); renderRight(); } syncUI(); }
function stopPick() { picking = false; if (!fire && !(train && train.active())) fireMode = false; applyView({ noFire: true }); renderRight(); syncUI(); }
$('bFire').onclick = () => (picking ? stopPick() : startPick());
$('bRepick').onclick = () => startPick();
$('bClr').onclick = () => { if (train && train.active()) train.quit(); if (followOn) follow.stop(); fire = null; picking = false; fireMode = false; rTab = 'info'; drawFire(); applyView(); renderRight(); syncUI(); };
function randomFire() {                                     // 훈련: 지상층 중 실이 있는 층 · 12㎡ 넘는 실 중심
  const cands = floors.filter((f) => f.level >= 0 && (BIM.floors[f.id]?.spaces || []).length);
  const f = cands[Math.floor(Math.random() * cands.length)] || floors[groundIdx];
  const sps = (BIM.floors[f.id]?.spaces || []).filter((s) => s.area > 12);
  const sp = sps[Math.floor(Math.random() * sps.length)];
  const c = sp ? centroid(sp.poly) : [(f.bb[0] + f.bb[2]) / 2, (f.bb[1] + f.bb[3]) / 2];
  fire = { fid: f.id, x: c[0], y: c[1] }; picking = false; fireMode = true; rTab = 'fire';
  selectFloor(f.id, false); flyTo(c[0], c[1], zOf(f.id), Math.max(70, SPAN * 0.8)); syncUI();
}
$('bTrain').onclick = randomFire;
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
const shown = (o) => { for (let p = o; p; p = p.parent) if (!p.visible) return false; return true; };
function rayAt(e) { const r = canvas.getBoundingClientRect(); mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(mouse, camera); }
// 선택 층 바닥면과 만나는 점(도면 좌표)
function floorHit() { const pl = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(G[selId].g.position.y + SLAB)); const p = new THREE.Vector3(); return ray.ray.intersectPlane(pl, p) ? [p.x, -p.z] : null; }
// 바닥 좌표 클릭(3D·2D 대체 보기 공통) → 훈련 선택·화점·도구. 처리했으면 true
function floorClick(x, y) {
  if (train && train.picking() && train.pickAt(selId, x, y)) return true;
  if (picking) { fire = { fid: selId, x, y }; picking = false; fireMode = true; rTab = 'fire'; applyView(); renderRight(); syncUI(); return true; }
  if (tools.mode()) { tools.click(selId, x, y); syncUI(); invalidate(); return true; }
  return false;
}
let downAt = null;
canvas.addEventListener('pointerdown', (e) => (downAt = [e.clientX, e.clientY]));
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5 || followOn) return;
  rayAt(e);
  if ((train && train.picking()) || picking || tools.mode()) { const p = floorHit(); if (p && floorClick(p[0], p[1])) return; if (picking || tools.mode()) return; }
  const hits = ray.intersectObjects(pickables.filter(shown), false);
  const h = hits.find((x) => x.object.isSprite) || hits.find((x) => !['slab', 'shell'].includes(x.object.userData.kind)) || hits[0];
  const box = $('info');
  if (!h) { box.style.display = 'none'; return; }
  const u = doorInfo(h);
  if (u.kind === 'slab' || u.kind === 'shell') {
    if ((u.fid !== selId || !focus) && viewMode !== 'one' && !fireActive()) selectFloor(u.fid, true);
    box.style.display = 'none'; return;
  }
  if (u.facHtml) { box.innerHTML = '<span class="x">✕</span>' + u.facHtml; box.style.display = 'block'; return; }
  let t = `<span class="x">✕</span><b>${esc(u.name || EQNAME[u.sub] || '')}</b>`;
  if (u.S) t += ` · 연결층 ${u.S.floors[0]}~${u.S.floors[u.S.floors.length - 1]}${u.S.note ? ' · ' + esc(u.S.note) : ''}${u.S.special ? `<br><b style="color:#fbbf24">${u.S.special.kind === 'sp' ? '특별피난계단' : '비상용 승강기'}${u.S.special.cert === 'cand' ? ' 후보' : ''}</b> <span class="k">${esc(u.S.special.why)}</span>` : ''}`;
  if (u.fid) t += ` · ${esc(FL[u.fid].name)}`;
  if (u.sub && EQNAME[u.sub]) t += `<br><span class="k">${EQNAME[u.sub]} · 평면도 기호 자동검출</span>`;
  if (u.gid) t += `<br><span class="k">IFC GlobalId ${esc(u.gid)}</span>`;
  if (u.S && u.S.gids) t += `<br><span class="k">IFC GlobalId(층별) ${esc(u.S.gids[u.S.floors.indexOf(u.fid)] || u.S.gids[0])}</span>`;
  box.innerHTML = t; box.style.display = 'block';
});
// 합친 문짝 형상: 맞은 면 번호 → 어느 문인지
function doorInfo(h) {
  const u = h.object.userData;
  if (!u.leafDoors || h.faceIndex == null) return u;
  const r = u.leafDoors.find(([a, b]) => h.faceIndex >= a && h.faceIndex < b); if (!r) return u;
  const d = u.doors[r[2]]; return { kind: 'door', fid: u.fid, name: d[5] || '문', gid: d[7], layer: 'walls', sub: d[0] };
}
$('info').addEventListener('click', (e) => { if (e.target.classList.contains('x')) $('info').style.display = 'none'; });
let lastHover = 0;
canvas.addEventListener('pointermove', (e) => {
  const now = performance.now(); if (now - lastHover < 70 || e.buttons || followOn) return; lastHover = now;
  rayAt(e);
  if (tools.mode() === 'measure') { const p = floorHit(); if (p) { tools.hover(selId, p[0], p[1]); invalidate(); } }
  const hits = ray.intersectObjects(pickables.filter((o) => o.userData.kind !== 'space' && shown(o)), false);
  const h = hits.find((x) => x.object.isSprite) || hits.find((x) => !['slab', 'shell'].includes(x.object.userData.kind)) || hits[0];
  const u = h ? doorInfo(h) : null, flo = u && (u.kind === 'shell' || u.kind === 'slab');
  const fid = flo ? u.fid : null;
  canvas.style.cursor = picking || tools.mode() ? 'crosshair' : h ? (flo && u.fid === selId && focus ? 'grab' : 'pointer') : 'grab';
  if (fid !== hoverFid) { hoverFid = fid; for (const f of floors) styleFloor(f); invalidate(); }
  // 마우스 올린 객체 설명(IFC 분류·층)
  const tip = $('tip'); let t = '';
  if (u && !picking && !tools.mode()) {
    const fl = u.fid && FL[u.fid] ? FL[u.fid].name : '';
    const IFC = { door: 'IfcDoor', equip: { hose: 'IfcFireSuppressionTerminal', panel: 'IfcUnitaryControlElement', gas: 'IfcValve' }[u.sub] || 'IfcFlowTerminal', stair: 'IfcStair', ev: 'IfcTransportElement', fac: '소방시설(정보카드)' };
    const col = u.layer === 'entrance' ? '#22d3ee' : u.layer === 'exit' ? '#22c55e' : u.sub && COL[u.sub] ? '#' + COL[u.sub].toString(16).padStart(6, '0') : u.kind === 'stair' ? '#4ade80' : u.kind === 'ev' ? '#60a5fa' : '#e6edf7';
    if (flo) { if (!(u.fid === selId && focus)) t = `<b>${esc(fl)}</b> <span class="k">${esc(FL[u.fid].use || '')}</span><div class="k">누르면 이 층 선택</div>`; }
    else if (u.S) t = `<b style="color:${col}">${esc(u.S.name)}</b> <span class="k">${IFC[u.kind]}</span><div class="k">${esc(fl)} · 연결층 ${esc(u.S.floors[0])}~${esc(u.S.floors[u.S.floors.length - 1])}${u.S.special ? ` · ${u.S.special.kind === 'sp' ? '특별피난계단' : '비상용 승강기'}${u.S.special.cert === 'cand' ? ' 후보' : ''}` : ''}</div>`;
    else if (IFC[u.kind]) t = `<b style="color:${col}">${esc(u.name || EQNAME[u.sub] || '')}</b> <span class="k">${IFC[u.kind]}</span><div class="k">${esc(fl)}${u.gid ? ' · 누르면 GlobalId' : ''}</div>`;
  }
  if (t) { const r = stage.getBoundingClientRect(); tip.innerHTML = t; tip.style.display = 'block'; tip.style.left = Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 8) + 'px'; tip.style.top = e.clientY - r.top + 16 + 'px'; }
  else tip.style.display = 'none';
});
canvas.addEventListener('pointerleave', () => { $('tip').style.display = 'none'; const et = $('evTip'); if (et) et.style.display = 'none'; if (hoverFid) { hoverFid = null; for (const f of floors) styleFloor(f); invalidate(); } });

// ═════════════════════════ 모듈 연결(기능 12·18·24 · 소방시설 · 대피로 · 인명검색 · 훈련)
function setFire(fid, x, y, opts) { fire = { fid, x, y, ...(opts || {}) }; picking = false; fireMode = true; rTab = 'fire'; selectFloor(fid, false); flyTo(x, y, zOf(fid), Math.max(60, SPAN * 0.75)); syncUI(); }
ext = initExt({ D, BIM, scene, G, floors, FL, V, SLAB, H, esc, $, boxGeom, zOf, selectFloor, setFire, getSel: modSel, getViewMode: () => viewMode, bx0, by0, bx1, by1 });
$('tZone').onchange = () => ext.onView();
fac = initFac({ D, G, floors, FL, V, SLAB, esc, $, pickables, getSel: modSel, activeFloor: () => selId, showAll: () => viewMode === 'exp' });
$('tFac').onchange = () => fac.onView();
evac = initEvac({ D, BIM, scene, G, floors, FL, V, SLAB, esc, $, zOf, getSel: modSel, getViewMode: () => viewMode, camera, canvas, gridFor, dijkstra, distAt, pathTo, selectFloor, isHidden: () => !!(train && train.hidden()) });
search = initSearch({ D, BIM, floors, FL, V, SLAB, esc, $, zOf, gridFor, dijkstra, distAt, onMode: () => { if (fire) drawFire(); }, floorVisible: (f) => G[f].g.visible });
train = initTrain({ D, BIM, FL, V, SLAB, esc, $, zOf, dijkstra, pathTo, scene, ACT, ground: () => floors[groundIdx].id, getFire: () => fire, randomFire, redraw: () => { fireMode = true; drawFire(); syncUI(); },
  setMode: (m) => search.setMode(m), selectFloor: (f) => selectFloor(f, true), getSel: () => selId });
$('tSite').onchange = () => { L.site = $('tSite').checked; ext.onView(); syncChips(); if ($('tSite').checked) { const t = V(CX, CY, 0), s = Math.max(SPAN, 70); flyRaw(t, t.clone().add(new THREE.Vector3(s * 1.1, s * 1.5, s * 1.9))); } };

// ═════════════════════════ v3 도구: 단면·투시·격리 · 거리·면적 · 미니맵 · 따라가기 · 저장·브리핑 · 그림자·저사양
const bounds = { bx0, by0, bx1, by1 };
const shaftsOf = (fid) => [...stairs, ...elevs].filter((s) => s.floors.includes(fid));
const section = initSection({ scene, floors, G, SLAB, WALLH, H, bounds, shells: SHELL, shaftsOf, invalidate, focusFid: () => (focus || viewMode === 'one' || fireActive() ? selId : null) });
section.attach();
const tools = initTools({ scene, V, SLAB, zOf: (fid) => G[fid].g.position.y, getSel: () => selId, D, BIM, FL, floors, esc, $, canvas, stage, render: () => { vert.sync(); renderer.render(scene, camera); labelR.render(scene, camera); },
  gridFor, dijkstra, distAt, stairs, elevs, equips, getPlan: () => (lastPlan && !(train && train.hidden()) ? lastPlan : null), getSimSummary: () => (fire && sim.last() && TL.on ? sim.summary(TL.t, lastPlan && lastPlan.attack) : null), getCrew: () => crewInfo, ACT,
  getBrief: () => [...document.querySelectorAll('#b30 li')].map((li) => ({ cls: li.className, k: li.querySelector('small').textContent, t: li.querySelector('b').textContent, s: li.querySelector('i').textContent })) });
let miniOn = true, followOn = false;
const minimap = initMinimap({ floors, FL, BIM, D, camera, bounds, getFid: () => (fire ? fire.fid : selId), getZ: (fid) => G[fid].g.position.y + SLAB,
  onPick: (x, y, drag) => { const fid = fire ? fire.fid : selId, t = V(x, y, G[fid].g.position.y + SLAB), d = t.clone().sub(controls.target); if (drag) { controls.target.add(d); camera.position.add(d); invalidate(); } else flyRaw(t, camera.position.clone().add(d), 450); } });
minimap.attach($('mini'));
const follow = initFollow({ camera, controls, $, esc, ACT, FL, onState: (on) => { followOn = on; fireG.children.forEach((o) => { if (o.userData.route || o.isInstancedMesh) o.visible = !on; }); styleAll(); syncUI(); } });   // 따라가는 동안 발밑 경로 관은 숨김(앞쪽 흐르는 점은 보임)
// 진입로 따라가기: 화재 가정 진입 경로(없으면 1층 진입구 → 가장 가까운 계단 진입로)
function startFollow() {
  if (followOn) { follow.stop(); return; }
  if (viewMode === 'one') setMode('all');
  let legs = lastPlan && lastPlan.legs && lastPlan.legs.length ? lastPlan.legs : null, label = '진입로 따라가기 — 화점까지';
  if (!legs && D.evac && D.evac.entry && D.evac.entry.length && D.evac.entry[0].routes[0]) { const r = D.evac.entry[0].routes[0]; legs = [{ fid: D.evac.ground, pts: r.path, note: `진입구 → ${r.name}` }]; label = `진입로 따라가기 — ${r.name}까지`; }
  if (!legs) { $('info').innerHTML = '<span class="x">✕</span>진입 경로가 없습니다 — 화재 가정으로 화점을 먼저 지정하세요(계단 미검출 층은 경로 없음).'; $('info').style.display = 'block'; return; }
  // 층 높이 전환이 끝난 위치 기준(zOf) 경로
  const seq = []; let t = 0;
  for (const Lg of legs) {
    if (Lg.vert) { const dir = FL[Lg.f1].idx > FL[Lg.f0].idx ? 1 : -1; for (let k = 1; k <= Lg.n; k++) { const f = floors[FL[Lg.f0].idx + dir * k]; t += ACT.stair_s_per_floor; seq.push({ p: V(Lg.x, Lg.y, zOf(f.id) + SLAB), t, fid: f.id, note: `${f.name} 계단` }); } continue; }
    Lg.pts.forEach(([x, y], i) => { if (i) { const [a, b] = Lg.pts[i - 1]; t += Math.hypot(x - a, y - b) / ACT.walk_mps; } seq.push({ p: V(x, y, zOf(Lg.fid) + SLAB), t, fid: Lg.fid, note: Lg.note }); });
  }
  follow.start(seq, label);
}
function setXray(on) { xray = on; if (xray) setDepth(fireG, false); else setDepth(fireG, true); applyView({ noFire: true }); syncUI(); }
function setIsolate(on) { isolate = on; if (on && viewMode !== 'all') { viewMode = 'all'; explode = 0; } focus = focus || on; applyView(); renderRight(); syncUI(); if (focus) flyFloor(); }   // 격리 = 모든 층을 제자리에 두고 고른 층만 진하게(단층 평면·층 분리에서 켜면 전체 보기로)
function setShadows(on) {
  if (!GL) return; shadows = on; renderer.shadowMap.enabled = on; renderer.shadowMap.type = THREE.PCFSoftShadowMap; sun.castShadow = on; shadowCatch.visible = on; for (const f of floors) styleFloor(f);
  scene.traverse((o) => { if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => (m.needsUpdate = true)); }); syncUI();
}
function setLowQ(on) { lowQ = on; perf.pr = on ? 1 : PR0; renderer.setPixelRatio(on ? Math.min(1, PR0) : PR0); resize(); if (on && shadows) setShadows(false); applyView({ noFire: true }); syncUI(); }
function toggleSection(force) {
  const on = force != null ? force : !section.active();
  section.set({ h: on ? $('secH').checked || !$('secV').checked : false, v: on ? $('secV').checked : false });
  if (on && !$('secH').checked && !$('secV').checked) { $('secH').checked = true; section.set({ h: true }); }
  $('secPanel').hidden = !on; if (on) { $('secHv').value = Math.round(section.S.hv * 1000); secText(); } syncUI();
}
$('secH').onchange = () => { section.set({ h: $('secH').checked }); syncUI(); };
$('secV').onchange = () => { section.set({ v: $('secV').checked }); syncUI(); };
$('secHv').addEventListener('input', () => { section.set({ hv: Number($('secHv').value) / 1000 }); secText(); });
const secText = () => { $('secHt').textContent = section.hText(); $('secVt').textContent = `${section.S.axis === 'x' ? '동서' : '남북'} ${(section.vPos() - (section.S.axis === 'x' ? bx0 : by0)).toFixed(1)}m 지점`; };
$('secVv').addEventListener('input', () => { section.set({ vv: Number($('secVv').value) / 1000 }); secText(); });
$('secAx').onchange = () => { section.set({ axis: $('secAx').value }); secText(); };
$('secFlip').onclick = () => section.set({ flip: !section.S.flip });
$('secClose').onclick = () => toggleSection(false);
function tool(k) {
  if (k === 'section') toggleSection();
  else if (k === 'xray') setXray(!xray);
  else if (k === 'isolate') setIsolate(!isolate);
  else if (k === 'measure' || k === 'area') { if (picking) stopPick(); if (!focus && viewMode !== 'one' && !fire) selectFloor(selId, true); tools.setMode(k); syncUI(); }
  else if (k === 'follow') startFollow();
  else if (k === 'minimap') { miniOn = !miniOn; $('mini').hidden = !miniOn; syncUI(); }
  else if (k === 'shot') tools.savePng();
  else if (k === 'brief') tools.saveBriefing();
  else if (k === 'shadow') setShadows(!shadows);
  else if (k === 'lowq') setLowQ(!lowQ);
  else if (k === 'help') { $('help').hidden = !$('help').hidden; helpPerf(); }
  invalidate();
}
$('dock').addEventListener('click', (e) => { const b = e.target.closest('[data-tool]'); if (b && !b.disabled) tool(b.dataset.tool); });
$('help').addEventListener('click', (e) => { if (e.target.id === 'help' || e.target.closest('.x')) $('help').hidden = true; });

// ── 키보드: 1~9 지상층 · 0 지하1층 · Shift+숫자 지하층 · F 화재 가정 · X 투시 · S 단면 · I 격리 · M 거리 · A 면적 · G 따라가기 · N 미니맵 · P 저장 · B 브리핑 · Space 재생 · H 그림자 · L 저사양 · ? 도움말 · Esc
addEventListener('keydown', (e) => {
  const ae = document.activeElement;                       // 글자 입력 칸에서만 단축키를 끔(슬라이더·체크 상자는 단축키 허용)
  if (ae && (/SELECT|TEXTAREA/.test(ae.tagName) || (ae.tagName === 'INPUT' && !/range|checkbox|radio|button/.test(ae.type))) && e.key !== 'Escape') return;
  if (ae && ae.tagName === 'INPUT' && ae.type === 'range' && /^Arrow/.test(e.key)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  invalidate();
  const k = e.key;
  if (k === 'Escape') {
    if (followOn) follow.stop();
    else if ($('modal').style.display === 'flex') closePlan();
    else if (!$('help').hidden) $('help').hidden = true;
    else if (picking) stopPick();
    else if (tools.mode()) { tools.setMode(null); syncUI(); }
    else if ($('info').style.display === 'block') $('info').style.display = 'none';
    else if ($('mkTip').style.display === 'block') mkHide(true);
    else if ($('layers').classList.contains('open')) $('layMore').click();
    else if (section.active()) toggleSection(false);
    else if (xray) setXray(false);
    else if (isolate) setIsolate(false);
    else if (!fireActive()) goHome();
    return;
  }
  if ((k === 'Delete' || k === 'Backspace') && (tools.measures().length || tools.area())) { tools.clear(); invalidate(); return; }   // 잰 거리·면적 지우기
  if (k === 'ArrowUp' || k === 'ArrowDown') { const i = FL[selId].idx + (k === 'ArrowUp' ? 1 : -1); if (floors[i]) { selectFloor(floors[i].id, true); e.preventDefault(); } return; }
  const dg = e.code && e.code.startsWith('Digit') ? Number(e.code.slice(5)) : /^[0-9]$/.test(k) ? Number(k) : null;
  if (dg != null) { const lv = e.shiftKey ? -dg : dg === 0 ? -1 : dg; const f = floors.find((q) => q.level === lv); if (f) selectFloor(f.id, true); e.preventDefault(); return; }
  const c = k.toLowerCase();
  const map = { x: 'xray', s: 'section', i: 'isolate', m: 'measure', a: 'area', g: 'follow', n: 'minimap', p: 'shot', b: 'brief', h: 'shadow', l: 'lowq' };
  if (c === 'f') { if (picking) stopPick(); else startPick(); }
  else if (c === 't') $('thBtn').click();
  else if (k === '+' || k === '=') setFont(FZ[Math.min(FZ.length - 1, FZ.indexOf(fz) + 1)]);
  else if (k === '-' || k === '_') setFont(FZ[Math.max(0, FZ.indexOf(fz) - 1)]);
  else if (map[c]) tool(map[c]);
  else if (k === ' ' && fire) { e.preventDefault(); $('tlPlay').click(); }
  else if (k === '?' || k === '/') tool('help');
});

// ── 터치: 한 손가락 회전 · 두 손가락 확대/이동(OrbitControls) + 두 손가락 비틀기 = 수평 회전
{
  const tp = new Map(); let ang = null;
  const A = () => { const [a, b] = [...tp.values()]; return Math.atan2(b[1] - a[1], b[0] - a[0]); };
  canvas.addEventListener('pointerdown', (e) => { if (e.pointerType === 'touch') { tp.set(e.pointerId, [e.clientX, e.clientY]); ang = tp.size === 2 ? A() : null; } });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'touch' || !tp.has(e.pointerId)) return; tp.set(e.pointerId, [e.clientX, e.clientY]);
    if (tp.size === 2 && ang != null) {
      const a = A(); let d = a - ang; if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; ang = a;
      if (Math.abs(d) > 1e-4) { const off = camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), -d); camera.position.copy(controls.target).add(off); invalidate(); }
    }
  });
  const up = (e) => { tp.delete(e.pointerId); ang = tp.size === 2 ? A() : null; };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
}
['pointerdown', 'pointermove', 'wheel', 'keydown', 'click', 'input', 'change'].forEach((ev) => addEventListener(ev, invalidate, { capture: true, passive: true }));
document.addEventListener('click', () => stage.classList.toggle('picking', !!(picking || tools.mode() || (train && train.picking && train.picking()))));   // 훈련 '실 고르기' 등 모듈 버튼 뒤에도 갱신

// ═════════════════════════ 정보카드 건물개황(표준양식) — 개요 탭
function cardPanel(B) {
  const C = B.card || {};
  if (C.양식 !== 'std') return;
  $('cardSec').style.display = '';
  const S = C.건물구조 || {}, G = C.거리 || {}, L = C.소재지 || {};
  const km = (v) => (v == null ? '-' : `${v}km`);
  let h = '';
  if ((C.위험물 || []).length) {
    h += (C.위험물).map((x) => `<div class="haz"><b>⚠ ${esc(x.구분)} — 진입 전 확인</b>${esc([x.시설명, x.품명, x.수량].filter(Boolean).join(' · '))}</div>`).join('');
  }
  h += '<table>';
  const row = (k, v) => { h += `<tr><td class="k">${k}</td><td>${esc(v == null || v === '' ? '-' : String(v))}</td></tr>`; };
  row('관리번호', C.관리번호);
  row('소재지', L.도로명 || L.지번);
  row('구조 · 지붕', `${S.구조 || '-'} · ${S.지붕 || '-'} (${C.내화구조 === true ? '내화 추정' : C.내화구조 === false ? '비내화 추정' : '미확인'})`);
  row('거리(관할/최단/소방서)', `${km(G.관할센터)} / ${km(G.최단센터)} / ${km(G.소방서)}`);
  row('소방용수', (C.소방용수 || []).map((w) => `${w.종류}-${w.관서}-${w.동}-${w.번호} ${Math.round(w.거리km * 1000)}m`).join(', '));
  if (!(C.위험물 || []).length) row('위험물 시설 등', '없음');
  const rel = C.관계인 || [];
  if (rel.some((r) => r.성명 || r.연락처)) for (const r of rel) row(r.구분, [r.성명, r.연락처].filter(Boolean).join(' '));
  else row('관계인', (rel.filter((r) => r.기재).map((r) => r.구분).join('·') || '-') + ' 기재 (성명·연락처는 내부 PC 옵션)');
  h += '</table>';
  const DL = C.동목록 || [], files = B.dong_files || [];
  if (DL.length > 1) {
    h += '<div class="dong" style="margin-top:6px"><span class="k">동 선택 (' + DL.length + '개 동)</span>';
    DL.forEach((d, i) => {
      const f = files[i];
      const lab = `${esc(d.동명 || '동' + (i + 1))} · ${esc(d.주용도 || '')} ${esc(d.층수)}${d.위험물 ? ' ⚠' : ''}`;
      h += `<div class="${d.선택 ? 'on' : ''}">${d.선택 ? '▶ ' : ''}${f && !d.선택 ? `<a href="${esc(f)}">${lab}</a>` : lab}${!f && !d.선택 ? ' <span class="k">(3D 없음 — 정보만)</span>' : ''}</div>`;
    });
    h += '</div>';
  }
  $('cardPanel').innerHTML = h;
}

// ═════════════════════════ 화면 속도: 바뀐 것이 있을 때만 그림 + 느리면 해상도 자동 낮춤
const perf = { frames: 0, t0: performance.now(), fps: 0, pr: PR0, slow: 0, fast: 0, auto: true, ms: [] };
function helpPerf() {
  const i = renderer.info.render;
  $('perfBox').innerHTML = GL ? `화면 ${perf.fps ? perf.fps.toFixed(0) : '-'} fps(움직일 때) · 그리기 ${i.calls}회 · 삼각형 ${i.triangles.toLocaleString()} · 해상도 배율 ${renderer.getPixelRatio().toFixed(2)}${lowQ ? ' · 저사양 모드' : ''}${shadows ? ' · 그림자' : ''}` : '2D 대체 보기(WebGL 없음)';
}
function governor(t, frameMs) {
  perf.frames++; perf.ms.push(frameMs); if (perf.ms.length > 120) perf.ms.shift();
  if (t - perf.t0 < 1000) return;
  perf.fps = (perf.frames * 1000) / (t - perf.t0); perf.frames = 0; perf.t0 = t;
  if (!perf.auto || !GL) return;
  const cur = renderer.getPixelRatio();
  if (perf.fps < 24) { if (++perf.slow >= 2 && cur > 0.75) { renderer.setPixelRatio(Math.max(0.75, cur - 0.25)); resize(); perf.slow = 0; } perf.fast = 0; }
  else if (perf.fps > 50) { if (++perf.fast >= 4 && cur < perf.pr) { renderer.setPixelRatio(Math.min(perf.pr, cur + 0.25)); resize(); perf.fast = 0; } perf.slow = 0; }
}

// ═════════════════════════ 시작
// ── 고정 UI 배치(v3.1): 서로 겹치지 않게 자리를 계산 — 위 범례 줄(넘치면 +N) · 왼쪽 도구 열(넘치면 2열·작게) · 오른쪽 층 열 ·
//    아래 보기 줄 + 나침반 · 그 위 시간축(경로 범례 포함)/실 용도 범례/객체 정보 · 오른쪽 아래 미니맵(좁으면 시간축 위로, 자리 없으면 숨김) · 위 가운데 안내 띠들
function uiRects() {                                           // 화면 위 고정 UI 사각형(라벨 겹침 회피·시험용)
  const out = [];
  for (const q of UI_SEL) for (const el of stage.querySelectorAll(q)) {
    if (!vis(el)) continue;
    if (el.id === 'fbar') { const bs = [...el.children].map((c) => c.getBoundingClientRect()).filter((r) => r.width); if (!bs.length) continue;   // 층 열은 단추가 있는 높이까지만
      out.push({ id: '#fbar', l: Math.min(...bs.map((r) => r.left)), t: Math.min(...bs.map((r) => r.top)), r: Math.max(...bs.map((r) => r.right)), b: Math.max(...bs.map((r) => r.bottom)) }); continue; }
    const r = el.getBoundingClientRect(); if (r.width && r.height) out.push({ id: el.id ? '#' + el.id : q + (el.dataset.l ? `[${el.dataset.l}]` : ''), l: r.left, t: r.top, r: r.right, b: r.bottom });
  }
  return out;
}
function layoutUI() {
  const W = stage.clientWidth, Hs = stage.clientHeight, M = 12, GAP = 8;
  if (!W || !Hs) return;
  const XR = W - M - $('fbar').offsetWidth - 6;               // 층 열 왼쪽 끝
  // 1) 범례 칩 한 줄(넘치면 +N)
  const lay = $('layers'), more = $('layMore'), open = lay.classList.contains('open');
  lay.style.right = px(W - XR); lay.querySelectorAll('.lay').forEach((b) => b.classList.remove('ovf'));
  let nOv = 0;
  if (!open) {
    const chips = [...lay.querySelectorAll('.lay')], top0 = chips.length ? chips[0].offsetTop : 0;
    let ov = chips.filter((b) => b.offsetTop > top0 + 4);
    if (ov.length) { lay.style.right = px(W - XR + 74); ov = chips.filter((b) => b.offsetTop > top0 + 4); }
    ov.forEach((b) => b.classList.add('ovf')); nOv = ov.length;
    more.hidden = !nOv;
    if (nOv) { const last = chips.filter((b) => !b.classList.contains('ovf')).pop(); more.style.left = px((last ? last.offsetLeft + last.offsetWidth : 0) + M + 6); more.innerHTML = `${svg('layers')}+${nOv}`; more.title = `나머지 레이어 ${nOv}개: ` + ov.map((b) => b.textContent).join(', '); more.setAttribute('aria-label', more.title); }
  } else { more.hidden = false; more.innerHTML = svg('close') + '접기'; more.style.left = px(Math.min(XR - 70, lay.offsetLeft + lay.offsetWidth - 72)); more.style.top = px(lay.offsetTop + lay.offsetHeight + 4); }
  if (!open) more.style.top = px(M + 1);
  const topY = M + 31 + 10;
  // 2) 도구 열: 한 열 → 두 열 → 작은 단추
  const dk = $('dock'); dk.style.top = px(topY);
  const avail = Hs - topY - M;
  dk.classList.remove('two', 'compact');                       // 한 열 → 한 열(작게) → 두 열 → 두 열(작게): 글자 이름은 항상 보임
  for (const c of [['compact'], ['two'], ['two', 'compact']]) { if (dk.offsetHeight <= avail) break; dk.classList.remove('two', 'compact'); dk.classList.add(...c); }
  const leftX = M + dk.offsetWidth + 10;
  // 3) 아래 보기 줄 + 나침반(층 열 왼쪽)
  const cp = $('compass'), cw = cp.offsetWidth || 44; cp.style.right = px(W - XR); cp.style.bottom = px(M);
  const vw = document.querySelector('#stage .views'); vw.style.left = px(leftX); vw.style.right = px(W - (XR - cw - 10)); vw.style.bottom = px(M);
  const vb = vw.offsetHeight, viewsTop = Hs - M - vb;
  // 4) 미니맵 크기·자리 / 시간축·실 용도 범례
  const mini = $('mini'), tl = $('tl'), vl = $('volLeg'), tlOn = !tl.hidden, vlOn = !vl.hidden;
  let mw = Math.round(Math.max(150, Math.min(240, W * 0.2))), mh = Math.round(mw * 0.74);
  const stackL = leftX;
  let stackR = XR - mw - 10, miniAbove = false;
  if (tlOn && stackR - stackL < 470) { stackR = XR; miniAbove = true; }
  if (!tlOn && vlOn && stackR - stackL < 260) { stackR = XR; miniAbove = true; }
  let y = viewsTop - GAP;                                      // 쌓기: 아래에서 위로
  for (const el of [tl, vl]) {
    if (el.hidden) continue;
    el.style.left = px(stackL); el.style.right = px(W - stackR); el.style.bottom = px(Hs - y); el.style.maxWidth = '';
    if (el === vl) { el.style.right = 'auto'; el.style.maxWidth = px(stackR - stackL); }
    y -= el.offsetHeight + GAP;
  }
  stage.style.setProperty('--vb', vb + 'px'); stage.style.setProperty('--tlh', (tlOn ? tl.offsetHeight : 0) + 'px');
  const stackTop = y + GAP;                                    // 쌓인 것 맨 위
  // 미니맵: 오른쪽 아래(나침반 위) 또는 쌓인 것 위
  const vRight = Math.max(0, ...[...vw.children].map((b) => vw.offsetLeft + b.offsetLeft + b.offsetWidth));
  let mBottom = miniAbove ? Hs - stackTop + GAP : vRight > XR - mw - 10 ? Hs - viewsTop + GAP : M + cw + GAP;   // 보기 줄이 미니맵 아래까지 오면 그 위로
  const room = Hs - mBottom - topY - GAP;
  if (mh > room) { mh = Math.max(0, room); mw = Math.round(mh / 0.74); }
  const fit = mh >= 96 && mw >= 130 && XR - mw >= leftX + 40;
  mini.classList.toggle('nofit', !fit);
  if (fit) { mini.style.width = px(mw); mini.style.height = px(mh); mini.style.right = px(W - XR); mini.style.bottom = px(mBottom); }
  // 5) 객체 정보 상자: 쌓인 것 위, 미니맵과 가로로 겹치지 않게
  const info = $('info'); info.style.left = px(stackL); info.style.bottom = px(Hs - stackTop + GAP);
  const miniL = fit && miniOn ? XR - mw : XR, miniT = fit && miniOn ? Hs - mBottom - mh : Hs;
  info.style.maxWidth = px(Math.max(200, Math.min(440, (miniAbove || stackTop > miniT ? miniL : stackR) - stackL - 10)));
  // 6) 위 가운데 띠(단면 창 · 따라가기 · WebGL 안내 · 클릭 안내): 왼쪽 도구 열과 층 열 사이, 앞의 것과 겹치면 아래로
  const placed = [];
  const sp = $('secPanel'); sp.style.left = px(leftX); sp.style.top = px(topY);
  if (!sp.hidden) placed.push([leftX, topY, leftX + sp.offsetWidth, topY + sp.offsetHeight]);
  for (const id of ['follow', 'glmsg', 'hint']) {
    const el = $(id); if (!vis(el)) continue;
    el.style.transform = 'none'; el.style.maxWidth = px(XR - leftX - 16);
    const w = el.offsetWidth, h = el.offsetHeight;
    let x = Math.round((leftX + XR) / 2 - w / 2), t = topY;
    for (const q of placed) {                                  // 왼쪽 단면 창 옆에 자리가 있으면 옆으로, 없으면 아래로
      if (x < q[2] && x + w > q[0] && t < q[3] && t + h > q[1]) {
        if (XR - q[2] - 10 >= w) x = Math.round(q[2] + 10 + (XR - q[2] - 10 - w) / 2); else t = q[3] + GAP;
      }
    }
    el.style.left = px(Math.max(leftX, x)); el.style.top = px(t); placed.push([x, t, x + w, t + h]);
  }
  Object.assign(LAY, { W, Hs, leftX, topY, XR, stackTop, topBand: placed.length ? Math.max(...placed.map((q) => q[3])) : topY });
  // 3D 화면의 '빈 곳' 가운데로 시점 중심을 옮김(setViewOffset): 아래 시간축·보기 줄, 위 범례 줄, 양옆 도구·층 열이 가리는 만큼 — 고른 층·화점이 UI 밑에 숨지 않게
  const occT = topY, occB = Hs - stackTop, occL = leftX, occR = W - XR;
  const ox = Math.round(Math.max(-W * 0.15, Math.min(W * 0.15, (occR - occL) / 2))), oy = Math.round(Math.max(-Hs * 0.2, Math.min(Hs * 0.2, (occB - occT) / 2)));
  const v = camera.view;
  if (!v || !v.enabled || v.fullWidth !== W || v.fullHeight !== Hs || v.offsetX !== ox || v.offsetY !== oy) { camera.setViewOffset(W, Hs, ox, oy, W, Hs); invalidate(); }
}
function resize() {
  const r = stage.getBoundingClientRect(), w = Math.max(50, r.width), h = Math.max(50, r.height);
  renderer.setSize(w, h, false); labelR.setSize(w, h); camera.aspect = w / h; camera.fov = w < h ? 60 : 42; camera.updateProjectionMatrix();
  // 핀(크기 고정 px): 화면 높이·시야각에 맞춰 크기 환산
  layoutUI();
  const k = 1 / (camera.projectionMatrix.elements[5] * h / 2);
  scene.traverse((o) => { if (o.isSprite && o.userData.pin) { if (o.material.sizeAttenuation) { o.material.sizeAttenuation = false; o.material.needsUpdate = true; } o.scale.set(o.userData.size * k, o.userData.size * k, 1); } });
  invalidate();
}
new ResizeObserver(resize).observe(stage); resize();
document.title = `${D.building.name} — AI 건축물 지휘정보 3D`;
panels.header(); panels.footer({ H, stat: $('bstat').textContent }); panels.issues(); panels.overview(userHidden); panels.haz(); panels.entry(); panels.basis();
cardPanel(D.building);
// 검수 반영본: review/common.py 가 HTML 의 '자동 변환 · 검수 전'(머리줄 원문)을 '… 검수 반영'으로 바꿈 → 번들 JS 안 문구는 유니코드 이스케이프라 치환이 안 닿으므로 머리줄 값을 읽어 씀
const reviewed = /검수 반영/.test($('bstat').textContent);
$('src').innerHTML = `${esc(D.building.source)}<br>벽·문·계단·승강기·소방시설 위치는 평면도에서 ${reviewed ? '자동 추출한 뒤 검수자가 확인·수정한 값입니다' : '자동 추출한 값이며 검수 전입니다'}.<br>IFC4 파일과 같은 객체 ID(GlobalId)를 씁니다.`;
if (innerWidth < 1360 && innerWidth > 1100) $('app').classList.add('noL');   // 작은 화면(1280 등): 왼쪽 정보 패널은 접고 시작(☰ 정보로 열기) — 3D·오른쪽 요약을 넓게
floorBar(); applyView({ instant: true }); renderRight(); syncUI(); panelAria();
{ const [t, p] = homePose(); controls.target.copy(t); camera.position.copy(p); }
// 2D 대체 보기(WebGL 없음): 큰 평면 캔버스 + 안내 띠, 3D 전용 도구는 끔
// 2D 대체 보기(WebGL 없음 · 실행 중 그래픽 중단): 큰 평면 캔버스 + 안내 띠, 3D 전용 도구는 끔
let fb2d = false;
function enable2D(why) {
  GL = false; if (fb2d) return; fb2d = true;
  if (why) $('glTxt').innerHTML = why;
  $('fallback').hidden = false; $('labels').style.display = 'none'; $('glmsg').hidden = false;
  document.querySelectorAll('#dock [data-tool="section"],#dock [data-tool="xray"],#dock [data-tool="isolate"],#dock [data-tool="follow"],#dock [data-tool="shadow"],#dock [data-tool="lowq"]').forEach((b) => (b.disabled = true));
  $('vExp').disabled = true; $('vTop').disabled = true; $('vIso').disabled = true; miniOn = false; $('mini').hidden = true;
  $('fallback').addEventListener('click', (e) => {
    const cv = $('fallback'), T = cv.__T; if (!T) return; const r = cv.getBoundingClientRect(); const [x, y] = T.inv((e.clientX - r.left) * cv.__dpr, (e.clientY - r.top) * cv.__dpr);
    if (!floorClick(x, y)) { focus = true; renderRight(); syncUI(); }
  });
  syncUI();
}
if (!GL) enable2D();
$('glX').onclick = () => { $('glmsg').hidden = true; layoutUI(); };
// 그래픽 드라이버가 3D 를 멈추면(내장 GPU 메모리 부족 등) 2D 로 이어서 씀 — 계산·패널은 그대로
canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); enable2D('3D 그래픽이 중단되어 <b>2D 평면 보기</b>로 바꿨습니다 — 화점·경로·브리핑은 그대로 사용<small>창을 닫았다 다시 열면 3D 로 돌아옵니다. 자주 생기면 저사양 모드(L)를 쓰세요.</small>'); });

// ═════════════════════════ v3.1 화면 설정: 테마(야간·주간·고대비) · 글자 크기 · 인쇄 · 시작 화면 · 아이콘
const THEMES = [['night', '야간', 'moon'], ['day', '주간', 'sun'], ['hc', '고대비', 'contrast']];
const store = { get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* 저장 불가(사생활 보호 창 등) */ } } };
function setTheme(t) {
  const th = THEMES.find((z) => z[0] === t) || THEMES[0];
  document.documentElement.dataset.theme = th[0]; store.set('firebim.theme', th[0]);
  const nx = THEMES[(THEMES.indexOf(th) + 1) % THEMES.length];
  $('thBtn').innerHTML = svg(th[2]) + `<span>${th[1]}</span>`; $('thBtn').setAttribute('aria-label', `화면 테마: ${th[1]} — 누르면 ${nx[1]}`); $('thBtn').title = `화면 테마: ${th[1]} (T) — 누르면 ${nx[1]}`;
  scene.fog.color.setHex(th[0] === 'hc' ? 0x000000 : th[0] === 'day' ? 0x122340 : 0x0b1220);
  invalidate(); setTimeout(layoutUI, 0); return th[0];
}
const FZ = [0.9, 1, 1.12, 1.25, 1.4];
let fz = Number(store.get('firebim.fz')) || 1;
function setFont(z) {
  fz = FZ.reduce((a, b) => (Math.abs(b - z) < Math.abs(a - z) ? b : a), 1);
  document.documentElement.style.setProperty('--fz', fz); document.documentElement.style.setProperty('--lz', fz); store.set('firebim.fz', fz);
  $('fzDn').disabled = fz <= FZ[0]; $('fzUp').disabled = fz >= FZ[FZ.length - 1];
  $('fzUp').title = `글자 크게 (+) — 지금 ${Math.round(fz * 100)}%`; $('fzDn').title = `글자 작게 (−) — 지금 ${Math.round(fz * 100)}%`;
  setTimeout(layoutUI, 0); invalidate(); return fz;
}
$('thBtn').onclick = () => setTheme(THEMES[(THEMES.findIndex((z) => z[0] === document.documentElement.dataset.theme) + 1) % THEMES.length][0]);
$('fzUp').innerHTML = svg('textup'); $('fzDn').innerHTML = svg('textdn'); $('prBtn').innerHTML = svg('print') + '<span>인쇄</span>';
$('fzUp').onclick = () => setFont(FZ[Math.min(FZ.length - 1, FZ.indexOf(fz) + 1)]); $('fzDn').onclick = () => setFont(FZ[Math.max(0, FZ.indexOf(fz) - 1)]);
$('prBtn').onclick = () => { buildPrint(); window.print(); };
// 인쇄(Ctrl+P · 인쇄 단추): 화면 대신 A4 지휘 요약(진입구·계단·소화전·위험물·확인사항·화재 가정 + 3D 그림) — tools.js 와 같은 내용
function buildPrint() { try { $('printSheet').innerHTML = tools.printSheet(); } catch (e) { console.warn('[FireBIM] 인쇄 요약 만들기 실패', e); } }
addEventListener('beforeprint', buildPrint);
// 아이콘(인라인 SVG): 도구 열 · 보기 줄 · 머리줄
const DOCKICON = { section: 'section', xray: 'xray', isolate: 'isolate', measure: 'measure', area: 'area', follow: 'follow', minimap: 'minimap', shot: 'shot', brief: 'brief', shadow: 'shadow', lowq: 'lowq', help: 'help' };
document.querySelectorAll('#dock [data-tool]').forEach((b) => { const sm = b.querySelector('small'); b.innerHTML = svg(DOCKICON[b.dataset.tool]) + (sm ? sm.outerHTML : ''); });
for (const [id, ic, t] of [['vAll', 'home', '전체'], ['vExp', 'explode', '층 분리'], ['vOne', 'plan', '단층 평면'], ['bFire', 'fire', '화재 가정'], ['vTop', 'top', '위'], ['vIso', 'iso', '입체']]) $(id).innerHTML = svg(ic) + `<span>${t}</span>`;
$('tlFollow').innerHTML = svg('follow'); $('glX').innerHTML = svg('close'); document.querySelector('#help .x').innerHTML = svg('close');
$('tgR').setAttribute('aria-expanded', 'true'); $('tgL').setAttribute('aria-expanded', 'true');
// 시작 화면 닫기(첫 화면을 그린 뒤)
function bootDone() { const b = $('boot'); if (!b || b.classList.contains('err')) return; b.classList.add('done'); setTimeout(() => b.remove(), 400); }
// ── 인명검색 번호(묶음 포함) 설명: 마우스 올림 = 잠깐, 누름 = 고정(태블릿) · 줄을 누르면 그 실로 이동
let mkPinned = false;
function mkShow(el, pinned) {
  const S = search && search.last(); if (!S) return;
  const ranks = (el.dataset.ranks || el.dataset.rank || el.textContent).split(',').map(Number);
  const items = ranks.map((k) => S.items.find((z) => z.rank === k)).filter(Boolean);
  if (!items.length) return;
  const CK = { iso: '고립', shared: '공격계단 공용', near: '화점 인접', detour: '우회', ok: '정상', fire: '화재실' };
  const tip = $('mkTip');
  tip.innerHTML = `<button class="x" aria-label="닫기">${svg('close')}</button><b style="color:#c4b5fd">인명검색 ${items.length > 1 ? `${items.length}곳(겹친 번호 묶음)` : `${items[0].rank}순위`}</b><table>` +
    items.map((z) => `<tr data-fly="${z.c[0]},${z.c[1]}" data-fl="${esc(z.fid)}" title="${esc((z.reasons || []).join(' · '))}"><td><b>${z.rank}</b></td><td style="text-align:left">${esc(FL[z.fid].name)} · ${esc(CK[z.cls] || z.cls || '')} · ${esc(CAT_KO[z.cat] || z.cat || '')}${z.area != null ? ` ${Math.round(z.area)}㎡` : ''}<br><span class="k">${esc((z.reasons || []).slice(0, 2).join(' · '))}</span></td><td>${z.d != null ? z.d.toFixed(0) + 'm' : '-'}</td></tr>`).join('') + '</table><div class="note" style="margin-top:2px">줄을 누르면 그 실로 이동</div>';
  tip.style.display = 'block';
  const sr = stage.getBoundingClientRect(), r = el.getBoundingClientRect();
  let x = r.right - sr.left + 8, y = r.top - sr.top - 6;
  if (x + tip.offsetWidth > LAY.XR) x = r.left - sr.left - tip.offsetWidth - 8;
  y = Math.max(LAY.topY, Math.min(y, sr.height - tip.offsetHeight - 8));
  tip.style.left = Math.max(LAY.leftX, x) + 'px'; tip.style.top = y + 'px'; mkPinned = !!pinned;
}
const mkHide = (force) => { if (mkPinned && !force) return; $('mkTip').style.display = 'none'; mkPinned = false; };
$('labels').addEventListener('pointerover', (e) => { const el = e.target.closest('.lb.srch'); if (el && !mkPinned) mkShow(el, false); });
$('labels').addEventListener('pointerout', (e) => { if (e.target.closest('.lb.srch')) mkHide(false); });
$('labels').addEventListener('click', (e) => { const el = e.target.closest('.lb.srch'); if (el) { e.stopPropagation(); mkShow(el, true); } });
$('mkTip').addEventListener('click', (e) => { if (e.target.closest('.x')) mkHide(true); else if (e.target.closest('[data-fly]')) mkHide(true); });
canvas.addEventListener('pointerdown', () => mkHide(true));
// ── 출동 전 30초 요약(오른쪽 맨 위)
function brief30() { panels.brief30({ plan: lastPlan && !(train && train.hidden()) ? lastPlan : null, search: search && fire && !(train && train.hidden()) ? search.last() : null }); }
$('b30').addEventListener('click', (e) => {
  if (e.target.closest('.hd')) { const o = !$('b30').classList.contains('open'); $('b30').classList.toggle('open', o); e.target.closest('.hd').setAttribute('aria-expanded', o); store.set('firebim.b30', o ? '1' : '0'); return; }
  const b = e.target.closest('[data-b30]'); if (!b) return;
  const k = b.dataset.b30;
  if (k === 'issues') $('bissue').click();
  else if (k === 'haz') { showTab('haz'); if (narrow()) { $('app').classList.add('showL'); $('app').classList.remove('showR'); } else $('app').classList.remove('noL'); }
});
if (store.get('firebim.b30') === '0') $('b30').classList.remove('open');
setTheme(document.documentElement.dataset.theme || 'night'); setFont(fz);
function miniDyn(fid) {
  const d = { fid };
  if (fire && fire.fid === fid) d.fire = [fire.x, fire.y];
  if (lastPlan && !(train && train.hidden())) {
    d.routes = []; for (const Lg of lastPlan.legs || []) if (!Lg.vert && Lg.fid === fid) d.routes.push([Lg.pts, '#ffd33d', 2.2]);
    if (lastPlan.hosePath && lastPlan.fid === fid) d.routes.push([lastPlan.hosePath, '#ff4d4f', 1.6]);
  }
  if (TL.on && sim.last()) { const nodes = sim.nodes(), R = sim.last(); d.rooms = []; for (let i = 1; i < nodes.length; i++) { const n = nodes[i]; if (n.fid === fid && R.ttu[i] <= TL.t) d.rooms.push({ poly: n.poly, col: sim.ttuCol(R.ttu[i]), a: 0.55 }); } }
  if (crewState.length && TL.crew) d.crew = crewState.filter((z) => z.p && z.fid === fid).map((z) => ({ p: [z.p.x, -z.p.z], col: z.col, name: z.name }));
  const ms = tools.measures().filter((m) => m.fid === fid); if (ms.length) d.measure = [ms[ms.length - 1].a, ms[ms.length - 1].b];
  return d;
}
let lastDc = 0, lastMini = 0, lastAttach = 0, lastRender = 0, prevT = performance.now();
let booted = false;
// 라벨 정리: 고정 UI 배치 → 라벨(말풍선·번호·꼬리표) 겹침 회피(고정 UI 를 장애물로)
function tidy() { layoutUI(); return declutter($('labels'), uiRects(), { simOn: stage.classList.contains('simOn') }); }
function loop(t) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (t - prevT) / 1000); prevT = t;
  for (const [g, w] of tw) { const k = Math.min(1, (t - w.t0) / 550), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; g.position.y = w.from + (w.to - w.from) * e; if (k >= 1) tw.delete(g); }
  if (fly) {
    const k = Math.min(1, (t - fly.t0) / fly.ms), e = k * k * (3 - 2 * k);
    controls.target.lerpVectors(fly.fromT, fly.toT, e); camera.position.lerpVectors(fly.fromP, fly.toP, e); if (k >= 1) fly = null;
  }
  if (fire && fire.ring) fire.ring.scale.setScalar(1 + 0.15 * Math.sin(t / 200));
  if (fire && fire.flame) { const n = 1 + 0.12 * Math.sin(t / 110) + 0.06 * Math.sin(t / 47); fire.flame.scale.set(4.2 * n, 5.4 * (2 - n), 1); }
  if (routeAnim && fireG.visible) { const R = routeAnim, M = new THREE.Matrix4(); for (let i = 0; i < R.n; i++) { const p = R.path.getPointAt(((t / 1000) * 6 / R.len + i / R.n) % 1); R.im.setMatrixAt(i, M.makeTranslation(p.x, p.y, p.z)); } R.im.instanceMatrix.needsUpdate = true; }
  for (let i = pings.length - 1; i >= 0; i--) { const p = pings[i], k = (t - p.t0) / 1600; if (k >= 1) { scene.remove(p.r); pings.splice(i, 1); continue; } const s = 1 + 4 * ((k * 2) % 1); p.r.scale.setScalar(s); p.r.material.opacity = 1 - ((k * 2) % 1); }
  if (TL.playing && fire) { TL.t = Math.min(TL.T, TL.t + dt * TL.speed); if (TL.t >= TL.T) TL.playing = false; applyTime(); }
  const fol = follow.step(t);
  const anim = tw.size || fly || fire || pings.length || TL.playing || fol;
  const moved = followOn ? false : controls.update();
  if (!GL) { if (!booted) { booted = true; bootDone(); } if (t - lastMini > 200) { lastMini = t; layoutUI(); minimap.draw($('fallback'), miniDyn(fire ? fire.fid : selId), { noFrustum: true, fitFloor: true, inset: { l: LAY.leftX + 6, t: LAY.topBand + 30, r: LAY.W - LAY.XR + 8, b: LAY.Hs - LAY.stackTop + 10 } }); } return; }
  if (!(moved || anim || dirty || t - lastRender > 1000)) return;
  dirty = false;
  const t0 = performance.now();
  vert.sync();
  if (section.active()) section.update();
  if (t - lastAttach > 500) { lastAttach = t; section.attach(); if (section.active()) section.labels(); }
  renderer.render(scene, camera); labelR.render(scene, camera);
  if (anim || moved) governor(t, performance.now() - t0); lastRender = t;
  if (t - lastDc > 180) { const fv = camera.position.distanceTo(controls.target) > Math.max(60, SPAN * 1.25); if (fv !== farView) { farView = fv; styleAll(); } }
  if (t - lastDc > 180) { lastDc = t; tidy(); $('compass').lastElementChild.style.transform = `rotate(${controls.getAzimuthalAngle()}rad)`; }
  if (miniOn && !$('mini').classList.contains('nofit') && t - lastMini > 150) { lastMini = t; minimap.draw($('mini'), miniDyn(fire ? fire.fid : selId)); }
  if (!booted) { booted = true; bootDone(); }
}
requestAnimationFrame(loop);
// 시험·평가용: 연속 N 프레임 그려서 평균 시간
function bench(n = 60) {
  const t0 = performance.now(); let calls = 0, tris = 0;
  for (let i = 0; i < n; i++) { controls.update(); vert.sync(); renderer.render(scene, camera); labelR.render(scene, camera); calls = renderer.info.render.calls; tris = renderer.info.render.triangles; }
  if (GL) { const gl = renderer.getContext(); const px = new Uint8Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); }   // GPU 끝날 때까지 기다림
  const ms = (performance.now() - t0) / n;
  return { ms, fps: 1000 / ms, calls, tris, pr: renderer.getPixelRatio(), gl: GL };
}
window.__viewer = { selectFloor, setFire: (fid, x, y, opts) => { fire = { fid, x, y, ...(opts || {}) }; picking = false; fireMode = true; rTab = 'fire'; selectFloor(fid, false); syncUI(); }, train: () => $('bTrain').click(), setView,
  alarm: (fid, no) => { $('alFl').value = fid; $('alNo').value = no; $('alGo').click(); }, site: (on) => { $('tSite').checked = on !== false; $('tSite').onchange(); },
  fac: (on, cats) => { if (cats) fac.cats(cats); L.fac = on !== false; fac.set(on); syncChips(); }, info: (html) => { $('info').innerHTML = html; $('info').style.display = 'block'; },
  zones: (on) => { $('tZone').checked = on !== false; L.zone = on !== false; ext.onView(); syncChips(); },
  // 인명검색·훈련·활동계획(시험·평가용)
  search: () => (search ? search.last() : null), searchMode: (m) => { search.setMode(m); if (fire) drawFire(); },
  plan: () => (lastPlan ? Object.fromEntries(Object.entries(lastPlan).filter(([k]) => !['grF', 'RF', 'grG'].includes(k))) : null), setAct: (o) => { Object.assign(ACT, o); if (fire) drawFire(); }, trainer: () => train,
  evalFire: (fid, x, y, mode) => { if (mode) search.setMode(mode); fire = { fid, x, y }; selId = fid; drawFire(); return search.last(); },     // 평가용: 한 번만 계산
  project: (fid, x, y) => { const v = V(x, y, zOf(fid) + SLAB).project(camera), r = canvas.getBoundingClientRect(); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; },
  view: (m) => (m === 'all' ? goHome() : setMode(m)), cam: (x, y, z, dx, dy, dz) => { controls.target.copy(V(x, y, z)); camera.position.copy(V(x + dx, y + dy, z + dz)); fly = null; invalidate(); },
  // v2 화면
  home: goHome, layer: (k, v) => setLayer(k, v !== false), tab: showTab,
  state: () => ({ selId, focus, viewMode, fire: !!fire, picking, fireMode, layers: { ...L }, visibleFloors: floors.filter((f) => G[f.id].g.visible).map((f) => f.id),
    xray, isolate, lowQ, shadows, section: { ...section.S, active: section.active() }, tool: tools.mode(), follow: followOn, minimap: miniOn, gl: GL, t: TL.t, playing: TL.playing, simOn: TL.on }),
  // v3: 단면·투시·격리·시뮬레이션·대원·따라가기·미니맵·도구·성능
  section: (o) => { if (o === false) toggleSection(false); else { if (o && o.v != null) $('secV').checked = !!o.v; if (o && o.h != null) $('secH').checked = !!o.h; if (!section.active()) toggleSection(true); section.set(o || {}); $('secHv').value = Math.round(section.S.hv * 1000); $('secVv').value = Math.round(section.S.vv * 1000); secText(); syncUI(); } return { ...section.S, hW: section.hWorld(), vP: section.vPos(), text: section.hText() }; },
  xray: (on) => setXray(on !== false), isolate: (on) => setIsolate(on !== false), shadows: (on) => setShadows(on !== false), lowQuality: (on) => setLowQ(on !== false),
  sim: { seek: (s) => { TL.t = Math.max(0, Math.min(TL.T, s)); applyTime(true); return TL.t; }, play: () => { TL.playing = true; applyTime(); }, pause: () => { TL.playing = false; applyTime(); },
    state: () => { const R = sim.last(); return R ? { t: TL.t, T: TL.T, playing: TL.playing, on: TL.on, nodes: R.n, src: R.src, summary: sim.summary(TL.t, lastPlan && lastPlan.attack), evac: simEv ? { ok: simEv.nOk, cut: simEv.nCut, smoke: simEv.nSmoke } : null } : null; },
    ttu: () => { const R = sim.last(); return R ? Array.from(R.ttu) : null; }, params: () => ({ ...SIMP }) },
  crew: () => ({ info: crewInfo, teams: crewState.map((z) => ({ id: z.id, name: z.name, state: z.state, rel: z.rel, bar: z.bar, lim: z.lim, fid: z.fid, at: z.at, p: z.p ? [z.p.x, -z.p.z, z.p.y] : null })), hose: crewState.hose || null }),
  follow: (on) => { if (on === false) follow.stop(); else if (!followOn) startFollow(); return follow.state(); }, followState: () => follow.state(),
  minimap: () => { const cv = $('mini'); return { on: miniOn, rect: cv.getBoundingClientRect().toJSON(), toPx: (x, y) => minimap.toPx(cv, x, y) }; }, miniPx: (x, y) => minimap.toPx($('mini'), x, y),
  target: () => [controls.target.x, -controls.target.z, controls.target.y], tool: (k) => tool(k), tools: () => ({ mode: tools.mode(), measures: tools.measures(), area: tools.area() }),
  floorPx: (x, y) => { const v = V(x, y, G[selId].g.position.y + SLAB).project(camera), r = canvas.getBoundingClientRect(); return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height]; },
  autoQuality: (on) => { perf.auto = on !== false; if (!perf.auto) { renderer.setPixelRatio(lowQ ? Math.min(1, PR0) : PR0); resize(); } return perf.auto; },
  snapshot: () => tools.snapshot(), briefingHtml: () => tools.briefingHtml(), bench, perf: () => ({ fps: perf.fps, pr: renderer.getPixelRatio(), calls: renderer.info.render.calls, tris: renderer.info.render.triangles, gl: GL }),
  stats: () => ({ shafts: vert.list(), walls: Object.fromEntries(Object.entries(SHELL).map(([k, s]) => [k, { segs: s.nSeg, leaves: s.leafDoors.length, ent: s.ent.length, glass: !!s.glass }])), vols: Object.fromEntries(floors.map((f) => [f.id, [...(G[f.id].volCats || [])]])) }),
  // v3.1: 테마·글자·인쇄·배치 점검(시험용)
  theme: (t) => (t ? setTheme(t) : document.documentElement.dataset.theme), fontScale: (z) => (z ? setFont(z) : fz),
  printSheet: () => { buildPrint(); return $('printSheet').innerHTML; },
  ui: () => ({ rects: uiRects(), lay: { ...LAY }, dock: { two: $('dock').classList.contains('two'), compact: $('dock').classList.contains('compact') }, layMore: $('layMore').hidden ? 0 : $('layers').querySelectorAll('.lay.ovf').length, mini: !$('mini').classList.contains('nofit') && miniOn }),
  tidy: () => { vert.sync(); renderer.render(scene, camera); labelR.render(scene, camera); return tidy(); },
  booted: () => booted,
  invalidate, _three: { scene, camera, renderer, G, vert } };
