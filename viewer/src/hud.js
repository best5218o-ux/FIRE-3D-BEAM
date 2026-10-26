// 화면 장식 — 지시선 말풍선(callout)·라벨 겹침 회피 · 핀 아이콘(Sprite) · 화염 텍스처
// main.js 에서 사용. 라벨은 모두 CSS2DRenderer 가 #labels 안에 그리므로, 여기서는 화면 좌표(getBoundingClientRect)만 보고 정리한다.
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { svg, drawIcon } from './icons.js';

// ── 지시선 말풍선: 기준점(바닥) 위에 세로 지시선 + 상자. 상자 높이(지시선 길이)·좌우 비킴은 겹침 회피가 정함
const COICON = { ent: 'entrance', stair: 'stair', ev: 'ev', panel: 'panel', hose: 'hose', gas: 'gas', exit: 'exit' };
export function callout(text, kind, pos, parent, pri = 1, onClick) {
  const el = document.createElement('div'); el.className = 'co t-' + kind; el.dataset.pri = pri;
  el.innerHTML = `<div class="bx" role="button"></div><div class="lg"></div><div class="pt"></div>`;
  el.firstChild.innerHTML = svg(COICON[kind.split(' ')[0]] || 'pin') + `<span></span>`; el.firstChild.lastChild.textContent = text;
  el.firstChild.setAttribute('aria-label', text);
  if (onClick) el.firstChild.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
  const o = new CSS2DObject(el); o.center.set(0.5, 1); o.position.copy(pos); parent.add(o);
  o.userData.co = { pri, kind }; return o;
}

// 라벨 정리(v3.1) — 고정 UI(범례·도구·층 열·시간축·미니맵 등)는 장애물. 순서:
//  ① 층 꼬리표(가리면 옆으로, 그래도 가리면 숨김) ② 꼭 보일 것(화점·대원·잰 값·공격계단 등: 겹치면 위아래로 비킴)
//  ③ 인명검색 번호: 화면에서 겹치면 묶어 가장 높은 순위 1개 + '+N' 배지(마우스 올림·누름 = 묶음 설명)
//  ④ 보조 고정 라벨(귀환 한계점·피난계단·문? 등: 겹치면 숨김) ⑤ 말풍선(우선순위 → 지시선 4단 × 좌우 비킴) ⑥ 작은 라벨(겹치면 숨김)
const MUST = /\b(fire|crew|meas|trn|atk)\b/;
const AUX = /\b(evac|scba|wseg|aer|pump|fdc|north|tick)\b/;
const LEGS = [16, 36, 58, 82];
const NUDGE = [0, -20, 20, -40, 40, -60];
const R = (el, pad = 0) => { const r = el.getBoundingClientRect(); return [r.left - pad, r.top - pad, r.right + pad, r.bottom + pad]; };
const X = (a, b) => !(a[2] <= b[0] || a[0] >= b[2] || a[3] <= b[1] || a[1] >= b[3]);
export function declutter(root, ui = [], opt = {}) {
  const obs = ui.map((q) => [q.l - 3, q.t - 3, q.r + 3, q.b + 3]);
  const placed = [];
  const hitAny = (r) => obs.some((q) => X(r, q)) || placed.some((q) => X(r, q));
  const stage = root.getBoundingClientRect();
  const off = (r) => r[2] < stage.left || r[0] > stage.right || r[3] < stage.top || r[1] > stage.bottom;   // 화면 밖(잘림)
  const kids = [...root.children].filter((el) => el.style.display !== 'none');
  const stat = { cluster: 0, culled: 0 };
  const hide = (el) => { el.classList.add('culled'); stat.culled++; };
  // ① 층 꼬리표
  for (const el of kids) {
    if (!el.classList.contains('ftag')) continue;
    el.classList.remove('culled'); el.style.marginLeft = '0px';
    let r = R(el); if (r[2] - r[0] < 1 || off(r)) continue;
    let dx = 0; for (const q of obs) if (X(r, q)) dx = Math.max(dx, q[2] - r[0] + 4);
    if (dx) { el.style.marginLeft = dx + 'px'; r = R(el); }
    if (hitAny(r)) hide(el); else placed.push(r);
  }
  // 분류
  const must = [], srch = [], aux = [], cos = [], small = [];
  for (const el of kids) {
    const c = el.className;
    if (el.classList.contains('ftag')) continue;
    el.classList.remove('culled'); el.style.marginTop = '';
    if (el.classList.contains('co')) { cos.push(el); continue; }
    if (c.includes('srch')) { if (!c.includes('fr0') && opt.simOn) continue; srch.push(el); continue; }
    if (MUST.test(c)) must.push(el); else if (AUX.test(c)) aux.push(el); else small.push(el);
  }
  // ② 꼭 보일 것: 고정 UI 에 가리면 숨김, 서로 겹치면 위아래로 비킴
  for (const el of must) {
    const r0 = R(el, 1); if (r0[2] - r0[0] < 2 || off(r0)) continue;
    if (obs.some((q) => X(r0, q))) { hide(el); continue; }
    let ok = false;
    for (const dy of NUDGE) { const r = [r0[0], r0[1] + dy, r0[2], r0[3] + dy]; if (!hitAny(r)) { if (dy) el.style.marginTop = dy + 'px'; placed.push(r); ok = true; break; } }
    if (!ok) placed.push(r0);
  }
  // ③ 인명검색 번호 묶기
  for (const el of srch) { if (!el.dataset.rank) el.dataset.rank = el.textContent; delete el.dataset.n; delete el.dataset.ranks; el.removeAttribute('aria-label'); }
  srch.sort((a, b) => Number(a.dataset.rank) - Number(b.dataset.rank));
  const leaders = [];
  for (const el of srch) {
    const r = R(el, 2); if (r[2] - r[0] < 2 || off(r)) continue;
    if (obs.some((q) => X(r, q))) { hide(el); continue; }
    const L = leaders.find((z) => X(r, z.r) || X(r, z.rb));
    if (L) { L.m.push(el.dataset.rank); hide(el); continue; }
    let rr = r, dy0 = 0;
    if (placed.some((q) => X(r, q))) { for (const dy of NUDGE.slice(1)) { const t = [r[0], r[1] + dy, r[2], r[3] + dy]; if (!hitAny(t)) { rr = t; dy0 = dy; break; } } }
    if (rr === r && placed.some((q) => X(r, q))) { hide(el); continue; }
    if (dy0) el.style.marginTop = dy0 + 'px';
    leaders.push({ el, r: rr, rb: [rr[0], rr[1] - 10, rr[2] + 16, rr[3]], m: [] });
  }
  for (const z of leaders) {
    if (z.m.length) { z.el.dataset.n = z.m.length; z.el.dataset.ranks = [z.el.dataset.rank, ...z.m].join(','); stat.cluster++; placed.push(z.rb); }
    else placed.push(z.r);
    z.el.setAttribute('aria-label', `인명검색 ${z.el.dataset.rank}순위${z.m.length ? ` 외 ${z.m.length}곳(${z.m.join('·')}순위)` : ''}`);
  }
  // ④ 보조 고정 라벨
  for (const el of aux) { const r = R(el, 1); if (r[2] - r[0] < 2 || off(r)) continue; if (hitAny(r)) hide(el); else placed.push(r); }
  // ⑤ 말풍선: 우선순위 → 화면 아래쪽(가까운 쪽) 먼저 · 지시선 4단 × 좌우 비킴(상자만 옆으로 — 지시선은 상자 모서리에 닿음)
  const meta = cos.map((el) => {
    const bx = el.firstChild; bx.style.transform = '';
    const r = el.getBoundingClientRect();
    return { el, bx, x: (r.left + r.right) / 2, y: r.bottom, w: bx.offsetWidth + 6, h: bx.offsetHeight + 4, pri: Number(el.dataset.pri || 1) };
  }).filter((m) => m.w > 6 && !(m.y < stage.top || m.x < stage.left - m.w || m.x > stage.right + m.w || m.y - 100 > stage.bottom));
  meta.sort((a, b) => a.pri - b.pri || b.y - a.y);
  for (const m of meta) {
    let pos = null;
    for (const dxk of [0, -0.42, 0.42]) {
      for (let i = 0; i < LEGS.length && !pos; i++) {
        const dx = dxk * m.w, top = m.y - 7 - LEGS[i] - m.h, r = [m.x - m.w / 2 + dx, top, m.x + m.w / 2 + dx, top + m.h];
        if (!hitAny(r) && !hitAny([m.x - 4, m.y - 8, m.x + 4, m.y])) pos = { i, dx, r };
      }
      if (pos) break;
    }
    if (!pos) { hide(m.el); continue; }
    m.el.style.setProperty('--leg', LEGS[pos.i] + 'px'); m.bx.style.transform = pos.dx ? `translateX(${Math.round(pos.dx)}px)` : '';
    placed.push(pos.r, [m.x - 4, m.y - 8, m.x + 4, m.y]);
  }
  // ⑥ 작은 라벨
  for (const el of small) { const r = R(el); if (r[2] - r[0] < 1 || off(r)) continue; if (hitAny(r)) hide(el); else placed.push(r); }
  return stat;
}

// ── 핀 아이콘(지도 핀 + 기호) — 깊이 무시(항상 보임). 기호는 icons.js 와 같은 경로(화면 전체 아이콘 통일)
const PIN = { entrance: ['#22d3ee', 'entrance'], exit: ['#22c55e', 'exit'], hose: ['#ef4444', 'hose'], panel: ['#a78bfa', 'panel'], gas: ['#f59e0b', 'gas'], stair: ['#4ade80', 'stair'], ev: ['#60a5fa', 'ev'] };
const texCache = {};
export function pinTexture(kind) {
  if (texCache[kind]) return texCache[kind];
  const [col, ic] = PIN[kind] || ['#94a3b8', 'pin'];
  const cv = document.createElement('canvas'); cv.width = cv.height = 128; const c = cv.getContext('2d');
  c.beginPath(); c.moveTo(64, 124); c.lineTo(40, 88); c.arc(64, 60, 36, Math.PI * 0.78, Math.PI * 0.22); c.closePath();
  c.fillStyle = col; c.fill(); c.lineWidth = 6; c.strokeStyle = '#0b1220'; c.stroke();
  drawIcon(c, ic, 64 - 21, 60 - 21, 42, '#0b1220', 2.6);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return (texCache[kind] = t);
}
// size = 화면 픽셀(크기 고정, main.js resize 가 화면 높이·시야각으로 환산 — 멀리서 봐도 진입구 표식이 건물을 덮지 않음)
export function pin(kind, pos, size = 24) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: pinTexture(kind), depthTest: false, transparent: true, sizeAttenuation: false }));
  s.scale.set(0.03, 0.03, 1); s.center.set(0.5, 0.02); s.position.copy(pos); s.renderOrder = 10; s.userData.pin = kind; s.userData.size = size; return s;
}

// ── 화염(방사 그라데이션 불꽃)
let flameTex = null;
export function flameTexture() {
  if (flameTex) return flameTex;
  const cv = document.createElement('canvas'); cv.width = cv.height = 128; const c = cv.getContext('2d');
  const g = c.createRadialGradient(64, 84, 6, 64, 70, 60);
  g.addColorStop(0, '#fff7cc'); g.addColorStop(0.35, '#fbbf24'); g.addColorStop(0.7, '#ef4444'); g.addColorStop(1, 'rgba(239,68,68,0)');
  c.fillStyle = g; c.beginPath(); c.moveTo(64, 6); c.bezierCurveTo(100, 40, 112, 70, 96, 100); c.bezierCurveTo(86, 120, 42, 120, 32, 100); c.bezierCurveTo(16, 70, 30, 44, 64, 6); c.fill();
  flameTex = new THREE.CanvasTexture(cv); flameTex.colorSpace = THREE.SRGBColorSpace; return flameTex;
}
