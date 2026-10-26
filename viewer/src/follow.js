import { svg } from './icons.js';
// 진입로 따라가기(1인칭) — 산출안 진입 경로(진입구 → 공격계단 → 화재층 → 화점)를 눈높이 1.6 m 의 부드러운 곡선(Catmull-Rom)으로 카메라가 따라감
// 화면 위 표시: 남은 거리·남은 시간(보행·계단 기준값) · 현재 층 · 배속.  Esc 로 끝냄(원래 조작으로 복귀).
import * as THREE from 'three';

export function initFollow(ctx) {
  const { camera, controls, $, esc, ACT, FL, onState } = ctx;
  let F = null;
  function start(seq, label = '진입로 따라가기') {
    if (!seq || seq.length < 2) return false;
    const pts = [], meta = [];
    for (const s of seq) { const p = s.p.clone(); p.y += 1.6; if (!pts.length || pts[pts.length - 1].distanceTo(p) > 0.05) { pts.push(p); meta.push(s); } }
    if (pts.length < 2) return false;
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
    const cum = [0]; for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    const Lc = cum[cum.length - 1];
    F = { curve, pts, meta, cum, Lc, t: 0, T: meta[meta.length - 1].t, speed: 6, paused: false, last: performance.now(), label,
      save: { near: camera.near, pos: camera.position.clone(), tgt: controls.target.clone() }, look: pts[1].clone() };
    controls.enabled = false; camera.near = 0.08; camera.updateProjectionMatrix();
    $('follow').hidden = false; onState(true); hud(); return true;
  }
  // 시각 → 꺾은선 위 거리(구간 시간 비례) → 곡선 매개변수
  function distAt(t) {
    const m = F.meta; if (t <= m[0].t) return 0;
    for (let i = 1; i < m.length; i++) if (t <= m[i].t) { const w = (t - m[i - 1].t) / Math.max(1e-6, m[i].t - m[i - 1].t); return F.cum[i - 1] + (F.cum[i] - F.cum[i - 1]) * w; }
    return F.Lc;
  }
  function step(now) {
    if (!F) return false;
    const dt = Math.min(0.1, (now - F.last) / 1000); F.last = now;
    if (!F.paused) F.t = Math.min(F.T, F.t + dt * F.speed);
    const u = F.Lc ? distAt(F.t) / F.Lc : 1, ua = F.Lc ? Math.min(1, (distAt(F.t) + 4.5) / F.Lc) : 1;
    const p = F.curve.getPointAt(Math.min(1, u)), a = F.curve.getPointAt(ua);
    if (Math.abs(a.y - p.y) > 0.5) a.y = p.y + Math.sign(a.y - p.y) * 0.5;      // 계단 구간: 너무 위·아래를 보지 않게
    if (a.distanceTo(p) < 0.3) a.copy(F.look);
    F.look.lerp(a, 0.12);
    camera.position.copy(p); camera.lookAt(F.look);
    hud();
    return true;
  }
  const mmss = (s) => `${Math.floor(s / 60)}분 ${String(Math.round(s % 60)).padStart(2, '0')}초`;
  function hud() {
    const m = F.meta, d = distAt(F.t);
    let k = m.findIndex((z) => z.t >= F.t); if (k < 0) k = m.length - 1;
    // 남은 수평 거리(평면) · 남은 층
    let rem = 0; for (let i = Math.max(1, k); i < m.length; i++) if (m[i].fid === m[i - 1].fid) rem += Math.hypot(m[i].p.x - m[i - 1].p.x, m[i].p.z - m[i - 1].p.z);
    if (k > 0 && m[k].fid === m[k - 1].fid) { const w = (F.t - m[k - 1].t) / Math.max(1e-6, m[k].t - m[k - 1].t); rem -= Math.hypot(m[k].p.x - m[k - 1].p.x, m[k].p.z - m[k - 1].p.z) * Math.min(1, Math.max(0, w)); }
    const fl = FL[m[k].fid] ? FL[m[k].fid].name : '';
    $('fwTxt').innerHTML = `<b>${esc(F.label)}</b> · ${esc(fl)} · ${esc(m[k].note || '')}<br>남은 거리 <b>${Math.max(0, rem).toFixed(0)}m</b> · 남은 시간 <b>${mmss(Math.max(0, F.T - F.t))}</b> <span class="k">(보행 ${ACT.walk_mps}m/s · 계단 ${ACT.stair_s_per_floor}초/층 기준, ×${F.speed} 재생)</span>`;
    $('fwBar').style.width = `${(100 * F.t) / Math.max(1, F.T)}%`;
    $('fwPause').innerHTML = svg(F.paused ? 'play' : 'pause'); $('fwPause').setAttribute('aria-label', F.paused ? '따라가기 계속' : '따라가기 멈춤');
    F.remain = Math.max(0, rem); F.d = d;
  }
  function stop() {
    if (!F) return;
    const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
    controls.target.copy(camera.position.clone().add(dir.multiplyScalar(6)));
    camera.position.add(new THREE.Vector3(0, 10, 0)).addScaledVector(dir.normalize(), -14);
    camera.near = F.save.near; camera.updateProjectionMatrix(); controls.enabled = true;
    F = null; $('follow').hidden = true; onState(false);
  }
  $('fwPause').onclick = () => { if (F) { F.paused = !F.paused; hud(); } };
  $('fwStop').onclick = () => stop();
  $('follow').addEventListener('click', (e) => { const b = e.target.closest('[data-sp]'); if (b && F) { F.speed = Number(b.dataset.sp); document.querySelectorAll('#follow [data-sp]').forEach((x) => x.classList.toggle('on', x === b)); hud(); } });
  return { start, stop, step, active: () => !!F, state: () => (F ? { t: F.t, T: F.T, remain: F.remain, d: F.d, Lc: F.Lc, speed: F.speed, cam: camera.position.toArray() } : null), setSpeed: (s) => { if (F) F.speed = s; } };
}
