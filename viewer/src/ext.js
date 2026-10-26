// 기능 12(수신기 경보 → 경계구역 화점 후보) · 18(건축물대장 3자 교차검증) · 24(외부 부서 위치) — main.js 에서 initExt(ctx) 로 연결
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export function initExt(ctx) {
  const { D, scene, G, floors, FL, V, SLAB, H, esc, $, boxGeom } = ctx;
  const Z = D.zones, S = D.site, DJ = D.daejang;
  const ground = floors.find((f) => f.level === 0) || floors[0];
  const ko = (fid) => (FL[fid] ? FL[fid].name : fid);
  const lab = (text, cls, x, y, z, parent) => {
    const el = document.createElement('div'); el.className = 'lb ' + cls; el.innerHTML = text;
    const o = new CSS2DObject(el); o.position.copy(V(x, y, z)); parent.add(o); return o;
  };
  const shapeOf = (ring) => new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y)));
  const lineLoop = (ring, z, color, op = 0.9) => {
    const g = new THREE.BufferGeometry().setFromPoints([...ring, ring[0]].map(([x, y]) => V(x, y, z)));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: op }));
  };
  const tubeLine = (pts3, color, r, parent, dashed) => {
    if (dashed) {
      const g = new THREE.BufferGeometry().setFromPoints(pts3);
      const l = new THREE.Line(g, new THREE.LineDashedMaterial({ color, dashSize: 2, gapSize: 1.2 })); l.computeLineDistances(); parent.add(l); return l;
    }
    const path = new THREE.CurvePath(); for (let k = 0; k < pts3.length - 1; k++) path.add(new THREE.LineCurve3(pts3[k], pts3[k + 1]));
    const m = new THREE.Mesh(new THREE.TubeGeometry(path, Math.max(4, pts3.length * 3), r, 6, false), new THREE.MeshBasicMaterial({ color }));
    parent.add(m); return m;
  };

  // ───────────────────────── 기능 12: 경계구역
  const ZG = {};
  const zoneList = [];
  const ZC = [0x58a6ff, 0x3fb950, 0xd29922, 0xa371f7, 0x39c5cf, 0xf778ba, 0x8ddb8c, 0xffa657];
  if (Z) {
    for (const f of floors) {
      const rec = Z.floors[f.id]; if (!rec || !G[f.id]) continue;
      const g = new THREE.Group(); g.visible = false; G[f.id].g.add(g); ZG[f.id] = g;
      rec.zones.forEach((z, i) => {
        const col = z.ok ? ZC[i % ZC.length] : 0xff4d4f;
        for (const ring of z.poly) {
          const sg = new THREE.ShapeGeometry(shapeOf(ring)); sg.rotateX(-Math.PI / 2); sg.translate(0, SLAB + 0.12, 0);
          const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide }));
          m.renderOrder = 4; g.add(m);
          g.add(lineLoop(ring, SLAB + 0.16, col, 0.95));
        }
        lab(`회로 ${z.no}${z.est ? '<span style="opacity:.7"> 추정</span>' : ''}`, 'zone', z.rep[0], z.rep[1], SLAB + 0.6, g);
        zoneList.push({ ...z, fid: f.id });
      });
    }
    // 경보 입력 UI
    const opts = floors.filter((f) => Z.floors[f.id] && Z.floors[f.id].zones.length).reverse();
    $('alFl').innerHTML = opts.map((f) => `<option value="${f.id}">${esc(f.name)} (${Z.floors[f.id].zones.length}회로)</option>`).join('') + '<option value="*">건물 일련번호</option>';
    $('alSec').style.display = '';
    const upd = () => {
      const v = $('alFl').value, rec = Z.floors[v];
      $('alNote').innerHTML = v === '*' ? `일련번호 1 ~ ${Z.total} (층 낮은 순)` : `${esc(ko(v))}: 정보카드 자동화재탐지기(회) <b>${rec.n_card ?? '-'}</b> · NFTC 최소 <b>${rec.n_req ?? '-'}</b> · 구역 출처 <b>${esc(rec.src)}</b>`;
    };
    $('alFl').onchange = upd; upd();
    $('alRnd').onclick = () => { const z = zoneList[Math.floor(Math.random() * zoneList.length)]; $('alFl').value = z.fid; $('alNo').value = z.no; upd(); alarm(z); };
    $('alGo').onclick = () => {
      const v = $('alFl').value, no = Number($('alNo').value);
      const z = v === '*' ? zoneList.find((q) => q.gno === no) : zoneList.find((q) => q.fid === v && q.no === no);
      if (!z) { $('alNote').innerHTML = '<span style="color:#ffa198">해당 회로(경계구역)가 없습니다</span>'; return; }
      alarm(z);
    };
    // 오른쪽 패널: 층별 회로 수 검증
    let h = `<table><tr><td class="k">층</td><td class="k">카드 회로</td><td class="k">NFTC 최소</td><td class="k">구역</td></tr>`;
    for (const f of [...floors].reverse()) {
      const r = Z.floors[f.id]; if (!r) continue;
      const bad = r.n_card != null && r.n_req != null && r.n_card < r.n_req;
      h += `<tr class="zrow" data-zf="${f.id}" style="cursor:pointer${bad ? ';color:#ffa198' : ''}"><td>${esc(f.name)}</td><td>${r.n_card ?? '-'}</td><td>${r.n_req ?? '-'}${r.open_hall && r.n_req_open < r.n_req ? `<span class="k">(대공간 ${r.n_req_open})</span>` : ''}</td><td>${r.zones.length}${r.src === '추정' ? ' <span class="k">추정</span>' : ''}</td></tr>`;
    }
    h += '</table>';
    h += `<div class="note">${esc(Z.basis)}<br>구역 경계는 벽선을 따라 회로 수만큼 나눈 <b>추정</b>입니다(경계구역 일람도 JSON 이 있으면 그대로 사용). 번호: ${esc(Z.numbering)}</div>`;
    const vs = Z.vertical || [];
    if (vs.length) h += `<div class="note">수직 경계구역(별도): ${vs.map((v) => `${v.kind}${v.idx} ${v.n}구역`).join(', ')}</div>`;
    $('zonePanel').innerHTML = h; $('zoneSec').style.display = '';
    $('zonePanel').addEventListener('click', (e) => { const r = e.target.closest('[data-zf]'); if (r) { $('tZone').checked = true; showZones(); ctx.selectFloor(r.dataset.zf, true); } });
  }
  function showZones() {
    const on = $('tZone') && $('tZone').checked;
    for (const [fid, g] of Object.entries(ZG)) g.visible = on && (fid === ctx.getSel() || ctx.getViewMode() === 'exp');
  }
  function alarm(z) {
    $('tZone').checked = true;
    ctx.setFire(z.fid, z.rep[0], z.rep[1], { zone: z });
    showZones();
  }
  function zoneAt(fid, x, y) {
    const pip = (ring) => { let c = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const [xi, yi] = ring[i], [xj, yj] = ring[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
    return zoneList.find((z) => z.fid === fid && z.poly.some(pip));
  }

  // ───────────────────────── 기능 24: 외부 부서 위치
  const siteG = new THREE.Group(); siteG.visible = false; scene.add(siteG);
  const WG = {};
  const vs = S ? H / (S.floor_h || 3.3) : 1;           // 실제 높이(m) → 화면 높이(층고 H 로 그린 층과 맞춤)
  if (S) {
    const pts = [[ctx.bx0, ctx.by0], [ctx.bx1, ctx.by1]];
    for (const r of S.roads) pts.push(...r.line);
    for (const h of S.hydrants) pts.push([h.x, h.y]);
    let gx0 = Math.min(...pts.map((p) => p[0])), gx1 = Math.max(...pts.map((p) => p[0])), gy0 = Math.min(...pts.map((p) => p[1])), gy1 = Math.max(...pts.map((p) => p[1]));
    // 지면: 건물 주변 ±150m (멀리 있는 소화전까지 포함)
    const cx = (ctx.bx0 + ctx.bx1) / 2, cy = (ctx.by0 + ctx.by1) / 2;
    gx0 = Math.max(gx0, cx - 600) - 20; gx1 = Math.min(gx1, cx + 600) + 20; gy0 = Math.max(gy0, cy - 600) - 20; gy1 = Math.min(gy1, cy + 600) + 20;
    const gp = new THREE.Mesh(new THREE.PlaneGeometry(gx1 - gx0, gy1 - gy0), new THREE.MeshBasicMaterial({ color: 0x18261c, transparent: true, opacity: 0.8, depthWrite: false }));
    gp.rotation.x = -Math.PI / 2; gp.position.copy(V((gx0 + gx1) / 2, (gy0 + gy1) / 2, -0.12)); gp.renderOrder = 0; siteG.add(gp);
    // 도로
    const cfg = S.cfg || {};
    for (const r of S.roads) {
      for (let i = 0; i < r.line.length - 1; i++) {
        const [a, b] = [r.line[i], r.line[i + 1]], L = Math.hypot(b[0] - a[0], b[1] - a[1]), ux = (b[0] - a[0]) / L, uy = (b[1] - a[1]) / L;
        const ok = r.width >= (cfg.engine_min_road_width_m || 4), aer = r.width >= (cfg.aerial_min_road_width_m || 6);
        const m = new THREE.Mesh(new THREE.PlaneGeometry(L, r.width), new THREE.MeshBasicMaterial({ color: aer ? 0x4a515c : ok ? 0x3d434c : 0x3a2f2f, transparent: true, opacity: 0.95, depthWrite: false }));
        m.rotation.x = -Math.PI / 2; m.rotation.z = Math.atan2(uy, ux); m.position.copy(V((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, -0.06)); m.renderOrder = 1; siteG.add(m);
        if (ok) tubeLine([V(a[0], a[1], 0), V(b[0], b[1], 0)], 0xd29922, 0, siteG, true);
        // 이름표: 건물에서 가장 가까운 점
        const t = Math.max(0, Math.min(L, (cx - a[0]) * ux + (cy - a[1]) * uy));
        lab(`${esc(r.name)} <span style="opacity:.75">폭 ${r.width}m${aer ? '' : ok ? ' · 사다리 전개 불가' : ' · 소방차 진입 불가'}</span>`, 'road', a[0] + ux * t, a[1] + uy * t, 1.5, siteG);
      }
    }
    for (const o of S.obstacles) {
      const xs = o.poly.map((p) => p[0]), ys = o.poly.map((p) => p[1]);
      const m = new THREE.Mesh(boxGeom(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), 0, 7 * vs / 1.3), new THREE.MeshBasicMaterial({ color: 0xf0883e, transparent: true, opacity: 0.25, depthWrite: false }));
      siteG.add(m); lab(`${esc(o.kind)} · 사다리 전개 불가`, 'obst', (Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, 7 * vs / 1.3 + 1, siteG);
    }
    // 외벽면별 소방차 접근
    for (const f of S.facades) {
      const col = f.engine ? 0x3fb950 : 0xff4d4f;
      const o = 0.8, A = [f.a[0] + f.n[0] * o, f.a[1] + f.n[1] * o], B = [f.b[0] + f.n[0] * o, f.b[1] + f.n[1] * o];
      tubeLine([V(A[0], A[1], 0.25), V(B[0], B[1], 0.25)], col, 0.35, siteG);
      const mx = (A[0] + B[0]) / 2 + f.n[0] * 3, my = (A[1] + B[1]) / 2 + f.n[1] * 3;
      lab(`${esc(f.name)} ${f.engine ? `소방차 접근 ✓ <span style="opacity:.75">(이격 ${f.setback}m)</span>` : '접근 불가'}`, f.engine ? 'facok' : 'facno', mx, my, 0.6, siteG);
    }
    // 고가사다리차: 차체 + 전개 반경 원 + 도달 범위(돔) + 최고 도달 진입창까지 사다리
    const Lh = cfg.aerial_ladder_len_m || 53, R = cfg.aerial_max_outreach_m || 25, h0 = cfg.aerial_turntable_h_m || 3;
    const roadDir = (name) => { const r = S.roads.find((q) => q.name === name); if (!r) return [1, 0]; const [a, b] = r.line; const L = Math.hypot(b[0] - a[0], b[1] - a[1]); return [(b[0] - a[0]) / L, (b[1] - a[1]) / L]; };
    const truck = (x, y, dir, color, len) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(len, 3.0, 2.5), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 }));
      m.position.copy(V(x, y, 1.5)); m.rotation.y = Math.atan2(dir[1], dir[0]); siteG.add(m); return m;
    };
    for (const a of S.aerial) {
      truck(a.x, a.y, roadDir(a.road), 0xd93b3b, 10);
      const ring = new THREE.Mesh(new THREE.RingGeometry(a.radius - 0.35, a.radius, 96), new THREE.MeshBasicMaterial({ color: 0xffa657, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(V(a.x, a.y, 0.05)); siteG.add(ring);
      const prof = [], topH = Math.min(h0 + Lh, (Math.max(...floors.map((f) => f.level)) + 1.6) * (S.floor_h || 3.3));   // 건물 높이 위까지만 표시
      for (let k = 0; k <= 40; k++) { const h = (k / 40) * topH; const r = Math.min(R, Math.sqrt(Math.max(0, Lh * Lh - (h - h0) ** 2))); prof.push(new THREE.Vector2(Math.max(0.01, r), h * vs)); }
      if (topH >= h0 + Lh - 1e-6) prof.push(new THREE.Vector2(0.01, (h0 + Lh) * vs));
      const dome = new THREE.Mesh(new THREE.LatheGeometry(prof, 48), new THREE.MeshBasicMaterial({ color: 0xffa657, transparent: true, opacity: 0.06, depthWrite: false, side: THREE.DoubleSide }));
      dome.position.copy(V(a.x, a.y, 0)); siteG.add(dome);
      const wire = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.LatheGeometry(prof.filter((_, i) => i % 5 === 0), 16), 1), new THREE.LineBasicMaterial({ color: 0xffa657, transparent: true, opacity: 0.12 }));
      wire.position.copy(dome.position); siteG.add(wire);
      const ws = S.windows.filter((w) => w.by === a.id).sort((p, q) => q.h - p.h);
      if (ws.length) {
        const w = ws[0], zw = FL[w.floor] ? ctx.zOf(w.floor) + SLAB + 0.8 : w.h * vs;
        tubeLine([V(a.x, a.y, h0 * vs * 0.5), V(w.x, w.y, zw)], 0xffd33d, 0.18, siteG);
      }
      lab(`🚒 고가사다리차 ${a.id} <span style="opacity:.8">반경 ${a.radius}m · ${a.floors.length ? a.floors[0] + '~' + a.top : '도달 층 없음'}</span>`, 'aer', a.x, a.y, 5, siteG);
    }
    // 소방관 진입창 후보(층 그룹 안 — 층 분해 보기와 함께 움직임)
    for (const w of S.windows) {
      if (!G[w.floor]) continue;
      const g = WG[w.floor] || (WG[w.floor] = Object.assign(new THREE.Group(), { visible: false }));
      if (!g.parent) G[w.floor].g.add(g);
      const fa = S.facades.find((f) => f.name === w.facade); const along = fa ? Math.abs(fa.n[1]) > Math.abs(fa.n[0]) : true;
      const m = new THREE.Mesh(along ? boxGeom(w.x - 0.6, w.y - 0.12, w.x + 0.6, w.y + 0.12, SLAB + 0.8, SLAB + 2.0) : boxGeom(w.x - 0.12, w.y - 0.6, w.x + 0.12, w.y + 0.6, SLAB + 0.8, SLAB + 2.0),
        new THREE.MeshStandardMaterial({ color: 0x39c5cf, emissive: 0x39c5cf, emissiveIntensity: 0.8 }));
      g.add(m);
      w.lab = lab(`진입창 후보 · ${w.by} ${w.dist}m`, 'win', w.x, w.y, SLAB + 2.6, g);
    }
    // 펌프차 → 송수구 · 소화전
    const P = S.pumper;
    if (P) {
      truck(P.x, P.y, roadDir(P.road), 0x1f6feb, 8);
      lab(`🚒 펌프차 <span style="opacity:.8">송수구 ${P.fdc_d}m · 호스 ${P.fdc_hose}본</span>`, 'pump', P.x, P.y, 4.5, siteG);
      tubeLine([V(P.x, P.y, 0.6), V(P.fdc_x, P.fdc_y, 0.6)], 0x58a6ff, 0.15, siteG);
      const h = P.hydrants[0];
      if (h) {
        const hy = S.hydrants.find((q) => q.id === h.id);
        tubeLine([V(P.x, P.y, 0.4), V(hy.x, hy.y, 0.4)], 0xff7b72, 0, siteG, true);
        lab(`급수 ${esc(h.id)}: 도로 ${h.road_d ?? h.straight}m · 호스 ${h.hose}본`, 'hyd', (P.x + hy.x) / 2, (P.y + hy.y) / 2, 2, siteG);
      }
    }
    for (const q of S.fdc) {
      siteG.add(new THREE.Mesh(boxGeom(q.x - 0.4, q.y - 0.4, q.x + 0.4, q.y + 0.1, 0.4, 1.4), new THREE.MeshStandardMaterial({ color: 0xffd33d, emissive: 0xffd33d, emissiveIntensity: 0.7 })));
      lab('송수구', 'fdc', q.x, q.y, 2.4, siteG);
    }
    for (const e of S.entrances || []) lab(`${e.kind === 'main' ? '주출입구' : '비상구'} <span style="opacity:.7">(진압작전도)</span>`, 'fdc', e.x, e.y, 1.2, siteG);   // 작전도 표시 출입구
    // 작전도 이격거리(치수선): 값 + 출처(글자 판독 / 선 길이 보정 / 선 길이 추정) · 확실도 · 글자↔선 길이 불일치 경고
    const DSRC = { ocr: '글자 판독', 'ocr+geom': '글자 판독·선 길이로 보정', geom: '선 길이 추정(글자 못 읽음)' };
    const DCERT = { high: '확실', medium: '보통', low: '낮음', estimated: '추정' };
    const dimTxt = (d) => `${d.approx ? '약 ' : ''}${d.val}m`;
    for (const d of S.dims || []) {
      if (d.val == null || !d.a || !d.b) continue;
      tubeLine([V(d.a[0], d.a[1], 0.35), V(d.b[0], d.b[1], 0.35)], d.geom_mismatch ? 0xffd33d : d.val_src === 'geom' ? 0x8b949e : 0x39c5cf, 0, siteG, true);
      const warn = d.geom_mismatch ? ` <span style="color:#ffd33d">⚠ 선 길이 ${d.geom_m}m</span>` : '';
      lab(`이격 ${dimTxt(d)} <span style="opacity:.75">(${DSRC[d.val_src] || '-'} · ${DCERT[d.certainty] || '-'})</span>${warn}`, d.val_src === 'geom' ? 'road' : 'win',
        (d.a[0] + d.b[0]) / 2, (d.a[1] + d.b[1]) / 2, 1.0, siteG);
    }
    for (const h of S.hydrants) {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.6, 1.6, 12), new THREE.MeshStandardMaterial({ color: 0xff4d4f, emissive: 0xff4d4f, emissiveIntensity: 0.6 }));
      m.position.copy(V(h.x, h.y, 0.8)); siteG.add(m);
      const ring = new THREE.Mesh(new THREE.RingGeometry(2.2, 3.0, 24), new THREE.MeshBasicMaterial({ color: 0xff4d4f, side: THREE.DoubleSide })); ring.rotation.x = -Math.PI / 2; ring.position.copy(V(h.x, h.y, 0.05)); siteG.add(ring);
      const wc = (S.water || []).find((w) => w.id === h.id);
      lab(`소화전 ${esc(h.id)} (${esc(h.type)})${wc && wc.ok === false ? ' <span style="color:#ffd33d">⚠ 카드 ' + Math.round(wc.km * 1000) + 'm / 계산 ' + wc.calc + 'm</span>' : ''}`, 'hyd', h.x, h.y, 3, siteG);
    }
    // 오른쪽 패널
    let t = `<table>`;
    for (const f of S.facades) t += `<tr><td class="k">${esc(f.name)} (${f.len}m)</td><td>${f.engine ? `<span style="color:#7ee787">소방차 접근</span> · ${esc(f.road)} ${f.setback}m` : '<span style="color:#ffa198">접근 불가</span>'}</td></tr>`;
    for (const a of S.aerial) t += `<tr><td class="k">사다리차 ${a.id}</td><td>${esc(a.road)} · ${a.floors.length ? a.floors[0] + '~' + a.top : '-'} · ${a.facades.map(esc).join('/')}</td></tr>`;
    if (P) {
      t += `<tr><td class="k">펌프차 → 송수구</td><td>${P.fdc_d}m · 호스 ${P.fdc_hose}본</td></tr>`;
      for (const h of P.hydrants) t += `<tr><td class="k">펌프차 → 소화전 ${esc(h.id)}</td><td>도로 ${h.road_d ?? '-'}m · 호스 ${h.hose}본</td></tr>`;
    }
    for (const d of S.dims || []) if (d.val != null) t += `<tr><td class="k">이격거리(작전도)</td><td>${dimTxt(d)} · ${DSRC[d.val_src] || '-'} · ${DCERT[d.certainty] || '-'}${d.geom_mismatch ? ` <b style="color:#ffd33d">선 길이 ${d.geom_m}m 불일치</b>` : ''}</td></tr>`;
    for (const w of S.water || []) t += `<tr><td class="k">카드 ⑥ ${esc(w.id)}</td><td>${Math.round(w.km * 1000)}m / 계산 ${w.calc ?? '-'}m ${w.ok === false ? '<b style="color:#ffd33d">불일치</b>' : '✓'}</td></tr>`;
    t += `</table><div style="margin-top:5px">`;
    for (const p of S.per_floor) t += `<span class="badge" style="margin:1px;${p.n ? '' : 'color:#ffa198;border-color:#f85149'}" title="${p.facades.join(',')}">${esc(ko(p.floor))} ${p.n ? '진입창 ' + p.n : '도달 불가'}</span>`;
    t += `</div><div class="note">층고 ${S.floor_h}m (${esc(S.floor_h_src || '')}) · 사다리 ${cfg.aerial_ladder_len_m}m · 전개 반경 ${cfg.aerial_max_outreach_m}m · 전개 도로폭 ${cfg.aerial_min_road_width_m}m · 호스 ${cfg.hose_len_m}m/본 — <b>기관 SOP 값으로 설정 필요</b><br>진입창 후보: 건축법 시행령 제51조 소방관 진입창 기준(2~${cfg.rescue_window_max_floor}층, ${cfg.rescue_window_spacing_m}m 간격)을 따라 사다리가 닿는 외벽면 위치를 표시(실제 창 위치는 도면 미표기 → 현장 확인)<br>외벽면: ${esc(S.reg.facade_src || '')} · 정합 ${esc(S.reg.ref || '')}${S.src === '진압작전도' ? ` · 부지 출처: 진압작전도 인식(축척 ${esc(S.drawing.reg.scale_src)}, 방위 ${S.drawing.reg.north_deg}°)` : ''}</div>`;
    $('sitePanel').innerHTML = t; $('siteSec').style.display = '';
  }
  function showSite() {
    const on = !!S && $('tSite').checked;
    siteG.visible = on;
    for (const [fid, g] of Object.entries(WG)) { g.visible = on; g.traverse((o) => { if (o.isCSS2DObject) o.visible = fid === ctx.getSel() || ctx.getViewMode() === 'exp'; }); }
  }

  // ───────────────────────── 기능 18: 건축물대장 3자 교차검증
  if (DJ) {
    const T = DJ.title || {}, C = DJ.card || {};
    const bad = (a, b) => (a != null && b != null && String(a) !== String(b) ? ' style="color:#ffa198"' : '');
    const d = T.useAprDay ? `${String(T.useAprDay).slice(0, 4)}.${String(T.useAprDay).slice(4, 6)}.${String(T.useAprDay).slice(6)}` : '-';
    let h = `<table><tr><td class="k">항목</td><td class="k">건축물대장</td><td class="k">정보카드</td></tr>`;
    h += `<tr${bad(T.mainPurpsCdNm, C.use)}><td class="k">주용도</td><td>${esc(T.mainPurpsCdNm)}</td><td>${esc(C.use)}</td></tr>`;
    h += `<tr${bad(`지하${T.ugrndFlrCnt}/지상${T.grndFlrCnt}`, C.floors)}><td class="k">층수</td><td>지하${T.ugrndFlrCnt}/지상${T.grndFlrCnt}</td><td>${esc(C.floors)}</td></tr>`;
    const gb = T.totArea && C.gfa && Math.abs(T.totArea - C.gfa) / T.totArea > 0.03;
    h += `<tr${gb ? ' style="color:#ffa198"' : ''}><td class="k">연면적</td><td>${T.totArea ? Number(T.totArea).toLocaleString() : '-'}㎡</td><td>${C.gfa ? Number(C.gfa).toLocaleString() : '-'}㎡</td></tr>`;
    h += `<tr><td class="k">승강기</td><td>${(T.rideUseElvtCnt || 0) + (T.emgenUseElvtCnt || 0)}대</td><td>${C.elev ?? '-'}대 (도면 ${C.elev_drawing ?? '-'})</td></tr>`;
    h += `<tr><td class="k">사용승인일 · 높이</td><td colspan="2" style="text-align:right">${d} · ${T.heit ?? '-'}m</td></tr></table>`;
    h += `<table style="margin-top:6px"><tr><td class="k">층</td><td class="k">대장</td><td class="k">카드</td><td class="k">도면</td></tr>`;
    for (const r of [...DJ.table].reverse()) {
      const col = r.verdict === '일치' ? '' : r.verdict.includes('도면 축척') ? 'color:#ffd33d' : 'color:#ffa198';
      const n = (v) => (v == null ? '<span class="k">없음</span>' : Math.round(v).toLocaleString());
      h += `<tr style="${col}" title="${esc(r.verdict)}"><td>${esc(ko(r.floor))}</td><td>${n(r.daejang)}</td><td>${n(r.card)}</td><td>${n(r.drawing)}</td></tr>`;
      if (r.verdict !== '일치') h += `<tr style="${col}"><td colspan="4" style="text-align:left;font-size:11px">↳ ${esc(r.verdict)}</td></tr>`;
    }
    h += `</table><div class="note">출처: ${esc(DJ.source)} (${esc(DJ.file || '')}) · 층 면적 판정은 대장·카드·도면 3자 다수결(도면은 축척 오차 감안)</div>`;
    $('djPanel').innerHTML = h; $('djSec').style.display = '';
  }

  // 화점 결과에 덧붙일 내용(경보 구역 강조 + 외부 구조 가능 여부)
  function fireExtra(fire, fireG, fz, env) {
    let top = '', bottom = '';
    const z = fire.zone || (Z ? zoneAt(fire.fid, fire.x, fire.y) : null);
    if (z) {
      for (const ring of z.poly) {
        const sg = new THREE.ShapeGeometry(shapeOf(ring)); sg.rotateX(-Math.PI / 2); sg.translate(0, fz + 0.2, 0);
        const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: fire.zone ? 0.35 : 0.15, depthWrite: false, side: THREE.DoubleSide }));
        m.renderOrder = 5; fireG.add(m);
        const pts = [...ring, ring[0]].map(([x, y]) => V(x, y, fz + 0.3));
        const path = new THREE.CurvePath(); for (let k = 0; k < pts.length - 1; k++) path.add(new THREE.LineCurve3(pts[k], pts[k + 1]));
        fireG.add(new THREE.Mesh(new THREE.TubeGeometry(path, pts.length * 4, 0.14, 5, false), new THREE.MeshBasicMaterial({ color: 0xff6b35 })));
      }
      // 구역 안 가장 먼 곳(대표점 기준) → 호스 여유
      let far = 0; const ds = [];
      for (const ring of z.poly) {                       // 구역 내부 2m 격자(경계에서 0.8m 안쪽) → 대표점에서의 보행 거리 95백분위
        const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
        const pip = (x, y) => { let c = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const [xi, yi] = ring[i], [xj, yj] = ring[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
        for (let x = Math.min(...xs) + 1; x < Math.max(...xs); x += 2) for (let y = Math.min(...ys) + 1; y < Math.max(...ys); y += 2) {
          if (!pip(x, y) || !pip(x + 0.8, y) || !pip(x - 0.8, y) || !pip(x, y + 0.8) || !pip(x, y - 0.8)) continue;
          ds.push(Math.hypot(x - z.rep[0], y - z.rep[1]));
        }
      }
      ds.sort((a, b) => a - b); if (ds.length) far = ds[Math.floor(ds.length * 0.95) - (ds.length > 1 ? 1 : 0)] || ds[ds.length - 1];
      if (fire.zone) {
        top += `<div style="margin:3px 0 5px;padding:4px 6px;background:#b6232455;border-radius:5px">🚨 수신기 경보 <b>${esc(ko(z.fid))} 회로 ${z.no}</b> (일련 ${z.gno}) → 경계구역 ${z.est ? '<b>추정</b>' : ''} 전체가 화점 후보<br>` +
          `<span class="k">구역 ${z.area}㎡ · 최장변 ${z.side}m · 대표점(구역 중앙)을 화점으로 경로 계산 · 구역 끝까지 직선 ${far.toFixed(0)}m</span></div>`;
        bottom += `<div>경보 구역 전체 수색: 대표점에서 가장 먼 곳 직선 <b>${far.toFixed(0)}m</b> → 호스 <b>+${Math.ceil(far / 15)}본</b> 여유 권장</div>`;
      } else top += `<div class="note">이 위치는 ${esc(ko(z.fid))} 경계구역 <b>회로 ${z.no}</b>${z.est ? '(추정)' : ''} — 수신기 경보 번호와 대조</div>`;
    }
    if (S && FL[fire.fid] && FL[fire.fid].level >= 1) {
      const pf = S.per_floor.find((p) => p.floor === fire.fid);
      const ws = S.windows.filter((w) => w.floor === fire.fid);
      if (pf && pf.n) bottom += `<div>외부 구조: 고가사다리차 <b>${pf.by.join(', ')}</b> → ${pf.facades.join('/')} 진입창 후보 ${pf.n}곳 (높이 ${pf.h}m${ws[0] ? `, 수평 ${ws[0].dist}m` : ''})</div>`;
      else if (pf) bottom += `<div style="color:#ffa198">외부 구조: 이 층은 고가사다리차가 닿지 않음 → 내부 진입·피난 유도 우선</div>`;
    }
    return { top, bottom };
  }
  return { fireExtra, onView: () => { showZones(); showSite(); }, siteOn: () => !!S, frameSite: () => S };
}
