// 인명검색 우선순위 — 화점 입력 후 층별 실 순서 + 근거 (firebim/search.py 의 실 용도 범주 + firebim/sop.py SEARCH 기준값 = D.sop_search)
// 정렬 = 사전식 키 [층 순위, 대피 곤란 분류 순위, 용도·시간대 등급, 공격계단(위층: 공격계단 또는 그 층 계단)에서 보행거리, 실 번호]
// 결정적: 같은 입력(화점·공격계단·대피로 분류·모드) → 같은 순서. 가중치는 모두 임시값(기관 SOP 로 설정).
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

const DEF = {                                   // D.sop_search 가 없을 때(옛 데이터)만 쓰는 최소 기본값 — 실제 값은 sop.py
  night_hours: [22, 6], cat_weight: { unk: [0.45, 0.45] }, mobility_bonus: 0, bands: [0.8, 0.45],
  class_order_fire: ['iso', 'fire', 'shared', 'near', 'detour', 'ok'], class_order_upper: ['none', 'above', 'hard', 'ok'],
  floors_above: 2, above_radius_m: 6, fire_cross_factor: 5, dist_ref: 'stair', walk_mps: 0.8, stair_s_per_floor: 25, markers_fire: 12, markers_upper: 6, route_n: 6,
};
export const CLASS_KO = { iso: '고립(대피로 없음)', fire: '화재실(화점 구역)', shared: '공격계단 공용', near: '화점 인접 경로', detour: '우회 증가', ok: '정상',
  none: '대피 경로 없음(평시)', above: '화점 직상부', hard: '평시 대피 곤란' };
export const CAT_KO = { crit: '거동 불가 실', sleep: '취침 가능 실', occupied: '주간 재실 실', hall: '대공간', corr: '복도·홀', svc: '설비·창고', unk: '용도 모름' };
const BAND_KO = ['높음', '보통', '낮음'];

export function initSearch(ctx) {
  const { D, BIM, floors, FL, V, SLAB, esc, $, zOf, gridFor, dijkstra, distAt } = ctx;
  const CFG = Object.assign({}, DEF, D.sop_search || {});
  const EV = D.evac || null;
  const hr = new Date().getHours(), [n0, n1] = CFG.night_hours;
  let mode = (n0 > n1 ? hr >= n0 || hr < n1 : hr >= n0 && hr < n1) ? 'night' : 'day';
  let last = null, lastF = null, group = null;

  const ctr = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  const cOf = (poly) => poly.reduce((a, p) => [a[0] + p[0] / poly.length, a[1] + p[1] / poly.length], [0, 0]);
  const inPoly = (x, y, poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; };
  // 층의 실 목록(대피로 실 분할 우선 — 화점 분류와 같은 실; 없으면 3D 실 구획)
  function rooms(fid) {
    const rec = EV && EV.floors[fid];
    if (rec) return { rec, list: rec.rooms.map((r, k) => ({ k, poly: r.poly, c: r.c, area: r.area, cat: r.cat || 'unk', cw: r.cw, why: r.why || '', d: r.d, over: r.over, n_reach: r.n_reach, wall: r.wall })) };
    const sp = (BIM.floors[fid] || {}).spaces || [];
    return { rec: null, list: sp.map((s, k) => ({ k, poly: s.poly, c: cOf(s.poly), area: s.area, cat: 'unk', cw: null, why: '대피로 계산 없음', d: null })) };
  }
  const wOf = (r) => (r.cw ? r.cw[mode === 'night' ? 1 : 0] : (CFG.cat_weight.unk || [0.45, 0.45])[mode === 'night' ? 1 : 0]);
  const bandOf = (w) => (w >= CFG.bands[0] ? 0 : w >= CFG.bands[1] ? 1 : 2);

  // F: { fid, x, y, attack(계단 객체|null), entry([x,y]|null — 지상층 화재 진입구), st(화재층 실별 대피 분류 배열|null) }
  function rank(F) {
    const { fid, x, y, attack } = F;
    const idx = floors.findIndex((f) => f.id === fid);
    const fl = [fid, ...floors.slice(idx + 1, idx + 1 + CFG.floors_above).map((f) => f.id)].filter((f) => BIM.floors[f]);
    const items = [], grids = {};
    fl.forEach((f, fr) => {
      const { rec, list } = rooms(f);
      // 보행거리 격자: 화재층은 화점 반경 안 비용 × fire_cross_factor(통과 억제), 위층은 그대로
      let gr = gridFor(f);
      if (fr === 0 && CFG.fire_cross_factor !== 1) {
        gr = { ...gr, cost: Float32Array.from(gr.cost) };
        for (let j = 0; j < gr.H; j++) for (let i = 0; i < gr.W; i++) {
          const px = gr.x0 + i * 0.25, py = gr.y1 - j * 0.25;
          if (Math.hypot(px - x, py - y) < CFG.above_radius_m) gr.cost[j * gr.W + i] *= CFG.fire_cross_factor;
        }
      }
      let src, srcName;
      if (CFG.dist_ref === 'fire' && fr === 0) { src = [[x, y]]; srcName = '화점'; }
      else if (attack && attack.floors.includes(f)) { src = [ctr(attack.box)]; srcName = '공격계단 ' + attack.name; }
      else if (fr === 0 && F.entry) { src = [F.entry]; srcName = '진입구'; }
      else {
        const ss = BIM.stairs.filter((s) => s.floors.includes(f));
        src = ss.length ? ss.map((s) => ctr(s.box)) : [[x, y]]; srcName = ss.length ? '이 층 계단' : '화점 위치';
      }
      const R = dijkstra(gr, src); grids[f] = { gr, src };
      const nT = rec ? (rec.targets || []).length : 0, order = fr === 0 ? CFG.class_order_fire : CFG.class_order_upper;
      for (const r of list) {
        let cls;
        if (fr === 0) cls = F.st && F.st[r.k] ? F.st[r.k] : Math.hypot(r.c[0] - x, r.c[1] - y) < CFG.above_radius_m || inPoly(x, y, r.poly) ? 'fire' : 'ok';
        else if (rec && r.d == null) cls = 'none';
        else if (inPoly(x, y, r.poly) || Math.hypot(r.c[0] - x, r.c[1] - y) < CFG.above_radius_m) cls = 'above';
        else if (r.over === 'high' || (r.n_reach === 1 && nT >= 2)) cls = 'hard';
        else cls = 'ok';
        const ci = Math.max(0, order.indexOf(cls)), w = wOf(r), b = bandOf(w);
        const d = distAt(gr, R, r.c[0], r.c[1]);
        const dUp = FL[f].ord - FL[fid].ord;
        const t = isFinite(d) ? d / CFG.walk_mps + dUp * CFG.stair_s_per_floor : null;
        items.push({ fid: f, fr, k: r.k, c: r.c, area: r.area, cls, ci, cat: r.cat, w, band: b, d: isFinite(d) ? d : null, t, src: srcName, why: r.why,
          key: [fr, ci, b, isFinite(d) ? d : 1e9, r.k] });
      }
    });
    items.sort((a, b) => { for (let i = 0; i < 5; i++) if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i]; return 0; });
    items.forEach((z, i) => {
      z.rank = i + 1;
      const role = z.fr === 0 ? '화재층' : z.fr === 1 ? '직상층' : `화재층 위 ${z.fr}개 층`;
      z.reasons = [`${role}(${FL[z.fid].name})`, `${CLASS_KO[z.cls]}`, `${CAT_KO[z.cat] || z.cat} — ${z.why}`, `${mode === 'night' ? '야간' : '주간'} 가중 ${z.w.toFixed(2)}(${BAND_KO[z.band]})`,
        z.d != null ? `${z.src}에서 ${z.d.toFixed(0)}m(≈${(z.t / 60).toFixed(1)}분)` : `${z.src}에서 경로 없음`];
    });
    return { items, mode, floors: fl, cfg: { dist_ref: CFG.dist_ref, floors_above: CFG.floors_above }, grids };
  }

  // 격자 최단경로 역추적(main.js pathTo 와 같은 격자 — 경고 상태는 건드리지 않음)
  function trace(gr, R, x, y) {
    let i = Math.round((x - gr.x0) / 0.25), j = Math.round((gr.y1 - y) / 0.25);
    i = Math.max(0, Math.min(gr.W - 1, i)); j = Math.max(0, Math.min(gr.H - 1, j));
    let u = j * gr.W + i, best = null;
    for (let r = 0; r < 8 && !isFinite(R.dist[u]); r++) for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
      const a = i + di, b = j + dj; if (a < 0 || b < 0 || a >= gr.W || b >= gr.H) continue; const v = b * gr.W + a; if (isFinite(R.dist[v]) && (best == null || R.dist[v] < R.dist[best])) best = v;
    }
    if (!isFinite(R.dist[u]) && best != null) u = best;
    const pts = []; while (u >= 0 && pts.length < 20000) { pts.push([gr.x0 + (u % gr.W) * 0.25, gr.y1 - ((u / gr.W) | 0) * 0.25]); u = R.prev[u]; }
    return pts.reverse();
  }
  // 검색 순서 경로(화재층 상위 route_n 실): 공격계단(기준점) → 1 → 2 → … 를 격자 최단경로로 잇는 보라 점선
  function route(parent) {
    const f = last.floors[0], G0 = last.grids[f]; if (!G0 || !CFG.route_n) return;
    const seq = last.items.filter((z) => z.fr === 0 && z.d != null).slice(0, CFG.route_n);
    let prev = G0.src[0], pts = [];
    for (const z of seq) { const R = dijkstra(G0.gr, [prev]); const P = trace(G0.gr, R, z.c[0], z.c[1]); pts = pts.concat(P); prev = z.c; }
    if (pts.length < 2) return;
    const zz = zOf(f) + SLAB + 0.55;
    const g = new THREE.BufferGeometry().setFromPoints(pts.map(([a, b]) => V(a, b, zz)));
    const ln = new THREE.Line(g, new THREE.LineDashedMaterial({ color: 0xd2a8ff, dashSize: 0.8, gapSize: 0.5 })); ln.computeLineDistances(); parent.add(ln);
  }
  function draw(parent) {
    if (!last) return;
    route(parent);
    const cnt = {};
    for (const z of last.items) {
      const lim = z.fr === 0 ? CFG.markers_fire : CFG.markers_upper;
      cnt[z.fid] = (cnt[z.fid] || 0) + 1; if (cnt[z.fid] > lim) continue;
      if (ctx.floorVisible && !ctx.floorVisible(z.fid)) continue;            // 숨긴 층(선택 층만 보기 등)의 번호는 그리지 않음
      const el = document.createElement('div'); el.className = 'lb srch b' + z.band + ' fr' + z.fr + (z.ci === 0 && z.cls !== 'ok' ? ' top' : ''); el.textContent = String(z.rank);
      el.title = z.reasons.join(' · ');
      const o = new CSS2DObject(el); o.position.copy(V(z.c[0], z.c[1], zOf(z.fid) + SLAB + 1.2)); parent.add(o);
    }
  }

  function panel() {
    if (!last) return '';
    const L = last.items, n = L.length;
    let h = `<div class="srres" id="srPanel"><div><b style="color:#d2a8ff">인명검색 우선순위</b> <select id="srMode"><option value="night"${mode === 'night' ? ' selected' : ''}>야간(취침)</option><option value="day"${mode === 'day' ? ' selected' : ''}>주간</option></select>`;
    h += ` <span class="k">${last.floors.map((f) => esc(FL[f].name)).join(' → ')} · ${n}실</span></div><table class="srt">`;
    const show = L.filter((z) => z.fr === 0).slice(0, 8).concat(L.filter((z) => z.fr > 0).slice(0, 4));
    for (const z of show) h += `<tr class="mk" data-fly="${z.c[0]},${z.c[1]}" data-fl="${z.fid}" title="${esc(z.reasons.join(' · '))}"><td class="k">${z.rank}</td><td style="text-align:left">${esc(FL[z.fid].name)} · <b>${esc(CLASS_KO[z.cls])}</b><br><span class="k">${esc(CAT_KO[z.cat] || z.cat)}(${BAND_KO[z.band]}) · ${z.area != null ? Number(z.area).toFixed(0) + '㎡' : ''}</span></td><td>${z.d != null ? z.d.toFixed(0) + 'm' : '-'}</td></tr>`;
    h += `</table><div class="note">순서 = 층(화재층→위층) → 대피 곤란 분류 → 용도·시간대 등급 → ${CFG.dist_ref === 'fire' ? '화점' : '공격계단'} 기준 보행거리. 줄에 마우스를 올리면 근거. 용도는 ${EV ? '정보카드 ⑩ 층 용도·실 모양(실명 있으면 실명)으로 추정' : '모름(대피로 없음)'} · 가중치는 임시값(SOP 조정).</div></div>`;
    return h;
  }

  function run(F, parent) {
    lastF = F; group = parent;
    if (!F) { last = null; return null; }
    last = rank(F); draw(parent);
    window.__search = last;
    return last;
  }
  document.addEventListener('change', (e) => {
    if (e.target && e.target.id === 'srMode') { mode = e.target.value; if (ctx.onMode) ctx.onMode(mode); }
  });
  return { run, panel, rank, getMode: () => mode, setMode: (m) => { mode = m; }, last: () => last, cfg: CFG };
}
