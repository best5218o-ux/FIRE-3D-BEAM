// 소방시설 레이어 — 정보카드 '층별 소방시설 설치현황' 전 항목을 법정 분류(시행령 별표 1)로, 위치 확실성과 함께 표시
//  located(도면 기호 위치) = 속이 찬 표식 · estimated(NFPC 502/NFTC 504 규칙 추정) = 속 빈 표식 + 점선 반경 · present(정보카드 수량만) = 층 배지 + 방호 범위 색조
// main.js 에서 initFac(ctx) 로 연결(피난 경로 등 다른 모듈과 분리)
import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

const CATCOL = { 소화설비: 0xff6b6b, 경보설비: 0xffb020, 피난구조설비: 0x3fb950, 소화용수설비: 0x39c5cf, 소화활동설비: 0xd058ff, '소방활동 관련 건축설비': 0x58a6ff, 기타: 0x8b949e };
const CERT = { located: ['위치 확인', '도면 범례 기호'], estimated: ['위치 추정', '규칙(NFPC/NFTC) 추정'], present: ['수량만', '정보카드 수량 — 위치 없음'], mixed: ['혼합', ''] };
const COVER = { SP: '스프링클러 헤드', SSP: '간이스프링클러 헤드', ESFR: '화재조기진압용 헤드', FOAM: '포 헤드', WSPRAY: '물분무 헤드', MIST: '미분무 헤드',
  CO2: '이산화탄소', HALON: '할론', CLEAN: '할로겐화합물·불활성기체', POWDER: '분말', DET: '감지기', SMOKE: '제연' };
const SHORT = { EXT: '소화기', STANDPIPE: '방수구', OUTLET: '비상콘센트', SIAMESE: '송수구', EMEV: '비상용승강기', HOSE: '옥내소화전', PANEL: '수신기' };

export function initFac(ctx) {
  const { D, G, floors, FL, V, SLAB, esc, $, pickables, getSel } = ctx;
  const FC = D.fac;
  if (!FC) return { onView() {}, set() {} };
  const cats = FC.cats || Object.keys(CATCOL);
  const on = Object.fromEntries(cats.map((c) => [c, true]));
  const hex = (c) => '#' + c.toString(16).padStart(6, '0');
  const lab = (html, x, y, z, parent, col) => {
    const el = document.createElement('div'); el.className = 'lb facl'; el.innerHTML = html; if (col) el.style.borderColor = hex(col);
    const o = new CSS2DObject(el); o.position.copy(V(x, y, z)); parent.add(o); return o;
  };
  const dashedCircle = (x, y, r, z, col) => {
    const pts = []; for (let k = 0; k <= 64; k++) { const a = (k / 64) * Math.PI * 2; pts.push(V(x + r * Math.cos(a), y + r * Math.sin(a), z)); }
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: col, dashSize: 0.6, gapSize: 0.45, transparent: true, opacity: 0.9 }));
    l.computeLineDistances(); return l;
  };
  const dashedLine = (a, b, col) => { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineDashedMaterial({ color: col, dashSize: 0.4, gapSize: 0.3 })); l.computeLineDistances(); return l; };
  const info = (it, f, how, q) => {
    const src = how === 'located' ? '도면 기호(범례 대조 자동검출)' : how === 'estimated' ? '규칙 추정 — ' + esc(it.note || '') : '정보카드 수량';
    return `<b>${esc(it.name)}</b> <span class="fcert ${how}">${CERT[how][0]}</span> · ${esc(f.name)}<br>` +
      `<span class="k">분류</span> ${esc(it.cat)} › ${esc(it.group)} · <span class="k">정보카드</span> ${it.card ?? '-'}${esc(it.unit ? ' ' + it.unit : '')} (원문 "${esc(it.raw)}") · <span class="k">도면 기호</span> ${it.drawing ?? '-'}<br>` +
      `<span class="k">출처</span> ${src}${q && q.alt ? (it.code === 'EMEV' ? ' · <b>후보</b>(도면 단서로 가리지 못함)' : ' · <b>후보 위치</b>(규칙상 계단 중 일부만 해당)') : ''}${q && q.why ? `<br><span class="k">근거</span> ${esc(q.why)}` : ''}${q && q.conf != null ? ` · 검출 신뢰도 ${q.conf}` : ''}${q && q.gid ? `<br><span class="k">IFC GlobalId ${esc(q.gid)}</span>` : ''}`;
  };

  const FG = {};
  for (const f of floors) {
    const items = FC.floors[f.id]; if (!items || !G[f.id]) continue;
    const g = new THREE.Group(); G[f.id].g.add(g);
    const parts = FG[f.id] = { g, byCat: {}, labels: [], badge: null };
    const catG = (c) => parts.byCat[c] || (parts.byCat[c] = (() => { const x = new THREE.Group(); g.add(x); return x; })());
    for (const it of items) {
      const col = CATCOL[it.cat] ?? CATCOL['기타'], cg = catG(it.cat);
      const nm = SHORT[it.code] || it.name;
      if (!['HOSE', 'PANEL'].includes(it.code)) {                  // 옥내소화전·수신기는 기존 설비 표식이 이미 있음
        for (const q of it.loc) {                                 // located: 속이 찬 기둥
          const m = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 1.2, 16), new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.55 }));
          m.position.copy(V(q.x, q.y, SLAB + 0.6)); m.userData = { kind: 'fac', fid: f.id, name: it.name, facHtml: info(it, f, 'located', q) }; cg.add(m); pickables.push(m);
          parts.labels.push(lab(esc(nm), q.x, q.y, SLAB + 1.9, cg, col));
        }
      }
      for (const q of it.est) {                                   // estimated: 속 빈(와이어) 표식 + 점선 반경·기준점
        if (q.anchor === 'elevator') {
          const [x0, y0, x1, y1] = q.box;
          const pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]].map(([a, b]) => V(a, b, SLAB + 0.25));
          const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: col, dashSize: 0.5, gapSize: 0.35 })); l.computeLineDistances(); cg.add(l);
        } else if (q.r) {
          cg.add(dashedCircle(q.ax, q.ay, q.r, SLAB + 0.2, col));
          cg.add(dashedLine(V(q.ax, q.ay, SLAB + 0.2), V(q.x, q.y, SLAB + 0.2), col));
        }
        const m = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.2, 12, 1, true), new THREE.MeshBasicMaterial({ color: col, wireframe: true, transparent: true, opacity: 0.95 }));
        m.position.copy(V(q.x, q.y, SLAB + 0.6)); m.userData = { kind: 'fac', fid: f.id, name: it.name, facHtml: info(it, f, 'estimated', q) }; cg.add(m); pickables.push(m);
        parts.labels.push(lab(esc(nm) + ' <span style="opacity:.75">추정' + (q.alt ? '·후보' : '') + '</span>', q.x, q.y, SLAB + 1.9, cg, col));
      }
      if (it.present && COVER[it.code] && f.poly.length) {        // present(방호 범위형): 층 전체 옅은 색조 — 개별 위치는 만들지 않음
        for (const ring of f.poly) {
          const sg = new THREE.ShapeGeometry(new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y)))); sg.rotateX(-Math.PI / 2); sg.translate(0, SLAB + 0.06 + (it.cat === '경보설비' ? 0.01 : 0), 0);
          const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
          m.renderOrder = 1; cg.add(m);
        }
      }
    }
    // 층 배지(수량만 항목 요약): 층 오른쪽 위 모서리
    const pres = items.filter((it) => it.present > 0);
    if (pres.length) {
      const byC = {}; for (const it of pres) (byC[it.cat] = byC[it.cat] || []).push(it);
      const html = Object.entries(byC).map(([c, a]) => `<span style="color:${hex(CATCOL[c] ?? 0x8b949e)}">■</span> ${a.map((it) => `${esc(SHORT[it.code] || it.raw)} ${it.present}`).join(' · ')}`).join('<br>');
      const el = document.createElement('div'); el.className = 'lb facbadge'; el.innerHTML = `<b>수량만(위치 없음)</b><br>${html}`;
      const o = new CSS2DObject(el); o.position.copy(V(f.bb[2] - 4, f.bb[3] - 4, SLAB + 0.5)); g.add(o); parts.badge = o;
    }
  }

  function table(fid) {
    const f = FL[fid], items = FC.floors[fid] || [];
    if (!items.length) return '<div class="note">정보카드 소방시설 항목 없음</div>';
    let h = `<div class="row" style="margin-bottom:5px">` + cats.filter((c) => Object.values(FC.floors).some((a) => a.some((it) => it.cat === c)))
      .map((c) => `<label class="tg"><input type="checkbox" data-fcat="${esc(c)}" ${on[c] ? 'checked' : ''}><span style="color:${hex(CATCOL[c] ?? 0x8b949e)}">■</span>${esc(c)}</label>`).join('') + `</div>`;
    h += `<div class="note" style="margin:0 0 4px">${esc(f.name)} · <span class="fcert located">위치 확인</span> 도면 기호 <span class="fcert estimated">위치 추정</span> 규칙 <span class="fcert present">수량만</span> 정보카드</div>`;
    h += '<table class="fact"><tr><td class="k">항목 (법정 명칭 · 정보카드 원문)</td><td class="k">카드</td><td class="k">도면</td><td class="k">위치</td></tr>';
    for (const c of cats) {
      for (const it of items.filter((x) => x.cat === c)) {
        const p = it.loc[0] || it.est[0];
        const cert = it.cert === 'mixed' ? (it.loc.length ? `<span class="fcert located">확인 ${it.loc.length}</span>` : '') + (it.est.length ? `<span class="fcert estimated">추정 ${it.est.length}</span>` : '') + (it.present ? `<span class="fcert present">수량 ${it.present}</span>` : '')
          : `<span class="fcert ${it.cert}">${CERT[it.cert][0]}${it.cert === 'estimated' && it.est.some((q) => q.alt) ? '(후보)' : ''}</span>`;
        h += `<tr ${p ? `class="mkrow" data-fly="${p.x},${p.y}" data-fl="${fid}"` : ''} title="${esc(it.cat + ' › ' + it.group + (it.note ? ' — ' + it.note : ''))}"><td><span style="color:${hex(CATCOL[c] ?? 0x8b949e)}">■</span> ${esc(it.name)}${it.raw && it.raw !== it.name ? `<span class="k"> "${esc(it.raw)}"</span>` : ''}</td>` +
          `<td>${it.card ?? '-'}</td><td>${it.drawing ?? '-'}</td><td>${cert}</td></tr>`;
      }
    }
    return h + '</table><div class="note">추정 위치는 NFPC 502 제6조·NFTC 504 2.1.5.2(계단 출입구 5 m 이내) 규칙으로 만든 후보이며 실제 위치가 아닙니다. 수량만 항목(헤드·감지기 등)은 층 전체 존재만 표시합니다.</div>';
  }
  $('facSec').style.display = '';
  $('facPanel').addEventListener('change', (e) => { const c = e.target.dataset.fcat; if (c) { on[c] = e.target.checked; onView(); } });

  function onView() {
    const show = $('tFac') ? $('tFac').checked : true, sel = getSel();
    for (const [fid, P] of Object.entries(FG)) {
      P.g.visible = show && (!ctx.showAll || ctx.showAll() || fid === sel);     // v2 화면: 선택 층만(층 분리 보기는 전 층)
      for (const [c, cg] of Object.entries(P.byCat)) cg.visible = on[c] !== false;
      const lv = show && $('tLab').checked && fid === sel && G[fid].g.visible;
      P.labels.forEach((o) => (o.visible = lv && on[o.parent === P.g ? '' : Object.keys(P.byCat).find((c) => P.byCat[c] === o.parent)] !== false));
      if (P.badge) P.badge.visible = lv;
    }
    $('facPanel').innerHTML = table(sel || (ctx.activeFloor ? ctx.activeFloor() : floors[0].id));     // 전체 요약 화면(선택 층 없음)이면 현재 층 표
  }
  return { onView, set: (v) => { if ($('tFac')) $('tFac').checked = v !== false; onView(); }, cats: (m) => { Object.assign(on, m); onView(); } };
}
