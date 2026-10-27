// 패널 HTML — 머리줄·왼쪽 탭(확인사항·개요·위험물·설비·진입·원리)·오른쪽(층 선택 전 요약 / 층 상세)·바닥줄
// 상태(선택 층 등)는 main.js 가 가지고, 여기서는 그 값으로 HTML 만 만든다. 클릭은 data-fly / data-go / data-act 속성으로 main.js 가 처리.
import { svg } from './icons.js';
const CATCOL = { 소화설비: '#ff6b6b', 경보설비: '#ffb020', 피난구조설비: '#3fb950', 소화용수설비: '#39c5cf', 소화활동설비: '#d058ff', '소방활동 관련 건축설비': '#58a6ff', 기타: '#8b949e' };
const ACT_KO = [['hose_len_m', '호스 1본 길이', 'm'], ['hose_carry', '기본 휴대 호스', '본'], ['hose_slack_m', '호스 여유 길이', 'm'], ['walk_mps', '보행 속도(진입)', 'm/s'], ['stair_s_per_floor', '계단 1개 층 이동', '초'],
  ['scba_cyl_l', '공기호흡기 용기', 'L'], ['scba_fill_bar', '충전 압력', 'bar'], ['scba_return_bar', '귀환 잔압', 'bar'], ['scba_lpm', '공기 소모량', 'L/분'], ['scba_margin_min', '안전 여유', '분'], ['standpipe_r_m', '방수구 인정 반경', 'm']];

export function initPanels(ctx) {
  const { D, BIM, floors, FL, stairs, elevs, equips, ACT, esc, $, groundId, SIMP, TTU_BANDS } = ctx;
  const B = D.building, C = B.card || {};
  const range = (a) => (a && a.length ? `${a[0]} ~ ${a[a.length - 1]}` : '-');
  const num = (v) => (v == null || v === '' ? '-' : Number(v).toLocaleString());
  const ctr = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  const shaftFl = (S) => (S.floors.includes(groundId) ? groundId : S.floors[Math.floor(S.floors.length / 2)]);
  const entsAll = [];                                                     // [fid, door]
  for (const f of [...floors].reverse()) for (const d of (BIM.floors[f.id] || {}).doors || []) if (d[0] === 'entrance') entsAll.push([f.id, d]);
  const exitsExt = (fid) => ((BIM.floors[fid] || {}).doors || []).filter((d) => d[0] === 'exit' && d[6]);
  const short = (s) => String(s || '').replace(/\s*\(폭 ([\d.]+m)[^)]*\)/, ' · $1');
  const cardStairs = ['일반계단', '피난계단', '특별피난계단', '옥외계단'].reduce((a, k) => a + ((B.stairs || {})[k] || 0), 0);
  const cardEv = ((B.stairs || {})['승강기_일반'] || 0) + ((B.stairs || {})['승강기_비상용'] || 0);
  const cardEm = (B.stairs || {})['승강기_비상용'];
  const facItems = () => {                                                // 건물 전체 소방시설(코드별 합)
    const FC = D.fac; if (!FC || !FC.floors) return [];
    const m = new Map();
    for (const f of floors) for (const it of FC.floors[f.id] || []) {
      const k = it.code + '|' + it.name;
      const z = m.get(k) || { name: it.name, code: it.code, cat: it.cat, card: 0, drawing: 0, loc: 0, est: 0, present: 0, fl: 0, unit: it.unit };
      z.card += Number(it.card) || 0; z.drawing += Number(it.drawing) || 0; z.loc += (it.loc || []).length; z.est += (it.est || []).length; z.present += it.present ? 1 : 0; z.fl++;
      m.set(k, z);
    }
    const order = FC.cats || Object.keys(CATCOL);
    return [...m.values()].sort((a, b) => order.indexOf(a.cat) - order.indexOf(b.cat) || b.card - a.card);
  };
  const certBadges = (z) => (z.loc ? `<span class="fcert located">확인 ${z.loc}</span>` : '') + (z.est ? `<span class="fcert estimated">추정 ${z.est}</span>` : '') + (!z.loc && !z.est ? '<span class="fcert present">수량만</span>' : '');
  const row = (dot, title, sub, attrs = '', right = '') => `<button class="rw" ${attrs}>${dot}<span>${title}${sub ? `<span class="s">${sub}</span>` : ''}</span>${right ? `<span class="r">${right}</span>` : ''}</button>`;
  const dot = (c) => `<span class="dot" style="background:${c};color:${c}"></span>`;
  const fidb = (f) => `<span class="fid">${esc(f)}</span>`;
  const aligned = (B.auto && B.auto.align) || [];
  const alignOf = {}; D.floors.forEach((f, i) => (alignOf[f.id] = aligned[i]));   // export.py: auto.align 순서 = D.floors 순서

  // ── 머리줄·바닥줄
  function header() {
    $('bname').textContent = B.name;
    const tags = [B.use && `<span class="tagx b">${esc(B.use)}</span>`, B.floors && `<span class="tagx b">${esc(B.floors)}</span>`, (C.위험물 || []).length && '<span class="tagx h">위험물</span>'].filter(Boolean);
    $('btags').innerHTML = tags.join(' ');
    const L = C.소재지 || {};
    $('baddr').textContent = [L.도로명 || L.지번, B.gfa ? `연면적 ${num(B.gfa)}㎡` : '', C.관리번호 ? `관리번호 ${C.관리번호}` : ''].filter(Boolean).join(' · ') || B.source || '';
    const hi = D.issues.filter((i) => i.lv === 'high').length;
    $('nIssues').textContent = hi || D.issues.length;
    $('bissue').classList.toggle('mid', !hi);
    $('bissue').title = `확인사항 전체 ${D.issues.length}건 (출동 전 확인 ${hi}건) — 누르면 목록`;
  }
  function footer(extra) {
    const mk = D.markers || [], auto = mk.filter((m) => m.src === 'auto').length, man = mk.filter((m) => m.src === 'manual').length;
    const nPlan = floors.filter((f) => f.plan || f.tex).length;
    const cnt = (v) => aligned.filter((a) => a === v).length;
    const low = floors.filter((f) => alignOf[f.id] === 'low').map((f) => f.id);
    const nObj = Object.values(BIM.floors).reduce((a, f) => a + f.walls.length + f.doors.length + f.spaces.length, 0) + BIM.equip.length;
    $('foot').innerHTML = [`원천: <b>${esc(B.source || '-')}</b>`, `정보카드 ${C.양식 === 'std' ? '표준양식' : '자동 판독'} · 도면 ${nPlan}개 층 판독`,
      `자동검출 ${auto}건${man ? ` · 도면판독 ${man}건` : ''} · BIM 객체 ${nObj.toLocaleString()}개`,
      aligned.length ? `층 정합 신뢰도 높음 ${cnt('high')} · 기준 ${cnt('ref')} · 낮음 ${cnt('low')}${low.length ? ` (${low.join('·')} — 위치 오차 클 수 있음)` : ''}` : '',
      `층고 ${extra.H}m 도식`, `<b id="fstat">${esc(extra.stat || '자동 변환 · 검수 전')}</b>`].filter(Boolean).join(' · ');
  }

  // ── 왼쪽 탭
  // 확인사항 문구: 현장 말로(개발용 세부 값은 접힌 '상세'). 옛 결과(축척 출처 'real:grid:tags+tread…' 문구)도 여기서 바꿈
  const SRC_KO = { sheet: '도면 표제란 축척', dims: '치수선', doors: '문 폭', tags: '창호 폭 표기', tread: '계단 디딤판', area: '정보카드 면적', grid: '다른 층 기준선', model: 'AI 모델' };
  const srcKo = (s) => [...new Set(String(s).replace(/^real:/, '').split(/[+:]/).map((t) => SRC_KO[t.trim()]).filter(Boolean))].join('·');
  function plainIssue(i) {
    if (i.x) return { d: i.d, x: i.x };
    const m = /^(.*?): 축척 ([\d.]+) px\/m \(([^)]*)\) — 이 층 출처 (.*?) → 축척 확인 필요$/.exec(i.d || '');
    if (m) {
      const c = m[4].split(',').map((z) => z.trim().split(' ')[0]);
      return { d: `${m[1]}: 도면 축척의 근거가 서로 달라 거리·면적이 틀릴 수 있음(${srcKo(c.join('+')) || '여러 근거'} 기준이 10% 넘게 차이) → 출동 전 실제 치수 1곳 확인`, x: `채택 축척 ${m[2]} px/m · 채택 근거 ${srcKo(m[3])} (${m[3]}) · 이 층 근거별 값: ${m[4]}` };
    }
    if (/px\/m|real:|ppm=/.test(i.d || '')) return { d: (i.d || '').replace(/\s*\([^)]*(?:px\/m|real:|ppm=)[^)]*\)/g, ''), x: i.d };
    return { d: i.d, x: null };
  }
  let ilv = 'all';
  function issues() {
    const n = (lv) => D.issues.filter((i) => lv === 'all' || i.lv === lv).length;
    $('ifilt').innerHTML = [['all', '전체'], ['high', '출동 전 확인'], ['mid', '현장 확인'], ['low', '참고']].filter(([k]) => n(k) || k === 'all')
      .map(([k, t]) => `<button data-ilv="${k}" class="${k === ilv ? 'on' : ''}">${t} ${n(k)}</button>`).join('');
    const ord = { high: 0, mid: 1, low: 2 };
    const L = D.issues.map((x, i) => [x, i]).filter(([x]) => ilv === 'all' || x.lv === ilv).sort((a, b) => (ord[a[0].lv] ?? 3) - (ord[b[0].lv] ?? 3) || a[1] - b[1]);
    const card = (i, sub) => { const p = plainIssue(i); return `<div class="issue ${esc(i.lv)}${sub ? ' sub' : ''}">${sub ? '' : `<b>${i.cat ? `<span class="cat">${esc(i.cat)}</span>` : ''}${esc(i.t)}</b>`}${esc(p.d)}${p.x ? `<details class="ix"><summary>상세(변환 기록)</summary><div>${esc(p.x)}</div></details>` : ''}</div>`; };
    // 같은 종류(심각도·분류·제목)가 여러 층에서 반복되면 한 카드로 묶음(층별 내용은 접힘) — 글 벽 줄이기
    const grp = new Map(); for (const [i] of L) { const k = `${i.lv}|${i.cat || ''}|${i.t}`; if (!grp.has(k)) grp.set(k, []); grp.get(k).push(i); }
    const flOf = (i) => (String(plainIssue(i).d).match(/^([^:：]{1,12})[:：]/) || [])[1];
    $('issues').innerHTML = L.length ? [...grp.values()].map((G) => {
      if (G.length < 3) return G.map((i) => card(i)).join('');
      const i = G[0], fls = G.map(flOf).filter(Boolean);
      return `<div class="igrp ${esc(i.lv)}"><b>${i.cat ? `<span class="cat">${esc(i.cat)}</span>` : ''}${esc(i.t)} <span class="cnt">${G.length}건</span></b><div class="sm">${fls.length ? esc(fls.join(' · ')) + ' — ' : ''}${esc(String(plainIssue(i).d).replace(/^[^:：]{1,12}[:：]\s*/, ''))}</div>` +
        `<details class="more"><summary>층별 ${G.length}건 보기</summary>${G.map((z) => card(z, true)).join('')}</details></div>`;
    }).join('') : '<div class="note">불일치 없음</div>';
    const t = document.querySelector('.tab[data-t="issues"]'); const hi = D.issues.filter((i) => i.lv === 'high').length;
    t.innerHTML = `확인사항${hi ? `<span class="n">${hi}</span>` : ''}`;
  }
  $('ifilt').addEventListener('click', (e) => { const b = e.target.closest('[data-ilv]'); if (b) { ilv = b.dataset.ilv; issues(); } });

  function overview(userHidden) {
    const S = C.건물구조 || {}, L = C.소재지 || {};
    const kv = (k, v) => (v == null || v === '' ? '' : `<div class="kv"><span>${k}</span><b>${esc(v)}</b></div>`);
    let h = kv('건물명', B.name) + kv('주용도', B.use) + kv('층수', B.floors) + kv('연면적', B.gfa ? num(B.gfa) + '㎡' : '') +
      kv('구조 · 지붕', S.구조 ? `${S.구조} · ${S.지붕 || '-'}` : '') + kv('소재지', L.도로명 || L.지번) + kv('관리번호', C.관리번호);
    const st = B.stairs || {};
    const chips = Object.entries(st).map(([k, v]) => `<span class="chip ${v === 0 ? 'zero' : ''}"><b>${v}</b>${esc(k.replace('_', ' '))}</span>`).join('');
    h += `<h4>계단 · 승강기 (정보카드 ↔ 도면)</h4><div class="chips">${chips}</div><div class="note">도면 판독: 계단 ${stairs.length}개소 · 승강기 ${elevs.length}개소 (정보카드 계단 ${cardStairs} · 승강기 ${cardEv})</div>`;
    $('bInfo').innerHTML = h;
    $('flList').innerHTML = [...floors].reverse().map((f) => `<div class="mk" data-go="${esc(f.id)}"><input type="checkbox" title="3D 표시" data-v="${esc(f.id)}" ${userHidden[f.id] ? '' : 'checked'}>` +
      `${fidb(f.id)}<span style="flex:1;min-width:0">${esc(f.name)} <span class="k">${esc(f.use || '')}</span></span><span class="k">${f.area ? num(f.area) + '㎡' : ''}</span></div>`).join('') +
      '<div class="note">체크를 풀면 그 층을 3D 에서 숨깁니다. 줄을 누르면 층 선택.</div>';
  }
  function haz() {
    const hz = C.위험물 || [];
    let h = hz.length ? hz.map((x) => `<div class="haz"><b>⚠ ${esc(x.구분)} — 진입 전 확인</b>${esc([x.시설명, x.품명, x.수량].filter(Boolean).join(' · '))}</div>`).join('')
      : `<div class="note" style="margin-top:0">${C.양식 === 'std' ? '정보카드 ⑨ 위험물 시설 등: 없음' : '정보카드에서 위험물 항목을 찾지 못함'}</div>`;
    const hi = D.issues.filter((i) => i.cat === '위험물');
    if (hi.length && !hz.length) h += hi.map((i) => `<div class="haz"><b>⚠ ${esc(i.t)}</b>${esc(i.d)}</div>`).join('');
    $('hazBox').innerHTML = h;
    const F = facItems();
    if (!F.length) { $('facAll').innerHTML = '<div class="note">정보카드 소방시설 항목 없음</div>'; return; }
    let t = '<div class="facs">';
    let cat = null;
    for (const z of F) {
      if (z.cat !== cat) { cat = z.cat; t += `<div class="k sm" style="grid-column:1/-1;margin-top:6px;font-weight:700">${esc(cat)}</div>`; }
      t += `<div class="nm"><span class="sq" style="background:${CATCOL[z.cat] || '#8b949e'}"></span><span title="${esc(z.name)}">${esc(z.name)} <span class="k">${z.card ? z.card.toLocaleString() + (z.unit ? esc(z.unit) : '') : ''}</span></span></div><div>${certBadges(z)}</div>`;
    }
    $('facAll').innerHTML = t + '</div><div class="note">정보카드 층별 설치현황을 건물 전체로 합한 값. 층을 고르면 오른쪽에 그 층 표가 나옵니다.</div>';
  }
  function entry() {
    const G = C.거리 || {}, km = (v) => (v == null ? '-' : `${v}km`);
    let h = '';
    if (C.양식 === 'std') {
      h += `<div class="kv"><span>출동 거리</span><b>관할 ${km(G.관할센터)} · 최단 ${km(G.최단센터)} · 소방서 ${km(G.소방서)}</b></div>`;
      const W = C.소방용수 || [];
      h += `<h4>소방용수 (정보카드)</h4>` + (W.length ? W.map((w) => `<div class="mk"><span class="dot" style="background:#39c5cf;color:#39c5cf"></span>${esc(w.종류)} ${esc(w.관서)}-${esc(w.동)}-${esc(w.번호)} <span class="k" style="margin-left:auto">${Math.round(w.거리km * 1000)}m</span></div>`).join('') : '<div class="note">없음</div>');
    }
    h += `<h4>지상 진입구 (도면)</h4>` + (entsAll.length ? entsAll.map(([fid, d]) => `<div class="mk" data-fly="${d[2]},${d[3]}" data-fl="${esc(fid)}">${fidb(fid)}${esc(short(d[5]))}</div>`).join('') : '<div class="note">도면에서 진입구를 찾지 못함 → 정보카드 진압작전도·현장 확인</div>');
    $('entryBox').innerHTML = h;
  }
  function basis() {
    const nObj = Object.values(BIM.floors).reduce((a, f) => a + f.walls.length + f.doors.length + f.spaces.length, 0) + BIM.equip.length;
    const steps = [['정보카드 읽기', `정보카드(${C.양식 === 'std' ? '표준양식' : 'PDF'})에서 층별 용도·면적·소방시설 수량, 계단·승강기, 위험물, 관계인(역할만)을 구조화`],
      ['평면도 좌표 맞춤', `층마다 축척·잘림이 다른 평면도를 한 좌표계(m)로 정합 — 층 ${floors.length}개`],
      ['기호 자동 검출', `도면 범례 기호(비상구·옥내소화전·수신기·가스밸브 등)를 찾아 좌표화 — ${(D.markers || []).length}건`],
      ['벽·문 벡터화', '벽 중심선·두께를 뽑고 같은 선 위 틈은 문으로, 외벽 틈은 진입구·비상구로 판정'],
      ['계단·승강기 도출', `여러 층에서 같은 자리에 반복되는 기호를 묶어 계단 ${stairs.length}개소 · 승강기 ${elevs.length}개소와 연결층 판정`],
      ['교차 검증', `정보카드 ↔ 도면 ↔ 건축물대장·부지 값을 대조해 확인사항 ${D.issues.length}건`],
      ['IFC4 BIM 생성', `IfcWall·IfcDoor·IfcSpace·IfcStair·IfcTransportElement·IfcFireSuppressionTerminal 등 ${nObj.toLocaleString()}개 객체 (같은 GlobalId 로 이 화면과 연결)`],
      ['화재 가정 경로', '층마다 벽·문으로 0.25m 보행 격자를 만들고 최단경로로 진입구→공격계단→화점, 가까운 옥내소화전·호스 본수·대피로·인명검색 순서를 계산'],
      ['3D 조립', '벽 중심선·두께로 두께 있는 유리 벽을 세우고 문 자리는 비움(문 위 인방·열린 문짝·회전 호), 외벽 유리 띠는 표현용(실제 창 위치 아님). 층을 누르면 위층이 들려 그 층이 드러나고, 층 분리·단층 평면·단면·투시·층 격리로 전환'],
      ['계단·승강로', '연결층 전체를 꿰는 빛나는 기둥(층마다 정지 고리). 특별피난계단은 정보카드 수와 도면 계단 수가 같으면 확정, 적으면 연결층이 긴 계단을 후보로 · 비상용승강기는 소방시설 추정 위치와 맞춤'],
      ['화재 확산 가정', '실 연결망(문·맞닿은 벽·계단실·승강로)에서 연기 지수를 시간에 따라 퍼뜨리고(계단은 위로 빠르게), 실마다 진입·대피 곤란 도달 시각과 그 시각의 대피로를 다시 계산 — 예시 모델'],
      ['대원 이동', '도착 시각(정보카드 관할 거리 ÷ 주행 속도) 뒤 진입로를 보행 속도·계단 층당 시간으로 따라 이동, 호스 전개·공기호흡기 잔압·귀환 한계 표시']];
    $('howBody').innerHTML = steps.map(([t, d], i) => `<div class="step"><b>${i + 1}</b><div><div class="it">${t}</div><div class="id">${esc(d)}</div></div></div>`).join('');
    $('actBox').innerHTML = ACT_KO.filter(([k]) => ACT[k] != null).map(([k, t, u]) => `<div class="kv"><span>${t}</span><b>${ACT[k]}${u}</b></div>`).join('') +
      '<div class="note">기관 SOP 로 바꿀 값입니다(fire3dbeam/sop.py). 계산 결과는 모두 이 가정값 기준.</div>' +
      (SIMP ? '<h4>화재 확산 가정 · 대원 이동 기준값 (예시·임시값 — sop.py SIM)</h4>' + [['smoke_max_m3s', '화원 연기 발생(최대)', '㎥/s'], ['grow_s', '성장 시간', '초'], ['untenable', '곤란 연기 지수', ''], ['c_door', '문 전도도', '㎥/s'], ['c_wall', '문 미검출 경계', '㎥/s'],
        ['c_stair_door', '계단실 문', '㎥/s'], ['c_stair_door_sp', '특별피난계단 문', '㎥/s'], ['c_stack_up', '계단 위로(굴뚝)', '㎥/s'], ['c_stack_dn', '계단 아래로', '㎥/s'], ['spread_door_s', '불 번짐(문)', '초'], ['spread_wall_s', '불 번짐(벽)', '초'],
        ['dispatch_s', '출동 준비', '초'], ['drive_kmh', '주행 속도', 'km/h'], ['setup_s', '현장 전개', '초'], ['search_dwell_s', '실 검색 시간', '초']].filter(([k]) => SIMP[k] != null).map(([k, t, u]) => `<div class="kv"><span>${t}</span><b>${SIMP[k]}${u}</b></div>`).join('') : '');
    $('limits').textContent = '벽·문·실은 평면도를 자동 벡터화한 LOD200 수준입니다. 얇은 칸막이 일부가 빠질 수 있고, 실은 문을 닫은 상태로 나눈 추정 영역입니다. ' +
      '층 위치는 도면 정합값(신뢰도 낮은 층은 바닥줄에 표시), 층고는 도식값입니다. 소방시설의 위치 추정·수량만 항목은 실제 위치가 아닙니다. 경로·시간은 추정이며 현장 확인이 우선입니다.';
  }

  // ── 오른쪽 맨 위: 출동 전 30초 요약(가장 중요한 5개 — 진입구 · 공격계단 · 위험물 · 인명 우선 구역 · 확인 필요). 누르면 그 위치/탭으로
  const EVR = (D.evac && D.evac.floors) || {};
  const CATK = { crit: '거동 불가 실', sleep: '취침 실', occupied: '재실', hall: '대공간', svc: '설비', corr: '복도', unk: '미상' };
  const CLS = { iso: '고립', shared: '공격계단 공용', near: '화점 인접', detour: '우회', ok: '정상', fire: '화재실' };
  const firstEntry = (D.evac && D.evac.entry) || [];
  function brief30(st) {
    const P = st.plan, SR = st.search, el = $('b30');
    const it = [];
    // 1) 진입구
    const en = P && P.entry ? { name: P.entry.name, x: P.entry.x, y: P.entry.y, fid: P.ground } : entsAll.length ? { name: entsAll[0][1][5], x: entsAll[0][1][2], y: entsAll[0][1][3], fid: entsAll[0][0] } : null;
    it.push({ ic: 'entrance', k: '진입구', cls: en ? '' : 'bad', t: en ? `${en.fid} ${short(en.name)}` : '도면에서 찾지 못함', s: en ? (P && P.entry ? '화재 가정 진입로의 진입구' : `지상 진입구 ${entsAll.length}곳${entsAll.length > 1 ? ` · 외 ${entsAll.length - 1}곳` : ''}`) : '진압작전도·현장 확인', at: en ? `data-fly="${en.x},${en.y}" data-fl="${esc(en.fid)}"` : '' });
    // 2) 공격계단: 화점 있으면 산출안, 없으면 진입구에서 가장 가까운 계단(후보)
    if (P && P.attackName) {
      const S = stairs.find((z) => z.id === P.attack) || {}; const c = S.box ? ctr(S.box) : null;
      it.push({ ic: 'stair', k: '공격계단', t: P.attackName + (S.special ? ' · 특별피난' : ''), s: `진입 ${(P.tIn / 60).toFixed(1)}분 · 1층 ${P.dG.toFixed(0)}m + ${P.floorsUp}개 층 + 화재층 ${(P.dF || 0).toFixed(0)}m`, at: c ? `data-fly="${c[0]},${c[1]}" data-fl="${esc(P.fid)}"` : '' });
    } else {
      const rs = firstEntry.flatMap((e) => (e.routes || []).map((r) => ({ ...r, ent: e }))).filter((r) => r.d != null).sort((a, b) => a.d - b.d);
      const S = rs.length ? stairs.find((z) => z.id === rs[0].stair) : stairs.find((z) => z.special) || stairs[0];
      if (S) it.push({ ic: 'stair', k: '공격계단(후보)', t: S.name + (S.special ? ' · 특별피난' + (S.special.cert === 'cand' ? '?' : '') : ''), s: rs.length ? `진입구에서 ${rs[0].d.toFixed(0)}m · ${range(S.floors)} · 화점을 정하면 자동 선정` : `${range(S.floors)} · 화점을 정하면 자동 선정`, at: `data-fly="${ctr(S.box)}" data-fl="${esc(shaftFl(S))}"` });
      else it.push({ ic: 'stair', k: '공격계단', cls: 'bad', t: '계단을 도면에서 찾지 못함', s: '정보카드 계단 ' + cardStairs + '개 — 현장 확인', at: '' });
    }
    // 3) 위험물
    const hz = C.위험물 || [], hzI = D.issues.filter((i) => i.cat === '위험물');
    if (hz.length) it.push({ ic: 'hazard', k: '위험물', cls: 'bad', t: hz.map((x) => x.구분).join(' · '), s: [hz[0].시설명, hz[0].품명, hz[0].수량].filter(Boolean).join(' · ') + (hz.length > 1 ? ` 외 ${hz.length - 1}` : ''), at: 'data-b30="haz"' });
    else if (hzI.length) it.push({ ic: 'hazard', k: '위험물', cls: 'warn', t: hzI[0].t, s: hzI[0].d, at: 'data-b30="haz"' });
    else it.push({ ic: 'hazard', k: '위험물', t: '없음', s: C.양식 === 'std' ? '정보카드 ⑨ 위험물 시설 등 — 기재 없음' : '정보카드에서 위험물 항목을 찾지 못함', at: 'data-b30="haz"' });
    // 4) 인명 우선 구역: 화점 있으면 인명검색 1순위, 없으면 거동 불가·취침 실 → 층 용도
    const z1 = SR && SR.items && SR.items[0];
    if (z1) it.push({ ic: 'life', k: '인명 우선(검색 1순위)', cls: 'warn', t: `${FL[z1.fid].name} · ${CATK[z1.cat] || z1.cat || '실'}${z1.area != null ? ' ' + Math.round(z1.area) + '㎡' : ''}`, s: `${CLS[z1.cls] || ''}${z1.d != null ? ` · 공격계단에서 ${z1.d.toFixed(0)}m` : ''} · 검색 대상 ${SR.items.length}실`, at: `data-fly="${z1.c[0]},${z1.c[1]}" data-fl="${esc(z1.fid)}"` });
    else {
      const pri = []; for (const f of floors) for (const r of (EVR[f.id] || {}).rooms || []) if ((r.cat === 'crit' || r.cat === 'sleep') && r.c) pri.push([f, r]);
      pri.sort((a, b) => (a[1].cat === 'crit' ? 0 : 1) - (b[1].cat === 'crit' ? 0 : 1) || b[0].level - a[0].level);
      const vuln = floors.filter((f) => /노유자|요양|의료|병원|숙박|기숙|수련|어린이|장애/.test(f.use || ''));
      if (pri.length) {
        const fl = [...new Set(pri.map(([f]) => f.id))], nC = pri.filter(([, r]) => r.cat === 'crit').length;
        it.push({ ic: 'life', k: '인명 우선 구역', cls: 'warn', t: nC ? `거동 불가 실 ${nC}곳` + (pri.length > nC ? ` · 취침 실 ${pri.length - nC}곳` : '') : `취침 실 ${pri.length}곳`, s: `${fl.slice(0, 4).join('·')}${fl.length > 4 ? ` 외 ${fl.length - 4}개 층` : ''} — 실 용도 추정(대피 분석)`, at: `data-fly="${pri[0][1].c[0]},${pri[0][1].c[1]}" data-fl="${esc(pri[0][0].id)}"` });
      } else if (vuln.length) it.push({ ic: 'life', k: '인명 우선 구역', cls: 'warn', t: `${vuln.map((f) => f.id).join('·')} ${vuln[0].use}`, s: '층 용도로 본 피난 약자 층 — 화재층·직상층과 함께 우선 검색', at: `data-go="${esc(vuln[0].id)}"` });
      else it.push({ ic: 'life', k: '인명 우선 구역', t: '거동 불가·취침 실 추정 없음', s: '주간 재실 위주 — 화재층 → 직상층 순 검색(화점을 정하면 순위 계산)', at: '' });
    }
    // 5) 확인 필요
    const hi = D.issues.filter((i) => i.lv === 'high'), mid = D.issues.filter((i) => i.lv === 'mid');
    const top = hi[0] || mid[0];
    it.push({ ic: 'alert', k: '확인 필요', cls: hi.length ? 'bad' : mid.length ? 'warn' : '', t: top ? `${top.t}${hi.length + mid.length > 1 ? ` 외 ${hi.length + mid.length - 1}건` : ''}` : '불일치 없음', s: `출동 전 확인 ${hi.length} · 현장 확인 ${mid.length}${top && top.d ? ' — ' + plainIssue(top).d : ''}`, at: 'data-b30="issues"' });
    const open = el.classList.contains('open');
    el.innerHTML = `<button class="hd" aria-expanded="${open}" aria-controls="b30l">${svg('clock')}출동 전 30초 요약<span>${P ? esc(P.fid) + ' 화재 반영' : '건물 전체'}</span>${svg('chev')}</button>` +
      `<ol id="b30l">${it.map((z) => `<li class="${z.cls || ''}"><button ${z.at || 'tabindex="-1"'} ${z.at ? '' : 'disabled style="cursor:default"'} aria-label="${esc(z.k)}: ${esc(z.t)}"><span class="bi">${svg(z.ic)}</span><span class="bt3"><small>${esc(z.k)}</small><b title="${esc(z.t)}">${esc(z.t)}</b><i title="${esc(z.s)}">${esc(z.s)}</i></span>${z.at ? `<span class="go">${svg('chev')}</span>` : '<span></span>'}</button></li>`).join('')}</ol>`;
  }
  const det = (title, cnt, body, open) => `<details class="more"${open ? ' open' : ''}><summary>${title}${cnt != null ? `<span class="cnt">${cnt}</span>` : ''}</summary>${body}</details>`;

  // ── 오른쪽: 층 선택 전(건물 요약) — 핵심만 펼치고 나머지는 접음(누르면 펼침)
  function home() {
    $('rpHead').innerHTML = `<div class="ph"><div class="ph-t">층을 선택하세요</div><div class="muted sm">3D 화면의 층 · 오른쪽 층 버튼 · 숫자 키(1~9, 0 = 지하1층)</div></div>`;
    let h = '<h4>🚒 소방활동 요약</h4>';
    const multi = new Set(entsAll.map(([f]) => f)).size > 1;
    const nx = floors.reduce((a, f) => a + exitsExt(f.id).length, 0);
    h += det(`${svg('entrance')} 지상 진입구${multi ? ' (층마다 다름 — 경사지)' : ''}`, `${entsAll.length}곳`, (entsAll.length ? entsAll.map(([fid, d]) => row(fidb(fid), esc(short(d[5])), '', `data-fly="${d[2]},${d[3]}" data-fl="${esc(fid)}"`)).join('') : '<div class="tip">도면에서 진입구를 찾지 못함 — 진압작전도·현장 확인</div>') +
      (nx ? `<div class="muted sm">외부로 바로 나가는 비상구·개구부 ${nx}곳 (녹색 문)</div>` : ''), entsAll.length <= 4);
    const spB = (s) => (s.special ? ` <span class="fcert ${s.special.cert === 'cand' ? 'estimated' : 'located'}">${s.special.kind === 'sp' ? '특별피난' : '비상용'}${s.special.cert === 'cand' ? ' 후보' : ''}</span>` : '');
    h += det(`${svg('stair')} 계단`, `${stairs.length}개소`, stairs.map((s) => row(dot(s.special ? '#facc15' : '#4ade80'), esc(s.name) + spB(s), `${range(s.floors)}${s.kind ? ' · ' + esc(s.kind) : ''}`, `data-fly="${ctr(s.box)}" data-fl="${esc(shaftFl(s))}"`)).join('') +
      (cardStairs !== stairs.length ? `<div class="tip">정보카드 계단 ${cardStairs} ↔ 도면 ${stairs.length}개소${cardStairs > stairs.length ? ' → 나머지 위치 현장 확인' : ''}</div>` : ''), true);
    const cars = elevs.reduce((a, e) => a + (e.cars || 1), 0);
    const em = facItems().find((z) => z.code === 'EMEV');
    h += det(`${svg('ev')} 승강기`, `${cars}대 · ${elevs.length}개소`, elevs.map((e) => row(dot(e.special ? '#fb923c' : '#60a5fa'), esc(e.name) + (e.cars > 1 ? ` (${e.cars}대)` : '') + spB(e), range(e.floors) + (e.note ? ' · ' + esc(e.note) : ''), `data-fly="${ctr(e.box)}" data-fl="${esc(shaftFl(e))}"`)).join(''), false);
    h += `<div class="og">` + (cardEm ? `<div class="sm">비상용 승강기: 정보카드 <b>${cardEm}</b>대${em ? ` · ${certBadges(em)}` : ' · 도면 위치 미확인'}</div>`
      : `<div class="tip">비상용 승강기 없음(정보카드${cardEm == null ? ' 기재 없음' : ''}) — 화재 시 일반 승강기 사용 금지</div>`) + '</div>';
    const hose = equips.filter((q) => q.type === 'hose').length, hoseC = floors.reduce((a, f) => a + (Number((f.fac || {})['옥내소화전']) || 0), 0);
    h += `<div class="og"><div class="ol">${svg('hose')} 옥내소화전</div><div class="muted sm">도면 ${hose}개 위치 · 정보카드 합계 ${hoseC}개 — 층을 고르면 위치 표시</div></div>`;
    const pn = equips.filter((q) => q.type === 'panel');
    if (pn.length) h += `<div class="og"><div class="ol">${svg('panel')} 수신기</div>${pn.map((q) => row(fidb(q.fid), esc(q.name), '', `data-fly="${q.x},${q.y}" data-fl="${esc(q.fid)}"`)).join('')}</div>`;
    const F = facItems();
    if (F.length) {
      const top = F.filter((z) => z.loc || z.est || ['SP', 'STANDPIPE', 'SIAMESE', 'HOSE', 'PANEL', 'DET'].includes(z.code)).slice(0, 9);
      h += det(`${svg('fac')} 소방시설 (위치 확실성)`, `${F.length}종`, `<div class="facs">${top.map((z) => `<div class="nm"><span class="sq" style="background:${CATCOL[z.cat] || '#8b949e'}"></span><span>${esc(z.name)}</span></div><div>${certBadges(z)}</div>`).join('')}</div>` +
        `<div class="note">전체 ${F.length}종 — 왼쪽 「위험물·설비」 탭</div>`, false);
    }
    h += det(`${svg('layers')} 층별 용도`, `${floors.length}개 층`, [...floors].reverse().map((f) => row(fidb(f.id), esc(f.use || f.name), '', `data-go="${esc(f.id)}"`, f.area ? `<span class="k sm">${num(f.area)}㎡</span>` : '')).join(''), false);
    $('key').innerHTML = h;
    $('rpFloor').classList.add('hide');
  }

  // ── 오른쪽: 층 상세 — 진입·계단·소화전은 펼침, 승강기·검출 수치는 접음
  function floor(fid) {
    const f = FL[fid];
    const lo = alignOf[fid] === 'low';
    $('rpHead').innerHTML = `<div class="ph"><div class="ph-t"><span class="fid big">${esc(f.id)}</span>${esc(f.name)}</div>
      <div class="ph-s">${esc(f.use || '-')}${f.area ? ` · ${num(f.area)}㎡` : ''}</div>
      ${f.grade ? `<div class="grade">⬒ 지상 출입: ${esc(f.grade)}</div>` : ''}${lo ? '<div class="grade lo">⚠ 층 위치 맞춤 신뢰도 낮음 — 위·아래 층과 위치가 어긋날 수 있음</div>' : ''}
      <div class="ph-b"><button class="pri" id="vPlan" ${f.plan ? '' : 'disabled title="원본 평면도 없음"'}>${svg('plan')}원본 평면도</button><button data-act="home">${svg('home')}전체 보기</button></div></div>`;
    const fb = BIM.floors[fid] || { doors: [] };
    const en = fb.doors.filter((d) => d[0] === 'entrance'), ex = exitsExt(fid);
    const st = stairs.filter((s) => s.floors.includes(fid)), ev = elevs.filter((e) => e.floors.includes(fid));
    const my = equips.filter((q) => q.fid === fid), hs = my.filter((q) => q.type === 'hose'), pn = my.filter((q) => q.type === 'panel'), gs = my.filter((q) => q.type === 'gas');
    let h = '<h4>🚒 소방활동 핵심</h4>';
    h += `<div class="og"><div class="ol">${svg('entrance')} 진입구·출입구</div>${en.length ? en.map((d) => row(dot('#22d3ee'), esc(short(d[5])), '', `data-fly="${d[2]},${d[3]}" data-fl="${esc(fid)}"`)).join('') : '<div class="muted sm">이 층에 지상 진입구 표기 없음</div>'}` +
      (ex.length ? `<div class="muted sm">외부로 나가는 비상구·개구부 ${ex.length}곳 (녹색 문)</div>` : '') + '</div>';
    h += `<div class="og"><div class="ol">${svg('stair')} 계단</div>${st.map((s) => row(dot(s.special ? '#facc15' : '#4ade80'), esc(s.name) + (s.special ? ` <span class="fcert located">${s.special.kind === 'sp' ? '특별피난' : '비상용'}${s.special.cert === 'cand' ? '?' : ''}</span>` : ''), `${range(s.floors)} 연결`, `data-fly="${ctr(s.box)}" data-fl="${esc(fid)}"`)).join('') || '<div class="tip">이 층에 닿는 계단을 도면에서 찾지 못함</div>'}</div>`;
    const hc = (f.fac || {})['옥내소화전'];
    h += `<div class="og"><div class="ol">${svg('hose')} 옥내소화전 <b>${hs.length}</b> <span class="muted sm">(도면 표기 · 정보카드 ${hc ?? '-'}개)</span></div><div class="chips">${hs.map((q, i) => `<button class="chip btn" data-fly="${q.x},${q.y}" data-fl="${esc(fid)}">소화전 ${i + 1}</button>`).join('') || '<span class="muted sm">도면 표기 없음 — 계단실 방수구·인접층 확인</span>'}</div></div>`;
    if (pn.length) h += `<div class="og"><div class="ol">${svg('panel')} 수신기</div>${pn.map((q) => row(dot('#a78bfa'), esc(q.name), '', `data-fly="${q.x},${q.y}" data-fl="${esc(fid)}"`)).join('')}</div>`;
    if (gs.length) h += `<div class="og"><div class="ol">${svg('gas')} 가스 차단밸브 <b>${gs.length}</b></div><div class="chips">${gs.map((q, i) => `<button class="chip btn am" data-fly="${q.x},${q.y}" data-fl="${esc(fid)}">밸브 ${i + 1}</button>`).join('')}</div></div>`;
    h += det(`${svg('ev')} 승강기`, ev.length ? `${ev.length}개소` : '없음', (ev.map((e) => row(dot('#60a5fa'), esc(e.name) + (e.cars > 1 ? ` (${e.cars}대)` : ''), `${range(e.floors)} 운행${e.note ? ' · ' + esc(e.note) : ''}`, `data-fly="${ctr(e.box)}" data-fl="${esc(fid)}"`)).join('') || '<div class="muted sm">없음</div>'), false) +
      (ev.length && !cardEm ? '<div class="tip">비상용 승강기 없음(정보카드) — 화재 시 일반 승강기 사용 금지</div>' : '');
    $('key').innerHTML = h;
    const dt = f.det || {};
    const k4 = Object.entries(dt).slice(0, 4).map(([k, v]) => `<div><b>${v}</b>${esc(k)}</div>`).join('');
    const fc = Object.entries(f.fac || {});
    $('flTitle').innerHTML = `도면 자동 검출 <span class="badge">AI</span>${fc.length ? ' · 정보카드 소방시설' : ''}`;
    $('flInfo').innerHTML = (k4 ? `<div class="kpis">${k4}</div>` : '') + '<div class="note">평면도 범례 기호를 색·형태로 찾아낸 값입니다. 도면에 표기된 것만 셉니다.</div>' +
      (fc.length ? `<h4>정보카드 소방시설</h4><div class="chips">${fc.map(([k, v]) => `<span class="chip"><b>${esc(v)}</b>${esc(k)}</span>`).join('')}</div>` : '');
    $('rpFloor').classList.remove('hide');
  }
  return { header, footer, issues, overview, haz, entry, basis, home, floor, brief30, alignOf };
}
