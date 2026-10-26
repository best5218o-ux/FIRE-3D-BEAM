// 아이콘 1벌(인라인 SVG, 오프라인) — 24×24 선 아이콘, 색 = currentColor. 같은 경로 문자열을 3D 핀(캔버스 Path2D)에도 써서 화면 전체 기호를 통일.
// 원·사각형도 path 로만 적음(Path2D 와 SVG 가 같은 자료를 씀). f: 채움(작은 점 등).
const c = (x, y, r) => `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
const rr = (x, y, w, h, r = 1.5) => `M${x + r} ${y}h${w - 2 * r}a${r} ${r} 0 0 1 ${r} ${r}v${h - 2 * r}a${r} ${r} 0 0 1 ${-r} ${r}h${-(w - 2 * r)}a${r} ${r} 0 0 1 ${-r} ${-r}v${-(h - 2 * r)}a${r} ${r} 0 0 1 ${r} ${-r}z`;
export const ICONS = {
  // 소방활동 기호
  entrance: [['M13 3h5.5a1.5 1.5 0 0 1 1.5 1.5v15a1.5 1.5 0 0 1-1.5 1.5H13'], ['M3 12h11'], ['M10 8l4 4-4 4']],
  exit: [['M11 3H5.5A1.5 1.5 0 0 0 4 4.5v15A1.5 1.5 0 0 0 5.5 21H11'], ['M10 12h11'], ['M17 8l4 4-4 4']],
  stair: [['M3 20h5v-5h5v-5h5V5h3']],
  ev: [[rr(4, 3, 16, 18, 2)], ['M8.5 10l3.5-3.5 3.5 3.5'], ['M8.5 14l3.5 3.5 3.5-3.5']],
  hose: [[rr(4, 3, 16, 18, 2)], [c(12, 12, 4.5)], [c(12, 12, 1.4), 1]],
  siamese: [[c(7, 9, 3.2)], [c(17, 9, 3.2)], ['M7 12.2v2.8h10v-2.8'], ['M12 15v6']],
  panel: [[rr(3, 4, 18, 13, 1.8)], ['M7 8.5h6'], ['M7 12.5h4'], [c(16.5, 10.5, 1.4), 1], ['M9 21h6'], ['M12 17v4']],
  gas: [[c(12, 12, 9)], ['M7 8.5l10 7v-7l-10 7z']],
  hazard: [['M12 3.2l9.5 16.8h-19z'], ['M12 9.5v5'], [c(12, 17.3, 0.9), 1]],
  life: [[c(12, 7, 3.4)], ['M5 21v-1.5A5.5 5.5 0 0 1 10.5 14h3a5.5 5.5 0 0 1 5.5 5.5V21']],
  alert: [[c(12, 12, 9)], ['M12 7.2v6'], [c(12, 16.6, 0.9), 1]],
  fire: [['M12 2.8c.8 3.6 5.8 5.2 5.8 10.6a5.8 5.8 0 0 1-11.6 0c0-2.8 1.6-4.2 2.4-6 .9 1.3 1.7 2 2.9 2.2-.4-2.2-.4-4.6.5-6.8z']],
  fac: [[c(12, 12, 8.5)], ['M12 7v5l3 2']],
  // 도구
  section: [['M12 3.5l8 4v9l-8 4-8-4v-9z'], ['M4 12l8 4 8-4', 'dash']],
  xray: [['M2 12s3.8-6.8 10-6.8S22 12 22 12s-3.8 6.8-10 6.8S2 12 2 12z'], [c(12, 12, 3)]],
  isolate: [['M12 3.5l9 4.2-9 4.2-9-4.2z', 0, 0.35], ['M3 12.2l9 4.2 9-4.2'], ['M3 16.4l9 4.2 9-4.2']],
  measure: [['M3.5 16.5L16.5 3.5l4 4-13 13z'], ['M7.5 12.5l2 2'], ['M10.5 9.5l2 2'], ['M13.5 6.5l2 2']],
  area: [[rr(4, 4, 16, 16, 1.5), 'dash'], ['M8 15l3-6 5 4z', 0, 0.35]],
  follow: [[c(6, 18, 2.2)], [c(18, 6, 2.2)], ['M8.2 18H14a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h5.8']],
  minimap: [['M9 4L3 6v14l6-2 6 2 6-2V4l-6 2z'], ['M9 4v14'], ['M15 6v14']],
  shot: [['M4 7.5h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8.5a1 1 0 0 1 1-1z'], [c(12, 13.2, 3.6)]],
  brief: [['M6 3h9l4 4v14H6z'], ['M15 3v4h4'], ['M9 12h7'], ['M9 16h7']],
  print: [['M6 9V3h12v6'], ['M6 18H4a1 1 0 0 1-1-1v-6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v6a1 1 0 0 1-1 1h-2'], ['M6 14h12v7H6z']],
  shadow: [[c(12, 12, 8.5)], ['M12 3.5a8.5 8.5 0 0 1 0 17z', 1]],
  lowq: [['M13 2.5L4.5 13.5h6.5l-1 8 8.5-11h-6.5z']],
  help: [[c(12, 12, 9)], ['M9.4 9.3a2.7 2.7 0 1 1 3.8 2.5c-.7.3-1.2 1-1.2 1.8v.6'], [c(12, 17.2, 0.9), 1]],
  theme: [[c(12, 12, 8.5)], ['M12 3.5a8.5 8.5 0 0 0 0 17z', 1]],
  sun: [[c(12, 12, 4)], ['M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6']],
  moon: [['M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z']],
  contrast: [[rr(3, 3, 18, 18, 2)], ['M3 21L21 3v18z', 1]],
  textup: [['M3 19l5.5-14h1L15 19'], ['M5.2 14h7.6'], ['M18 7v6M15 10h6']],
  textdn: [['M3 19l5.5-14h1L15 19'], ['M5.2 14h7.6'], ['M15 10h6']],
  home: [['M3 11.5l9-8 9 8'], ['M5.5 9.5V20h13V9.5']],
  layers: [['M12 3.5l9 4.2-9 4.2-9-4.2z'], ['M3 12.2l9 4.2 9-4.2'], ['M3 16.4l9 4.2 9-4.2']],
  explode: [['M12 2.5l8 3.5-8 3.5-8-3.5z'], ['M4 11.5l8 3.5 8-3.5'], ['M4 18l8 3.5 8-3.5']],
  plan: [[rr(3.5, 3.5, 17, 17, 1.5)], ['M3.5 12h8v8.5'], ['M11.5 3.5v5'], ['M15 12h5.5']],
  top: [['M12 3v9'], ['M8.5 8.5L12 12l3.5-3.5'], ['M3.5 16.5l8.5 4 8.5-4-8.5-4z']],
  iso: [['M12 3l8 4.5v9L12 21l-8-4.5v-9z'], ['M4 7.5l8 4.5 8-4.5'], ['M12 12v9']],
  play: [['M7 4.5l12.5 7.5L7 19.5z', 1]],
  pause: [['M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z', 1]],
  close: [['M6 6l12 12M18 6L6 18']],
  chev: [['M9 6l6 6-6 6']],
  info: [[c(12, 12, 9)], ['M12 11v6'], [c(12, 7.6, 0.9), 1]],
  pin: [['M12 21s-6.5-6.2-6.5-11a6.5 6.5 0 0 1 13 0c0 4.8-6.5 11-6.5 11z'], [c(12, 10, 2.3)]],
  clock: [[c(12, 12, 9)], ['M12 7v5.2l3.4 2']],
  scba: [[rr(8, 6, 8, 15, 4)], ['M10 6V3.5h4V6'], ['M8 12h8']],
  hoseRoute: [['M3 18c4 0 4-12 9-12s5 12 9 12']],
};
// SVG 문자열(버튼·패널) — 크기는 CSS(.ic) 로
export function svg(name, extra = '') {
  const parts = ICONS[name] || ICONS.info;
  const body = parts.map(([d, f, op]) => f === 1 ? `<path d="${d}" fill="currentColor" stroke="none"/>` : `<path d="${d}"${f === 'dash' ? ' stroke-dasharray="3 2.4"' : ''}${op ? ` fill="currentColor" fill-opacity="${op}"` : ''}/>`).join('');
  return `<svg class="ic${extra ? ' ' + extra : ''}" viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
}
// 캔버스에 아이콘 그리기(3D 핀 텍스처·미니맵) — (x,y) 왼쪽 위, s = 한 변(px)
export function drawIcon(ctx, name, x, y, s, color, lw = 2) {
  const parts = ICONS[name]; if (!parts) return;
  ctx.save(); ctx.translate(x, y); ctx.scale(s / 24, s / 24);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (const [d, f, op] of parts) {
    const p = new Path2D(d);
    if (f === 1) ctx.fill(p);
    else { if (op) { ctx.globalAlpha = op; ctx.fill(p); ctx.globalAlpha = 1; } ctx.setLineDash(f === 'dash' ? [3, 2.4] : []); ctx.stroke(p); }
  }
  ctx.restore();
}
