# -*- coding: utf-8 -*-
"""가상 건물 → 스캔 느낌의 층별 평면도 이미지 (정보카드 첨부 도면 스타일) + 좌표 정답"""
import io, math, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_fc = {}
def font(px):
    px = max(6, int(px))
    if px not in _fc: _fc[px] = ImageFont.truetype(FONT, px, index=1)
    return _fc[px]

GREEN, DGREEN, BLUE, RED, ORANGE = (0, 150, 70), (0, 100, 45), (0, 112, 192), (230, 0, 0), (237, 125, 49)
WALLGRAY, GRID = (182, 182, 182), (150, 150, 150)


def floor_bounds(b, f):
    xs, ys = [], []
    for r in f['rects']: xs += [r[0], r[2]]; ys += [r[1], r[3]]
    for S in b.stairs + b.elevs:
        if f['id'] in S['floors']: xs += [S['box'][0], S['box'][2]]; ys += [S['box'][1], S['box'][3]]
    return min(xs), min(ys), max(xs), max(ys)


def render_floor(b, fids, rng, legend_items, meta=None):
    """fids: 같은 도면을 쓰는 층들(기준층). 첫 층 기준으로 그림."""
    f = b.floors[fids[0]]
    SS = 2                                         # 과표본 후 축소(안티에일리어싱)
    s = rng.uniform(6.5, 11.0)
    S = s * SS
    x0, y0, x1, y1 = floor_bounds(b, f)
    ml, mr, mb, mt = rng.uniform(4, 18), rng.uniform(4, 18), rng.uniform(4, 14), rng.uniform(16, 24)
    ox, oy = x0 - ml, y1 + mt
    W, H = int((x1 + mr - ox) * S), int((oy - (y0 - mb)) * S)
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    P = lambda x, y: ((x - ox) * S, (oy - y) * S)

    # 주변 잡음: 나무·주차선
    for _ in range(rng.randint(6, 20)):
        tx = rng.uniform(ox + 1, ox + W / S - 1); ty = rng.uniform(oy - H / S + 1, oy - 1)
        if x0 - 1 < tx < x1 + 1 and y0 - 1 < ty < y1 + 1: continue
        cx, cy = P(tx, ty); r = rng.uniform(1.0, 2.2) * S
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 120, 120), width=max(1, SS))
        for a in range(0, 360, 60):
            d.line([cx, cy, cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))], fill=(140, 140, 140), width=1)
    # 그리드선 + 치수
    bay = b.meta.get('bay', 8.4)
    gx = [x0 + k * bay for k in range(int((x1 - x0) / bay) + 1)]
    gy = sorted(set([y0, y1] + [y0 + k * bay for k in range(int((y1 - y0) / bay) + 1)]))
    for x in gx:
        px, _ = P(x, 0); d.line([px, P(0, y1 + mt - 6)[1], px, H], fill=GRID, width=SS)
    for y in gy:
        _, py = P(0, y); d.line([0, py, W, py], fill=GRID, width=SS)
    dy_txt = P(0, y1 + 3.2)[1]
    for a, c in zip(gx, gx[1:]):
        txt = f'{int(round((c - a) * 1000)):,}'
        fn = font(0.9 * S)
        tw = d.textlength(txt, font=fn)
        d.text(((P(a, 0)[0] + P(c, 0)[0]) / 2 - tw / 2, dy_txt - 0.9 * S), txt, fill=(60, 60, 60), font=fn)
    # 기둥·랙
    for c in f['columns']:
        cx, cy = P(*c); r = 0.35 * S; d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 30))
    for rk in f['racks']:
        a, bb = P(rk[0], rk[3]), P(rk[2], rk[1]); d.rectangle([a[0], a[1], bb[0], bb[1]], outline=(175, 175, 175), width=1)
    # 계단 디딤판·승강기 X
    for St in b.stairs:
        if f['id'] not in St['floors']: continue
        bx0, by0, bx1, by1 = St['box']
        along_y = (by1 - by0) >= (bx1 - bx0)
        if along_y:
            mx = (bx0 + bx1) / 2; d.line([*P(mx, by0 + 1.2), *P(mx, by1 - 1.2)], fill=(40, 40, 40), width=SS)
            yy = by0 + 1.2
            while yy < by1 - 1.2:
                d.line([*P(bx0 + 0.2, yy), *P(bx1 - 0.2, yy)], fill=(70, 70, 70), width=1); yy += 0.28
        else:
            my = (by0 + by1) / 2; d.line([*P(bx0 + 1.2, my), *P(bx1 - 1.2, my)], fill=(40, 40, 40), width=SS)
            xx = bx0 + 1.2
            while xx < bx1 - 1.2:
                d.line([*P(xx, by0 + 0.2), *P(xx, by1 - 0.2)], fill=(70, 70, 70), width=1); xx += 0.28
    for E in b.elevs:
        if f['id'] not in E['floors']: continue
        ex0, ey0, ex1, ey1 = E['box']
        d.line([*P(ex0 + 0.2, ey0 + 0.2), *P(ex1 - 0.2, ey1 - 0.2)], fill=(90, 90, 90), width=1)
        d.line([*P(ex0 + 0.2, ey1 - 0.2), *P(ex1 - 0.2, ey0 + 0.2)], fill=(90, 90, 90), width=1)
    # 벽
    for w in f['walls']:
        t = w['t']
        if w['y0'] == w['y1']: a, c = P(w['x0'], w['y0'] + t / 2), P(w['x1'], w['y0'] - t / 2)
        else: a, c = P(w['x0'] - t / 2, w['y1']), P(w['x0'] + t / 2, w['y0'])
        if w['kind'] == 'part':
            d.rectangle([a[0], a[1], c[0], c[1]], fill=(35, 35, 35))
        else:
            d.rectangle([a[0], a[1], c[0], c[1]], fill=WALLGRAY, outline=(40, 40, 40), width=1)
    # 문 (호 + 문짝)
    for dr in f['doors']:
        w = dr['w']; cx, cy = P(dr['x'], dr['y']); r = w * S / 2
        if dr['o'] == 'h':
            hx = cx - r
            d.arc([hx - 2 * r, cy - 2 * r, hx + 2 * r, cy + 2 * r], 270, 360, fill=(80, 80, 80), width=1)
            d.line([hx, cy, hx, cy - 2 * r], fill=(60, 60, 60), width=1)
        else:
            hy = cy + r
            d.arc([cx - 2 * r, hy - 2 * r, cx + 2 * r, hy + 2 * r], 0, 90, fill=(80, 80, 80), width=1)
            d.line([cx, hy, cx + 2 * r, hy], fill=(60, 60, 60), width=1)
    # 실 이름 (요주의 실은 주황 테두리 + 큰 글씨)
    for rm in f['rooms']:
        xs = [p[0] for p in rm['poly']]; ys = [p[1] for p in rm['poly']]
        cx, cy = P((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        if rm['type'] == 'critical':
            a, c = P(min(xs) + 0.3, max(ys) - 0.3), P(max(xs) - 0.3, min(ys) + 0.3)
            d.rectangle([a[0], a[1], c[0], c[1]], outline=ORANGE, width=3 * SS)
            fn = font(1.5 * S)
        else:
            fn = font((0.75 if rm['type'] != 'ward' else 0.95) * S)
        tw = d.textlength(rm['name'], font=fn)
        d.text((cx - tw / 2, cy - fn.size / 2), rm['name'], fill=(20, 20, 20), font=fn)
    # 대피 방향 (빨간 선 + 화살촉)
    for line in f['evac']:
        pts = [P(*p) for p in line]
        d.line(pts, fill=RED, width=3 * SS)
        for end, prev in ((pts[0], pts[1]), (pts[-1], pts[-2])):
            ang = math.atan2(end[1] - prev[1], end[0] - prev[0]); L = 0.9 * S
            d.polygon([end, (end[0] - L * math.cos(ang - 0.45), end[1] - L * math.sin(ang - 0.45)),
                       (end[0] - L * math.cos(ang + 0.45), end[1] - L * math.sin(ang + 0.45))], fill=RED)
    # 소방 기호 (범례와 같은 모양)
    Z = rng.uniform(1.6, 2.1) * S                 # 기호 크기(px)
    for sy in f['symbols']:
        cx, cy = P(sy['x'], sy['y'])
        draw_symbol(d, sy['type'], cx, cy, Z, SS)
    # 제목·범례
    tfn = font(2.2 * S)
    title = f['ko'] if len(fids) == 1 else f"{b.floors[fids[0]]['ko']}~{b.floors[fids[-1]]['ko']}"
    tw = d.textlength(title, font=tfn)
    tx, ty = 1.2 * S, 1.2 * S
    d.rounded_rectangle([tx, ty, tx + tw + 1.6 * S, ty + 3.4 * S], radius=0.6 * S, outline=(0, 0, 0), width=2 * SS)
    d.text((tx + 0.8 * S, ty + 0.4 * S), title, fill=(0, 0, 0), font=tfn)
    lx, ly = tx + tw + 5 * S, ty
    lfn = font(1.2 * S); items = legend_items
    col_w = 9.5 * S
    lw = col_w * math.ceil(len(items) / 2) + 1.5 * S
    d.rounded_rectangle([lx, ly, lx + lw, ly + 5.4 * S], radius=0.6 * S, outline=(68, 114, 196), width=2 * SS)
    for i, (t, label) in enumerate(items):
        cx = lx + 1.6 * S + (i // 2) * col_w; cy = ly + 1.5 * S + (i % 2) * 2.4 * S
        if t == 'arrow':
            d.line([cx - 0.9 * S, cy, cx + 0.9 * S, cy], fill=RED, width=2 * SS)
            d.polygon([(cx + 1.1 * S, cy), (cx + 0.5 * S, cy - 0.4 * S), (cx + 0.5 * S, cy + 0.4 * S)], fill=RED)
        else:
            draw_symbol(d, t, cx, cy, 1.5 * S, SS)
        d.text((cx + 1.4 * S, cy - lfn.size * 0.6), label, fill=(0, 0, 0), font=lfn)

    # 축소 + 스캔 열화
    img = img.resize((W // SS, H // SS), Image.LANCZOS)
    if rng.random() < 0.6: img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.6)))
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rng.randint(55, 90)); buf.seek(0)
    img = Image.open(buf).convert('RGB')
    tf = dict(s=s, ox=ox, oy=oy, w=img.width, h=img.height)
    if meta is not None: meta.update(s=s, ox=ox, oy=oy, mt=mt, lx=lx / SS, ly=ly / SS, lw=lw / SS, W=img.width, H=img.height, x1=x1, y1=y1, y0=y0)
    return img, tf


def draw_symbol(d, t, cx, cy, Z, SS):
    h = Z / 2
    box = [cx - h, cy - h, cx + h, cy + h]
    if t == 'exit':
        d.rectangle(box, fill=GREEN, outline=DGREEN, width=SS)
        w = Z * 0.12
        d.rectangle([cx + h * 0.25, cy - h * 0.7, cx + h * 0.75, cy + h * 0.8], outline='white', width=max(1, int(w / 2)))
        d.ellipse([cx - h * 0.35, cy - h * 0.75, cx - h * 0.05, cy - h * 0.45], fill='white')
        d.line([cx - h * 0.2, cy - h * 0.4, cx - h * 0.3, cy + h * 0.2], fill='white', width=max(1, int(w)))
        d.line([cx - h * 0.3, cy + h * 0.2, cx - h * 0.6, cy + h * 0.75], fill='white', width=max(1, int(w)))
        d.line([cx - h * 0.3, cy + h * 0.2, cx + h * 0.05, cy + h * 0.75], fill='white', width=max(1, int(w)))
        d.line([cx - h * 0.65, cy - h * 0.2, cx + h * 0.1, cy - h * 0.1], fill='white', width=max(1, int(w)))
    elif t == 'hose':
        b2 = [cx - h, cy - h * 0.6, cx + h, cy + h * 0.6]
        d.rectangle(b2, fill='white', outline=DGREEN, width=SS)
        d.polygon([(b2[2], b2[1]), (b2[2], b2[3]), (b2[0], b2[3])], fill=GREEN)
    elif t == 'panel':
        b2 = [cx - h, cy - h * 0.7, cx + h, cy + h * 0.7]
        d.rectangle(b2, fill=GREEN, outline=DGREEN, width=SS)
        for k in (1, 2):
            yy = b2[1] + (b2[3] - b2[1]) * k / 3; d.line([b2[0] + 2, yy, b2[2] - 2, yy], fill='white', width=SS)
        d.line([cx, b2[1] + 2, cx, b2[3] - 2], fill='white', width=SS)
        d.line([b2[0] + 2, b2[1] + 2, b2[2] - 2, b2[3] - 2], fill='white', width=SS)
    elif t == 'gas':
        r = h * 0.55
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0), width=2 * SS)
        for a in (90, 210, 330):
            d.line([cx, cy, cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))], fill=(0, 0, 0), width=2 * SS)
    elif t == 'ev':
        # 승강기 표기: 파란 테두리 + EV 글자 (실제 정보카드 도면 표기 방식)
        b2 = [cx - h * 0.9, cy - h * 0.7, cx + h * 0.9, cy + h * 0.7]
        d.rectangle(b2, outline=BLUE, width=3 * SS)
        fn = font(h * 0.9); tw = d.textlength('EV', font=fn)
        d.text((cx - tw / 2, cy - fn.size * 0.62), 'EV', fill=(0, 0, 0), font=fn)


MAGENTA = (205, 0, 190)


def draw_fac_symbol(d, t, cx, cy, Z):
    """소방시설 추가 기호(자홍 단색 채움): 소화기 ▲ · 방수구 ◯(굵은 고리) · 비상콘센트 ■ · 송수구 ◆"""
    h = Z / 2
    if t == 'ext': d.polygon([(cx, cy - h), (cx + h, cy + h), (cx - h, cy + h)], fill=MAGENTA)
    elif t == 'outlet': d.ellipse([cx - h, cy - h, cx + h, cy + h], outline=MAGENTA, width=max(2, int(round(Z * 0.3))))
    elif t == 'socket': d.rectangle([cx - h * 0.85, cy - h * 0.85, cx + h * 0.85, cy + h * 0.85], fill=MAGENTA)
    elif t == 'siamese': d.polygon([(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)], fill=MAGENTA)


def _stamp(img, t, cx, cy, Z, col=MAGENTA, check=False):
    """최종 해상도 도면에 기호 하나를 4배 과표본으로 그려 알파 합성 — 기호 밖 화소는 바이트 단위로 그대로.
    check=True: 기호가 덮을 화소(+1px)에 무채색 선·글자(벽·그리드·나무 등)가 있으면 그리지 않고 False"""
    import numpy as np
    K = 4; n = int(math.ceil(Z)) + 4
    pat = Image.new('L', (n * K, n * K), 0); dd = ImageDraw.Draw(pat)
    c0 = n * K / 2 + (cx - int(cx)) * K - K / 2; c1 = n * K / 2 + (cy - int(cy)) * K - K / 2
    fill_ = 255
    class _D:                                   # draw_fac_symbol 이 쓰는 도형 호출을 알파 판에 그대로
        def polygon(self, pts, fill=None): dd.polygon(pts, fill=fill_)
        def ellipse(self, box, fill=None, outline=None, width=1): dd.ellipse(box, fill=fill_ if fill else None, outline=fill_ if outline else None, width=width)
        def rectangle(self, box, fill=None): dd.rectangle(box, fill=fill_)
    draw_fac_symbol(_D(), t, c0, c1, Z * K)
    a = pat.resize((n, n), Image.LANCZOS)
    x0, y0 = int(cx) - n // 2 + 1, int(cy) - n // 2 + 1
    if check:
        if x0 < 1 or y0 < 1 or x0 + n >= img.width - 1 or y0 + n >= img.height - 1: return False
        A = np.asarray(a) > 0
        A = A | np.roll(A, 1, 0) | np.roll(A, -1, 0) | np.roll(A, 1, 1) | np.roll(A, -1, 1)
        R = np.asarray(img.crop((x0, y0, x0 + n, y0 + n))).astype(int)
        lum = R.mean(2); sat = R.max(2) - R.min(2)
        if ((lum < 238) & (sat < 45) & A).any(): return False
    solid = Image.new('RGB', (n, n), col)
    img.paste(solid, (x0, y0), a)
    return True


def draw_fac_overlay(img, meta, syms, legend, b, fids):
    """스캔 열화 뒤 도면에 소방시설 기호 + 별도 범례 상자(자홍 테두리)를 덧그림 → 그린 경우 True.
    기존 범례(파란 상자)·도면 화소는 기호·범례 글자가 놓인 곳 말고는 바뀌지 않는다(기존 인식 결과 불변 목적).
    범례 자리가 없으면(좁은 도면) 아무것도 그리지 않는다(정보카드 수량만 → 추정/수량 표시)."""
    if not syms or not legend: return None
    import numpy as np
    s = meta['s']; W, H = meta['W'], meta['H']
    P = lambda X, Y: ((X - meta['ox']) * s, (meta['oy'] - Y) * s)
    out = img.copy()
    drawn = [(t, X, Y) for t, X, Y in syms if _stamp(out, t, *P(X, Y), 1.4 * s, check=True)]
    if not drawn: return None
    used = {t for t, _, _ in drawn}; legend = [(t, lb) for t, lb in legend if t in used]
    d = ImageDraw.Draw(out)
    lfn = font(1.0 * s)
    col_w = max(d.textlength(lb, font=lfn) for _, lb in legend) + 2.8 * s
    # 범례 자리: 도면 아래쪽 절반(치수 OCR 는 위쪽 절반만 읽음)에서 무채색 잉크(나무·그리드선·벽·글자)가 전혀 없는 빈 사각형
    R = np.asarray(out).astype(np.int16)
    ink = ((R.mean(2) < 242) | ((R.max(2) - R.min(2)) > 20)).astype(np.int32)
    I = np.pad(ink.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ok = None
    for rows in sorted({len(legend), 1, 2}):
        ncol = math.ceil(len(legend) / rows)
        w_, h_ = int(ncol * col_w + 0.8 * s), int((0.5 + 2.1 * rows) * s)
        pad = 4
        for y in range(int(0.55 * H), H - h_ - pad, 2):
            for x in range(pad, W - w_ - pad, 2):
                y0_, y1_, x0_, x1_ = y - pad, y + h_ + pad, x - pad, x + w_ + pad
                if I[y1_, x1_] - I[y0_, x1_] - I[y1_, x0_] + I[y0_, x0_] == 0: ok = (x, y, w_, h_, rows); break
            if ok: break
        if ok: break
    if ok is None: return None
    x, y, w_, h_, rows = ok
    d.rounded_rectangle([x, y, x + w_, y + h_], radius=0.5 * s, outline=MAGENTA, width=1)
    for j, (t, lb) in enumerate(legend):
        c_, r_ = j // rows, j % rows
        cx = x + 1.2 * s + c_ * col_w; cy = y + 1.3 * s + r_ * 2.1 * s
        _stamp(out, t, cx, cy, 1.4 * s)
        d.text((cx + 1.1 * s, cy - lfn.size * 0.6), lb, fill=(0, 0, 0), font=lfn)
    img.paste(out)
    return drawn
