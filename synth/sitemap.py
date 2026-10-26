# -*- coding: utf-8 -*-
"""가상 진압작전도(정보카드 ⑪ '0 진압작전도') 생성 + 정답.

표준양식 ⑪ 진압작전도는 범례가 필수: 주출입구 · 비상구 · 소화전 · 연결송수구 · 이격거리.
실제 작전도처럼 위에서 본 배치도 — 대상 건물 외곽(1층), 인접 건물, 주변 도로(폭 표기), 소방차 진입로,
외벽의 주출입구·비상구 표시, 소화전(지상식·지하식, 도면 밖이면 방향 화살표 + 거리), 외벽 연결송수구,
이격거리 치수선(소화전→송수구, 건물 간격, 도로 이격), 방위표, 축척 막대(없을 때도 있음), 범례 상자(5종, 건물마다 아이콘 모양·색이 다름),
스캔 열화(흐림·잡음·JPEG).

부지 배치는 synth.extras.make_site 와 같은 난수로 만든다 → 같은 건물의 site.json 과 도로·소화전·송수구가 일치(정답 공유).
도면에만 있는 것(인접 건물, 도면 안 소화전 가상-3·4, 이격거리 치수)은 정답 dict 의 'site_true' 에 합쳐 둔다.

사용:  from synth.sitemap import sitemap_image
       img, gt = sitemap_image(b)            # b = synth.model 건물(B) 또는 gt.json dict. img = PIL RGB
       python -m synth.sitemap data/core_1404 [--seed 7] [--out x.png]   (시험 — 기본 출력 out_sitemap/<건물>_sitemap.png)
정답 좌표: 'm' = 건물 정답 좌표계(동 +x, 북 +y), 'px' = 최종 이미지 화소."""
import io, json, math, os, random, re, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from shapely.geometry import box, Point, Polygon, LineString
from shapely.ops import unary_union, nearest_points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from synth.extras import make_site

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
_fc = {}


def font(px, bold=False):
    k = (int(px), bold)
    if k not in _fc:
        p = FONTB if bold and os.path.exists(FONTB) else FONT
        _fc[k] = ImageFont.truetype(p, int(px), index=1)
    return _fc[k]


TYPES = ['main', 'exit', 'hyd', 'fdc', 'dim']
WORDS = dict(main='주출입구', exit='비상구', hyd='소화전', fdc='연결송수구', dim='이격거리')
SHAPES = dict(main=['tri', 'arrow', 'pent'], exit=['square', 'tri', 'arrow', 'diamond', 'squarex'],
              hyd=['circle', 'circlex', 'dcircle', 'ring', 'squarex'], fdc=['Y', 'twin', 'diamond', 'square', 'hex', 'star'])
COLORS = {'red': (220, 30, 30), 'blue': (30, 70, 210), 'green': (20, 150, 60), 'orange': (240, 130, 0), 'magenta': (200, 30, 170),
          'purple': (120, 50, 190), 'teal': (0, 150, 150), 'brown': (140, 80, 30)}
PASTEL = [(255, 236, 205), (225, 238, 255), (235, 250, 225), (255, 228, 235), (245, 240, 215)]


def seed_of(G):
    m = re.search(r'(\d+)$', G['name']); return int(m.group(1)) if m else 0


def _site_for(G):
    """synth.extras.extend 와 같은 난수 → site.json 과 같은 배치"""
    d = f"{G['btype']}_{seed_of(G)}"
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(d))
    return make_site(G, random.Random(seed + 1))


def pick_style(rng):
    """종류별 아이콘 (모양, 색). 한 도면 안에서 (모양,색) 이 겹치지 않고, 같은 색이면 모양이 달라야 함"""
    st = {}
    used = []
    cols = list(COLORS)
    for t in ('main', 'exit', 'hyd', 'fdc'):
        for _ in range(200):
            sh = rng.choice(SHAPES[t])
            if t == 'hyd': c = rng.choice(['red', 'red', 'blue', 'magenta', 'orange'])
            elif t == 'exit': c = rng.choice(['green', 'green', 'teal', 'blue', 'purple'])
            else: c = rng.choice(cols)
            fam = {'tri': 'tri', 'arrow': 'arrow', 'pent': 'arrow', 'square': 'sq', 'squarex': 'sq', 'diamond': 'dia', 'circle': 'cir', 'circlex': 'cir',
                   'dcircle': 'cir', 'ring': 'ring', 'Y': 'Y', 'twin': 'twin', 'hex': 'hex', 'star': 'star'}[sh]
            if any(u[1] == c and (u[2] == fam or rng.random() < 0.7) for u in used): continue       # 같은 색은 드물게, 같은 모양 계열이면 금지
            used.append((t, c, fam)); st[t] = dict(shape=sh, color=c, rgb=COLORS[c]); break
    dc = rng.choice([c for c in ('blue', 'red', 'magenta', 'green', 'orange', 'purple')])
    st['dim'] = dict(shape='dim', color=dc, rgb=COLORS[dc], dashed=rng.random() < 0.3, head=rng.choice(['arrow', 'tick', 'arrow']))
    return st


def draw_icon(d, shape, rgb, cx, cy, s, ang=0.0):
    """아이콘(중심 cx,cy, 크기 s px). ang: 방향형 아이콘이 가리키는 방향(화면 각, 0=오른쪽, 시계방향 +, 도)"""
    r = s / 2
    a = math.radians(ang)
    def rot(pts): return [(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)) for x, y in pts]
    W = (255, 255, 255)
    if shape == 'tri': d.polygon(rot([(r, 0), (-r, -r * 0.9), (-r, r * 0.9)]), fill=rgb)
    elif shape == 'arrow': d.polygon(rot([(r, 0), (0, -r), (0, -r * 0.45), (-r, -r * 0.45), (-r, r * 0.45), (0, r * 0.45), (0, r)]), fill=rgb)
    elif shape == 'pent': d.polygon(rot([(r, 0), (r * 0.2, -r * 0.8), (-r, -r * 0.8), (-r, r * 0.8), (r * 0.2, r * 0.8)]), fill=rgb)
    elif shape == 'square': d.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=rgb)
    elif shape == 'squarex':
        d.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=rgb)
        w = max(2, int(s * 0.12))
        d.line([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], fill=W, width=w); d.line([cx - r * 0.55, cy + r * 0.55, cx + r * 0.55, cy - r * 0.55], fill=W, width=w)
    elif shape == 'diamond': d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb)
    elif shape == 'circle': d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == 'circlex':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb); w = max(2, int(s * 0.13))
        d.line([cx - r * 0.6, cy, cx + r * 0.6, cy], fill=W, width=w); d.line([cx, cy - r * 0.6, cx, cy + r * 0.6], fill=W, width=w)
    elif shape == 'dcircle':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb); d.ellipse([cx - r * 0.62, cy - r * 0.62, cx + r * 0.62, cy + r * 0.62], fill=W)
        d.ellipse([cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38], fill=rgb)
    elif shape == 'ring':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb); d.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], fill=W)
    elif shape == 'Y':
        w = max(3, int(s * 0.2))
        d.line([cx, cy + r, cx, cy], fill=rgb, width=w); d.line([cx, cy, cx - r * 0.8, cy - r * 0.85], fill=rgb, width=w); d.line([cx, cy, cx + r * 0.8, cy - r * 0.85], fill=rgb, width=w)
        d.ellipse([cx - w * 0.6, cy - w * 0.6, cx + w * 0.6, cy + w * 0.6], fill=rgb)
    elif shape == 'twin':
        q = r * 0.52
        d.ellipse([cx - 2 * q, cy - q, cx, cy + q], fill=rgb); d.ellipse([cx, cy - q, cx + 2 * q, cy + q], fill=rgb)
        d.rectangle([cx - q * 0.4, cy - q * 1.6, cx + q * 0.4, cy], fill=rgb)
    elif shape == 'hex': d.polygon([(cx + r * math.cos(math.radians(60 * k)), cy + r * math.sin(math.radians(60 * k))) for k in range(6)], fill=rgb)
    elif shape == 'star':
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45; t = math.radians(-90 + 36 * k); pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
        d.polygon(pts, fill=rgb)


def draw_dim(d, p0, p1, st, width, head_len):
    rgb = st['rgb']
    L = math.dist(p0, p1)
    if L < 1: return
    ux, uy = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
    if st.get('dashed'):
        k, on = 0.0, True
        seg = max(6, head_len * 0.9)
        while k < L:
            e = min(L, k + seg)
            if on: d.line([p0[0] + ux * k, p0[1] + uy * k, p0[0] + ux * e, p0[1] + uy * e], fill=rgb, width=width)
            k = e + (seg * 0.5 if on else 0); on = not on
            if not on: continue
    else:
        d.line([p0, p1], fill=rgb, width=width)
    for (q, sg) in ((p0, 1), (p1, -1)):
        if st.get('head') == 'tick':
            nx, ny = -uy, ux
            d.line([q[0] + (nx - ux * sg) * head_len * 0.6, q[1] + (ny - uy * sg) * head_len * 0.6,
                    q[0] - (nx - ux * sg) * head_len * 0.6, q[1] - (ny - uy * sg) * head_len * 0.6], fill=rgb, width=width)
        else:
            bx, by = q[0] + ux * sg * head_len, q[1] + uy * sg * head_len
            nx, ny = -uy * head_len * 0.45, ux * head_len * 0.45
            d.polygon([q, (bx + nx, by + ny), (bx - nx, by - ny)], fill=rgb)


def _text(d, xy, t, px, fill=(0, 0, 0), anchor='mm', bold=False, bg=None):
    f = font(px, bold)
    if bg is not None:
        bb = d.textbbox(xy, t, font=f, anchor=anchor); d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=bg)
    d.text(xy, t, font=f, fill=fill, anchor=anchor)
    return d.textbbox(xy, t, font=f, anchor=anchor)


def _dimvar_text(img, d, rv, v, qa, qb, tp, fpx, fill, bg, SS, dimvar=True):
    """[dimvar] 치수 글자 변형: 형식('12,5m' 소수 쉼표·'약 12m'·'12미터'·'12.0m'·'L=12m'…) + 배치(가로 / 선 방향으로 돌림 / 선 위에 흰 바탕으로 끊고 얹음).
    → (글자, 2배 화소 상자)"""
    vt = f"{v:g}"
    if dimvar == 'v2':                                   # v2(dev_var2): '약'·'~' 앞붙임, 소수 쉼표·점, 미터, km, 천 단위 쉼표 mm 를 많이
        pre = rv.choice(['', '', '약 ', '약 ', '약', '~', '약 '])
        body = rv.choice(['{}m', '{} m', 'comma', 'comma', 'dot0', 'dot0', '{}미터', '{} 미터', 'dot0미터', 'km', 'mm'])
        if body == 'mm': pre = ''                        # 천 단위 쉼표 mm(단위 없음): '12,500'
        t = pre + ((f"{v:.1f}".replace('.', ',') + 'm') if body == 'comma' else f"{v:.1f}m" if body == 'dot0' else f"{v:.1f}미터" if body == 'dot0미터'
                   else f"{v / 1000:.3f}km" if body == 'km' else f"{int(round(v * 1000)):,}" if body == 'mm' else body.format(vt))
    else:
        fmt = rv.choice(['{}m', '{} m', '{}M', 'L={}m', '약 {}m', '약{}m', '{}미터', 'comma', 'dot0'])
        t = (f"{v:.1f}".replace('.', ',') + 'm') if fmt == 'comma' else (f"{v:.1f}m" if fmt == 'dot0' else fmt.format(vt))
    mode = rv.choice(['horiz', 'along', 'along', 'along_gap'])
    mx, my = (qa[0] + qb[0]) / 2, (qa[1] + qb[1]) / 2
    L = math.dist(qa, qb) or 1; nx, ny = -(qb[1] - qa[1]) / L, (qb[0] - qa[0]) / L
    sg = 1 if (tp[0] - mx) * nx + (tp[1] - my) * ny >= 0 else -1                           # 원래 글자 쪽
    f = font(fpx * SS)
    if mode == 'horiz':                                                                 # 바뀐 글자 폭으로 선에서 띄움
        bb = d.textbbox((0, 0), t, font=f, anchor='mm'); tw, th = (bb[2] - bb[0]) / SS, (bb[3] - bb[1]) / SS
        off = abs(nx) * tw / 2 + abs(ny) * th / 2 + rv.uniform(5, 9)
        return t, _text(d, ((mx + sg * nx * off) * SS, (my + sg * ny * off) * SS), t, fpx * SS, fill=fill, bg=bg)
    ang = math.degrees(math.atan2(qb[1] - qa[1], qb[0] - qa[0])); ang = (ang + 90) % 180 - 90       # 읽을 수 있는 방향
    bb = d.textbbox((0, 0), t, font=f, anchor='mm'); tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 3 * SS
    timg = Image.new('RGBA', (int(tw + 2 * pad), int(th + 2 * pad)), (0, 0, 0, 0)); td = ImageDraw.Draw(timg)
    if mode == 'along_gap' or bg is not None: td.rectangle([0, 0, timg.width - 1, timg.height - 1], fill=(255, 255, 255, 255))
    td.text((timg.width / 2, timg.height / 2), t, font=f, fill=tuple(fill) + (255,), anchor='mm')
    rot = timg.rotate(-ang, expand=True, resample=Image.BICUBIC)
    if mode == 'along':                                    # 선 옆(글자 상자가 선에 닿지 않게), 원래 글자 쪽
        off = th / SS / 2 + pad / SS + rv.uniform(2, 5)
        mx, my = mx + sg * nx * off, my + sg * ny * off
    x0, y0 = int(mx * SS - rot.width / 2), int(my * SS - rot.height / 2)
    img.paste(rot, (x0, y0), rot)
    a = np.asarray(rot)[..., 3]; ys, xs = np.where(a > 0)
    return t, (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


def sitemap_image(b, rng=None, planted=True, dimvar=None):
    """→ (PIL RGB 이미지, 정답 dict). b: synth.model.B 또는 gt.json dict
    dimvar(또는 환경변수 SITEMAP_DIMVAR=1): 치수 글자 변형(기울어진·세로 글자, 선 위 글자, 소수 쉼표, '약', '미터'). 별도 난수라 끄면 기존 그림과 같고,
    dimvar='v2'(SITEMAP_DIMVAR=v2): 형식을 '약'·'~'·소수 쉼표/점·미터·km·천 단위 쉼표 mm 위주로(배치는 같음, dimvar=True 그림은 그대로).
    켜도 치수 글자 말고는 같은 그림."""
    G = b if isinstance(b, dict) else b.to_json()
    if rng is None: rng = random.Random(seed_of(G) * 31 + 7)
    if dimvar is None: dimvar = {'1': True, 'v2': 'v2'}.get(os.environ.get('SITEMAP_DIMVAR'), False)
    rv = random.Random(hash(tuple(rng.getstate()[1][:8])) ^ 0x5D1A) if dimvar else None       # 기본 난수 흐름을 건드리지 않는 별도 난수
    site, sgt = _site_for(G)
    f1 = G['floors']['1F']
    fp = unary_union([box(*r) for r in f1['rects']])
    x0, y0, x1, y1 = fp.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    st = pick_style(rng)
    SS = 2                                                           # 2배로 그린 뒤 축소(선 매끄럽게)
    W, H = rng.choice([(1500, 1060), (1400, 1000), (1600, 1130), (1300, 950)])
    leg_pos = rng.choice(['right', 'bottom', 'inset'])
    title = rng.random() < 0.6
    top = 60 if title else 14
    if leg_pos == 'right': MR = [14, top, W - 250, H - 14]
    elif leg_pos == 'bottom': MR = [14, top, W - 14, H - 120]
    else: MR = [14, top, W - 14, H - 14]
    mw, mh = MR[2] - MR[0], MR[3] - MR[1]
    # 방위: 55% 북쪽 위, 20% 약간 돌림, 25% 90°배수
    u = rng.random()
    theta = 0.0 if u < 0.55 else (rng.choice([-1, 1]) * rng.uniform(6, 25) if u < 0.75 else rng.choice([90.0, -90.0, 180.0]))
    th = math.radians(theta)
    # 축척: 건물(회전 외접)이 지도 짧은 변의 30~50%
    cs = [(x, y) for x in (x0, x1) for y in (y0, y1)]
    ext_u = max(abs((x - cx) * math.cos(th) + (y - cy) * math.sin(th)) for x, y in cs) * 2
    ext_v = max(abs((x - cx) * math.sin(th) - (y - cy) * math.cos(th)) for x, y in cs) * 2
    s = min(mw * rng.uniform(0.3, 0.5) / max(ext_u, 1), mh * rng.uniform(0.3, 0.5) / max(ext_v, 1))
    s = min(s, 14.0, (min(mw, mh) - 40) / (max(ext_u, ext_v) + 70))       # 도로(최대 ~35m 떨어짐)까지 보이게
    u0 = MR[0] + mw / 2 + rng.uniform(-0.08, 0.08) * mw
    v0 = MR[1] + mh / 2 + rng.uniform(-0.08, 0.08) * mh
    def P(x, y):
        dx, dy = x - cx, y - cy
        return (u0 + s * (dx * math.cos(th) + dy * math.sin(th)), v0 + s * (dx * math.sin(th) - dy * math.cos(th)))
    def Pinv(uu, vv):
        a, c = (uu - u0) / s, (vv - v0) / s
        return (cx + a * math.cos(th) + c * math.sin(th), cy + a * math.sin(th) - c * math.cos(th))
    view = Polygon([Pinv(MR[0], MR[1]), Pinv(MR[2], MR[1]), Pinv(MR[2], MR[3]), Pinv(MR[0], MR[3])])
    Z = int(rng.uniform(22, 32))                                       # 아이콘 크기(px)
    zmap = Z * rng.uniform(0.85, 1.1)                                   # 도면 안 아이콘 크기
    img = Image.new('RGB', (W * SS, H * SS), (255, 255, 255))
    d = ImageDraw.Draw(img)
    S2 = lambda p: (p[0] * SS, p[1] * SS)
    PS = lambda x, y: S2(P(x, y))
    # 도로
    roads = site['roads']
    road_fill = rng.choice([(205, 205, 205), (215, 215, 215), (222, 222, 222), (198, 200, 204)])
    rpolys = []
    for r in roads:
        rp = LineString(r['line']).buffer(r['width'] / 2, cap_style=2)
        rpolys.append(rp)
    rU = unary_union(rpolys)
    for g in (rU.geoms if rU.geom_type == 'MultiPolygon' else [rU]):
        d.polygon([PS(*p) for p in g.exterior.coords], fill=road_fill, outline=(60, 60, 60), width=2 * SS)
    for r in roads:                                                    # 넓은 도로 중앙선
        if r['width'] >= 10:
            a, bb = r['line'][0], r['line'][-1]
            d.line([PS(*a), PS(*bb)], fill=rng.choice([(240, 200, 40), (255, 255, 255)]), width=2 * SS)
    # 인접 건물
    neigh = []
    tgt_buf = fp.buffer(4)
    for k in range(rng.randint(1, 3)):
        for _ in range(60):
            w_, h_ = rng.uniform(10, 35), rng.uniform(10, 30)
            side = rng.choice(['N', 'E', 'W', 'N'])
            g = rng.uniform(4, 14)
            if side == 'N': nx0, ny0 = rng.uniform(x0 - 15, x1 - 5), y1 + g
            elif side == 'E': nx0, ny0 = x1 + g, rng.uniform(y0, y1)
            else: nx0, ny0 = x0 - g - w_, rng.uniform(y0, y1)
            nb = box(nx0, ny0, nx0 + w_, ny0 + h_)
            if nb.intersects(rU.buffer(1.5)) or nb.intersects(tgt_buf) or any(nb.buffer(3).intersects(q) for q in neigh): continue
            if not view.buffer(-5).contains(nb.centroid): continue
            neigh.append(nb); break
    nfill = rng.choice([(245, 245, 245), (236, 236, 236), (255, 255, 255)])
    for nb in neigh:
        d.polygon([PS(*p) for p in nb.exterior.coords], fill=nfill, outline=(90, 90, 90), width=2 * SS)
    # 대상 건물
    bfill = rng.choice(PASTEL)
    for g in (fp.geoms if fp.geom_type == 'MultiPolygon' else [fp]):
        d.polygon([PS(*p) for p in g.exterior.coords], fill=bfill, outline=(20, 20, 20), width=3 * SS)
    if rng.random() < 0.4:                                            # 빗금
        bx0, by0, bx1, by1 = fp.bounds; step = rng.uniform(2.5, 4.0)
        k = bx0 - (by1 - by0)
        while k < bx1:
            ln = LineString([(k, by0), (k + (by1 - by0), by1)]).intersection(fp.buffer(-0.4))
            if not ln.is_empty and ln.geom_type == 'LineString': d.line([PS(*p) for p in ln.coords], fill=(170, 170, 170), width=SS)
            k += step
    bc = P(cx, cy)
    npx = int(rng.uniform(15, 20))
    if font(npx, True).getlength(G['name']) < 0.8 * min(ext_u, ext_v) * s:          # 건물 안에 들어갈 때만 이름
        _text(d, S2((bc[0], bc[1] - (6 if rng.random() < 0.5 else -6))), G['name'], npx * SS, bold=True)
    # 소방차 진입로(전면 도로 위 굵은 점선 화살표)
    R1 = roads[0]; yR1 = R1['line'][0][1]
    ta = (cx - rng.uniform(30, 55) * rng.choice([1, -1]), yR1)
    tb = (cx, yR1)
    tcol = (100, 100, 100)
    pa, pb = P(*ta), P(*tb)
    Lp = math.dist(pa, pb)
    for k in range(0, int(Lp), 16):
        e = min(Lp, k + 10); ux, uy = (pb[0] - pa[0]) / Lp, (pb[1] - pa[1]) / Lp
        d.line([S2((pa[0] + ux * k, pa[1] + uy * k)), S2((pa[0] + ux * e, pa[1] + uy * e))], fill=tcol, width=3 * SS)
    ux, uy = (pb[0] - pa[0]) / Lp, (pb[1] - pa[1]) / Lp
    d.polygon([S2(pb), S2((pb[0] - ux * 14 - uy * 7, pb[1] - uy * 14 + ux * 7)), S2((pb[0] - ux * 14 + uy * 7, pb[1] - uy * 14 - ux * 7))], fill=tcol)
    _text(d, S2(((pa[0] + pb[0]) / 2 - uy * 16, (pa[1] + pb[1]) / 2 + ux * 16)), '소방차 진입로', 14 * SS, fill=(70, 70, 70))
    # 도로 이름·폭 글자
    labels = []
    for r in roads:
        ln = LineString(r['line']).intersection(view.buffer(-8))
        if ln.is_empty: continue
        if ln.geom_type != 'LineString': ln = max(ln.geoms, key=lambda q: q.length)
        fr = rng.uniform(0.2, 0.8); q = ln.interpolate(fr, normalized=True)
        a, bb = r['line'][0], r['line'][-1]
        L_ = math.dist(a, bb); nrm = (-(bb[1] - a[1]) / L_, (bb[0] - a[0]) / L_)
        wtxt = f"{r['width']:g}m"
        t = rng.choice([f"{r['name']} (폭 {wtxt})", f"{r['name']} {wtxt}", f"도로폭 {wtxt}", f"B={wtxt}"])
        tw_px = font(14 * SS).getlength(t) / SS
        along_h = abs(math.cos(math.atan2(*(np.subtract(P(*r['line'][-1]), P(*r['line'][0]))[::-1]))))     # 도로가 화면 가로에 가까운 정도
        fits = r['width'] * s >= 26 and (along_h > 0.7 or tw_px < r['width'] * s - 4)
        off = 0 if fits else (r['width'] / 2 + 12 / s)
        side_ = 1 if (fp.centroid.x - q.x) * nrm[0] + (fp.centroid.y - q.y) * nrm[1] < 0 else -1
        qq = (q.x + nrm[0] * off * side_, q.y + nrm[1] * off * side_)
        bb_ = _text(d, S2(P(*qq)), t, 14 * SS, fill=(30, 30, 30), bg=road_fill if off == 0 else None)       # 도로 밖 글자는 바탕 없이(외곽선을 가리지 않게)
        labels.append(dict(road=r['id'], text=t, width=r['width'], px=[v / SS for v in bb_]))
    # 점 기호 정답 수집 ------------------------------------------------------
    items = []
    def outward(x, y):
        """외벽 위 점 → 바깥 법선(m)"""
        best = None
        for (ax, ay, bx_, by_, n) in [(x0, y0, x1, y0, (0, -1)), (x1, y0, x1, y1, (1, 0)), (x0, y1, x1, y1, (0, 1)), (x0, y0, x0, y1, (-1, 0))]:
            dd = LineString([(ax, ay), (bx_, by_)]).distance(Point(x, y))
            if best is None or dd < best[0]: best = (dd, n)
        return best[1]
    def scr_ang(n):                                                    # 부지 방향 벡터 → 화면 각(도)
        p0, p1 = P(0, 0), P(n[0], n[1]); return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    fac_name = {'N': '북측면', 'E': '동측면', 'S': '남측면', 'W': '서측면'}
    def facade_of(n): return {(0, -1): '남측면', (1, 0): '동측면', (0, 1): '북측면', (-1, 0): '서측면'}[tuple(n)]
    doors = [q for q in f1['doors'] if (q['kind'] == 'entrance' and q['name'] == '주출입구') or q['kind'] == 'exit']
    doors = [q for q in doors if min(abs(q['x'] - x0), abs(q['x'] - x1), abs(q['y'] - y0), abs(q['y'] - y1)) < 0.4]
    plant = []
    marks = []
    for q in doors:
        t = 'main' if q['kind'] == 'entrance' else 'exit'
        marks.append(dict(type=t, x=q['x'], y=q['y']))
    if planted and rng.random() < 0.25:                               # 도면과 작전도 불일치 심기
        exits = [m for m in marks if m['type'] == 'exit']
        if exits and rng.random() < 0.5:
            m = rng.choice(exits); marks.remove(m); plant.append(dict(kind='sitemap_exit_missing', x=m['x'], y=m['y']))
        else:
            for _ in range(30):                                        # 도면에 없는 외벽 위치에 비상구 표시
                e = rng.choice([((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))])
                fr = rng.uniform(0.15, 0.85); xx, yy = e[0][0] + (e[1][0] - e[0][0]) * fr, e[0][1] + (e[1][1] - e[0][1]) * fr
                if all(math.dist((xx, yy), (m['x'], m['y'])) > 8 for m in marks) and all(math.dist((xx, yy), (q2['x'], q2['y'])) > 6 for q2 in f1['doors']):
                    marks.append(dict(type='exit', x=round(xx, 2), y=round(yy, 2))); plant.append(dict(kind='sitemap_exit_extra', x=round(xx, 2), y=round(yy, 2))); break
    for m in marks:
        n = outward(m['x'], m['y'])
        off = zmap * 0.75 / s
        ix, iy = m['x'] + n[0] * off, m['y'] + n[1] * off
        ang = scr_ang((-n[0], -n[1])) if m['type'] == 'main' else scr_ang(n)          # 주출입구: 안쪽, 비상구: 바깥쪽
        items.append(dict(type=m['type'], x=m['x'], y=m['y'], icon=P(ix, iy), ang=ang, facade=facade_of(n)))
    for fd in site['fdc']:
        n = outward(fd['x'], fd['y']); off = zmap * 0.7 / s
        # 주출입구 표시와 겹치면 옆으로
        fx, fy = fd['x'], fd['y']
        ic = P(fx + n[0] * off, fy + n[1] * off)
        if any(math.dist(ic, it['icon']) < 1.3 * zmap for it in items):                 # 주출입구 표시와 겹치면 바깥으로 더
            off = zmap * 2.0 / s; ic = P(fx + n[0] * off, fy + n[1] * off)
        items.append(dict(type='fdc', x=fx, y=fy, icon=ic, ang=0, facade=facade_of(n), name=fd['name']))
    # 도면 안 소화전(도면에만 있는 가상-3·4) — 전면 도로 보도 쪽 또는 측면 도로
    extra_h = []
    fdc0 = site['fdc'][0]
    wR1 = R1['width']
    for k in range(rng.choice([1, 1, 2])):
        for _ in range(80):
            if rng.random() < 0.7:
                hx = fdc0['x'] + rng.choice([1, -1]) * rng.uniform(12, 55); hy = yR1 + (wR1 / 2 - 0.8)
            else:
                R2 = roads[1]; xR2 = R2['line'][0][0]; hx = xR2 + (-1 if xR2 > cx else 1) * (R2['width'] / 2 - 0.6); hy = rng.uniform(y0, y1)
            pt = P(hx, hy)
            if not (MR[0] + 50 < pt[0] < MR[2] - 50 and MR[1] + 50 < pt[1] < MR[3] - 50): continue
            if any(math.dist((hx, hy), (e['x'], e['y'])) < 12 for e in extra_h): continue
            if any(abs(((math.atan2(hy - fdc0['y'], hx - fdc0['x']) - math.atan2(e['y'] - fdc0['y'], e['x'] - fdc0['x'])) + math.pi) % (2 * math.pi) - math.pi) < math.radians(25) for e in extra_h): continue
            if any(math.dist(pt, it['icon']) < 2.5 * zmap for it in items): continue
            extra_h.append(dict(id=f'가상-{3 + len(extra_h)}', type=rng.choice(['지상식', '지하식']), x=round(hx, 2), y=round(hy, 2))); break
    hyd_all = [dict(h) for h in site['hydrants']] + extra_h
    for h in hyd_all:
        h['dist'] = round(fp.distance(Point(h['x'], h['y'])), 1)
        pt = P(h['x'], h['y'])
        inside = MR[0] + 20 < pt[0] < MR[2] - 20 and MR[1] + 20 < pt[1] < MR[3] - 20
        if leg_pos == 'inset': inside = inside  # (범례 가림은 아래에서 처리)
        h['offmap'] = not inside
        if inside: items.append(dict(type='hyd', x=h['x'], y=h['y'], icon=pt, ang=0, id=h['id'], htype=h['type']))
        else:
            # 도면 가장자리: 건물 중심→소화전 방향 선이 지도 틀과 만나는 곳 안쪽에 아이콘 + 바깥 화살표
            c0 = P(cx, cy); dxp, dyp = pt[0] - c0[0], pt[1] - c0[1]; L_ = math.hypot(dxp, dyp)
            ux_, uy_ = dxp / L_, dyp / L_
            tmax = min([((MR[2] - 30) - c0[0]) / ux_ if ux_ > 1e-6 else 1e9, ((MR[0] + 30) - c0[0]) / ux_ if ux_ < -1e-6 else 1e9,
                        ((MR[3] - 30) - c0[1]) / uy_ if uy_ > 1e-6 else 1e9, ((MR[1] + 30) - c0[1]) / uy_ if uy_ < -1e-6 else 1e9])
            e = (c0[0] + ux_ * (tmax - 1.6 * zmap), c0[1] + uy_ * (tmax - 1.6 * zmap))
            # 도로 위(보도 쪽)에 붙인다: 소화전이 전면 도로 위이므로 같은 y(부지) 선으로 투영
            em = Pinv(*e); e2 = P(em[0], h['y'])
            if MR[0] + 1.2 * zmap < e2[0] < MR[2] - 1.2 * zmap and MR[1] + 1.2 * zmap < e2[1] < MR[3] - 1.2 * zmap: e = e2
            items.append(dict(type='hyd', x=None, y=None, icon=e, ang=0, id=h['id'], htype=h['type'], offmap=True, dir=[ux_, uy_], dist=h['dist'], em=em))
    # 이격거리 치수선
    dims = []
    for h in extra_h:
        a, bb = (h['x'], h['y']), (fdc0['x'], fdc0['y'])
        if LineString([a, bb]).intersection(fp.buffer(-0.5)).length > 0.5: continue       # 건물을 가로지르는 치수선은 그리지 않음
        dims.append(dict(kind='hyd_fdc', a=a, b=bb, val=round(math.dist(a, bb))))
    if neigh:
        nb = min(neigh, key=lambda q: q.distance(fp))
        pa_, pb_ = nearest_points(fp, nb)
        a, bb = (pa_.x, pa_.y), (pb_.x, pb_.y)
        # 축 맞춘 치수: 투영이 겹치면 겹친 구간 가운데
        ox0, ox1 = max(x0, nb.bounds[0]), min(x1, nb.bounds[2]); oy0, oy1 = max(y0, nb.bounds[1]), min(y1, nb.bounds[3])
        if ox1 - ox0 > 2: xm = (ox0 + ox1) / 2; a, bb = (xm, a[1]), (xm, bb[1])
        elif oy1 - oy0 > 2: ym = (oy0 + oy1) / 2; a, bb = (a[0], ym), (bb[0], ym)
        if math.dist(a, bb) * s > 3 * Z: dims.append(dict(kind='bldg_sep', a=a, b=bb, val=round(math.dist(a, bb) * 2) / 2))
    if rng.random() < 0.5:
        xm = rng.uniform(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0))
        a, bb = (xm, y0), (xm, yR1 + wR1 / 2)
        if math.dist(a, bb) * s > 3 * Z: dims.append(dict(kind='setback', a=a, b=bb, val=round(math.dist(a, bb) * 2) / 2))
    for dm in dims:
        dm['text_val'] = dm['val']
        if planted and dm['kind'] == 'hyd_fdc' and rng.random() < 0.15:       # 치수 글자와 실제 거리가 다른 작전도
            dm['text_val'] = round(dm['val'] * rng.choice([0.6, 1.5]) + rng.choice([-3, 3])); plant.append(dict(kind='sitemap_dim', val=dm['val'], text=dm['text_val']))
    # 그리기: 치수 → 아이콘 → 글자
    lw = max(2, int(rng.uniform(2, 3)))
    for dm in dims:
        pa, pb = P(*dm['a']), P(*dm['b'])
        # 아이콘 중심 사이면 아이콘 가장자리에서 끊기
        L_ = math.dist(pa, pb); ux, uy = (pb[0] - pa[0]) / L_, (pb[1] - pa[1]) / L_
        sa = zmap * 0.6 if dm['kind'] == 'hyd_fdc' else 0; sb = zmap * 0.6 if dm['kind'] == 'hyd_fdc' else 0
        qa, qb = (pa[0] + ux * sa, pa[1] + uy * sa), (pb[0] - ux * sb, pb[1] - uy * sb)
        draw_dim(d, S2(qa), S2(qb), st['dim'], lw * SS, 10 * SS)
        dm['pa'], dm['pb'] = qa, qb
        v = dm['text_val']; vt = f"{v:g}"
        t = rng.choice([f"{vt}m", f"{vt}m", f"{vt} m", f"{vt}M", f"L={vt}m"])
        mx, my = (qa[0] + qb[0]) / 2, (qa[1] + qb[1]) / 2
        nx, ny = -uy, ux
        fpx_ = int(rng.uniform(15, 19))
        tb_ = d.textbbox((0, 0), t, font=font(fpx_ * SS), anchor='mm'); tw_, th_ = (tb_[2] - tb_[0]) / SS, (tb_[3] - tb_[1]) / SS
        off = abs(nx) * tw_ / 2 + abs(ny) * th_ / 2 + rng.uniform(5, 9)                 # 글자 상자가 선에 닿지 않게
        sides = [(mx + nx * off, my + ny * off), (mx - nx * off, my - ny * off)]
        bpoly_px = Polygon([P(*p) for p in cs]).convex_hull
        def tscore(q):
            tb = box(q[0] - tw_ / 2 - 3, q[1] - th_ / 2 - 3, q[0] + tw_ / 2 + 3, q[1] + th_ / 2 + 3)
            return (-(tb.intersection(bpoly_px.exterior.buffer(3)).area), min([math.dist(q, it['icon']) for it in items] + [1e9]))
        tp = max(sides, key=tscore)                                   # 건물 외곽선·아이콘과 겹치지 않는 쪽
        fill_ = st['dim']['rgb'] if rng.random() < 0.5 else (0, 0, 0)                  # (난수 소비 순서는 예전과 같게)
        bg_ = (255, 255, 255) if tscore(tp)[0] == 0 and rng.random() < 0.6 else None
        if dimvar: t, bb_ = _dimvar_text(img, d, rv, v, qa, qb, tp, fpx_, fill_, bg_, SS, dimvar)
        else: bb_ = _text(d, S2(tp), t, fpx_ * SS, fill=fill_, bg=bg_)
        dm['text'] = t; dm['text_px'] = [v_ / SS for v_ in bb_]
    for it in items:
        draw_icon(d, st[it['type']]['shape'], st[it['type']]['rgb'], it['icon'][0] * SS, it['icon'][1] * SS, zmap * SS,
                  it['ang'] if st[it['type']]['shape'] in ('tri', 'arrow', 'pent') else 0)
        if it['type'] == 'hyd':
            lx, ly = it['icon'][0] + zmap * 0.7 + 30, it['icon'][1] - zmap * 0.9
            if it.get('offmap'):
                ux_, uy_ = it['dir']; a0 = (it['icon'][0] + ux_ * zmap * 0.8, it['icon'][1] + uy_ * zmap * 0.8)
                a1 = (a0[0] + ux_ * zmap * 1.1, a0[1] + uy_ * zmap * 1.1)
                d.line([S2(a0), S2(a1)], fill=(0, 0, 0), width=2 * SS)
                d.polygon([S2(a1), S2((a1[0] - ux_ * 9 - uy_ * 5, a1[1] - uy_ * 9 + ux_ * 5)), S2((a1[0] - ux_ * 9 + uy_ * 5, a1[1] - uy_ * 9 - ux_ * 5))], fill=(0, 0, 0))
                dv = it['dist']
                dt = rng.choice([f"{dv:.0f}m", f"{dv / 1000:.3f}km", f"{dv:.0f}m"])
                t = f"{it['id']} ({it['htype']}) {dt}" if rng.random() < 0.5 else f"{it['id']} {dt}"
                lx, ly = it['icon'][0] - uy_ * zmap * 1.2 - ux_ * zmap * 2, it['icon'][1] + ux_ * zmap * 1.2 - uy_ * zmap * 1.2
                it['dist_text'] = dt
            else:
                t = it['id'] if rng.random() < 0.6 else f"{it['id']}({it['htype']})"
            f_ = font(14 * SS); tw = d.textlength(t, font=f_) / SS
            if not it.get('offmap'): lx = it['icon'][0] + zmap * 0.6 + tw / 2 + 4
            lx = min(max(lx, MR[0] + tw / 2 + 4), MR[2] - tw / 2 - 4); ly = min(max(ly, MR[1] + 12), MR[3] - 12)
            tb_ = box(lx - tw / 2 - 3, ly - 10, lx + tw / 2 + 3, ly + 10)
            bpx = Polygon([P(*p) for p in cs]).convex_hull
            bb_ = _text(d, S2((lx, ly)), t, 14 * SS, fill=(0, 0, 0), bg=None if tb_.intersects(bpx.exterior.buffer(3)) else (255, 255, 255))
            it['label'] = t; it['label_px'] = [v_ / SS for v_ in bb_]
    # 지도 틀 밖 지우기 → 틀·제목
    img_a = np.asarray(img).copy(); m_ = np.ones(img_a.shape[:2], bool); m_[int(MR[1] * SS):int(MR[3] * SS), int(MR[0] * SS):int(MR[2] * SS)] = False
    img_a[m_] = 255; img = Image.fromarray(img_a); d = ImageDraw.Draw(img)
    if rng.random() < 0.7: d.rectangle([S2((MR[0], MR[1])), S2((MR[2], MR[3]))], outline=(0, 0, 0), width=2 * SS)
    if title: _text(d, S2((MR[0] + 10, 32)), rng.choice(['진압작전도', '0 진압작전도', f"진압작전도 - {G['name']}"]), 26 * SS, anchor='lm', bold=True)
    # 방위표
    na_style = rng.choice(['line', 'half', 'circle'])
    corners = [(MR[2] - 60, MR[1] + 75), (MR[0] + 60, MR[1] + 75)]
    nc = corners[0] if leg_pos != 'inset' else corners[0]
    bpx = [P(*p) for p in cs]
    if any(abs(p[0] - nc[0]) < 70 and abs(p[1] - nc[1]) < 70 for p in bpx): nc = corners[1]
    nL = rng.uniform(26, 36)
    ndir = (math.sin(th), -math.cos(th))
    tip = (nc[0] + ndir[0] * nL, nc[1] + ndir[1] * nL); tail = (nc[0] - ndir[0] * nL, nc[1] - ndir[1] * nL)
    px_, py_ = -ndir[1], ndir[0]
    if na_style == 'line':
        d.line([S2(tail), S2(tip)], fill=(0, 0, 0), width=3 * SS)
        d.polygon([S2(tip), S2((tip[0] - ndir[0] * 16 + px_ * 8, tip[1] - ndir[1] * 16 + py_ * 8)), S2((tip[0] - ndir[0] * 16 - px_ * 8, tip[1] - ndir[1] * 16 - py_ * 8))], fill=(0, 0, 0))
    elif na_style == 'half':
        d.polygon([S2(tip), S2((tail[0] + px_ * 10, tail[1] + py_ * 10)), S2(nc)], fill=(0, 0, 0))
        d.polygon([S2(tip), S2((tail[0] - px_ * 10, tail[1] - py_ * 10)), S2(nc)], outline=(0, 0, 0), width=2 * SS)
    else:
        d.ellipse([S2((nc[0] - nL * 0.8, nc[1] - nL * 0.8)), S2((nc[0] + nL * 0.8, nc[1] + nL * 0.8))], outline=(0, 0, 0), width=2 * SS)
        d.polygon([S2(tip), S2((nc[0] + px_ * 9, nc[1] + py_ * 9)), S2((tail[0], tail[1])), S2((nc[0] - px_ * 9, nc[1] - py_ * 9))], fill=(0, 0, 0))
    nt = (tip[0] + ndir[0] * 16, tip[1] + ndir[1] * 16)
    _text(d, S2(nt), 'N', 20 * SS, bold=True)
    # 축척 막대
    sbar = None
    if rng.random() < 0.7:
        for L_m in (10, 20, 25, 50, 100):
            if L_m * s >= 90: break
        n = 4 if L_m in (20, 100) else 5
        if L_m == 25: n = 5
        bx0 = MR[0] + 30; by = MR[3] - 30
        if leg_pos == 'inset': bx0 = MR[2] - 30 - L_m * s; by = MR[3] - 30
        seg = L_m * s / n
        for k in range(n):
            d.rectangle([S2((bx0 + k * seg, by - 5)), S2((bx0 + (k + 1) * seg, by + 5))], fill=(0, 0, 0) if k % 2 == 0 else (255, 255, 255), outline=(0, 0, 0), width=SS)
        _text(d, S2((bx0, by - 16)), '0', 13 * SS); _text(d, S2((bx0 + L_m * s, by - 16)), f'{L_m}m', 13 * SS)
        if rng.random() < 0.5: _text(d, S2((bx0 + L_m * s / 2, by - 16)), f'{L_m / 2:g}', 13 * SS)
        sbar = dict(px=[bx0, by - 5, bx0 + L_m * s, by + 5], m=L_m)
    # 범례
    lg = {}
    iz = Z
    if leg_pos == 'right':
        LB = [W - 236, top + rng.uniform(0, 80), W - 14, 0]; LB[3] = LB[1] + 50 + 5 * (iz + 18)
        rows = [(LB[0] + 22 + iz / 2, LB[1] + 50 + k * (iz + 18) + iz / 2) for k in range(5)]
        wpos = lambda c: (c[0] + iz / 2 + 14, c[1])
    elif leg_pos == 'bottom':
        LB = [MR[0], H - 108, MR[0] + min(W - 28, 5 * 200 + 20), H - 14]
        cw = (LB[2] - LB[0] - 20) / 5
        rows = [(LB[0] + 10 + cw * k + 16 + iz / 2, LB[1] + 58) for k in range(5)]
        wpos = lambda c: (c[0] + iz / 2 + 12, c[1])
    else:
        far = max([(MR[0] + 10, MR[3] - 10), (MR[2] - 10, MR[3] - 10), (MR[0] + 10, MR[1] + 10)], key=lambda q: min(math.dist(q, P(*p)) for p in cs))
        lw_, lh_ = 200, 50 + 5 * (iz + 16)
        lx0 = far[0] if far[0] < W / 2 else far[0] - lw_; ly0 = far[1] - lh_ if far[1] > H / 2 else far[1]
        LB = [lx0, ly0, lx0 + lw_, ly0 + lh_]
        rows = [(LB[0] + 22 + iz / 2, LB[1] + 48 + k * (iz + 16) + iz / 2) for k in range(5)]
        wpos = lambda c: (c[0] + iz / 2 + 14, c[1])
    d.rectangle([S2((LB[0], LB[1])), S2((LB[2], LB[3]))], fill=(255, 255, 255), outline=(0, 0, 0), width=2 * SS)
    _text(d, S2(((LB[0] + LB[2]) / 2, LB[1] + 22)), rng.choice(['범 례', '범례', '범례(필수)']), 17 * SS, bold=True)
    order = list(TYPES) if rng.random() < 0.7 else rng.sample(TYPES, 5)
    for k, t in enumerate(order):
        c = rows[k]
        if t == 'dim':
            draw_dim(d, S2((c[0] - iz * 0.8, c[1])), S2((c[0] + iz * 0.8, c[1])), st['dim'], lw * SS, 9 * SS)
            wp = (c[0] + iz * 0.8 + 12, c[1])
        else:
            draw_icon(d, st[t]['shape'], st[t]['rgb'], c[0] * SS, c[1] * SS, iz * SS, -90 if st[t]['shape'] in ('tri', 'arrow', 'pent') else 0)
            wp = wpos(c)
        _text(d, S2(wp), WORDS[t], 15 * SS, anchor='lm')
        lg[t] = dict(px=c, **{k2: v for k2, v in st[t].items() if k2 != 'rgb'})
    # 범례 상자에 가려진 도면 요소는 정답에서 뺀다
    def hidden(p): return LB[0] - 4 <= p[0] <= LB[2] + 4 and LB[1] - 4 <= p[1] <= LB[3] + 4
    items = [it for it in items if not hidden(it['icon'])]
    dims = [dm for dm in dims if not (hidden(dm['pa']) or hidden(dm['pb']))]
    # 축소 + 스캔 열화
    img = img.resize((W, H), Image.LANCZOS)
    q = rng.random()
    skew = rng.uniform(-0.8, 0.8) if q < 0.5 else 0.0
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=(255, 255, 255))      # 반시계(도) — 중심 기준
    if rng.random() < 0.8: img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    a = np.asarray(img).astype(np.float32)
    a = a * rng.uniform(0.9, 1.0) + rng.uniform(0, 18)
    a += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 7), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rng.randint(40, 80)); img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    def SK(p):                                                              # 기울임 반영(화소)
        if not skew: return [round(p[0], 1), round(p[1], 1)]
        a_ = math.radians(skew); ccx, ccy = W / 2, H / 2
        dx, dy = p[0] - ccx, p[1] - ccy
        return [round(ccx + dx * math.cos(a_) + dy * math.sin(a_), 1), round(ccy - dx * math.sin(a_) + dy * math.cos(a_), 1)]
    # 정답
    for it in items: it['px'] = SK(it.pop('icon')); it.pop('ang', None)
    for dm in dims: dm['pa'], dm['pb'] = SK(dm['pa']), SK(dm['pb'])
    site_true = dict(site, hydrants=[{k: v for k, v in h.items() if k not in ('dist', 'offmap')} for h in hyd_all])
    gt = dict(name=G['name'], size=[W, H], px_per_m=round(s, 4), north_deg=round(theta + (-skew), 3), theta_design=theta, skew=skew,
              center_m=[cx, cy], center_px=SK((u0, v0)), map_rect=MR, legend_box=[round(v, 1) for v in LB], legend=lg, legend_pos=leg_pos,
              footprint_m=[list(p) for p in fp.exterior.coords][:-1] if fp.geom_type == 'Polygon' else None,
              footprint_px=[SK(P(*p)) for p in list(fp.exterior.coords)[:-1]] if fp.geom_type == 'Polygon' else None,
              neighbors_m=[[list(p) for p in nb.exterior.coords][:-1] for nb in neigh],
              items=items, dims=[dict(kind=dm['kind'], a=list(dm['a']), b=list(dm['b']), val=dm['val'], text_val=dm['text_val'], text=dm['text'], pa=dm['pa'], pb=dm['pb'], text_px=[SK(dm['text_px'][:2]), SK(dm['text_px'][2:])]) for dm in dims],
              hydrants=[dict(h) for h in hyd_all], road_labels=labels, scalebar=sbar, planted=plant, site_true=site_true, site_gt=sgt)
    return img, gt


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--seed', type=int, default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    G = json.load(open(os.path.join(a.dir, 'gt.json'), encoding='utf-8'))
    img, gt = sitemap_image(G, random.Random(a.seed) if a.seed is not None else None)
    out = a.out or os.path.join(ROOT, 'out_sitemap', os.path.basename(a.dir.rstrip('/')) + '_sitemap.png')   # data/ 안에 두면 run_all 기능 24 입력이 바뀌므로 기본은 밖에
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out); json.dump(gt, open(out.rsplit('.', 1)[0] + '_gt.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(out, gt['px_per_m'], gt['north_deg'], len(gt['items']), len(gt['dims']), gt['planted'])


def card_hook(b, ctx=None):
    """표준양식 카드 생성기 훅(synth.make.sitemap_image) 연결용: SYNTH_SITEMAP=synth.sitemap:card_hook
    첫 동만 진압작전도를 그리고 정답을 gt['sitemap_gt'] 에 넣는다(다른 동은 자리표시)."""
    if ctx and ctx.get('dong', 0) != 0: return None
    img, sgt = sitemap_image(b)
    if ctx is not None and isinstance(ctx.get('gt'), dict): ctx['gt']['sitemap_gt'] = sgt
    return img
