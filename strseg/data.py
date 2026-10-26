# -*- coding: utf-8 -*-
"""AI허브 「건축 도면 데이터」 STR 라벨 → 학습·평가용 라벨 영상.

원천: ./aihub/img/STR/*.PNG (긴 변 2048 로 축소, ≈124dpi) + meta.json {w,h 원본 px, f 축소 배율}
라벨: ./aihub/label_index.json (대문자 stem → labels/TL_STR/...json, COCO 유사, 좌표 = 원본 px)
클래스: 0 배경, 1 벽(구조_벽체), 2 창(구조_창호), 3 문(구조_출입문), 255 무시(라벨 없는 영역)

라벨 없는 영역 처리(선택한 방법):
  한 시트에 평면이 여러 개인데 일부만 라벨된 경우가 있다. 'background' 주석은 시트 전체라 쓸모없음.
  → 벽·창·문 다각형을 원본 기준 ~1.6 m(2048 영상 64px) 팽창해 묶은 덩어리마다 외접 상자 + 여백 40px(≈8mm 용지, 치수선 일부 포함)
    = '유효 영역'. 그 밖은 255(무시): 손실·지표 모두 유효 영역 안에서만 계산.
분할: 파일명(stem) md5 → 80% dev / 20% held-out. dev 안에서 다시 10% 를 dev-val(체크포인트 선택용)로.
※ AI허브 자료는 로컬 전용 — 영상·겹침 그림 공개 금지.
"""
import os, json, hashlib
import numpy as np, cv2

AIHUB = os.environ.get('AIHUB', './aihub')
IMG_DIR = os.path.join(AIHUB, 'img', 'STR')
CACHE = os.environ.get('STR_CACHE', './cache/strm/cache/str')
CACHE_R = 0.6            # 캐시 배율(2048 영상 대비) → 긴 변 ≈1229px, 원본(300dpi) 대비 ≈0.248 (≈74dpi)
CLS = {'구조_벽체': 1, '구조_창호': 2, '구조_출입문': 3}
NAMES = ['bg', 'wall', 'window', 'door']
IGN = 255


def h01(stem, salt=''):
    return int(hashlib.md5((salt + stem.upper()).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(stem):
    """'held' (20%) / 'val' (dev 의 10%) / 'train'.  검수 라벨(REV_*)은 항상 train"""
    if stem.startswith(('REV_', 'SYN_')): return 'train'
    if h01(stem) < 0.2: return 'held'
    return 'val' if h01(stem, 'val') < 0.1 else 'train'


_IDX = None


def label_index():
    global _IDX
    if _IDX is None: _IDX = json.load(open(os.path.join(AIHUB, 'label_index.json')))
    return _IDX


def list_images():
    """현재 폴더에 있는(meta 기록 + 라벨 있는) 영상 stem 목록 — 매번 다시 훑는다(자료가 추가되는 중)"""
    try: meta = json.load(open(os.path.join(IMG_DIR, 'meta.json')))
    except Exception: meta = {}
    idx = label_index()
    out = []
    for fn, m in sorted(meta.items()):
        stem = os.path.splitext(fn)[0]
        if stem.upper() in idx and os.path.exists(os.path.join(IMG_DIR, fn)):
            out.append(dict(stem=stem, fn=fn, **m))
    return out


def load_ann(stem):
    d = json.load(open(os.path.join(AIHUB, label_index()[stem.upper()])))
    cat = {c['id']: c['name'] for c in d['categories']}
    anns = []
    for a in d['annotations']:
        n = cat.get(a['category_id'])
        if n not in CLS: continue
        polys = [np.array(s, float).reshape(-1, 2) for s in a['segmentation'] if len(s) >= 6]
        if not polys:
            x, y, w, h = a['bbox']; polys = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)]
        at = a.get('attributes', {})
        sub = at.get(n) or at.get('창호') or ''
        anns.append(dict(c=CLS[n], sub=sub, polys=polys, bbox=a['bbox']))
    return anns


def raster(anns, shape, s, valid_margin=None):
    """anns(원본 좌표) → shape 크기 라벨(배율 s). 벽→창→문 순서로 칠함(문이 위). valid_margin=None 이면 무시영역 없음"""
    H, W = shape
    lab = np.zeros((H, W), np.uint8)
    for c in (1, 2, 3):
        ps = [np.round(p * s).astype(np.int32) for a in anns if a['c'] == c for p in a['polys']]
        if ps: cv2.fillPoly(lab, ps, c)
    return lab


def valid_mask(anns, shape, s, join_px=None, margin_px=None):
    """유효 영역: 라벨 다각형 덩어리(join 팽창으로 묶음)마다 외접 상자 + 여백. s = 원본→이 해상도 배율"""
    H, W = shape
    k = s / (2048 / 4963)                       # 2048 영상 기준 대비 배율
    join_px = join_px or max(3, int(64 * k)); margin_px = margin_px or max(2, int(40 * k))
    m = np.zeros((H, W), np.uint8)
    ps = [np.round(p * s).astype(np.int32) for a in anns for p in a['polys']]
    if not ps: return m
    cv2.fillPoly(m, ps, 1)
    for p in ps: cv2.polylines(m, [p], True, 1, 1)
    d = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (join_px, join_px)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    v = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        v[max(0, y - margin_px):y + h + margin_px, max(0, x - margin_px):x + w + margin_px] = 1
    return v


def gt_at(stem, meta, long_side):
    """평가용: 2048 영상을 긴 변 long_side 로. 반환 gray, lab(0..3, 무시=255), anns, 배율 s(원본→평가)"""
    g = cv2.imread(os.path.join(IMG_DIR, meta['fn']), cv2.IMREAD_GRAYSCALE)
    r = long_side / max(g.shape)
    ge = cv2.resize(g, (round(g.shape[1] * r), round(g.shape[0] * r)), interpolation=cv2.INTER_AREA)
    s = meta['f'] * r
    anns = load_ann(stem)
    lab = raster(anns, ge.shape, s)
    v = valid_mask(anns, ge.shape, s)
    lab[v == 0] = IGN
    return ge, lab, anns, s


def cache_one(it):
    """학습 캐시: 배율 CACHE_R 회색조 + 라벨(무시 포함) PNG"""
    stem = it['stem']
    pi, pl = os.path.join(CACHE, stem + '_i.png'), os.path.join(CACHE, stem + '_l.png')
    if os.path.exists(pl): return stem
    g = cv2.imread(os.path.join(IMG_DIR, it['fn']), cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    gi = cv2.resize(g, (round(g.shape[1] * CACHE_R), round(g.shape[0] * CACHE_R)), interpolation=cv2.INTER_AREA)
    s = it['f'] * CACHE_R
    anns = load_ann(stem)
    lab = raster(anns, gi.shape, s); v = valid_mask(anns, gi.shape, s); lab[v == 0] = IGN
    cv2.imwrite(pi, gi, [cv2.IMWRITE_PNG_COMPRESSION, 3]); cv2.imwrite(pl + '.tmp.png', lab)
    os.replace(pl + '.tmp.png', pl)
    return stem


def build_cache(workers=1):
    os.makedirs(CACHE, exist_ok=True)
    its = [it for it in list_images() if not os.path.exists(os.path.join(CACHE, it['stem'] + '_l.png'))]
    n = 0
    for it in its:
        try:
            if cache_one(it): n += 1
        except Exception as e:
            print('cache fail', it['stem'], e)
    return n


def review_items():
    """추가 학습 캐시: ⑤ 검수 라벨(REV_*, strseg.review_import) + 가상 정보카드 정답(SYN_*, strseg.synth_import)"""
    import glob
    return [dict(stem=os.path.basename(p)[:-6], fn=None, review=True)
            for pre in ('REV_', 'SYN_') for p in sorted(glob.glob(os.path.join(CACHE, pre + '*_l.png')))]


def cached(split=None):
    its = list_images()
    out = [it for it in its if os.path.exists(os.path.join(CACHE, it['stem'] + '_l.png'))] + review_items()
    if split: out = [it for it in out if split_of(it['stem']) in (split if isinstance(split, (list, tuple)) else [split])]
    return out


if __name__ == '__main__':
    import sys, time
    loop = len(sys.argv) > 1 and sys.argv[1] == 'loop'
    while True:
        t = time.time(); n = build_cache()
        c = cached()
        from collections import Counter
        print(time.strftime('%H:%M:%S'), 'new', n, 'cached', len(c), Counter(split_of(i['stem']) for i in c), f'{time.time() - t:.0f}s', flush=True)
        if not loop: break
        time.sleep(120)
