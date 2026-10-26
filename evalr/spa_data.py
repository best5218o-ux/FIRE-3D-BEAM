# -*- coding: utf-8 -*-
"""AI허브 「건축 도면 데이터」 SPA(공간) 라벨 로더 — 평가(spa_eval)·학습(spa_train) 공용.
이미지: AIHUB_SPA_IMG(기본 ./aihub/img/SPA/*.png, 긴 변 2048 축소본 + meta.json{w,h,f})
라벨:   AIHUB_ROOT/label_index.json (대문자 stem → json 상대경로), 좌표는 원본 픽셀 → ×f.
공개 불가 자료 — 로컬 전용(이미지·오버레이를 외부로 내보내지 말 것)."""
import os, json, glob, hashlib
import numpy as np, cv2

ROOT = os.environ.get('AIHUB_ROOT', './aihub')
IMG = os.environ.get('AIHUB_SPA_IMG', os.path.join(ROOT, 'img', 'SPA'))

SPA_CLASSES = ['공간_침실', '공간_거실', '공간_주방', '공간_화장실', '공간_발코니', '공간_현관', '공간_드레스룸', '공간_실외기실',
               '공간_계단실', '공간_엘리베이터', '공간_엘리베이터홀', '공간_다목적공간', '공간_기타']
SHORT = {c: c.split('_', 1)[1] for c in SPA_CLASSES}


def split_of(stem):
    """결정적 분할: 파일명(대문자 stem) md5 → 20% 보류(held), 80% 개발(dev). 이미지가 추가돼도 기존 분할 불변."""
    h = int(hashlib.md5(stem.upper().encode()).hexdigest()[:8], 16)
    return 'held' if h % 5 == 0 else 'dev'


HELD198_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spa_held198.json')
_H198 = None


def held198():
    """SPA v2 보류 수치(2026-09-30)를 낸 보류 198장 stem 고정 목록(evalr/spa_held198.json, STR 의 held674.json 과 같은 역할).
    모두 split_of()=='held' 이므로 자료가 늘어도 학습에 들어가지 않음. 자료가 늘면 held 는 커지지만 이 198장은 그대로 비교 기준."""
    global _H198
    if _H198 is None: _H198 = frozenset(s.upper() for s in json.load(open(HELD198_JSON)))
    return _H198


def trainable(stem):
    """학습·캐시에 넣어도 되는 영상인가: 개발 split 이면서 고정 보류 목록에 없음(이중 안전장치)."""
    s = stem.upper()
    return split_of(s) == 'dev' and s not in held198()


def list_items(split='all'):
    """split: all | dev | held | held198(고정 보류 198장 — 영상이 있는 것만)"""
    meta = json.load(open(os.path.join(IMG, 'meta.json')))
    idx = json.load(open(os.path.join(ROOT, 'label_index.json')))
    out = []
    for p in sorted(glob.glob(os.path.join(IMG, '*'))):
        fn = os.path.basename(p)
        if not fn.lower().endswith('.png') or fn not in meta: continue
        stem = os.path.splitext(fn)[0].upper()
        if stem not in idx: continue
        sp = split_of(stem)
        if split == 'held198':
            if stem not in held198(): continue
        elif split != 'all' and sp != split: continue
        out.append(dict(stem=stem, path=p, label=os.path.join(ROOT, idx[stem]), f=meta[fn]['f'], split=sp))
    return out


def load_gt(it):
    """→ [{cls:'공간_침실', poly:(N,2) float (축소 영상 좌표), area}] (background 제외, 다중 조각은 각각)"""
    d = json.load(open(it['label'], encoding='utf-8'))
    cat = {c['id']: c['name'] for c in d['categories']}
    out = []
    for a in d['annotations']:
        n = cat.get(a['category_id'])
        if n not in SPA_CLASSES: continue
        for s in a['segmentation']:
            p = np.array(s, float).reshape(-1, 2) * it['f']
            if len(p) < 3: continue
            ar = cv2.contourArea(p.astype(np.float32))
            if ar < 4: continue
            out.append(dict(cls=n, poly=p, area=float(ar)))
    return out


def labeled_zone(gt, shape, margin=40):
    """라벨 작업이 된 영역(모든 공간 다각형 합집합을 margin px 팽창 + 구멍 메움). 밖의 검출은 '미라벨 영역'으로 정밀도에서 분리."""
    m = np.zeros(shape, np.uint8)
    for r in gt: cv2.fillPoly(m, [np.round(r['poly']).astype(np.int32)], 1)
    if margin: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cs, -1, 1, -1)
    return m


def raster(gt, shape, cls_index, scale=1.0):
    """클래스 id 래스터(0=배경). 큰 다각형부터 칠해 작은 것(승강기 등)이 위에 오게."""
    m = np.zeros(shape, np.uint8)
    for r in sorted(gt, key=lambda r: -r['area']):
        k = cls_index.get(r['cls'])
        if k is None: continue
        cv2.fillPoly(m, [np.round(r['poly'] * scale).astype(np.int32)], int(k))
    return m
