# -*- coding: utf-8 -*-
"""가상 정보카드(정답 포함) → STR 학습 캐시(SYN_*): 정보카드 평면도 표현(회색 띠 벽·그리드선·소방기호)을 모델에 학습시킨다.

사용: python -m strseg.synth_import [data 폴더=data] [--cache DIR]
- card.pdf 에서 평면도 이미지를 꺼내(fire3dbeam.card.extract_plans) 정답 gt.json 의 벽(m, 두께 t)·문을 transforms 로 화소에 그림.
- 그리드선·기호·치수는 배경(0) → '그리드선=벽' 오검출을 줄이는 음성 예. 창은 가상 도면에 없음(배경).
- 작업 해상도: 1/100 기준 1 m ≈ 29 px (fire3dbeam.str_seg.WORK_PPM_1_100).
- 보류 평가는 다른 시드 세트(SYNTH_DATA=data_hold)로 따로 — 여기서는 data/ 만 학습에 쓴다.
"""
import argparse, glob, json, os
import numpy as np, cv2, pymupdf
from strseg import data as D
from fire3dbeam.card import extract_plans
from fire3dbeam.str_seg import WORK_PPM_1_100


def rect_poly(x0, y0, x1, y1, t):
    d = np.array([x1 - x0, y1 - y0], float); L = np.hypot(*d) or 1.0; n = np.array([-d[1], d[0]]) / L * t / 2
    a, b = np.array([x0, y0]), np.array([x1, y1])
    return np.array([a + n, b + n, b - n, a - n])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data', nargs='?', default='data'); ap.add_argument('--cache', default=D.CACHE)
    a = ap.parse_args(); os.makedirs(a.cache, exist_ok=True); n = 0
    for gp in sorted(glob.glob(os.path.join(a.data, '*', 'gt.json'))):
        bd = os.path.dirname(gp); gt = json.load(open(gp, encoding='utf-8'))
        plans = extract_plans(pymupdf.open(os.path.join(bd, 'card.pdf')))
        done = set()
        for fid, T in gt['transforms'].items():
            cap = T.get('caption')
            if cap in done: continue
            done.add(cap)
            P = [p for p in plans if p['caption'].replace(' ', '') == (cap or '').replace(' ', '')]
            if not P: continue
            img = cv2.cvtColor(P[0]['img'], cv2.COLOR_BGR2GRAY)
            if abs(img.shape[1] - T['w']) > 2 or abs(img.shape[0] - T['h']) > 2:
                continue                                           # 크기가 다르면 좌표계 불확실 → 건너뜀
            F = gt['floors'][T['group'][0] if T.get('group') else fid]
            s = T['s']; k = WORK_PPM_1_100 / s
            gi = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
            lab = np.zeros(gi.shape, np.uint8)
            tp = lambda P2: np.round(np.c_[(P2[:, 0] - T['ox']) * s, (T['oy'] - P2[:, 1]) * s] * k).astype(np.int32)
            for w in F['walls']:
                cv2.fillPoly(lab, [tp(rect_poly(w['x0'], w['y0'], w['x1'], w['y1'], max(0.15, w.get('t', 0.2))))], 1)
            for d in F['doors']:
                hw, ht = d['w'] / 2, 0.25
                x0, y0, x1, y1 = ((d['x'] - hw, d['y'] - ht, d['x'] + hw, d['y'] + ht) if d['o'] == 'h' else (d['x'] - ht, d['y'] - hw, d['x'] + ht, d['y'] + hw))
                cv2.fillPoly(lab, [tp(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))], 3)
            stem = f"SYN_{os.path.basename(bd)}_{fid}"
            cv2.imwrite(os.path.join(a.cache, stem + '_i.png'), gi); cv2.imwrite(os.path.join(a.cache, stem + '_l.png'), lab); n += 1
    print(f'가상 정답 → 학습 캐시 {n}장 ({a.cache})')


if __name__ == '__main__':
    main()
