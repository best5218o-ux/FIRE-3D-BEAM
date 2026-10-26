# -*- coding: utf-8 -*-
"""가상 정보카드(data/*/card.pdf) 평면 이미지에 분할 모델을 돌려 겹침 그림 + 가상 정답 벽과의 간이 비교(정성 확인용).
기존 파이프라인 기본 경로는 건드리지 않는다.  python -m strseg.synth_check [n_buildings] [--out out_str/synth]"""
import os, sys, json, glob, argparse
import numpy as np, cv2, pymupdf
from firebim.card import extract_plans
from firebim import str_seg as SS

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('n', type=int, nargs='?', default=6); ap.add_argument('--out', default='out_str/synth')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seg = SS.StrSeg()
    dirs = sorted(glob.glob('data/*/card.pdf'))
    pick = [dirs[i] for i in np.linspace(0, len(dirs) - 1, min(a.n, len(dirs))).astype(int)]
    rows = []
    for pdf in pick:
        name = os.path.basename(os.path.dirname(pdf))
        outj = f'out/{name}.json'
        S = {p['i']: p['s'] for p in json.load(open(outj))['plans']} if os.path.exists(outj) else {}
        plans = extract_plans(pymupdf.open(pdf))
        for i, p in enumerate(plans[:2]):
            g = cv2.cvtColor(p['img'], cv2.COLOR_BGR2GRAY)
            s = S.get(i)
            lab, prob, k = seg.predict(g, px_per_m=s) if s else seg.predict(g, src_dpi=100)
            lab_in = cv2.resize(lab, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            v = SS.vectorize(lab, px_per_m=None)
            # 간이 비교: 회색 띠 벽(가상 양식: 채도 낮은 중간 회색 굵은 띠) ↔ 모델 벽
            gray_band = ((g > 60) & (g < 200)).astype(np.uint8)
            gray_band = cv2.morphologyEx(gray_band, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            wm = lab_in == 1
            cov = float((wm & (gray_band > 0)).sum() / max(1, wm.sum()))
            img = SS.overlay(cv2.resize(g, None, fx=max(1, 900 / max(g.shape)), fy=max(1, 900 / max(g.shape)), interpolation=cv2.INTER_NEAREST),
                             cv2.resize(lab_in, None, fx=max(1, 900 / max(g.shape)), fy=max(1, 900 / max(g.shape)), interpolation=cv2.INTER_NEAREST))
            cv2.imwrite(os.path.join(a.out, f'{name}_p{i}.png'), np.hstack([cv2.cvtColor(cv2.resize(g, (img.shape[1], img.shape[0])), cv2.COLOR_GRAY2BGR), img]))
            rows.append(dict(name=name, plan=i, caption=p['caption'], s=s, k=round(k, 3), wall_px_frac=round(float(wm.mean()), 4),
                             wall_on_grayband=round(cov, 3), n_walls=len(v['walls']), n_doors=len(v['doors']), n_windows=len(v['windows'])))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    json.dump(rows, open(os.path.join(a.out, 'synth_check.json'), 'w'), ensure_ascii=False, indent=1)
