# -*- coding: utf-8 -*-
"""가상 정보카드 보류 세트(다른 시드)에서 STR 모델 화소 IoU 평가.
사용: python -m strseg.synth_import data_hold --cache <DIR>  →  python -m strseg.synth_eval <DIR> [--model models/str_seg.onnx]"""
import argparse, glob, json, os
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cache'); ap.add_argument('--model', default=None); a = ap.parse_args()
    from fire3dbeam.str_seg import StrSeg, WORK_PPM_1_100
    seg = StrSeg(a.model) if a.model else StrSeg()
    I = np.zeros(4); U = np.zeros(4); n = 0
    for lp in sorted(glob.glob(os.path.join(a.cache, 'SYN_*_l.png'))):
        g = cv2.imread(lp[:-6] + '_i.png', 0); l = cv2.imread(lp, 0)
        p, _, _ = seg.predict(g, px_per_m=WORK_PPM_1_100, out_shape=g.shape)
        v = l != 255
        for c in (1, 3):
            I[c] += ((p == c) & (l == c) & v).sum(); U[c] += (((p == c) | (l == c)) & v).sum()
        n += 1
    r = dict(n=n, wall_iou=round(I[1] / max(1, U[1]), 3), door_iou=round(I[3] / max(1, U[3]), 3))
    print(json.dumps(r)); return r


if __name__ == '__main__':
    main()
