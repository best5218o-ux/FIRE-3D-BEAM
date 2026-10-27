# -*- coding: utf-8 -*-
"""⑤ 학습 순환 연결: 검수 화면에서 확정한 라벨(review.labels 의 COCO 출력) → STR 모델 학습 캐시(REV_*).

사용: python -m strseg.review_import <review.labels --out 폴더> [--cache DIR] [--all]
  → CACHE/REV_<이미지>_i.png(작업 해상도 회색조) + _l.png(0 배경,1 벽,3 문, 255 무시)
  이후 python -m strseg.train --warm models/ckpt/best.pt --minutes N  (REV_REPEAT=3: 검수 라벨을 3배로 더 자주 사용)

라벨 규칙
- 벽: 검수 확정 벽(walls_final − 삭제) 다각형 → 1.   문: 검수 문 상자 → 3.
- 창: 검수 화면에는 창 표시가 없으므로 현재 모델이 '창'으로 본 화소는 255(무시) — '창이 아님'을 잘못 가르치지 않게.
- 검수에서 확정되지 않았는데 모델이 벽·문으로 본 화소도 255(무시): 검수자가 손대지 않은 벽을 '벽 아님'으로 학습하지 않게.
- 단, 교차검증 판정으로 검수자가 '벽 아님'이라고 명시한 곳(wall_rejected: 규칙만 본 벽에 「모델이 맞음」, 모델만 본 벽에 「규칙이 맞음」)은
  모델이 벽으로 봐도 0(배경)으로 가르친다(명시적 음성). door_rejected(모델 문에 「규칙이 맞음」)도 모델 문 화소를 0으로.
  확정 벽·문(1·3)이 음성 영역과 겹치면 확정이 이긴다. 모델이 '창'으로 본 화소는 음성 영역 안에서도 무시(255) 유지.
- 계단·승강기 샤프트(stair·elevator_shaft) 주석은 이 STR 캐시에는 쓰지 않는다(SPA 학습용).
- 유효 영역: 벽 전체 외접 상자 + 여백(2 m). 그 밖은 255.
- 기본은 '검수 완료(reviewed)' 도면만. 내부 PC에서 실제 정보카드로 만든 라벨은 반출 금지(학습도 내부 PC에서).
"""
import argparse, json, os
import numpy as np, cv2
from strseg import data as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('review_dir'); ap.add_argument('--cache', default=D.CACHE); ap.add_argument('--all', action='store_true')
    ap.add_argument('--no-model', action='store_true', help='창 무시 영역 계산에 모델을 쓰지 않음')
    a = ap.parse_args()
    from fire3dbeam.str_seg import WORK_PPM_1_100
    coco = json.load(open(os.path.join(a.review_dir, 'annotations.json'), encoding='utf-8'))
    cat = {c['id']: c['name'] for c in coco['categories']}
    seg = None
    if not a.no_model:
        try:
            from fire3dbeam.str_seg import StrSeg
            seg = StrSeg()
        except Exception as e:
            print('모델 없음 → 창 무시 영역 생략:', e)
    os.makedirs(a.cache, exist_ok=True)
    n = n_neg_all = 0; stats = []
    for im in coco['images']:
        if not a.all and not im.get('reviewed'): continue
        g = cv2.imdecode(np.fromfile(os.path.join(a.review_dir, im['file_name']), np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        ppm = float(im.get('px_per_m') or 10.0)
        k = WORK_PPM_1_100 / ppm
        gi = cv2.resize(g, (max(8, round(g.shape[1] * k)), max(8, round(g.shape[0] * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        H, W = gi.shape
        lab = np.zeros((H, W), np.uint8)
        anns = [x for x in coco['annotations'] if x['image_id'] == im['id']]
        walls = [x for x in anns if cat.get(x['category_id']) == 'wall']
        if not walls: continue
        for x in walls:
            poly = x.get('segmentation', [None])[0]
            if poly is None:
                bx, by, bw, bh = x['bbox']; poly = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
            cv2.fillPoly(lab, [np.round(np.array(poly, float).reshape(-1, 2) * k).astype(np.int32)], 1)
        for x in anns:
            if cat.get(x['category_id']) == 'door':
                bx, by, bw, bh = x['bbox']
                cv2.rectangle(lab, (round(bx * k), round(by * k)), (round((bx + bw) * k), round((by + bh) * k)), 3, -1)
        # 명시적 음성(검수 판정): 벽 아님 / 문 아님
        neg_w = np.zeros((H, W), np.uint8); neg_d = np.zeros((H, W), np.uint8)
        for x in anns:
            nm = cat.get(x['category_id'])
            if nm == 'wall_rejected':
                poly = (x.get('segmentation') or [None])[0]
                if poly is None:
                    bx, by, bw, bh = x['bbox']; poly = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]
                cv2.fillPoly(neg_w, [np.round(np.array(poly, float).reshape(-1, 2) * k).astype(np.int32)], 1)
            elif nm == 'door_rejected':
                bx, by, bw, bh = x['bbox']
                cv2.rectangle(neg_d, (round(bx * k), round(by * k)), (round((bx + bw) * k), round((by + bh) * k)), 1, -1)
        n_neg = 0
        if seg is not None:
            try:
                p, _, _ = seg.predict(gi, px_per_m=WORK_PPM_1_100, out_shape=gi.shape)
                lab[(p == 2) & (lab == 0)] = D.IGN
                # 검수자가 확정하지 않은 곳을 모델이 벽·문으로 본 경우 → '벽 아님'으로 가르치지 않도록 무시(보수적)
                #   예외: 검수자가 교차검증에서 '벽 아님/문 아님'으로 판정한 곳은 0(배경)으로 둔다
                keep0 = ((p == 1) & (neg_w == 1)) | ((p == 3) & (neg_d == 1))
                lab[np.isin(p, (1, 3)) & (lab == 0) & ~keep0] = D.IGN
                n_neg = int((keep0 & (lab == 0)).sum())
            except Exception as e:
                print('창 추정 실패', im['file_name'], e)
        ys, xs = np.nonzero((lab == 1) | (lab == 3) | (neg_w == 1) | (neg_d == 1))
        mg = int(2 * WORK_PPM_1_100)
        v = np.zeros_like(lab); v[max(0, ys.min() - mg):ys.max() + mg, max(0, xs.min() - mg):xs.max() + mg] = 1
        lab[v == 0] = D.IGN
        stem = 'REV_' + os.path.splitext(os.path.basename(im['file_name']))[0]
        cv2.imwrite(os.path.join(a.cache, stem + '_i.png'), gi); cv2.imwrite(os.path.join(a.cache, stem + '_l.png'), lab)
        n += 1; n_neg_all += n_neg
        stats.append(dict(stem=stem, wall=int((lab == 1).sum()), door=int((lab == 3).sum()), ign=int((lab == D.IGN).sum()),
                          neg_regions=int(sum(1 for x in anns if cat.get(x['category_id']) in ('wall_rejected', 'door_rejected'))), neg_model_px=n_neg))
    json.dump(stats, open(os.path.join(a.cache, 'review_import_stats.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'검수 라벨 → 학습 캐시 {n}장 ({a.cache}) · 명시적 음성(모델은 벽·문으로 봤지만 검수자가 아니라고 한 화소→0) {n_neg_all}px')


if __name__ == '__main__':
    main()
