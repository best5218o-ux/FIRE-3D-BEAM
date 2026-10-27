# -*- coding: utf-8 -*-
"""실제 도면 외곽(건물 바닥 윤곽, footprint) 정확도 — AI허브 STR 라벨(벽·창·문 다각형)로 직접 채점.

정답 외곽: 라벨 다각형(벽∪창∪문)을 평가 해상도(원본 300dpi 의 1/4 = Q, 75dpi)로 그려
  ≈1.3 m 팽창으로 묶은 덩어리(= 라벨된 평면 하나)마다 → 작은 닫기(k=3,5,9,15 px 중 채움이 안정되는 가장 작은 값) → 바깥과 안 이어진 곳 채움.
예측 외곽: fire3dbeam.realplan.process_gray(원본 크기로 되돌린 영상, 모델은 124dpi 축소본을 그대로 src_dpi 로)
  의 'plan' 영역 외곽 다각형을 Q 해상도로 채움.
채점(정답 덩어리마다): 정답과 겹치는 예측 성분 전체(치수 띠·상세도로 번진 부분 포함) − 다른 정답 덩어리(라벨된 다른 평면) → IoU, 재현(=|P∩G|/|G|), 정밀(|P∩G|/|P|).
  예측이 없으면 IoU 0. 라벨 안 된 평면(한 시트에 일부만 라벨)은 정답 덩어리와 겹치지 않으면 채점하지 않음.
분할: strseg.data.split_of — dev = train+val, held = 보류(report-only). 순서 = 파일명 해시 고정(D.h01(stem,'order')).
사용:
  python -m evalr.footprint_eval run --split dev --limit 80 [--rules] [--tag T] [--overlays 10] [--nodims]
  python -m evalr.footprint_eval run --split held --limit 150 --tag final
출력: out_fp/<tag>_<split>.jsonl (시트별), out_fp/<tag>_<split>.json (요약), out_fp/<tag>_ov/*.png(정답 초록·예측 빨강)
※ AI허브 자료 — 결과·겹침 그림은 로컬 전용, 공개 금지.
"""
import os, sys, json, time, argparse, traceback
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strseg import data as D

Q = 0.25                     # 평가 해상도 = 원본(300dpi) × 1/4
OUT = 'out_fp'
MCACHE = os.environ.get('FP_MODEL_CACHE', '')    # 모델 라벨 캐시 폴더(선택)


def items(split, limit=0):
    sp = ('train', 'val') if split == 'dev' else (split,)
    its = [it for it in D.list_images() if D.split_of(it['stem']) in sp]
    its.sort(key=lambda it: D.h01(it['stem'], 'order'))
    return its[:limit] if limit else its


def _fill(m):
    """바깥(영상 테두리)과 4-연결로 안 이어진 곳 채움"""
    H, W = m.shape
    inv = cv2.copyMakeBorder((1 - m).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    ff = inv.copy(); msk = np.zeros((H + 4, W + 4), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 2)
    return (ff[1:-1, 1:-1] != 2).astype(np.uint8)


def gt_clusters(it, shape_q, gq=None):
    """정답 외곽 덩어리 목록 [(fp 마스크 bool, 라벨 화소 마스크)] (Q 해상도).
    gq: Q 해상도 회색조 — 있으면 덩어리마다 라벨 벽 화소가 잉크(±1px) 위에 있는 비율 ink_cov (라벨 위치 어긋남 검사)"""
    anns = D.load_ann(it['stem'])
    H, W = shape_q
    inkd = cv2.dilate((gq < 225).astype(np.uint8), np.ones((3, 3), np.uint8)) if gq is not None else None
    wl = np.zeros((H, W), np.uint8)
    wp = [np.round(p * Q).astype(np.int32) for a in anns if a['c'] == 1 for p in a['polys']]
    if wp: cv2.fillPoly(wl, wp, 1)
    lab = np.zeros((H, W), np.uint8)
    ps = [np.round(p * Q).astype(np.int32) for a in anns for p in a['polys']]
    if not ps: return []
    cv2.fillPoly(lab, ps, 1)
    for p in ps: cv2.polylines(lab, [p], True, 1, 1)
    join = max(3, int(64 * Q / (2048 / 4963)))
    d = cv2.dilate(lab, cv2.getStructuringElement(cv2.MORPH_RECT, (join, join)))
    n, cl, st, _ = cv2.connectedComponentsWithStats(d, 8)
    out = []
    for i in range(1, n):
        m = (lab & (cl == i)).astype(np.uint8)
        if m.sum() < 200: continue
        fs = []
        for k in (3, 5, 9, 15, 25):
            c = cv2.morphologyEx(cv2.copyMakeBorder(m, k, k, k, k, cv2.BORDER_CONSTANT, 0), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))[k:-k, k:-k]
            fs.append(_fill(c))
        amax = fs[-1].sum()
        fp = next(f for f in fs if f.sum() >= 0.92 * amax)       # 채움이 거의 다 된 가장 작은 닫기(오목한 곳을 덜 메움)
        wm = (wl > 0) & (cl == i)
        ic = float(inkd[wm].mean()) if inkd is not None and wm.any() else None
        out.append(dict(fp=fp.astype(bool), lab=m.astype(bool), box=[int(v) for v in st[i, :4]], ink_cov=ic))
    if out:                                                   # 떨어진 작은 조각(실외기 선반·벽 토막 라벨)은 평면이 아님 → 채점 제외
        amax = max(g['fp'].sum() for g in out)
        out = [g for g in out if g['fp'].sum() >= 0.1 * amax]
    return out


MASS = os.environ.get('FP_MASS', '0') == '1'      # (2026-10-03) 예측 외곽 = fire3dbeam/realcard.mass(STR 라벨 → 건물 덩어리·마당형 벗기기) — 정보카드 실제 도면 경로의 외곽


def mass_outline(ml, r, Hq, Wq):
    """realplan 'plan' 영역 → realcard.mass 외곽(Q 해상도 마스크). ml = 시트 1/2 해상도 STR 라벨, r = 영역 결과"""
    from fire3dbeam import realcard as RC, str_seg as SS
    x0, y0, x1, y1 = [v * 2 for v in r['box_Q']]                     # 1/4 → 1/2
    sub = ml[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20]
    ox, oy = max(0, x0 - 20), max(0, y0 - 20)
    s = r.get('px_per_mm')
    ppm_h = s * 1000 / 2 if s else 300 / 25.4 * 10 / 2          # 축척 없으면 1/100@300dpi 가정
    k = SS.WORK_PPM_1_100 / ppm_h
    lw = cv2.resize(sub, (max(8, round(sub.shape[1] * k)), max(8, round(sub.shape[0] * k))), interpolation=cv2.INTER_NEAREST)
    v = SS.vectorize(lw, SS.WORK_PPM_1_100)
    M = RC.mass(lw, SS.WORK_PPM_1_100, walls=v['walls'])['mask']
    mq = np.zeros((Hq, Wq), np.uint8)
    h2, w2 = sub.shape                                                # 1/2 해상도 → Q(1/4)
    mm = cv2.resize(M, (max(1, w2 // 2), max(1, h2 // 2)), interpolation=cv2.INTER_NEAREST)
    yq, xq = oy // 2, ox // 2
    hh, ww = min(mm.shape[0], Hq - yq), min(mm.shape[1], Wq - xq)
    if hh > 0 and ww > 0: mq[yq:yq + hh, xq:xq + ww] = mm[:hh, :ww]
    return mq


def predict(it, model=True, want_dims=True):
    from fire3dbeam import realplan as R
    if model and R._SEG is None:
        from fire3dbeam import str_seg as SS
        R._SEG = SS.StrSeg(threads=1)
    g = cv2.imread(os.path.join(D.IMG_DIR, it['fn']), cv2.IMREAD_GRAYSCALE)
    gF = cv2.resize(g, (it['w'], it['h']), interpolation=cv2.INTER_CUBIC)          # realplan 은 300dpi 원본 기준
    msrc = None
    if model:
        cp = os.path.join(MCACHE, it['stem'] + '_mH.png') if MCACHE else None
        if cp and os.path.exists(cp): msrc = dict(lab=cv2.imread(cp, cv2.IMREAD_GRAYSCALE))
        else:
            msrc = dict(img=g, dpi=300 * it['f'])
    out, vizs, q, fr = R.process_gray(gF, it['fn'], want_dims, False, model, model_src=msrc)
    if model and MCACHE and msrc.get('lab') is None and R._LAST_ML is not None:
        os.makedirs(MCACHE, exist_ok=True); cv2.imwrite(os.path.join(MCACHE, it['stem'] + '_mH.png'), R._LAST_ML)
    Hq, Wq = round(it['h'] * Q), round(it['w'] * Q)
    pm = np.zeros((Hq, Wq), np.uint8)
    regs = []
    for r, v in zip(out['regions'], vizs):
        if r['kind'] != 'plan' or not r.get('outline'): continue
        ox, oy = r['offset_F']
        if MASS and R._LAST_ML is not None:
            pm |= mass_outline(R._LAST_ML, r, Hq, Wq)
        else:
            ps = [np.round((np.array(p, float) + [ox, oy]) * Q).astype(np.int32) for p in r['outline']]
            cv2.fillPoly(pm, ps, 1)
        regs.append(dict(style=r['style'], s=r.get('px_per_mm'), conf=r.get('scale_conf'), n_dims=r.get('n_dims'),
                         err=(r.get('total_check') or {}).get('err'), scale_1_to=r.get('scale_1_to')))
    return pm, regs, out['runtime_s']


def score(gts, pm):
    n, pl = cv2.connectedComponents(pm, connectivity=8)
    allg = np.zeros(pm.shape, bool)
    for g in gts: allg |= g['fp']
    res = []
    for j, g in enumerate(gts):
        G = g['fp']
        ids = np.unique(pl[G & (pm > 0)])
        ids = [i for i in ids if i > 0 and (pl[G] == i).sum() >= 0.02 * G.sum()]
        P = np.isin(pl, ids) if ids else np.zeros_like(G)
        other = np.zeros_like(G)
        for k, g2 in enumerate(gts):
            if k != j: other |= g2['fp']
        P = P & ~(other & ~G)
        inter = float((P & G).sum()); uni = float((P | G).sum())
        res.append(dict(ink_cov=None if g.get('ink_cov') is None else round(g['ink_cov'], 3), iou=round(inter / uni, 4) if uni else 0.0, rec=round(inter / G.sum(), 4), prec=round(inter / P.sum(), 4) if P.sum() else 0.0,
                        area_gt=int(G.sum()), area_pred=int(P.sum()), n_comp=len(ids)))
    return res


def work(it, model, want_dims, ovdir=None):
    t0 = time.time()
    try:
        Hq, Wq = round(it['h'] * Q), round(it['w'] * Q)
        g0 = cv2.imread(os.path.join(D.IMG_DIR, it['fn']), cv2.IMREAD_GRAYSCALE)
        gts = gt_clusters(it, (Hq, Wq), cv2.resize(g0, (Wq, Hq), interpolation=cv2.INTER_AREA))
        pm, regs, rt = predict(it, model, want_dims)
        sc = score(gts, pm)
        if ovdir:
            g = cv2.imread(os.path.join(D.IMG_DIR, it['fn']), cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(cv2.resize(g, (Wq, Hq), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            img = (0.6 * img + 0.4 * 255).astype(np.uint8)
            for gg in gts:
                cs, _ = cv2.findContours(gg['fp'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (0, 170, 0), 2)
            cs, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 0, 230), 1)
            txt = ' '.join(f"{s['iou']:.2f}" for s in sc)
            cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
            cv2.imwrite(os.path.join(ovdir, it['stem'] + '.png'), img)
        return dict(stem=it['stem'], type=it['stem'].split('_')[0], clusters=sc, regions=regs, runtime_s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(stem=it['stem'], error=repr(e), tb=traceback.format_exc()[-1500:])


def pct(a, b): return round(100.0 * a / b, 1) if b else None


def summarize(rows):
    ok = [r for r in rows if 'error' not in r]
    C = [(r, c) for r in ok for c in r['clusters']]
    def st(cs):
        v = np.array([c['iou'] for _, c in cs], float)
        if not len(v): return None
        return dict(n=len(v), mean_iou=round(float(v.mean()), 4), median_iou=round(float(np.median(v)), 4),
                    ge90=pct((v >= 0.9).sum(), len(v)), ge80=pct((v >= 0.8).sum(), len(v)), ge50=pct((v >= 0.5).sum(), len(v)),
                    missed=pct(sum(1 for _, c in cs if c['area_pred'] == 0), len(v)),
                    mean_rec=round(float(np.mean([c['rec'] for _, c in cs])), 4),
                    mean_prec=round(float(np.mean([c['prec'] for _, c in cs if c['area_pred'] > 0] or [0])), 4))
    S = dict(n_sheets=len(rows), crash=len(rows) - len(ok), all=st(C),
             clean=st([(r, c) for r, c in C if (c.get('ink_cov') or 0) >= 0.6]),   # 라벨 위치 어긋남 의심(라벨 벽의 잉크 위 비율<0.6) 제외
             runtime_mean_s=round(float(np.mean([r['runtime_s'] for r in ok])), 1) if ok else None)
    S['by_type'] = {t: st([(r, c) for r, c in C if r['type'] == t]) for t in sorted(set(r['type'] for r in ok))}
    R_ = [g for r in ok for g in r['regions']]
    S['plan_regions'] = len(R_)
    S['scale_found'] = pct(sum(1 for g in R_ if g['s']), len(R_))
    S['scale_conf_ge_0.5'] = pct(sum(1 for g in R_ if (g['conf'] or 0) >= 0.5), len(R_))
    E = np.array([g['err'] for g in R_ if g['err'] is not None], float)
    S['dim_err_le5'] = pct((E <= 0.05).sum(), len(E)); S['dim_checked'] = len(E)
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'summ'])
    ap.add_argument('--split', default='dev', choices=['dev', 'held', 'val', 'train'])
    ap.add_argument('--limit', type=int, default=80)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--rules', action='store_true', help='규칙 벽(기본은 모델 경로)')
    ap.add_argument('--nodims', action='store_true')
    ap.add_argument('--tag', default='cur')
    ap.add_argument('--overlays', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--stems', nargs='*')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f'{a.tag}_{a.split}')
    if a.cmd == 'summ':
        rows = [json.loads(l) for l in open(fn + '.jsonl')]
    else:
        its = items(a.split)[a.offset:a.offset + a.limit] if a.limit else items(a.split)
        if a.stems: its = [it for it in items(a.split) if it['stem'] in a.stems]
        ovd = None
        if a.overlays:
            ovd = fn + '_ov'; os.makedirs(ovd, exist_ok=True)
        rows = []; t0 = time.time()
        with open(fn + '.jsonl', 'w') as fo:
            for k, it in enumerate(its):
                r = work(it, not a.rules, not a.nodims, ovd if k < a.overlays else None)
                rows.append(r); fo.write(json.dumps(r, ensure_ascii=False) + '\n'); fo.flush()
                if 'error' in r: print('ERROR', r['stem'], r['error'], r['tb'], flush=True)
                if (k + 1) % 10 == 0: print(f'{k + 1}/{len(its)} {time.time() - t0:.0f}s', flush=True)
    S = summarize(rows); S.update(tag=a.tag, split=a.split, model=not a.rules)
    json.dump(S, open(fn + '.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))
