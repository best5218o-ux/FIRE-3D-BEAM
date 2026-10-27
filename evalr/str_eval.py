# -*- coding: utf-8 -*-
"""AI허브 STR 라벨(벽·창·문) 보류 세트 채점 — 규칙 기반 realplan(+realsym 문 호) vs 분할 모델(str_seg).

분할: 파일명 md5 → 20% 보류(held) / 80% 개발(dev; 그중 10% dev-val). strseg/data.py split_of 와 같음.
평가 해상도: 2048 축소 영상의 1/2 = 긴 변 1024 (≈62dpi, 원본 300dpi 의 0.206).
라벨 없는 영역: 라벨 다각형 덩어리마다 외접 상자+여백(strseg/data.valid_mask) 밖은 채점 제외(예측도 무시).
지표:
  - 벽 화소 IoU / F1, 창·문 화소 IoU (유효 영역 안, 전체 화소 합산)
  - 벽 중심선 재현/정밀: 골격 화소가 상대 골격에서 tol px 안(기본 5px ≈ 1/100 도면 0.2 m)
  - 문·창 개체 재현/정밀: 정답 개체 중심이 예측 성분(2px 여유) 안 또는 IoU>0.3 이면 일치. 예측 성분(면적≥4px, 중심이 유효 영역)마다 일치 여부
  - 벽 표현(realplan wall_style: filled/filled_gray/hatched/double/single)별, 도면 유형(APT/ROW/DEH)별, 벽 속성(철근콘크리트벽/기타벽)별 재현
방법:
  baseline   realplan(sheet_layout→wall_style→wall_mask→keep_main, 원본 크기로 되돌린 영상) 벽 + realsym.detect_doors 문(호 4분면 상자). 창 없음
  model      str_seg ONNX 분할(작업 해상도 ≈74dpi) 확률 → 평가 해상도 argmax
  model_vec  model 을 벡터화(축 선분+두께, 사선 벽, 문·창 개체 상자) 후 다시 그린 것 — 엔진에 들어가는 표현의 정확도
사용:
  python -m evalr.str_eval pred baseline --limit 150 [--workers 1]
  python -m evalr.str_eval pred model [--onnx models/str_seg.onnx]     (pred 의 방법 이름은 자유: 예 model_v1 --onnx 옛모델.onnx)
  python -m evalr.str_eval score baseline model_v1 model [--subset out_str/orig83.json --tag _orig83]
  python -m evalr.str_eval score [--out out_str]   → out_str/str_eval.json (+ 표 출력)
  python -m evalr.str_eval overlay STEM ...          → out_str/overlay/STEM.png (정답|기준|모델)
※ AI허브 자료 — 예측·겹침 그림은 로컬 전용, 공개 금지.
"""
import os, sys, json, time, argparse, traceback
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strseg import data as D

EVAL_LONG = 1024
OUT = 'out_str'
TOL = 5


SPLIT = 'held'


def held_items(limit=0, split=None):
    its = [it for it in D.list_images() if D.split_of(it['stem']) == (split or SPLIT)]
    its.sort(key=lambda it: D.h01(it['stem'], 'order'))           # 고정된 무작위 순서(자료가 늘어도 앞부분 유지 경향)
    return its[:limit] if limit else its


def eval_shape(it):
    g = cv2.imread(os.path.join(D.IMG_DIR, it['fn']), cv2.IMREAD_GRAYSCALE)
    r = EVAL_LONG / max(g.shape)
    return g, (round(g.shape[0] * r), round(g.shape[1] * r)), r


# ───────────────────────── 예측: 기준(realplan) ─────────────────────────
def pred_baseline(it):
    from fire3dbeam import realplan as R, realsym as S
    g, es, r = eval_shape(it)
    gF = cv2.resize(g, (it['w'], it['h']), interpolation=cv2.INTER_CUBIC)       # realplan 은 300dpi 원본 기준
    q = R.minpool(gF, 4); regs, caps, fr = R.sheet_layout(q)
    Hh, Wh = gF.shape[0] // 2, gF.shape[1] // 2
    wall = np.zeros((Hh, Wh), np.uint8)
    k = es[1] / gF.shape[1]                                                      # 원본 → 평가
    doors, styles = [], []
    for rg in regs:
        res, viz = R.analyze_region(gF, rg['box'], fr, True)
        ob = res.get('outline_bbox_F')
        kind = 'plan' if ob and min(ob[2] - ob[0], ob[3] - ob[1]) >= 240 and res['n_wall_segs'] >= 8 else 'other'
        box_e = [round(v * 4 * k) for v in rg['box']]
        styles.append(dict(kind=kind, style=res['style'], box=box_e, px_per_mm=res.get('px_per_mm')))
        if kind != 'plan': continue
        ox, oy = res['offset_F'][0] // 2, res['offset_F'][1] // 2
        m = viz['m']; hh, ww = min(m.shape[0], Hh - oy), min(m.shape[1], Wh - ox)
        wall[oy:oy + hh, ox:ox + ww] |= m[:hh, :ww]
        try:
            for d in S.detect_doors(gF, box=[v * 4 for v in rg['box']]):
                (hx, hy), rx, ry = d['hinge'], d['rx'], d['ry']
                sx = 1 if d['swing'] in ('SE', 'NE') else -1; sy = 1 if d['swing'] in ('SE', 'SW') else -1
                x0, x1 = sorted([hx, hx + sx * rx]); y0, y1 = sorted([hy, hy + sy * ry])
                doors.append([x0 * k, y0 * k, x1 * k, y1 * k])
        except Exception:
            pass
    wf = cv2.resize(wall.astype(np.float32), (es[1], es[0]), interpolation=cv2.INTER_AREA)
    lab = (wf >= 0.3).astype(np.uint8)
    for x0, y0, x1, y1 in doors:
        lab[int(round(y0)):int(round(y1)) + 1, int(round(x0)):int(round(x1)) + 1] = 3
    return lab, dict(styles=styles, n_doors=len(doors))


# ───────────────────────── 예측: 모델 ─────────────────────────
_SEG = None


def pred_model(it, onnx=None):
    global _SEG
    from fire3dbeam import str_seg as SS
    if _SEG is None: _SEG = SS.StrSeg(onnx or SS.MODEL, threads=2)
    g, es, r = eval_shape(it)
    t = time.time()
    lab_w, p, k = _SEG.predict(g, src_dpi=300 * it['f'])
    t_inf = time.time() - t
    pe = np.stack([cv2.resize(c, (es[1], es[0]), interpolation=cv2.INTER_AREA) for c in p])
    lab = pe.argmax(0).astype(np.uint8)
    t = time.time()
    v = SS.vectorize(lab_w, P=_SEG.vec_param)
    t_vec = time.time() - t
    lab_v = SS.render_vectors(v, tuple(es), scale=vec_scale(lab_w.shape, es))   # 평가 해상도에 바로 그림(최근접 축소는 ≈0.5px 밀림)
    return lab, lab_v, dict(t_infer=round(t_inf, 2), t_vec=round(t_vec, 2), n_walls=len(v['walls']), n_diag=len(v['walls_diag']),
                            n_doors=len(v['doors']), n_windows=len(v['windows']))


def vec_scale(ws, es):
    """작업 해상도 → 평가 해상도 배율(가로·세로 평균)"""
    return (es[1] / ws[1] + es[0] / ws[0]) / 2


# ───────────────────────── 채점 ─────────────────────────
def _skel(m):
    from fire3dbeam.str_seg import skeleton
    return skeleton(m) if m.any() else m.astype(np.uint8)


def gt_instances(anns, s, shape, cls):
    out = []
    for a in anns:
        if a['c'] != cls: continue
        ps = [np.round(p * s).astype(np.int32) for p in a['polys']]
        x0 = min(p[:, 0].min() for p in ps); y0 = min(p[:, 1].min() for p in ps)
        x1 = max(p[:, 0].max() for p in ps); y1 = max(p[:, 1].max() for p in ps)
        out.append(dict(ps=ps, box=(int(x0), int(y0), int(x1), int(y1)), c=((x0 + x1) / 2, (y0 + y1) / 2)))
    return out


def inst_match(pred_c, gts, valid, min_area=4):
    n, lab, st, cen = cv2.connectedComponentsWithStats(pred_c.astype(np.uint8), 8)
    ok = np.zeros(n, bool)
    for i in range(1, n):
        cx, cy = cen[i]
        ok[i] = st[i, 4] >= min_area and valid[min(valid.shape[0] - 1, int(cy)), min(valid.shape[1] - 1, int(cx))]
    labd = cv2.dilate(lab.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)   # 2px 여유(라벨 번호 최대값)
    hit_pred = np.zeros(n, bool); hit_gt = 0
    H, W = lab.shape
    for gt in gts:
        cx, cy = int(round(gt['c'][0])), int(round(gt['c'][1]))
        js = set()
        if 0 <= cy < H and 0 <= cx < W:
            j = labd[cy, cx]
            if j and ok[j]: js.add(j)
        x0, y0, x1, y1 = gt['box']; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W - 1, x1), min(H - 1, y1)
        if x1 >= x0 and y1 >= y0:
            gm = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            cv2.fillPoly(gm, [p - [x0, y0] for p in gt['ps']], 1)
            sub = lab[y0:y1 + 1, x0:x1 + 1][gm > 0]
            if sub.size:
                cnt = np.bincount(sub, minlength=n)
                for j in np.nonzero(cnt[1:])[0] + 1:
                    if not ok[j]: continue
                    inter = cnt[j]; iou = inter / (gm.sum() + st[j, 4] - inter)
                    if iou > 0.3: js.add(j)
        if js: hit_gt += 1
        for j in js: hit_pred[j] = True
    return dict(gt=len(gts), gt_hit=hit_gt, pred=int(ok.sum()), pred_hit=int((hit_pred & ok).sum()))


def score_image(it, pred, tol=TOL):
    g, es, r = eval_shape(it)
    s = it['f'] * r
    anns = D.load_ann(it['stem'])
    gt = D.raster(anns, es, s); valid = D.valid_mask(anns, es, s) > 0
    if pred.shape != es: pred = cv2.resize(pred, (es[1], es[0]), interpolation=cv2.INTER_NEAREST)
    R = {}
    for c, nm in ((1, 'wall'), (2, 'window'), (3, 'door')):
        a, b = (pred == c) & valid, (gt == c) & valid
        R[nm] = dict(tp=int((a & b).sum()), fp=int((a & ~b).sum()), fn=int((~a & b).sum()))
    # 중심선
    gs, ps_ = _skel((gt == 1) & valid), _skel((pred == 1) & valid)
    dg = cv2.distanceTransform((1 - gs).astype(np.uint8), cv2.DIST_L2, 3); dp = cv2.distanceTransform((1 - ps_).astype(np.uint8), cv2.DIST_L2, 3)
    R['cl'] = dict(gt=int(gs.sum()), gt_hit=int((dp[gs > 0] <= tol).sum()), pred=int(ps_.sum()), pred_hit=int((dg[ps_ > 0] <= tol).sum()))
    # 벽 속성별 재현(화소)
    for sub in ('철근콘크리트벽', '기타벽'):
        m = np.zeros(es, np.uint8)
        ps = [np.round(p * s).astype(np.int32) for a in anns if a['c'] == 1 and a['sub'] == sub for p in a['polys']]
        if ps: cv2.fillPoly(m, ps, 1)
        m = (m > 0) & valid
        R['wall_' + sub] = dict(gt=int(m.sum()), hit=int((m & (pred == 1)).sum()))
    for c, nm in ((2, 'window_inst'), (3, 'door_inst')):
        R[nm] = inst_match(pred == c, gt_instances(anns, s, es, c), valid)
    R['valid_frac'] = round(float(valid.mean()), 3)
    return R


def image_style(bj, valid_box=None):
    """기준 실행의 평면 영역 중 가장 큰 것의 wall_style"""
    if not bj: return 'unknown'
    P = [x for x in bj.get('styles', []) if x['kind'] == 'plan'] or bj.get('styles', [])
    if not P: return 'none'
    return max(P, key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]))['style']


def agg(rows):
    def sm(k, f): return sum(r[k][f] for r in rows)
    S = {}
    for nm in ('wall', 'window', 'door'):
        tp, fp, fn = sm(nm, 'tp'), sm(nm, 'fp'), sm(nm, 'fn')
        S[nm] = dict(iou=round(tp / max(1, tp + fp + fn), 4), f1=round(2 * tp / max(1, 2 * tp + fp + fn), 4),
                     p=round(tp / max(1, tp + fp), 4), r=round(tp / max(1, tp + fn), 4))
    S['wall_centerline'] = dict(r=round(sm('cl', 'gt_hit') / max(1, sm('cl', 'gt')), 4), p=round(sm('cl', 'pred_hit') / max(1, sm('cl', 'pred')), 4))
    for nm in ('door_inst', 'window_inst'):
        S[nm] = dict(r=round(sm(nm, 'gt_hit') / max(1, sm(nm, 'gt')), 4), p=round(sm(nm, 'pred_hit') / max(1, sm(nm, 'pred')), 4),
                     n_gt=sm(nm, 'gt'), n_pred=sm(nm, 'pred'))
    for sub in ('철근콘크리트벽', '기타벽'):
        S['wall_recall_' + sub] = round(sm('wall_' + sub, 'hit') / max(1, sm('wall_' + sub, 'gt')), 4)
    S['n'] = len(rows)
    return S


def pred_dir(out, method): return os.path.join(out, 'pred', method)


def run_pred(method, items, out, onnx=None):
    mv = method + '_vec'
    for m in ([method] if method == 'baseline' else [method, mv]): os.makedirs(pred_dir(out, m), exist_ok=True)
    t0 = time.time(); n = 0
    for it in items:
        st = it['stem']
        if os.path.exists(os.path.join(pred_dir(out, method), st + '.png')): continue
        try:
            if method == 'baseline':
                lab, info = pred_baseline(it)
                json.dump(info, open(os.path.join(pred_dir(out, 'baseline'), st + '.json'), 'w'))
                cv2.imwrite(os.path.join(pred_dir(out, 'baseline'), st + '.png'), lab)
            else:
                lab, lab_v, info = pred_model(it, onnx)
                json.dump(info, open(os.path.join(pred_dir(out, method), st + '.json'), 'w'))
                cv2.imwrite(os.path.join(pred_dir(out, mv), st + '.png'), lab_v)
                cv2.imwrite(os.path.join(pred_dir(out, method), st + '.png'), lab)
        except Exception as e:
            print('ERR', st, repr(e), traceback.format_exc()[-400:], flush=True)
        n += 1
        if n % 10 == 0: print(f'{method} {n}/{len(items)} {time.time() - t0:.0f}s', flush=True)


def run_score(out, methods=('baseline', 'model', 'model_vec'), tol=TOL, subset=None, tag=''):
    items = held_items()
    if subset: keep = set(json.load(open(subset))); items = [it for it in items if it['stem'] in keep]
    have = {m: {os.path.splitext(f)[0] for f in os.listdir(pred_dir(out, m)) if f.endswith('.png')} if os.path.isdir(pred_dir(out, m)) else set() for m in methods}
    rows = {m: [] for m in methods}; meta = {}
    for it in items:
        st = it['stem']
        bj = os.path.join(pred_dir(out, 'baseline'), st + '.json')
        style = image_style(json.load(open(bj))) if os.path.exists(bj) else 'unknown'
        meta[st] = dict(style=style, type=st.split('_')[0])
        for m in methods:
            if st not in have[m]: continue
            p = cv2.imread(os.path.join(pred_dir(out, m), st + '.png'), cv2.IMREAD_GRAYSCALE)
            r = score_image(it, p, tol); r['stem'] = st; rows[m].append(r)
    common = set.intersection(*[set(r['stem'] for r in rows[m]) for m in methods if rows[m]]) if any(rows.values()) else set()
    S = dict(split=SPLIT, eval_long=EVAL_LONG, tol_px=tol, n_held_total=len(items), methods={})
    for m in methods:
        if not rows[m]: continue
        R = rows[m]; C = [r for r in R if r['stem'] in common]
        d = dict(all=agg(R), common=agg(C) if C else None)
        d['by_style_common'] = {st: agg([r for r in C if meta[r['stem']]['style'] == st]) for st in sorted({meta[r['stem']]['style'] for r in C})}
        d['by_style_all'] = {st: agg([r for r in R if meta[r['stem']]['style'] == st]) for st in sorted({meta[r['stem']]['style'] for r in R})}
        d['by_type_all'] = {t: agg([r for r in R if meta[r['stem']]['type'] == t]) for t in sorted({meta[r['stem']]['type'] for r in R})}
        S['methods'][m] = d
    S['n_common'] = len(common)
    os.makedirs(out, exist_ok=True)
    S['subset'] = subset
    json.dump(S, open(os.path.join(out, f'str_eval{tag}.json'), 'w'), ensure_ascii=False, indent=1)
    with open(os.path.join(out, f'str_eval{tag}_rows.jsonl'), 'w') as fo:
        for m in methods:
            for r in rows[m]: fo.write(json.dumps(dict(method=m, **r), ensure_ascii=False) + '\n')
    return S


def table(S, key='common'):
    lines = [f"[{key}] n={S['n_common'] if key == 'common' else '-'}  (eval long side {S['eval_long']}, centerline tol {S['tol_px']}px)",
             f"{'method':10s} {'n':>4s} {'wallIoU':>7s} {'wallF1':>6s} {'clR':>5s} {'clP':>5s} {'winIoU':>6s} {'doorIoU':>7s} {'doorR':>5s} {'doorP':>5s} {'winR':>5s} {'winP':>5s} {'RC_R':>5s} {'etc_R':>5s}"]
    for m, d in S['methods'].items():
        a = d[key]
        if not a: continue
        lines.append(f"{m:10s} {a['n']:4d} {a['wall']['iou']:7.3f} {a['wall']['f1']:6.3f} {a['wall_centerline']['r']:5.3f} {a['wall_centerline']['p']:5.3f} "
                     f"{a['window']['iou']:6.3f} {a['door']['iou']:7.3f} {a['door_inst']['r']:5.3f} {a['door_inst']['p']:5.3f} {a['window_inst']['r']:5.3f} {a['window_inst']['p']:5.3f} "
                     f"{a['wall_recall_철근콘크리트벽']:5.3f} {a['wall_recall_기타벽']:5.3f}")
    return '\n'.join(lines)


def style_table(S, key='by_style_common'):
    lines = [f"{key}: wall IoU / centerline R,P / door R,P"]
    for m, d in S['methods'].items():
        for st, a in d[key].items():
            lines.append(f"  {m:10s} {st:12s} n={a['n']:3d} IoU {a['wall']['iou']:.3f}  cl {a['wall_centerline']['r']:.3f}/{a['wall_centerline']['p']:.3f}  door {a['door_inst']['r']:.3f}/{a['door_inst']['p']:.3f}")
    return '\n'.join(lines)


def overlay(stem, out):
    it = [i for i in D.list_images() if i['stem'] == stem][0]
    g, es, r = eval_shape(it)
    ge = cv2.resize(g, (es[1], es[0]), interpolation=cv2.INTER_AREA)
    s = it['f'] * r; anns = D.load_ann(stem)
    gt = D.raster(anns, es, s); v = D.valid_mask(anns, es, s)
    from fire3dbeam.str_seg import overlay as ov
    tiles = []
    for nm, lab in [('GT', gt)] + [(m, cv2.imread(os.path.join(pred_dir(out, m), stem + '.png'), 0)) for m in ('baseline', 'model', 'model_vec')]:
        if lab is None: continue
        im = ov(ge, lab); im[v == 0] = (im[v == 0] * 0.7).astype(np.uint8)
        cv2.putText(im, nm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        tiles.append(im)
    ys, xs = np.nonzero(v)
    if len(ys): tiles = [t[max(0, ys.min() - 10):ys.max() + 10, max(0, xs.min() - 10):xs.max() + 10] for t in tiles]
    od = os.path.join(out, 'overlay'); os.makedirs(od, exist_ok=True)
    img = np.hstack(tiles) if tiles[0].shape[1] < tiles[0].shape[0] * 1.2 else np.vstack(tiles)
    cv2.imwrite(os.path.join(od, stem + '.png'), img)
    return os.path.join(od, stem + '.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pred', 'score', 'overlay'])
    ap.add_argument('arg', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--onnx', default=None)
    ap.add_argument('--tol', type=int, default=TOL)
    ap.add_argument('--shard', default='0/1', help='i/n: 여러 프로세스로 나눠 돌릴 때')
    ap.add_argument('--split', default='held', help='held(보고용) | val(dev-val: 후처리 조정용)')
    ap.add_argument('--subset', default=None, help='채점할 stem 목록 JSON (예: out_str/orig83.json = 첫 보류 83장)')
    ap.add_argument('--tag', default='', help='결과 파일 접미사 str_eval<tag>.json')
    a = ap.parse_args()
    SPLIT = a.split
    if a.cmd == 'pred':
        its = held_items(a.limit)
        if a.subset:                                                   # 예: evalr/held674.json (이전 보고와 같은 674장)
            keep = set(json.load(open(a.subset))); its = [it for it in its if it['stem'] in keep]
        i, n = map(int, a.shard.split('/')); its = its[i::n]
        run_pred(a.arg[0], its, a.out, a.onnx)
    elif a.cmd == 'score':
        S = run_score(a.out, tuple(a.arg) if a.arg else ('baseline', 'model', 'model_vec'), a.tol, a.subset, a.tag)
        print(table(S, 'common')); print(table(S, 'all')); print(style_table(S))
    else:
        for st in a.arg: print(overlay(st, a.out))
