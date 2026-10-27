# -*- coding: utf-8 -*-
"""실제 도면 평가 하네스 (라벨 없음 → 치수 자기지도 지표)

사용:
  python -m evalr.realplan_eval ./aihub/obj2 --split dev  [--limit 60] [--out out_real] [--overlays 25]
  python -m evalr.realplan_eval ./aihub/obj2 --split hold          # 보류 세트(파일명 정렬 후 5번째마다) — 마지막에만
  python -m evalr.realplan_eval ./aihub/obj2 --split all --limit 60 [--rules] --out out_real_model   # 기본 벽·창·문 = 분할 모델(2026-09-30~), --rules = 규칙 벽
출력: <out>/summary_<split>.json (+ 마지막 실행은 summary.json 으로 복사), results_<split>.jsonl, overlays/*.png
수동 판정: evalr/realplan_manual.json (있으면 자동 지표와 비교해 summary 에 기록)
※ AI허브 원천 이미지·겹침 그림은 로컬 시험용. 외부 공개·재배포 금지.
"""
import os, sys, json, glob, time, argparse, random, traceback, shutil
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANUAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realplan_manual.json')
MCACHE = os.environ.get('REALPLAN_MODEL_CACHE', '')    # (선택) 모델 라벨 캐시 폴더 — <폴더>/<모델이름>/<stem>_mH.png


def split_files(folder):
    fs = sorted(f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    dev = [f for i, f in enumerate(fs) if i % 5 != 4]
    hold = [f for i, f in enumerate(fs) if i % 5 == 4]
    return dev, hold


def work(args):
    path, od, draw, model = args
    import cv2
    from fire3dbeam import realplan as R
    t0 = time.time()
    try:
        stem = os.path.splitext(os.path.basename(path))[0]
        msrc, cp = None, None
        if model:
            from fire3dbeam import str_seg as SS
            if MCACHE:                                   # 모델 라벨 캐시(모델 파일 이름별 폴더) — 축척·외곽 규칙만 바꿔 다시 돌릴 때
                cp = os.path.join(MCACHE, os.path.splitext(os.path.basename(SS.MODEL))[0], stem + '_mH.png')
                if os.path.exists(cp): msrc = dict(lab=cv2.imread(cp, cv2.IMREAD_GRAYSCALE))
            if msrc is None and R._SEG is None: R._SEG = SS.StrSeg(threads=1)
        gF = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gF is None: raise IOError(path)
        out, vizs, q, fr = R.process_gray(gF, os.path.basename(path), True, True, model, model_src=msrc)
        if cp and msrc is None and R._LAST_ML is not None:
            os.makedirs(os.path.dirname(cp), exist_ok=True); cv2.imwrite(cp, R._LAST_ML)
        if draw:
            cv2.imwrite(os.path.join(od, stem + '_sheet.png'), R.draw_sheet(out, q, fr))
            for i, (r, v) in enumerate(zip(out['regions'], vizs)):
                if r['kind'] == 'plan': cv2.imwrite(os.path.join(od, f'{stem}_r{i}.png'), R.draw_region(r, v))
        for r in out['regions']:
            r.pop('outline', None)
            r['dims'] = [dict(mm=d['mm'], seg=round(d['seg'], 1), rot=d['rot'], inlier=d.get('inlier')) for d in r['dims']]
        return out
    except Exception as e:
        return dict(file=os.path.basename(path), error=repr(e), tb=traceback.format_exc()[-800:], runtime_s=round(time.time() - t0, 2))


def pct(a, b): return round(100.0 * a / b, 1) if b else None


def summarize(rows):
    ok = [r for r in rows if 'error' not in r]
    regs = [(r['file'], i, g) for r in ok for i, g in enumerate(r['regions'])]
    plans = [(f, i, g) for f, i, g in regs if g['kind'] == 'plan']
    S = dict(n_images=len(rows), crash=len(rows) - len(ok), crash_rate=pct(len(rows) - len(ok), len(rows)),
             runtime_mean_s=round(float(np.mean([r['runtime_s'] for r in rows])), 2) if rows else None,
             runtime_p90_s=round(float(np.percentile([r['runtime_s'] for r in rows], 90)), 2) if rows else None,
             n_regions=len(regs), n_plan_regions=len(plans),
             images_with_plan=pct(sum(1 for r in ok if any(g['kind'] == 'plan' for g in r['regions'])), len(rows)),
             plans_per_image=round(len(plans) / max(1, len(ok)), 2))
    P = [g for _, _, g in plans]
    S['outline_found'] = pct(sum(1 for g in P if g.get('outline_bbox_F')), len(P))
    S['scale_found'] = pct(sum(1 for g in P if g.get('px_per_mm')), len(P))
    S['scale_conf_ge_0.5'] = pct(sum(1 for g in P if (g.get('scale_conf') or 0) >= 0.5), len(P))
    src = lambda g: g.get('scale_src') or ('region' if g.get('px_per_mm') else None)
    S['scale_found_dims'] = pct(sum(1 for g in P if src(g) in ('region', 'sheet')), len(P))
    S['scale_src'] = {k: sum(1 for g in P if src(g) == k) for k in ('region', 'sheet', 'doors', None)}
    # 독립 검사: 치수 축척 ↔ 모델 문 폭(중앙값 ≈1040 mm) 축척이 ±20% 안 (문 3개 이상인 영역)
    ag = [abs(np.log(np.median([d['w'] for d in g['doors']]) / 1040.0 / g['px_per_mm'])) < np.log(1.2)
          for g in P if src(g) in ('region', 'sheet') and len(g.get('doors') or []) >= 3]
    S['scale_door_agree'] = pct(sum(ag), len(ag)); S['scale_door_n'] = len(ag)
    S['style_dist'] = {st: sum(1 for g in P if g['style'] == st) for st in sorted(set(g['style'] for g in P))}
    S['scale_1_to_hist'] = {}
    for g in P:
        if g.get('scale_1_to'):
            k = int(round(g['scale_1_to'] / 25.0) * 25); S['scale_1_to_hist'][k] = S['scale_1_to_hist'].get(k, 0) + 1
    S['scale_1_to_hist'] = dict(sorted(S['scale_1_to_hist'].items()))
    E = [g['total_check'] for g in P if g.get('total_check')]
    S['checked'] = len(E); S['checked_pct_of_plans'] = pct(len(E), len(P))

    def stats(v):
        v = np.array(v, float)
        if not len(v): return None
        return dict(n=len(v), median=round(float(np.median(v)), 4), mean=round(float(np.mean(v)), 4),
                    le2=pct((v <= 0.02).sum(), len(v)), le5=pct((v <= 0.05).sum(), len(v)), le10=pct((v <= 0.10).sum(), len(v)))
    S['err_union'] = stats([e['err'] for e in E])                  # 주지표: 치수 사슬 범위 vs 외곽 범위
    S['err_union_axis'] = stats([e[a]['err'] for e in E for a in ('x', 'y') if a in e])
    S['err_total'] = stats([e['err_total'] for e in E if e.get('err_total') is not None])   # 최외곽 단일 전체치수 vs 외곽
    S['plans_err_le5_of_all_plans'] = pct(sum(1 for e in E if e['err'] <= 0.05), len(P))
    by = {}
    for t in sorted(set(f.split('_')[0] for f, _, _ in plans)):
        Pt = [g for f, _, g in plans if f.split('_')[0] == t]; Et = [g['total_check'] for g in Pt if g.get('total_check')]
        by[t] = dict(plans=len(Pt), scale_found=pct(sum(1 for g in Pt if g.get('px_per_mm')), len(Pt)),
                     outline_found=pct(sum(1 for g in Pt if g.get('outline_bbox_F')), len(Pt)), err_union=stats([e['err'] for e in Et]),
                     styles={st: sum(1 for g in Pt if g['style'] == st) for st in sorted(set(g['style'] for g in Pt))})
    S['by_type'] = by
    by = {}
    for st in sorted(set(g['style'] for g in P)):
        Pt = [g for g in P if g['style'] == st]; Et = [g['total_check'] for g in Pt if g.get('total_check')]
        by[st] = dict(plans=len(Pt), scale_found=pct(sum(1 for g in Pt if g.get('px_per_mm')), len(Pt)), err_union=stats([e['err'] for e in Et]))
    S['by_style'] = by
    return S


def manual_compare(rows):
    """수동 판정(realplan_manual.json) ↔ 자동 지표. 같은 파일·영역 번호가 있을 때만"""
    if not os.path.exists(MANUAL): return None
    man = json.load(open(MANUAL))
    idx = {(r['file'], i): g for r in rows if 'error' not in r for i, g in enumerate(r['regions'])}
    out = dict(n=0, outline_ok=0, walls_ok=0, dims_ok=0, auto_agree=0, n_auto=0)
    for m in man.get('verdicts', []):
        g = idx.get((m['file'], m['region']))
        if g is None: continue
        out['n'] += 1
        for k in ('outline_ok', 'walls_ok', 'dims_ok'): out[k] += int(bool(m.get(k)))
        tc = g.get('total_check')
        if tc is not None:
            out['n_auto'] += 1
            out['auto_agree'] += int((tc['err'] <= 0.05) == bool(m.get('outline_ok') and m.get('dims_ok')))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--split', default='dev', choices=['dev', 'hold', 'all'])
    ap.add_argument('--limit', type=int, default=0, help='무작위 표본 수(0=전부)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='out_real')
    ap.add_argument('--overlays', type=int, default=25, help='겹침 그림을 그릴 시트 수(무작위)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--files', nargs='*', help='특정 파일명만')
    ap.add_argument('--model', action='store_true', help='(기본) 벽·창·문을 분할 모델(fire3dbeam/str_seg, models/str_seg.onnx)로')
    ap.add_argument('--rules', action='store_true', help='규칙 벽 경로(예전 기본)')
    a = ap.parse_args()
    from fire3dbeam import realplan as _R
    a.model = (not a.rules) and _R.model_available()
    dev, hold = split_files(a.folder)
    fs = dict(dev=dev, hold=hold, all=dev + hold)[a.split]
    if a.files: fs = [f for f in dev + hold if os.path.basename(f) in a.files]
    rnd = random.Random(a.seed)
    if a.limit and a.limit < len(fs): fs = sorted(rnd.sample(fs, a.limit))
    od = os.path.join(a.out, 'overlays'); os.makedirs(od, exist_ok=True)
    drawset = set(rnd.sample(fs, min(a.overlays, len(fs))))
    t0 = time.time()
    rows = []
    with Pool(a.workers, maxtasksperchild=10) as pool:
        for r in pool.imap_unordered(work, [(f, od, f in drawset, a.model) for f in fs]):
            rows.append(r)
            if len(rows) % 20 == 0: print(f'{len(rows)}/{len(fs)}  {time.time() - t0:.0f}s', flush=True)
    rows.sort(key=lambda r: r['file'])
    tag = a.split + (f'_n{a.limit}' if a.limit else '') + ('_model' if a.model else '')
    with open(os.path.join(a.out, f'results_{tag}.jsonl'), 'w') as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    S = summarize(rows)
    S.update(split=a.split, limit=a.limit, model=a.model, wall_s=round(time.time() - t0, 1), manual=manual_compare(rows))
    for r in rows:
        if 'error' in r: print('ERROR', r['file'], r['error'], r['tb'])
    json.dump(S, open(os.path.join(a.out, f'summary_{tag}.json'), 'w'), ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(a.out, f'summary_{tag}.json'), os.path.join(a.out, 'summary.json'))
    print(json.dumps(S, ensure_ascii=False, indent=1))
