# -*- coding: utf-8 -*-
"""가상 건물 일괄: 생성(없으면) → 변환 → 채점 → 유형별 요약.  python run_all.py [n_per_type] [--model]   (RESCORE=1: 저장된 변환 JSON 으로 채점만, WORKERS=1: 1 프로세스)
표준양식 세트: SYNTH_FORM=mix SYNTH_DATA=data_std OUT=out_std python run_all.py 6  /  SYNTH_FORM=mix SYNTH_DATA=data_std_hold SYNTH_SEED_OFS=5000 OUT=out_std_hold python run_all.py 6
--model (또는 FIRE3DBEAM_MODEL=1): STR 모델 교차검증 경로(fire3dbeam/xcheck.py) — 기본은 규칙만. 교차검증 채점은 python -m evalr.xcheck_eval DATA OUT"""
import json, os, sys, glob, subprocess, asyncio
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalr.score import score
DATA = os.environ.get('SYNTH_DATA', 'data')          # 보류 세트 평가: SYNTH_DATA=data_hold OUT=out_hold
OUT = os.environ.get('OUT', 'out')


def work(d):
    name = os.path.basename(d)
    from evalr.card_eval import score_dir as card_std_score      # 표준양식 카드면 항목 해석 정확도 + 개인정보 유출 검사(없으면 {})
    if os.environ.get('RESCORE') and os.path.exists(f'{OUT}/{name}.json'):       # 변환 결과 JSON 으로 채점만 다시
        r = json.load(open(f'{OUT}/{name}.json'))
        return dict(score(json.load(open(os.path.join(d, 'gt.json'))), r), **card_std_score(d, r))
    try:
        from fire3dbeam.pipeline import run
        r = run(os.path.join(d, 'card.pdf'))
        json.dump(r, open(f'{OUT}/{name}.json', 'w'), ensure_ascii=False)
        return dict(score(json.load(open(os.path.join(d, 'gt.json'))), r), **card_std_score(d, r))
    except Exception as e:
        import traceback
        return dict(name=name, error=repr(e), tb=traceback.format_exc()[-600:])


def frac(v):
    a, b = v.split('/'); return (int(a), int(b))


if __name__ == '__main__':
    if '--model' in sys.argv: sys.argv.remove('--model'); os.environ['FIRE3DBEAM_MODEL'] = '1'
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(OUT, exist_ok=True)
    if len(glob.glob(f'{DATA}/*/card.pdf')) < 3 * n:
        subprocess.run([sys.executable, '-m', 'synth.make', str(n)], check=True)
    dirs = sorted(glob.glob(f'{DATA}/*/'))
    with Pool(int(os.environ.get('WORKERS') or min(8, os.cpu_count() or 2))) as pool:      # WORKERS=1: CPU 공유 시
        rows = pool.map(work, [d.rstrip('/') for d in dirs])
    json.dump(rows, open(f'{OUT}/score_all.json', 'w'), ensure_ascii=False, indent=1)
    errs = [r for r in rows if 'error' in r]
    for e in errs: print('ERROR', e['name'], e['error'], e['tb'])
    ok = [r for r in rows if 'error' not in r]
    summ = {}
    for bt in sorted(set(r['btype'] for r in ok)) + ['ALL']:
        R = [r for r in ok if bt == 'ALL' or r['btype'] == bt]
        def mean(k): v = [r[k] for r in R if r.get(k) is not None]; return round(float(np.mean(v)), 3) if v else None
        def ratio(k): t = [frac(r[k]) for r in R]; return f"{sum(a for a, b in t)}/{sum(b for a, b in t)}"
        sym = {}
        for typ in ('exit', 'hose', 'panel', 'gas', 'ev'):
            v = [r['symbols'][typ] for r in R if typ in r['symbols']]
            if v: sym[typ] = f"R{np.mean([x['recall'] for x in v]):.2f}/P{np.mean([x['precision'] for x in v]):.2f}"
        summ[bt] = dict(n=len(R), align_ok=ratio('align_ok'), align_mean_m=mean('align_mean_m'), wall_R=mean('wall_recall'), wall_P=mean('wall_precision'),
                        door_R=mean('door_recall'), door_P=mean('door_precision'),
                        entrance=ratio('entrance_found') if all('entrance_found' in r for r in R) else f"{sum(frac(r['entrance_found'])[0] for r in R if 'entrance_found' in r)}/{sum(frac(r['entrance_found'])[1] for r in R if 'entrance_found' in r)}",
                        extopen_R=mean('extopen_recall'), extopen_P=mean('extopen_precision'), stairs=ratio('stairs_found'), stairs_ok=ratio('stairs_correct'),
                        elevs=ratio('elevs_found'), flags=ratio('flags_planted_found'), card=ratio('card_floors_ok'), symbols=sym)
        def ratio2(k):                                  # 기능 12·18·24 추가 항목(있는 건물만)
            t = [frac(r[k]) for r in R if k in r]; return f"{sum(a for a, b in t)}/{sum(b for a, b in t)}" if t else None
        summ[bt].update(daejang_found=ratio2('daejang_found'), daejang_correct=ratio2('daejang_correct'), zone_flag_agree=ratio2('zone_flag_agree'),
                        site_facade_ok=ratio2('site_facade_ok'), water_found=ratio2('water_found'),
                        water_false=sum(r.get('water_false', 0) for r in R), zones_total=sum(r.get('zones_total') or 0 for r in R))
        # 소방시설(추가): 새 기호 P/R(합산), 카드 항목 해석, 규칙 추정 위치 오차, 비상용승강기, 수량 불일치 플래그
        fs = {}
        for r in R:
            for t, v in (r.get('fac_sym') or {}).items(): fs[t] = [a + b for a, b in zip(fs.get(t, [0, 0, 0]), v)]
        fac = dict(sym={t: f"R{v[0] / max(1, v[0] + v[1]):.3f}/P{v[0] / max(1, v[0] + v[2]):.3f} (n={v[0] + v[1]})" for t, v in sorted(fs.items())}, parse=ratio2('fac_parse'),
                   emev=ratio2('fac_emev'), emev_wrong=sum(r.get('fac_emev_wrong', 0) for r in R), flags=ratio2('fac_flags'), flags_false=sum(r.get('fac_flags_false', 0) for r in R))
        for c in ('STANDPIPE', 'OUTLET'):
            e = [x for r in R for x in (r.get('fac_est_err') or {}).get(c, [])]
            v = [x for x in e if x is not None]
            if e: fac['est_' + c] = dict(n=len(e), no_est=len(e) - len(v), mean_m=round(float(np.mean(v)), 2) if v else None,
                                         median_m=round(float(np.median(v)), 2) if v else None, within5m=f"{sum(1 for x in v if x <= 5)}/{len(e)}",
                                         in_rule_circle=f"{sum((r.get('fac_est_inradius') or {}).get(c, [0, 0])[0] for r in R)}/{sum((r.get('fac_est_inradius') or {}).get(c, [0, 0])[1] for r in R)}")
        nc = [x for r in R for x in r.get('fac_est_ncand', [])]
        if nc: fac['est_candidates_mean'] = round(float(np.mean(nc)), 2)
        summ[bt]['fac'] = fac
        from evalr.evac_score import summarize as ev_summ                  # 대피로 채점(evalr/evac_score.py)
        summ[bt].update(ev_summ(R))
        from evalr.card_eval import summarize as card_summ                 # 표준양식 카드 항목 해석(있는 세트만)
        cs = card_summ(R)
        if cs: summ[bt]['card_std'] = cs
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)
    for k, v in summ.items(): print(k, json.dumps(v, ensure_ascii=False))
    if os.environ.get('FIRE3DBEAM_MODEL', '').lower() in ('1', 'true', 'on', 'yes'):
        from evalr.xcheck_eval import main as xmain
        xs = xmain(DATA, OUT); json.dump(xs, open(f'{OUT}/xcheck_eval.json', 'w'), ensure_ascii=False, indent=1)
        print('XCHECK', json.dumps({k: xs[k] for k in ('rule_only', 'model_only', 'items_p>=0.5', 'capture', 'added_doors') if k in xs}, ensure_ascii=False))
