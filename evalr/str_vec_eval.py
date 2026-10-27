# -*- coding: utf-8 -*-
"""STR 벡터화 비교 채점 — 같은 모델 확률(캐시)에서 벡터화기만 바꿔 채점(evalr.str_eval.score_image 와 같은 지표).

  python -m evalr.str_vec_eval cache --split val  [--cache DIR]      # 네트워크 1회: 작업 해상도 확률(uint8 npz) 캐시
  python -m evalr.str_vec_eval score --split val --vec old new pix  # old = 이전 벡터화(str_seg.vectorize_v1), new = 현재 vectorize,
                                                                    # pix = 화소 예측(평가 해상도 argmax, str_eval 'model' 과 같음)
  --model X.onnx : 캐시를 만든 모델(cache: 추론 모델, score: 그 모델 .json 의 vec_param 사용). 기본 models/str_seg.onnx
  벡터는 평가 해상도에 바로 그려 채점(2026-09-30 밤). 이름 뒤 '@nn' = 예전 방식(작업 해상도에 그린 뒤 최근접 축소, ≈0.5px 오른쪽·아래 밀림) — 옛 수치 재현용
※ AI허브 자료 — 캐시·결과 로컬 전용.
"""
import os, sys, json, time, argparse
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalr import str_eval as E
from fire3dbeam import str_seg as SS

CACHE = './cache/strm/vec_cache'


def run_cache(split, cache, threads=1, limit=0, model=None):
    E.SPLIT = split
    its = E.held_items(limit, split)
    os.makedirs(cache, exist_ok=True)
    seg = SS.StrSeg(model or SS.MODEL, threads=threads)
    t0 = time.time(); n = 0
    for it in its:
        fp = os.path.join(cache, it['stem'] + '.npz')
        if os.path.exists(fp): continue
        g, es, r = E.eval_shape(it)
        lab_w, p, k = seg.predict(g, src_dpi=300 * it['f'])
        pe = np.stack([cv2.resize(c, (es[1], es[0]), interpolation=cv2.INTER_AREA) for c in p])
        np.savez_compressed(fp, p=np.round(p * 255).astype(np.uint8), lab_w=lab_w, lab_e=pe.argmax(0).astype(np.uint8))
        n += 1
        if n % 10 == 0: print(f'cache {split} {n}/{len(its)} {time.time() - t0:.0f}s', flush=True)


def vec_fn(name, model=None):
    P = SS.model_vec_param(model)
    if name == 'old': return lambda z: SS.vectorize_v1(z['lab_w'])
    if name == 'new': return lambda z: SS.vectorize(z['lab_w'], prob=z['p'].astype(np.float32) / 255, P=P)
    if name == 'base': return lambda z: SS.vectorize(z['lab_w'], prob=z['p'].astype(np.float32) / 255, P=SS.VEC_PARAM)   # 모델별 조정 없이
    if name == 'new_noprob': return lambda z: SS.vectorize(z['lab_w'], P=P)
    raise ValueError(name)


def run_score(split, cache, vecs, limit=0, out=None, model=None):
    E.SPLIT = split
    its = [it for it in E.held_items(limit, split) if os.path.exists(os.path.join(cache, it['stem'] + '.npz'))]
    rows = {v: [] for v in vecs}; tv = {v: 0.0 for v in vecs}
    for it in its:
        z = dict(np.load(os.path.join(cache, it['stem'] + '.npz')))
        for v in vecs:
            if v == 'pix': lab = z['lab_e']
            elif v == 'lab_w': lab = z['lab_w']
            else:
                nm, nn = v.split('@')[0], v.endswith('@nn')
                t = time.time(); V = vec_fn(nm, model)(z); tv[v] += time.time() - t
                _, es, _ = E.eval_shape(it)
                if nn: lab = cv2.resize(SS.render_vectors(V, z['lab_w'].shape), (es[1], es[0]), interpolation=cv2.INTER_NEAREST)
                else: lab = SS.render_vectors(V, tuple(es), scale=E.vec_scale(z['lab_w'].shape, es))   # 평가 해상도에 바로 그림
            _, es, _ = E.eval_shape(it)
            if lab.shape != tuple(es): lab = cv2.resize(lab, (es[1], es[0]), interpolation=cv2.INTER_NEAREST_EXACT)
            r = E.score_image(it, lab); r['stem'] = it['stem']; rows[v].append(r)
    res = {}
    for v in vecs:
        a = E.agg(rows[v]); res[v] = a
        print(f"{v:10s} n={a['n']:3d} wallIoU {a['wall']['iou']:.4f} F1 {a['wall']['f1']:.4f} P {a['wall']['p']:.3f} R {a['wall']['r']:.3f} | "
              f"cl R/P {a['wall_centerline']['r']:.4f}/{a['wall_centerline']['p']:.4f} | door R/P {a['door_inst']['r']:.4f}/{a['door_inst']['p']:.4f} "
              f"IoU {a['door']['iou']:.3f} | win R/P {a['window_inst']['r']:.4f}/{a['window_inst']['p']:.4f} IoU {a['window']['iou']:.3f} | t {tv[v] / max(1, len(its)):.2f}s", flush=True)
    if out:
        json.dump(dict(split=split, n=len(its), res=res), open(out, 'w'), ensure_ascii=False, indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cache', 'score'])
    ap.add_argument('--split', default='val')
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--vec', nargs='*', default=['old', 'new', 'pix'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', default=None, help='캐시를 만든(만들) 모델 onnx — score 는 그 .json 의 vec_param')
    a = ap.parse_args()
    if a.cmd == 'cache': run_cache(a.split, a.cache, a.threads, a.limit, a.model)
    else: run_score(a.split, a.cache, a.vec, a.limit, a.out, a.model)
