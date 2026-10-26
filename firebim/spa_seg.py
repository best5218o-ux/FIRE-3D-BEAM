# -*- coding: utf-8 -*-
"""(선택) 학습 분할망 models/spa_seg.onnx 로 실 유형·계단실·승강기 검출 — onnxruntime 만 필요(torch 불필요, 오프라인 Windows 설치판용).
AI허브 「건축 도면 데이터」 SPA 공간 라벨(개발 split)로 학습: python -m evalr.spa_train. 모델이 없으면 available()=False.
입력 축척: 학습은 A3 시트를 긴 변 1024 px(≈62 dpi, 문 폭 0.9 m ≈ 15 px)로 줄여서 함. 임의 영상은
 ① px_per_m(문 호로 추정)이 있으면 그 기준으로, ② 없으면 긴 변 1024 로 맞춤(시트 전체 입력 가정). scale 로 직접 지정 가능."""
import os, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get('SPA_SEG_ONNX', os.path.join(HERE, 'models', 'spa_seg.onnx'))
REF_PPM = 17.5          # 학습 축척에서 px/m (2048 축소본 문 반경 중앙값 ≈ 31.5 px/0.9 m → ×0.5)
STAIR, ELEV = '계단실', '엘리베이터'
# 기본 임계(개발 split 의 devval 로 조정 — evalr/spa_eval.py). 면적은 학습 축척 px²
# min_score: 연결 성분의 평균 확률 하한(성분 전체를 버림 — 상자·실 점 모두). v2(815장) 모델의 계단 오검출 억제용,
#   devval 124장: 계단 R .907→.880, P_라벨영역 .825→.936 (2026-09-30, 보류 미사용 조정). 승강기·실은 적용 안 함.
PARAM = dict(min_area={STAIR: 200, ELEV: 50, '_room': 150}, min_prob={STAIR: 0.4, ELEV: 0.5, '_room': 0.45},
             min_score={STAIR: 0.7})
_S = {}


def available(path=MODEL):
    return os.path.exists(path)


def model_param(path=MODEL):
    """모델 .json 의 "param"(모델별 후처리 조정값, devval 에서만 고른 값) — 없으면 None → PARAM 그대로.
    메타의 "post" 는 기록용이라 적용하지 않음(v2 이전 호환). 우선순위: PARAM < json "param" < detect(param=…)."""
    return _sess(path)[1].get('param')


def _sess(path=MODEL):
    if path not in _S:
        import onnxruntime as ort
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        if os.environ.get("SPA_SEG_ARENA", "0") != "1": o.enable_cpu_mem_arena = False      # 실행 뒤 작업 메모리 반환(결과 동일)
        mj = path.replace('.onnx', '.json')
        if not os.path.exists(mj): mj = os.path.join(HERE, 'models', 'spa_seg.json')     # 같은 클래스 구성의 이전 모델(spa_seg_v1.onnx 등)
        meta = json.load(open(mj, encoding='utf-8')) if os.path.exists(mj) else {}
        _S[path] = (ort.InferenceSession(path, o, providers=['CPUExecutionProvider']), meta)
    return _S[path]


def sheet_scale(long_side, path=MODEL):
    """시트 전체(긴 변 long_side px) 기준 모델 입력 배율 = 학습 축척(2048 축소본 × json "scale", 기본 0.5 → 긴 변 1024).
    잘린 평면 영상(realsym 의 box)도 시트 긴 변으로 배율을 정해야 학습 축척과 같다."""
    ms = float(_sess(path)[1].get('scale', 0.5)) / 0.5
    return 1024.0 * ms / float(long_side)


def probs(g, scale=None, ppm=None, path=MODEL):
    """→ (P: (C,h,w) float32 확률(학습 축척), 클래스 이름 목록, 사용한 scale)"""
    sess, meta = _sess(path)
    names = meta.get('classes')
    if scale is None:                                  # 학습 축척 = 2048 축소본 × meta['scale'](기본 0.5 → 긴 변 1024, 17.5 px/m)
        ms = float(meta.get('scale', 0.5)) / 0.5
        scale = REF_PPM * ms / ppm if ppm else 1024.0 * ms / max(g.shape[:2])
    s = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    H, W = s.shape
    Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    x = np.full((Hp, Wp), 255, np.uint8); x[:H, :W] = s
    x = ((255 - x.astype(np.float32)) / 255.0)[None, None]
    lo = sess.run(None, {'img': x})[0][0]
    lo = lo - lo.max(0, keepdims=True); e = np.exp(lo); p = e / e.sum(0, keepdims=True)
    P = np.stack([cv2.resize(c, (Wp, Hp), interpolation=cv2.INTER_LINEAR) for c in p])[:, :H, :W]
    return P, names, scale


def detect(g, scale=None, ppm=None, path=MODEL, param=None):
    """→ {rooms:[{name,spa,xy,score,area_m2?}], stairs:[{box,score}], elevators:[{box,score}]} (좌표=입력 영상 픽셀)"""
    prm = {**PARAM, **(model_param(path) or {}), **(param or {})}
    P, names, sc = probs(g, scale, ppm, path)
    am = P.argmax(0); pm = P.max(0)
    out = dict(rooms=[], stairs=[], elevators=[], scale=sc)
    for k in range(1, len(names)):
        nm = names[k]
        m = ((am == k) & (pm >= prm['min_prob'].get(nm, prm['min_prob']['_room']))).astype(np.uint8)
        if not m.any(): continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        amin = prm['min_area'].get(nm, prm['min_area']['_room'])
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < amin: continue
            comp = lab[y:y + h, x:x + w] == i
            scr = float(P[k, y:y + h, x:x + w][comp].mean())
            if scr < (prm.get('min_score') or {}).get(nm, 0.0): continue
            if nm in (STAIR, ELEV):
                d = dict(box=[x / sc, y / sc, (x + w) / sc, (y + h) / sc], score=round(scr, 3))
                out['stairs' if nm == STAIR else 'elevators'].append(d)
                if nm == ELEV: continue
            dt = cv2.distanceTransform(np.pad(comp.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
            pts = [np.unravel_index(int(dt.argmax()), dt.shape)]
            if prm.get('peaks', True) and a >= 3 * amin:      # 문으로 이어진 같은 유형 실 여럿(침실1·2) → 거리변환 봉우리마다 한 점
                R = prm.get('peak_r', 12)
                mx = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1)))
                ys, xs = np.nonzero((dt >= mx - 1e-6) & (dt >= max(4.0, 0.45 * dt.max())))
                for yy, xx in sorted(zip(ys, xs), key=lambda q: -dt[q]):
                    if all(np.hypot(yy - py, xx - px) > 2 * R for py, px in pts): pts.append((yy, xx))
            for yy, xx in pts:
                out['rooms'].append(dict(name=nm, spa='공간_' + nm, xy=[(x + xx) / sc, (y + yy) / sc], score=round(scr, 3),
                                         area_px=int(a / sc / sc), box=[x / sc, y / sc, (x + w) / sc, (y + h) / sc]))
    # 계단 전용 모델(모델 json "stair_model"): devval 에서 계단은 이전 모델이, 실·승강기는 새 모델이 나아서 계단만 바꿔 끼움
    sm = _sess(path)[1].get('stair_model')
    if sm and not prm.get('no_stair_model'):
        sp = os.path.join(os.path.dirname(path), sm)
        if os.path.exists(sp) and os.path.abspath(sp) != os.path.abspath(path):
            # 계단 모델의 학습 축척(json "scale")이 다르면 배율도 그 비율로(scale 을 직접 받은 경우 — realsym 은 시트 긴 변 기준 배율을 넘김)
            s2 = None if scale is None else scale * float(_sess(sp)[1].get('scale', 0.5)) / float(_sess(path)[1].get('scale', 0.5))
            o2 = detect(g, s2, ppm, sp)
            out['stairs'] = o2['stairs']
            out['rooms'] = [r for r in out['rooms'] if r['name'] != STAIR] + [r for r in o2['rooms'] if r['name'] == STAIR]
            out['stair_model'] = sm
    return out
