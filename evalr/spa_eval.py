# -*- coding: utf-8 -*-
"""realsym(규칙) · spa_seg(학습 모델) 을 AI허브 SPA 공간 라벨로 채점.
python -m evalr.spa_eval run  --split dev|held|devsub [--tag rule_v1] [--ocr-up 2] [--limit N]   # 규칙 검출 → 캐시
python -m evalr.spa_eval model --split dev|held|held198|devsub|devval [--tag model] [--onnx x.onnx] [--cdir D]   # ONNX 모델 → 캐시
python -m evalr.spa_eval score --split ... --tag rule_v1 [--tag2 model fuse:rule_v1+model] [--json out.json] [--rematch]
   fuse:A+B = 규칙 캐시 A 와 모델 캐시 B 를 realsym.fuse_model 로 합쳐 채점. --rematch = 캐시된 OCR 글자로 현재 어휘 매칭 재적용.
캐시: out/spa_cache/<tag>/<stem>.json (규칙 ≈10 s/장 — 2배 확대 OCR, 모델 ≈0.3 s/장).
분할: 파일명 md5 → 20% 보류(held)/80% 개발(dev). devsub = dev 중 md5 기반 고정 표본(규칙 반복 개선용, 약 1/5). devval = dev 중 학습에서 뺀 1/8(모델 비교용).
채점 정의
 (a) 실 유형: 라벨 다각형마다, 같은 유형으로 매핑된 검출 점(실명 글자 중심)이 다각형 안에 있으면 TP → 유형별 재현율.
     정밀도 = 라벨 다각형 안에 떨어진 검출 중 유형 일치 비율(가장 작은 포함 다각형 기준). 라벨 밖(미라벨 영역) 검출 수는 따로.
     글자 있음 추정 = 다각형 안에 OCR 이 한글 1자 이상(또는 영문 2자)을 읽은 글자 덩어리가 있는 비율(인쇄 실명 유무 근사, 상한 아님).
 (b) 계단: 검출 상자 중심이 공간_계단실 다각형 안이거나 (교집합/검출면적)>0.5 → 적중. (c) 승강기: 공간_엘리베이터 동일.
     정밀도_라벨영역 = 라벨 작업 영역(모든 공간 다각형 +40px) 안 검출만 대상(평면 일부만 라벨된 시트가 많음)."""
import os, sys, json, time, argparse, hashlib, re
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalr import spa_data as SD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.environ.get('SPA_EVAL_CACHE', os.path.join(HERE, 'out', 'spa_cache'))

# realsym 표준명 → SPA 범주 (실명 단위; realsym 의 소방 범주보다 세분)
NAME2SPA = {}
for n in ('침실', '안방', '방', '자녀방', '부부침실', '게스트룸', '작은방', '큰방'): NAME2SPA[n] = '공간_침실'
for n in ('거실', '가족실', '라운지'): NAME2SPA[n] = '공간_거실'
for n in ('주방', '식당', '부엌', '주방식당', '주방및식당', '주방및거실'): NAME2SPA[n] = '공간_주방'
for n in ('욕실', '화장실', '파우더룸', '샤워실'): NAME2SPA[n] = '공간_화장실'
for n in ('발코니', '베란다', '테라스', '데크', '대피공간', '대피실'): NAME2SPA[n] = '공간_발코니'
NAME2SPA.update({'현관': '공간_현관', '계단실': '공간_계단실', '계단': '공간_계단실', '승강기': '공간_엘리베이터', '엘리베이터': '공간_엘리베이터',
                 '승강장': '공간_엘리베이터홀', '전실': '공간_엘리베이터홀', '홀': '공간_엘리베이터홀', '엘리베이터홀': '공간_엘리베이터홀',
                 '드레스룸': '공간_드레스룸', '실외기실': '공간_실외기실', '알파룸': '공간_다목적공간', '다목적실': '공간_다목적공간'})
ROOM_EVAL = ['공간_침실', '공간_거실', '공간_주방', '공간_화장실', '공간_발코니', '공간_현관', '공간_계단실', '공간_드레스룸', '공간_실외기실', '공간_엘리베이터홀']
FIRE = ['공간_침실', '공간_거실', '공간_화장실', '공간_발코니', '공간_현관', '공간_계단실']


def is_devsub(stem):
    return int(hashlib.md5(stem.upper().encode()).hexdigest()[16:24], 16) % 5 == 0


def items(split):
    if split == 'devsub': return [it for it in SD.list_items('dev') if is_devsub(it['stem'])]
    if split == 'devval':                # 개발 split 중 모델 학습에서 뺀 1/8 (모델·규칙 공정 비교용)
        from evalr.spa_train import is_devval
        return [it for it in SD.list_items('dev') if is_devval(it['stem'])]
    return SD.list_items(split)          # dev | held | held198(고정 보류 198장, evalr/spa_held198.json) | all


def _j(o):
    if isinstance(o, dict): return {k: _j(v) for k, v in o.items() if k not in ('mask', 'chars')}
    if isinstance(o, (list, tuple)): return [_j(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o


# ─────────────── 검출 실행(캐시) ───────────────
def run_rule(split, tag, limit=None, **kw):
    from fire3dbeam import realsym as RS
    od = os.path.join(CDIR, tag); os.makedirs(od, exist_ok=True)
    its = items(split)[:limit] if limit else items(split)
    t00 = time.time()
    for k, it in enumerate(its):
        fn = os.path.join(od, it['stem'] + '.json')
        if os.path.exists(fn): continue
        g = cv2.imread(it['path'], 0)
        t0 = time.time()
        out = RS.analyze(g, None, dets=('rooms', 'doors', 'stairs', 'elev'), **kw)
        res = dict(sec=round(time.time() - t0, 1), px_per_m=out.get('px_per_m'),
                   rooms=[dict(name=r['name'], cat=r['cat'], xy=r['xy'], box=r['box'], score=r['score'], text=r['text']) for r in out['rooms']],
                   words=[dict(text=c['text'], box=c['box'], conf=round(c.get('conf', -1), 1), how=c.get('how')) for c in out.get('words', []) if c['text'].strip()],
                   stairs=[dict(box=s['box'], label=s.get('label'), walk=s.get('walk'), n=s.get('n')) for s in out['stairs']],
                   elevators=[dict(box=e['box'], label=e.get('label')) for e in out['elevators']])
        json.dump(_j(res), open(fn, 'w'), ensure_ascii=False)
        print(f'[{k + 1}/{len(its)}] {it["stem"]} {res["sec"]}s rooms {len(res["rooms"])} st {len(res["stairs"])} ev {len(res["elevators"])}  (누적 {time.time() - t00:.0f}s)', flush=True)


def run_model(split, tag, limit=None, onnx=None):
    from fire3dbeam import spa_seg as SS
    onnx = onnx or SS.MODEL
    od = os.path.join(CDIR, tag); os.makedirs(od, exist_ok=True)
    its = items(split)[:limit] if limit else items(split)
    for it in its:
        fn = os.path.join(od, it['stem'] + '.json')
        if os.path.exists(fn): continue
        g = cv2.imread(it['path'], 0)
        t0 = time.time()
        r = SS.detect(g, path=onnx)
        res = dict(sec=round(time.time() - t0, 2), rooms=[dict(name=x['name'], spa=x['spa'], xy=x['xy'], score=x['score'], box=x['box']) for x in r['rooms']],
                   stairs=[dict(box=s['box'], score=s['score']) for s in r['stairs']], elevators=[dict(box=e['box'], score=e['score']) for e in r['elevators']],
                   room_votes=None, scale=r.get('scale'))
        json.dump(_j(res), open(fn, 'w'), ensure_ascii=False)
    print('model done', len(its))


# ─────────────── 채점 ───────────────
def _inside(poly, x, y):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


def _box_hit(poly, box):
    from shapely.geometry import Polygon, box as sbox
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if _inside(poly, cx, cy): return True
    try:
        P = Polygon(poly).buffer(0); B = sbox(*box)
        return B.area > 0 and P.intersection(B).area / B.area > 0.5
    except Exception:
        return False


HANGUL = re.compile('[가-힣]')


def rematch_rooms(res):
    """캐시된 OCR 글자(words)로 현재 realsym.select_rooms 다시 적용(OCR 재실행 없이 어휘·임계 조정 평가)"""
    from fire3dbeam import realsym as RS
    items = [dict(text=w['text'], box=w['box'], xy=[(w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2],
                  how=w.get('how') or 'g', conf=w.get('conf', 60)) for w in res.get('words', []) if w.get('how') != '2line']
    keep, _ = RS.select_rooms(items, want_area=False)
    return [dict(name=r['name'], cat=r['cat'], xy=r['xy'], box=r['box'], score=r['score'], text=r['text']) for r in keep]


def fuse_res(rr, mr):
    from fire3dbeam import realsym as RS
    out = dict(rooms=[dict(r) for r in rr['rooms']], stairs=[dict(d) for d in rr['stairs']], elevators=[dict(d) for d in rr['elevators']],
               ocr_up=rr.get('ocr_up', 2.0))                 # SPA 원천은 긴 변 2048 축소본 → realsym 'auto' 는 2배 확대 OCR
    mr = dict(mr); mr.setdefault('scale', 1024.0 / 2048)    # 모델 입력 배율(긴 변 2048 시트 → 1024)
    RS.fuse_model(out, mr)
    return dict(sec=rr.get('sec', 0) + mr.get('sec', 0), rooms=out['rooms'], words=rr.get('words', []), stairs=out['stairs'], elevators=out['elevators'])


def score(split, tag, verbose=False, rematch=False):
    its = items(split)
    od = os.path.join(CDIR, tag)
    R = dict(n_img=0, rooms={c: dict(n=0, tp=0, txt=0) for c in ROOM_EVAL}, det_in=0, det_ok=0, det_out=0, conf={},
             stairs=dict(n=0, tp=0, det=0, det_ok=0, det_lab=0, det_lab_ok=0), elev=dict(n=0, tp=0, det=0, det_ok=0, det_lab=0, det_lab_ok=0), sec=[])
    for it in its:
        if tag.startswith('fuse:'):                       # fuse:규칙태그+모델태그 → realsym.fuse_model 로 합친 결과
            ta, tb = tag[5:].split('+')
            fa, fb = os.path.join(CDIR, ta, it['stem'] + '.json'), os.path.join(CDIR, tb, it['stem'] + '.json')
            if not (os.path.exists(fa) and os.path.exists(fb)): continue
            ra = json.load(open(fa))
            if rematch: ra['rooms'] = rematch_rooms(ra)          # 재매칭은 규칙 쪽에만(합친 뒤에 하면 모델 실이 사라짐)
            res = fuse_res(ra, json.load(open(fb)))
        else:
            fn = os.path.join(CDIR, tag, it['stem'] + '.json')
            if not os.path.exists(fn): continue
            res = json.load(open(fn))
            if rematch and 'words' in res: res['rooms'] = rematch_rooms(res)
        gt = SD.load_gt(it)
        R['n_img'] += 1; R['sec'].append(res.get('sec', 0))
        g_shape = cv2.imread(it['path'], 0).shape if False else None
        # (a) 실 유형
        D = []
        for r in res['rooms']:
            spa = r.get('spa') or NAME2SPA.get(r['name'])
            if spa: D.append((spa, r['xy'][0], r['xy'][1]))
        words = [w for w in res.get('words', []) if len(HANGUL.findall(w['text'])) >= 1 or len(re.findall('[A-Za-z]', w['text'])) >= 2]
        for gr in gt:
            c = gr['cls']
            if c not in R['rooms']: continue
            R['rooms'][c]['n'] += 1
            if any(s == c and _inside(gr['poly'], x, y) for s, x, y in D): R['rooms'][c]['tp'] += 1
            if any(_inside(gr['poly'], (w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2) for w in words): R['rooms'][c]['txt'] += 1
        for s, x, y in D:
            if s not in R['rooms']: continue
            cont = [gr for gr in gt if _inside(gr['poly'], x, y)]
            if not cont: R['det_out'] += 1; continue
            gc = min(cont, key=lambda gr: gr['area'])['cls']
            R['det_in'] += 1; R['det_ok'] += gc == s
            if gc != s: R['conf'][f'{SD.SHORT[s]}→{SD.SHORT.get(gc, gc)}'] = R['conf'].get(f'{SD.SHORT[s]}→{SD.SHORT.get(gc, gc)}', 0) + 1
        # (b)(c) 계단·승강기
        H, W = int(max(p['poly'][:, 1].max() for p in gt)) + 200 if gt else 2048, int(max(p['poly'][:, 0].max() for p in gt)) + 200 if gt else 2048
        zone = SD.labeled_zone(gt, (max(H, 2048), max(W, 2048)), margin=40) if gt else None
        for key, cls, dk in (('stairs', '공간_계단실', 'stairs'), ('elev', '공간_엘리베이터', 'elevators')):
            P = [gr['poly'] for gr in gt if gr['cls'] == cls]
            B = [d['box'] for d in res[dk]]
            S = R[key]; S['n'] += len(P)
            S['tp'] += sum(1 for p in P if any(_box_hit(p, b) for b in B))
            for b in B:
                ok = any(_box_hit(p, b) for p in P)
                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                lab = zone is not None and 0 <= cy < zone.shape[0] and 0 <= cx < zone.shape[1] and zone[cy, cx] > 0
                S['det'] += 1; S['det_ok'] += ok; S['det_lab'] += lab or ok; S['det_lab_ok'] += ok
    return R


def fmt(R, name):
    r = R['rooms']
    L = [f'### {name}  (영상 {R["n_img"]}장, 평균 {np.mean(R["sec"]) if R["sec"] else 0:.1f}s/장)']
    L.append('| 실 유형 | 라벨 수 | 재현율 | 글자 있음 추정 |')
    L.append('|---|---|---|---|')
    tn = tt = 0
    for c in ROOM_EVAL:
        x = r[c]
        if not x['n']: continue
        L.append(f'| {SD.SHORT[c]} | {x["n"]} | {x["tp"] / x["n"]:.3f} | {x["txt"] / x["n"]:.2f} |')
        tn += x['n']; tt += x['tp']
    fn = sum(r[c]['n'] for c in FIRE); ft = sum(r[c]['tp'] for c in FIRE)
    L.append(f'| **전체(위 {len(ROOM_EVAL)}종)** | {tn} | {tt / max(tn, 1):.3f} | {sum(r[c]["txt"] for c in ROOM_EVAL) / max(tn, 1):.2f} |')
    L.append(f'| **소방 6종**(침실·거실·화장실·발코니·현관·계단실) | {fn} | {ft / max(fn, 1):.3f} | {sum(r[c]["txt"] for c in FIRE) / max(fn, 1):.2f} |')
    L.append(f'실 유형 정밀도(라벨 다각형 안 검출 {R["det_in"]}개): {R["det_ok"] / max(R["det_in"], 1):.3f}  · 라벨 밖 검출 {R["det_out"]}개')
    if R['conf']: L.append('혼동 상위: ' + ', '.join(f'{k} {v}' for k, v in sorted(R['conf'].items(), key=lambda kv: -kv[1])[:6]))
    for key, nm in (('stairs', '계단(공간_계단실)'), ('elev', '승강기(공간_엘리베이터)')):
        S = R[key]
        L.append(f'{nm}: 재현율 {S["tp"]}/{S["n"]} = {S["tp"] / max(S["n"], 1):.3f} · 정밀도 {S["det_ok"]}/{S["det"]} = {S["det_ok"] / max(S["det"], 1):.3f}'
                 f' · 정밀도_라벨영역 {S["det_lab_ok"]}/{S["det_lab"]} = {S["det_lab_ok"] / max(S["det_lab"], 1):.3f}')
    return '\n'.join(L)


def summary(R):
    r = R['rooms']
    fn = sum(r[c]['n'] for c in FIRE); ft = sum(r[c]['tp'] for c in FIRE)
    return dict(n_img=R['n_img'], room_recall={SD.SHORT[c]: round(r[c]['tp'] / r[c]['n'], 3) for c in ROOM_EVAL if r[c]['n']},
                room_text_frac={SD.SHORT[c]: round(r[c]['txt'] / r[c]['n'], 3) for c in ROOM_EVAL if r[c]['n']},
                fire6_recall=round(ft / max(fn, 1), 3), room_precision=round(R['det_ok'] / max(R['det_in'], 1), 3), room_det_outside=R['det_out'],
                stairs=dict(R['stairs'], recall=round(R['stairs']['tp'] / max(R['stairs']['n'], 1), 3), precision=round(R['stairs']['det_ok'] / max(R['stairs']['det'], 1), 3),
                            precision_lab=round(R['stairs']['det_lab_ok'] / max(R['stairs']['det_lab'], 1), 3)),
                elev=dict(R['elev'], recall=round(R['elev']['tp'] / max(R['elev']['n'], 1), 3), precision=round(R['elev']['det_ok'] / max(R['elev']['det'], 1), 3),
                          precision_lab=round(R['elev']['det_lab_ok'] / max(R['elev']['det_lab'], 1), 3)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'model', 'score'])
    ap.add_argument('--split', default='devsub'); ap.add_argument('--tag', default='rule'); ap.add_argument('--tag2', nargs='*', default=[])
    ap.add_argument('--limit', type=int); ap.add_argument('--ocr-up', type=float, default=None); ap.add_argument('--json'); ap.add_argument('--rematch', action='store_true')
    ap.add_argument('--onnx', default=None, help='model: 평가할 onnx(기본 models/spa_seg.onnx 또는 SPA_SEG_ONNX)')
    ap.add_argument('--cdir', default=None, help='검출 캐시 폴더(기본 out/spa_cache 또는 SPA_EVAL_CACHE)')
    a = ap.parse_args()
    if a.cdir: CDIR = a.cdir
    if a.cmd == 'run':
        kw = {} if a.ocr_up is None else dict(ocr_up=a.ocr_up)
        run_rule(a.split, a.tag, a.limit, **kw)
    elif a.cmd == 'model':
        run_model(a.split, a.tag, a.limit, a.onnx)
    else:
        allj = {}
        for t in [a.tag] + a.tag2:
            R = score(a.split, t, rematch=a.rematch)
            print(fmt(R, f'{t} @ {a.split}')); print()
            allj[t] = summary(R)
        if a.json: json.dump(_j(allj), open(a.json, 'w'), ensure_ascii=False, indent=1)
