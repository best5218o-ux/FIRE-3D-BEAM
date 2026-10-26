# -*- coding: utf-8 -*-
"""Colab(또는 로컬)에서 AI허브 TS_SPA 원천 zip → SPA 학습 캐시 + 평가용 2048 축소본. colab/FireBIM_SPA_GPU학습.ipynb 에서 사용.
zip 을 풀지 않고 한 장씩 읽어(PIL LANCZOS → 긴 변 2048, 로컬 축소본과 같은 방식 — downscale()) 역할별로 저장:
  - 개발(dev, 학습 가능 = spa_data.trainable)  → SPA_SEG_CACHE/<STEM>.npz (spa_train.prep_arrays: 1/2 축척 영상 + 마스크)
  - 개발 중 devval                            → 위 npz + AIHUB_SPA_IMG/<파일>.PNG (spa_eval devval 용 2048 축소본)
  - 보류(held) 중 고정 198장(spa_held198.json) → AIHUB_SPA_IMG/<파일>.PNG 만 (npz 안 만듦 = 학습 불가)
  - 나머지 보류                               → 건너뜀(save_held_all=True 면 PNG 저장)
원본 크기는 원본 영상에서 정확히 읽음(meta.json {w,h,f}, 라벨 좌표 × f). 이미 만든 파일은 건너뜀(이어하기).
python -m evalr.spa_colab extract <TS_SPA_x.zip> [--limit N]   (AIHUB_ROOT·AIHUB_SPA_IMG·SPA_SEG_CACHE 환경변수 사용)
공개 불가 자료 — 캐시·영상은 본인 Colab/Drive 안에서만."""
import os, io, sys, json, time, zipfile, tarfile, threading
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalr import spa_data as SD
from evalr import spa_train as ST

L = 2048


def downscale(buf):
    """원본 PNG 바이트 → (긴 변 2048 회색조 uint8, W, H, f). 로컬 축소본(./aihub/prep.py)과 **같은 방법**
    (PIL convert('L') + LANCZOS) — 같은 원본이면 로컬 2048 영상과 바이트 단위로 같아 보류 198장 수치를 그대로 비교 가능.
    (STR 노트북의 cv2 IMREAD_REDUCED_GRAYSCALE_2+INTER_AREA 는 PNG 에선 빠르지도 않음: 전체 해독 후 축소)"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(io.BytesIO(buf)); W, H = im.size
    im = im.convert('L'); f = L / max(W, H)
    im = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
    return np.asarray(im), W, H, f


def role_of(stem, save_held_all=False):
    """→ 'train' | 'devval' | 'held198' | 'held' | None(건너뜀)"""
    if SD.trainable(stem): return 'devval' if ST.is_devval(stem) else 'train'
    if stem in SD.held198(): return 'held198'
    return 'held' if save_held_all else None


def outputs(stem, base, role, img_dir, cache):
    o = []
    if role in ('train', 'devval'): o.append(os.path.join(cache, stem + '.npz'))
    if role in ('devval', 'held198', 'held'): o.append(os.path.join(img_dir, base))
    return o


def list_members(z):
    return [n for n in z.namelist() if n.upper().endswith('.PNG') and '_SPA_' in os.path.basename(n).upper()]


def extract_zip(zpath, root=None, img_dir=None, cache=None, threads=3, save_held_all=False, limit=None, log_every=500):
    """원천 zip 한 개 처리. → dict(통계, files=[이 zip 에서 나온 결과 파일 경로], meta={파일: {w,h,f}})"""
    root = root or SD.ROOT; img_dir = img_dir or SD.IMG; cache = cache or ST.CACHE
    os.makedirs(img_dir, exist_ok=True); os.makedirs(cache, exist_ok=True)
    idx = json.load(open(os.path.join(root, 'label_index.json')))
    mp = os.path.join(img_dir, 'meta.json')
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    if not zipfile.is_zipfile(zpath):   # Drive 의 0바이트 자리표시·받다 만 zip(예: VS_SPA.zip + .irx35) → 건너뜀
        print('  ! zip 아님/손상 — 건너뜀:', os.path.basename(zpath), os.path.getsize(zpath), 'bytes')
        return dict(zip=os.path.basename(zpath), skipped=True, members=0, files=[], meta={})
    z = zipfile.ZipFile(zpath)
    names = list_members(z)
    if limit: names = names[:limit]
    st = dict(zip=os.path.basename(zpath), members=len(names), nolabel=0, skip_held=0, done_before=0, new=0, err=0,
              roles={}, files=[], meta={})
    todo = []
    for n in names:
        base = os.path.basename(n); stem = os.path.splitext(base)[0].upper()
        if stem not in idx: st['nolabel'] += 1; continue
        role = role_of(stem, save_held_all)
        if role is None: st['skip_held'] += 1; continue
        st['roles'][role] = st['roles'].get(role, 0) + 1
        outs = outputs(stem, base, role, img_dir, cache)
        st['files'] += outs
        if all(os.path.exists(p) for p in outs) and (not any(p.lower().endswith('.png') for p in outs) or base in meta):
            st['done_before'] += 1
            if base in meta: st['meta'][base] = meta[base]
            continue
        todo.append((n, base, stem, role))
    print(f"{st['zip']}: PNG {len(names)} · 라벨 없음 {st['nolabel']} · 보류 건너뜀 {st['skip_held']} · 역할 {st['roles']} · 이미 됨 {st['done_before']} · 할 것 {len(todo)}", flush=True)
    lock = threading.Lock()

    def one(t):
        try: return _one(t)
        except Exception as e: return t[1], None, repr(e)[:200]

    def _one(t):
        n, base, stem, role = t
        with lock: buf = z.read(n)
        g, W, H, f = downscale(buf)
        m = dict(w=int(W), h=int(H), f=f)
        if role in ('train', 'devval'):
            gt = SD.load_gt(dict(label=os.path.join(root, idx[stem]), f=f))
            s, mk = ST.prep_arrays(g, gt)
            ST.save_npz(stem, s, mk, cache)
        if role in ('devval', 'held198', 'held'):
            p = os.path.join(img_dir, base); tmp = p[:-4] + '.tmp.png'
            cv2.imwrite(tmp, g); os.replace(tmp, p)
        return base, m, None

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); k = 0
    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(todo), 64):
            for base, m, err in ex.map(one, todo[i:i + 64]):
                if err: st['err'] += 1; print('오류', base, err, flush=True); continue
                meta[base] = m; st['meta'][base] = m; st['new'] += 1
            json.dump(meta, open(mp + '.tmp', 'w')); os.replace(mp + '.tmp', mp)
            k += len(todo[i:i + 64])
            if k % log_every < 64 or k == len(todo):
                el = time.time() - t0
                print(f"  {st['zip']} {k}/{len(todo)}  {el / 60:.1f}분 경과, 남은 ≈{el / k * (len(todo) - k) / 60:.1f}분", flush=True)
    st['files'] = [p for p in st['files'] if os.path.exists(p)]
    return st


def pack(st, tar_path, img_dir=None, cache=None):
    """extract_zip 결과를 tar 한 개로(Drive 보관 → 연결이 끊겨도 다시 zip 을 읽지 않음). npz·PNG 는 이미 압축이라 무압축 tar."""
    if st.get('skipped'): return 0   # 건너뛴 zip 은 tar 를 만들지 않음(나중에 정상 zip 이 오면 다시 처리)
    img_dir = img_dir or SD.IMG; cache = cache or ST.CACHE
    tmp = tar_path + '.tmp'
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with tarfile.open(tmp, 'w') as t:
        for p in st['files']:
            t.add(p, arcname=('seg/' if p.endswith('.npz') else 'img/') + os.path.basename(p))
        mb = json.dumps(st['meta']).encode()
        ti = tarfile.TarInfo('meta_part.json'); ti.size = len(mb); t.addfile(ti, io.BytesIO(mb))
    os.replace(tmp, tar_path)
    return os.path.getsize(tar_path)


def unpack(tar_path, img_dir=None, cache=None):
    img_dir = img_dir or SD.IMG; cache = cache or ST.CACHE
    os.makedirs(img_dir, exist_ok=True); os.makedirs(cache, exist_ok=True)
    mp = os.path.join(img_dir, 'meta.json')
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    n = 0
    with tarfile.open(tar_path) as t:
        for m in t:
            if not m.isfile(): continue
            if m.name == 'meta_part.json':
                meta.update(json.load(t.extractfile(m))); continue
            d = cache if m.name.startswith('seg/') else img_dir
            fn = os.path.join(d, os.path.basename(m.name))
            if os.path.exists(fn): continue
            with open(fn, 'wb') as fo: fo.write(t.extractfile(m).read())
            n += 1
    json.dump(meta, open(mp, 'w'))
    return n


def summary(img_dir=None, cache=None):
    img_dir = img_dir or SD.IMG; cache = cache or ST.CACHE
    tr, va = ST.cached_stems('train', cache), ST.cached_stems('val', cache)
    pngs = {os.path.splitext(f)[0].upper() for f in os.listdir(img_dir) if f.lower().endswith('.png')} if os.path.isdir(img_dir) else set()
    h = SD.held198()
    return dict(train=len(tr), devval=len(va), png=len(pngs), held198_png=len(pngs & h), held198_in_train=len(set(tr) & h))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['extract', 'summary']); ap.add_argument('zip', nargs='*')
    ap.add_argument('--limit', type=int); ap.add_argument('--threads', type=int, default=3); ap.add_argument('--held_all', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'extract':
        for zp in a.zip:
            r = extract_zip(zp, threads=a.threads, save_held_all=a.held_all, limit=a.limit)
            print({k: v for k, v in r.items() if k not in ('files', 'meta')})
    print(summary())
