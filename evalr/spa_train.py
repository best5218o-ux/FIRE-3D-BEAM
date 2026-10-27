# -*- coding: utf-8 -*-
"""SPA(공간) 라벨로 소형 분할망 학습 → models/spa_seg.onnx (onnxruntime 전용 추론: fire3dbeam/spa_seg.py).
python -m evalr.spa_train prep [--workers 2]   # 개발 split 영상·라벨을 1/2 축척 npz 캐시로 (out/spa_cache/seg, SPA_SEG_CACHE 로 변경)
python -m evalr.spa_train train --minutes 50  # CPU 1스레드 학습(시간 제한), 체크포인트 out/spa_cache/spa_seg.pt  (이전과 같은 기본값)
python -m evalr.spa_train export              # → models/spa_seg.onnx (+ .json)
GPU(Colab T4, colab/FIRE-3D-BEAM_SPA_GPU학습.ipynb):
python -m evalr.spa_train train --device cuda --width 2.0 --bs 32 --workers 2 --minutes 120 --aug 1 --warmup 200 \
       --out <Drive>/spa_gpu/ckpt [--resume <Drive>/spa_gpu/ckpt/last.pt]
python -m evalr.spa_train export --ckpt <..>/best.pt --onnx <..>/spa_seg_gpu.onnx
보류(held) split 은 학습·조정에 절대 쓰지 않음(캐시·학습 모두 SD.trainable 로 거름, 고정 보류 198장 evalr/spa_held198.json 포함).
개발 split 중 md5 기반 1/8 은 'devval'(임계값 조정·모니터링·best 선택)로 따로 둠.
체크포인트 = dict(model, opt, scaler, elapsed, step, val, c=채널 구성, w=클래스 가중, ...). 예전 형식(state_dict 만)도 읽음."""
import os, sys, time, json, math, glob, hashlib, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalr import spa_data as SD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get('SPA_SEG_CACHE', os.path.join(HERE, 'out', 'spa_cache', 'seg'))
CKPT = os.path.join(HERE, 'out', 'spa_cache', 'spa_seg.pt')
ONNX = os.path.join(HERE, 'models', 'spa_seg.onnx')
SCALE = 0.5                                   # 2048 축소본 → 1024 (≈62 dpi)
CLASSES = ['배경'] + [SD.SHORT[c] for c in SD.SPA_CLASSES]      # 0=배경, 1..13
CIDX = {c: i + 1 for i, c in enumerate(SD.SPA_CLASSES)}
IGN = 255
C0 = (16, 32, 48, 64, 96)                     # 기본 채널(0.54M 파라미터) — --width 로 배율
FIRE_EVAL = ['침실', '거실', '화장실', '발코니', '현관', '계단실', '엘리베이터']


def is_devval(stem):
    return int(hashlib.md5(stem.upper().encode()).hexdigest()[8:16], 16) % 8 == 0


def width_channels(width=1.0):
    return tuple(int(round(c * width / 4) * 4) for c in C0)


# ─────────────── 캐시 ───────────────
def prep_arrays(g, gt):
    """2048 축소본 회색조 + 라벨 → (1/2 축척 영상 uint8, 클래스 마스크 uint8, 라벨 밖=255). Colab 추출(spa_colab)과 공용."""
    h, w = int(round(g.shape[0] * SCALE)), int(round(g.shape[1] * SCALE))
    s = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    m = SD.raster(gt, (h, w), CIDX, SCALE)
    z = SD.labeled_zone([dict(r, poly=r['poly'] * SCALE) for r in gt], (h, w), margin=30)
    m[z == 0] = IGN
    return s, m


def save_npz(stem, s, m, cache=None):
    fn = os.path.join(cache or CACHE, stem + '.npz')
    tmp = fn[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, img=s, mask=m)
    os.replace(tmp, fn)


def _prep_one(it):
    fn = os.path.join(CACHE, it['stem'] + '.npz')
    if os.path.exists(fn): return 0
    g = cv2.imread(it['path'], 0)
    s, m = prep_arrays(g, SD.load_gt(it))
    save_npz(it['stem'], s, m)
    return 1


def prep(workers=1):
    os.makedirs(CACHE, exist_ok=True)
    its = [it for it in SD.list_items('dev') if SD.trainable(it['stem'])]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex: n = sum(ex.map(_prep_one, its, chunksize=8))
    else:
        n = sum(_prep_one(it) for it in its)
    print('prep', n, 'new ->', CACHE)


def cached_stems(which='all', cache=None):
    """캐시(npz)에 있는 학습 가능 stem. which: train(devval 제외) | val(devval) | all. 보류는 여기서도 한 번 더 거름."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache or CACHE, '*.npz'))):
        st = os.path.basename(p)[:-4]
        if st.endswith('.tmp') or not SD.trainable(st): continue
        if which == 'train' and is_devval(st): continue
        if which == 'val' and not is_devval(st): continue
        out.append(st)
    return out


def load_npz(stem, cache=None):
    z = np.load(os.path.join(cache or CACHE, stem + '.npz'))
    return z['img'], z['mask']


def load_cache(which):
    """(이전 호환) 메모리로 전부 읽기 — 작은 자료용. 큰 자료는 cached_stems + load_npz(지연 읽기)."""
    return [(st,) + load_npz(st) for st in cached_stems('val' if which == 'val' else 'train')]


# ─────────────── 모델 ───────────────
def build(nc=len(CLASSES), c=None):
    import torch, torch.nn as nn, torch.nn.functional as F
    c1, c2, c3, c4, c5 = tuple(c) if c else C0

    def cbr(i, o, s=1, d=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Net(nn.Module):
        """입력 1/2 축척 회색조(잉크=1). 출력 1/4 해상도 logits(추론에서 보간). 기본 채널 0.54M 파라미터(width 2.0 ≈ 2.1M)."""
        def __init__(s):
            super().__init__()
            s.e1 = nn.Sequential(cbr(1, c1, 2), cbr(c1, c1))            # 1/2
            s.e2 = nn.Sequential(cbr(c1, c2, 2), cbr(c2, c2))           # 1/4
            s.e3 = nn.Sequential(cbr(c2, c3, 2), cbr(c3, c3))           # 1/8
            s.e4 = nn.Sequential(cbr(c3, c4, 2), cbr(c4, c4), cbr(c4, c4, d=2))          # 1/16
            s.e5 = nn.Sequential(cbr(c4, c5, 2), cbr(c5, c5, d=2), cbr(c5, c5, d=4))      # 1/32 (방 단위 맥락)
            s.d4 = cbr(c5 + c4, c4); s.d3 = cbr(c4 + c3, c3); s.d2 = cbr(c3 + c2, c2)
            s.head = nn.Conv2d(c2, nc, 1)

        def forward(s, x):
            x1 = s.e1(x); x2 = s.e2(x1); x3 = s.e3(x2); x4 = s.e4(x3); x5 = s.e5(x4)
            up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False)
            y = s.d4(torch.cat([up(x5, x4), x4], 1))
            y = s.d3(torch.cat([up(y, x3), x3], 1))
            y = s.d2(torch.cat([up(y, x2), x2], 1))
            return s.head(y)
    return Net()


def load_ckpt(path, map_location='cpu'):
    """→ (ckpt dict). 예전 형식(state_dict 만)은 {'model': sd, 'c': None} 으로 감쌈."""
    import torch
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if not (isinstance(ck, dict) and 'model' in ck): ck = dict(model=ck, c=None)
    return ck


# ─────────────── 표본 ───────────────
def aug_ink(a, rng):
    """(--aug 1) 가벼운 화질 흔들기 — uint8 회색조(흰 바탕). 선 굵기·감마·대비·흐림·잡음·JPEG."""
    g = a.astype(np.float32)
    r = rng.random()
    if r < 0.2: g = cv2.erode(g, np.ones((2, 2), np.uint8))           # 선 굵게
    elif r < 0.3: g = cv2.dilate(g, np.ones((2, 2), np.uint8))        # 얇게
    if rng.random() < 0.4:
        gam = math.exp(rng.uniform(-0.4, 0.4)); g = 255 * (np.clip(g, 0, 255) / 255) ** gam
        lo, hi = rng.uniform(0, 50), rng.uniform(205, 255); g = lo + g * (hi - lo) / 255
    if rng.random() < 0.25: g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < 0.25: g = g + rng.normal(0, rng.uniform(2, 10), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        _, b = cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 85))]); g = cv2.imdecode(b, 0)
    return g


def crop(img, m, rng, S=384, aug=0):
    """무작위 자르기. 절반은 드문 클래스(계단실·승강기·현관·승강기홀) 화소를 중심으로."""
    H, W = m.shape
    sc = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
    cs = int(S / sc)
    rare = [CIDX['공간_계단실'], CIDX['공간_엘리베이터'], CIDX['공간_현관'], CIDX['공간_엘리베이터홀']]
    cy = cx = None
    if rng.random() < 0.5:
        k = rare[rng.integers(len(rare))]
        ys, xs = np.nonzero(m[::4, ::4] == k)
        if len(ys): j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
    if cy is None:
        ys, xs = np.nonzero((m[::8, ::8] != IGN))
        if len(ys): j = rng.integers(len(ys)); cy, cx = ys[j] * 8, xs[j] * 8
        else: cy, cx = H // 2, W // 2
    y0 = int(np.clip(cy - cs // 2 + rng.integers(-cs // 4, cs // 4 + 1), 0, max(0, H - cs)))
    x0 = int(np.clip(cx - cs // 2 + rng.integers(-cs // 4, cs // 4 + 1), 0, max(0, W - cs)))
    a = np.full((cs, cs), 255, np.uint8); b = np.full((cs, cs), IGN, np.uint8)
    ca, cb = img[y0:y0 + cs, x0:x0 + cs], m[y0:y0 + cs, x0:x0 + cs]
    a[:ca.shape[0], :ca.shape[1]] = ca; b[:cb.shape[0], :cb.shape[1]] = cb
    if cs != S:
        a = cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA if cs > S else cv2.INTER_LINEAR)
        b = cv2.resize(b, (S, S), interpolation=cv2.INTER_NEAREST)
    k = rng.integers(4); a, b = np.rot90(a, k), np.rot90(b, k)
    if rng.random() < 0.5: a, b = a[:, ::-1], b[:, ::-1]
    if aug: a = aug_ink(np.ascontiguousarray(a), rng)
    a = (255 - a.astype(np.float32)) / 255.0
    if rng.random() < 0.3: a = np.clip(a * rng.uniform(0.6, 1.3), 0, 1)          # 선 농도 변화
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def sample(X, rng, S=384):
    """(이전 호환) 메모리 목록 X=[(stem,img,mask)] 에서 한 장 뽑아 자르기."""
    _, img, m = X[rng.integers(len(X))]
    return crop(img, m, rng, S)


def _crops_dataset():
    import torch
    from torch.utils.data import IterableDataset

    class Crops(IterableDataset):
        """캐시 npz 를 지연 읽기(수천 장이라 메모리에 다 올리지 않음). 한 장당 per_img 개 자르기 → 섞음 버퍼."""
        def __init__(s, stems, seed=0, per_img=2, S=384, aug=0, buf=128, cache=None):
            s.stems, s.seed, s.per_img, s.S, s.aug, s.buf, s.cache = stems, seed, per_img, S, aug, buf, cache

        def __iter__(s):
            cv2.setNumThreads(1)
            wi = torch.utils.data.get_worker_info()
            if wi is not None: torch.set_num_threads(1)
            rng = np.random.default_rng(s.seed * 1000 + (wi.id if wi else 0))
            B = []
            while True:
                st = s.stems[rng.integers(len(s.stems))]
                try: img, m = load_npz(st, s.cache)
                except Exception as e:
                    print('skip', st, e, flush=True); continue
                for _ in range(s.per_img): B.append(crop(img, m, rng, s.S, s.aug))
                while len(B) > s.buf:
                    a, b = B.pop(int(rng.integers(len(B))))
                    yield torch.from_numpy(a)[None], torch.from_numpy(b)          # 마스크 uint8(255=무시) → 장치에서 long
    return Crops


def class_weights(stems, maxn=800, seed=0):
    rng = np.random.default_rng(seed)
    pick = stems if len(stems) <= maxn else [stems[i] for i in rng.choice(len(stems), maxn, replace=False)]
    cnt = np.zeros(len(CLASSES))
    for st in pick:
        _, m = load_npz(st)
        cnt += np.bincount(m[m != IGN].ravel(), minlength=len(CLASSES))[:len(CLASSES)]
    w = 1.0 / np.sqrt(cnt / cnt.sum() + 1e-4); w = w / w[1:].mean()
    w[CIDX['공간_계단실']] *= 1.5; w[CIDX['공간_엘리베이터']] *= 1.5
    return w


def _save(ck, path):
    import torch
    tmp = path + '.tmp'
    torch.save(ck, tmp); os.replace(tmp, path)          # 저장 도중 끊겨도 이전 파일은 온전


# ─────────────── 학습 ───────────────
def train(minutes=50, bs=8, lr=3e-3, resume=False, device='auto', width=1.0, workers=0, warm='', out='', val_every=0.0,
          nval=0, save_every=10.0, aug=0, warmup=0, per_img=2, threads=1, seed=0, log_every=50):
    """minutes: 전체 시간 예산(이어 학습 시 누적). resume: True(예전 CKPT) 또는 체크포인트 경로. warm: 가중치만 불러와 새 일정.
    out: 체크포인트 폴더(last.pt·best.pt·val_log.json). 비우면 예전처럼 CKPT(out/spa_cache/spa_seg.pt, best 는 _best.pt).
    val_every(분): devval 주기 평가(0 = 끝에서만, 예전 동작). nval: devval 평가 장수(0 = 전부)."""
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    dev = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu')
    torch.set_num_threads(threads)
    cuda = dev.type == 'cuda'
    if cuda: torch.backends.cudnn.benchmark = True
    if out:
        os.makedirs(out, exist_ok=True); p_last, p_best = os.path.join(out, 'last.pt'), os.path.join(out, 'best.pt')
    else:
        p_last, p_best = CKPT, CKPT.replace('.pt', '_best.pt')
    os.makedirs(os.path.dirname(p_last), exist_ok=True)
    if resume is True: resume = p_last if os.path.exists(p_last) else ''
    ck0 = load_ckpt(resume, dev) if resume else None
    C = tuple(ck0['c']) if ck0 and ck0.get('c') else width_channels(width)
    if warm:
        ckw = load_ckpt(warm, dev)
        if ckw.get('c'): C = tuple(ckw['c'])
    Xs = cached_stems('train'); Vs = cached_stems('val')
    if nval and len(Vs) > nval: Vs = [Vs[i] for i in np.linspace(0, len(Vs) - 1, nval).round().astype(int)]
    nheld = sum(1 for p in glob.glob(os.path.join(CACHE, '*.npz')) if not SD.trainable(os.path.basename(p)[:-4]))
    print(f'device {dev} channels {C} train imgs {len(Xs)} devval {len(Vs)} (캐시 안 보류 {nheld}장 — 학습 제외)', flush=True)
    assert Xs, f'학습 캐시 없음: {CACHE}'
    assert not (set(Xs) & SD.held198()), '보류 198장이 학습 목록에 있음'
    w = np.array(ck0['w']) if ck0 and ck0.get('w') is not None else class_weights(Xs, seed=seed)
    print('class w', dict(zip(CLASSES, np.round(w, 2))), flush=True)
    net = build(c=C).to(dev)
    if cuda: net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    t_off, step, best, log = 0.0, 0, -1.0, []
    if warm: net.load_state_dict(ckw['model']); print('warm', warm, flush=True)
    if ck0:
        net.load_state_dict(ck0['model'])
        if ck0.get('opt'): opt.load_state_dict(ck0['opt'])
        if ck0.get('scaler') and cuda: scaler.load_state_dict(ck0['scaler'])
        t_off, step = ck0.get('elapsed', 0.0), ck0.get('step', 0)
        best, log = ck0.get('best', -1.0), ck0.get('log', [])
        print(f'resume {resume} step {step} elapsed {t_off / 60:.1f}m best {best}', flush=True)
    params = int(sum(p.numel() for p in net.parameters()))
    print('params', params, flush=True)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    Crops = _crops_dataset()
    ds = Crops(Xs, seed=seed + step + (1 if resume else 0), per_img=per_img, aug=aug)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, prefetch_factor=4 if workers else None, pin_memory=cuda)

    def ckpt():
        return dict(model=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict() if cuda else None, elapsed=time.time() - t0,
                    step=step, val=log[-1] if log else None, best=best, log=log, c=list(C), w=[float(x) for x in w], classes=CLASSES,
                    scale=SCALE, params=params, bs=bs, lr=lr, aug=aug, ntrain=len(Xs))

    def do_val(final=False):
        nonlocal best
        r = evaluate(net, Vs, dev) if Vs else {}
        r.update(step=step, min=round((time.time() - t0) / 60, 1), nval=len(Vs), ntrain=len(Xs), final=final)
        log.append(r); print('VAL', json.dumps(r, ensure_ascii=False), flush=True)
        try: json.dump(log, open(os.path.join(os.path.dirname(p_last), 'val_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
        except OSError: pass
        if r.get('_mean', -1) > best:
            best = r['_mean']; _save(ckpt(), p_best)
        return r

    T = minutes * 60; t0 = time.time() - t_off; tv = ts = time.time(); ema = None; tw = 0.0; tl = time.time()
    for x, y in dl:
        el = time.time() - t0
        tw += time.time() - tl
        if el >= T: break
        frac = el / T
        g = lr * 0.5 * (1 + np.cos(np.pi * min(frac, 1)))
        if warmup: g *= min(1.0, (step + 1) / warmup)
        for gr in opt.param_groups: gr['lr'] = g
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True).long()
        if cuda: x = x.contiguous(memory_format=torch.channels_last)
        net.train()
        with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
            lo = net(x)
        yd = y[:, 2::4, 2::4]                               # 1/4 해상도 정답(중앙 표본)
        loss = F.cross_entropy(lo.float(), yd, weight=wt, ignore_index=IGN)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
        if step % log_every == 0:
            print(f'{time.strftime("%H:%M:%S")} it {step} t {el:.0f}s ({el / 60:.1f}/{minutes:g}m) loss {ema:.3f} lr {g:.2e} '
                  f'data_wait {tw / max(1e-6, time.time() - ts + 1e-6):.0%}', flush=True)
        if val_every and time.time() - tv > val_every * 60:
            do_val(); tv = time.time()
            _save(ckpt(), p_last); ts = time.time(); tw = 0.0
        elif save_every and time.time() - ts > save_every * 60:
            _save(ckpt(), p_last); ts = time.time(); tw = 0.0
        tl = time.time()
    r = do_val(final=True)
    _save(ckpt(), p_last)
    print('saved', p_last, 'iters', step, 'best', p_best, round(best, 4), flush=True)
    return r


def evaluate(net, V, dev=None):
    """devval 화소 IoU(라벨 영역만). V = stem 목록 또는 [(stem,img,mask)]. '_mean' = 클래스 평균, '_fire' = 소방 7종 평균."""
    import torch
    dev = dev or next(net.parameters()).device
    cuda = getattr(dev, 'type', str(dev)) == 'cuda'
    net.eval()
    inter = np.zeros(len(CLASSES)); uni = np.zeros(len(CLASSES))
    with torch.no_grad():
        for v in V:
            _, img, m = v if isinstance(v, tuple) else (v,) + load_npz(v)
            H, W = img.shape
            Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
            a = np.full((Hp, Wp), 255, np.uint8); a[:H, :W] = img
            x = torch.from_numpy((255 - a.astype(np.float32)) / 255.0)[None, None].to(dev)
            if cuda: x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
                lo = net(x)
            p = lo.float()[0].argmax(0).cpu().numpy().astype(np.uint8)
            p = cv2.resize(p, (Wp, Hp), interpolation=cv2.INTER_NEAREST)[:H, :W]
            vm = m != IGN
            for k in range(1, len(CLASSES)):
                a_, b_ = (p == k) & vm, (m == k)
                inter[k] += (a_ & b_).sum(); uni[k] += (a_ | b_).sum()
    net.train()
    r = {CLASSES[k]: round(inter[k] / uni[k], 3) for k in range(1, len(CLASSES)) if uni[k]}
    r['_mean'] = round(float(np.mean([r[c] for c in CLASSES[1:] if c in r])), 4) if r else 0.0
    f = [r[c] for c in FIRE_EVAL if c in r]
    r['_fire'] = round(float(np.mean(f)), 4) if f else 0.0
    return r


# ─────────────── 내보내기 ───────────────
def export(ckpt=None, onnx_path=None, version=None):
    import torch
    ckpt, onnx_path = ckpt or CKPT, onnx_path or ONNX
    ck = load_ckpt(ckpt)
    C = tuple(ck['c']) if ck.get('c') else C0
    net = build(c=C); net.load_state_dict(ck['model']); net.eval()
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    x = torch.zeros(1, 1, 512, 512)
    torch.onnx.export(net, x, onnx_path, input_names=['img'], output_names=['logits'], opset_version=17,
                      dynamic_axes={'img': {2: 'h', 3: 'w'}, 'logits': {2: 'h4', 3: 'w4'}}, dynamo=False)
    err = None
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        x2 = torch.rand(1, 1, 384, 544)
        with torch.no_grad(): ref = net(x2).numpy()
        err = float(np.abs(ref - s.run(None, {'img': x2.numpy()})[0]).max())
    except ImportError:
        pass
    try:
        from fire3dbeam.spa_seg import PARAM
        post = json.loads(json.dumps(PARAM, ensure_ascii=False))
    except Exception:
        post = None
    meta = dict(classes=CLASSES, scale=SCALE, out_stride=4, input='(255-gray)/255, 1x1xHxW, 2048축소본을 scale 배',
                version=version or f'c={list(C)} step {ck.get("step")} ntrain {ck.get("ntrain")}',
                c=list(C), params=int(sum(p.numel() for p in net.parameters())), ckpt=os.path.basename(ckpt), step=ck.get('step'),
                train_minutes=round(ck.get('elapsed', 0) / 60, 1) if ck.get('elapsed') else None, ntrain=ck.get('ntrain'),
                dev_val=ck.get('val'), onnx_vs_torch_maxabs=err, size_mb=round(os.path.getsize(onnx_path) / 1e6, 2),
                post=post, post_note='기록용 — 실제 적용값은 fire3dbeam/spa_seg.PARAM (이 json 에 "param" 이 있으면 그 위에 덮음)',
                created=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(meta, open(os.path.splitext(onnx_path)[0] + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(onnx_path, os.path.getsize(onnx_path), 'maxabs', err)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prep', 'train', 'export'])
    ap.add_argument('--minutes', type=float, default=50); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--resume', nargs='?', const=True, default=False, help='이어 학습: 경로 생략 시 last(예전 CKPT)')
    ap.add_argument('--warm', default='', help='가중치만 불러오고 새 일정으로 학습(.pt, 같은 채널 구성)')
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda')
    ap.add_argument('--width', type=float, default=1.0, help='채널 배율(1.0 = 0.54M 파라미터, 2.0 ≈ 2.1M)')
    ap.add_argument('--workers', type=int, default=0, help='DataLoader 프로세스 수(0 = 본 프로세스)')
    ap.add_argument('--threads', type=int, default=1, help='torch CPU 스레드')
    ap.add_argument('--out', default='', help='체크포인트 폴더(last.pt·best.pt). 비우면 out/spa_cache/spa_seg.pt')
    ap.add_argument('--val_every', type=float, default=0, help='분(0 = 끝에서만)')
    ap.add_argument('--nval', type=int, default=0, help='devval 평가 장수(0 = 전부)')
    ap.add_argument('--save_every', type=float, default=10, help='분 — last.pt 저장 주기')
    ap.add_argument('--aug', type=int, default=0, help='1 = 화질 흔들기(선 굵기·감마·흐림·잡음·JPEG)')
    ap.add_argument('--warmup', type=int, default=0, help='학습률 선형 warmup 스텝')
    ap.add_argument('--per_img', type=int, default=2, help='읽은 영상 1장당 자르기 수')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log_every', type=int, default=50, help='스텝마다 손실 출력')
    ap.add_argument('--ckpt', default='', help='export: 체크포인트(기본 out/spa_cache/spa_seg.pt)')
    ap.add_argument('--onnx', default='', help='export: 출력 onnx(기본 models/spa_seg.onnx)')
    ap.add_argument('--version', default=None)
    a = ap.parse_args()
    if a.cmd == 'prep': prep(a.workers or 1)
    elif a.cmd == 'train':
        train(a.minutes, bs=a.bs, lr=a.lr, resume=a.resume, device=a.device, width=a.width, workers=a.workers, warm=a.warm, out=a.out,
              val_every=a.val_every, nval=a.nval, save_every=a.save_every, aug=a.aug, warmup=a.warmup, per_img=a.per_img,
              threads=a.threads, seed=a.seed, log_every=a.log_every)
    elif a.cmd == 'export': export(a.ckpt or None, a.onnx or None, a.version)
