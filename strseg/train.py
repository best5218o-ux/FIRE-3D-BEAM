# -*- coding: utf-8 -*-
"""STR 벽·창·문 분할 학습 (CPU 2스레드, bf16 autocast + channels_last).
  python -m strseg.train --minutes 110 --out models/ckpt
- 자료: strseg.data 캐시(배율 0.6 ≈ 74dpi). 매 에폭(=N 스텝)마다 폴더를 다시 훑어 새로 들어온 영상을 포함.
- 무작위 자르기 384px, 배율 0.67~1.33(유효 dpi ≈50~100), 뒤집기·90°회전·±3°기울임, 대비·감마·흐림·잡음·JPEG,
  선 굵기 흔들기(최소/최대 필터), 벽 회색 칠(가상 정보카드의 회색 띠 벽 대비).
- 손실: CE(무시 255, 가중 1/1.5/3/3) + 0.5·soft dice(벽·창·문). 학습률: 시간 기준 cosine.
- dev-val(해시 10%) 에서 주기적으로 IoU → best.pt"""
import os, sys, time, json, math, argparse, random
import numpy as np, cv2, torch, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from strseg import data as D
from strseg.model import StrNet

CROP = 384


def load_pair(stem):
    g = cv2.imread(os.path.join(D.CACHE, stem + '_i.png'), cv2.IMREAD_GRAYSCALE)
    l = cv2.imread(os.path.join(D.CACHE, stem + '_l.png'), cv2.IMREAD_GRAYSCALE)
    return g, l


def aug_img(g, l, rng):
    g = g.astype(np.float32)
    if rng.random() < 0.15:                                    # 벽 회색 칠(회색 띠 벽 양식)
        wm = (l == 1)
        if wm.any():
            lvl = rng.uniform(90, 200)
            g[wm & (g < 170)] = lvl + rng.normal(0, 4, (wm & (g < 170)).sum())
    if rng.random() < 0.3:                                     # 선 굵기: 어두운 획 굵게
        k = int(rng.choice([2, 3]))
        g = cv2.erode(g, np.ones((k, k), np.uint8))
    elif rng.random() < 0.12:                                  # 얇게
        g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:                                     # 감마·대비·바탕 회색
        gam = math.exp(rng.uniform(-0.5, 0.5)); g = 255 * (np.clip(g, 0, 255) / 255) ** gam
        lo, hi = rng.uniform(0, 60), rng.uniform(200, 255)
        g = lo + g * (hi - lo) / 255
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.3:
        g = g + rng.normal(0, rng.uniform(2, 12), g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        ok, b = cv2.imencode('.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 85))]); g = cv2.imdecode(b, 0)
    return g


def sample_crop(g, l, rng, crop=CROP):
    H, W = l.shape
    sc = math.exp(rng.uniform(math.log(0.67), math.log(1.33)))
    cs = int(round(crop / sc))
    vy, vx = np.nonzero(l[::4, ::4] != 255)
    if len(vy) and rng.random() < 0.9:
        k = rng.integers(len(vy)); cy, cx = vy[k] * 4, vx[k] * 4
    else:
        cy, cx = rng.integers(H), rng.integers(W)
    y0 = int(np.clip(cy - cs // 2, 0, max(0, H - cs))); x0 = int(np.clip(cx - cs // 2, 0, max(0, W - cs)))
    gi = np.full((cs, cs), 255, np.uint8); li = np.full((cs, cs), 255, np.uint8)
    gg = g[y0:y0 + cs, x0:x0 + cs]; ll = l[y0:y0 + cs, x0:x0 + cs]
    gi[:gg.shape[0], :gg.shape[1]] = gg; li[:ll.shape[0], :ll.shape[1]] = ll
    if rng.random() < 0.3:                                     # 작은 기울임
        M = cv2.getRotationMatrix2D((cs / 2, cs / 2), rng.uniform(-3, 3), 1.0)
        gi = cv2.warpAffine(gi, M, (cs, cs), flags=cv2.INTER_LINEAR, borderValue=255)
        li = cv2.warpAffine(li, M, (cs, cs), flags=cv2.INTER_NEAREST, borderValue=255)
    gi = cv2.resize(gi, (crop, crop), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    li = cv2.resize(li, (crop, crop), interpolation=cv2.INTER_NEAREST)
    k = int(rng.integers(4)); gi = np.rot90(gi, k); li = np.rot90(li, k)
    if rng.random() < 0.5: gi = gi[:, ::-1]; li = li[:, ::-1]
    gi = np.ascontiguousarray(gi); li = np.ascontiguousarray(li)
    return aug_img(gi, li, rng), li


class Crops(IterableDataset):
    def __init__(self, seed=0, per_img=2):
        self.seed, self.per_img = seed, per_img

    def __iter__(self):
        cv2.setNumThreads(1); torch.set_num_threads(1)
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) + int(time.time()))
        items, t_scan = [], 0
        while True:
            if time.time() - t_scan > 180:                     # 새로 들어온 영상 포함
                items = [it['stem'] for it in D.cached('train')]; t_scan = time.time()
                rev = [x for x in items if x.startswith('REV_')]          # 검수 라벨은 REV_REPEAT 배로 더 자주 뽑음
                items += rev * max(0, int(os.environ.get('REV_REPEAT', '3')) - 1)
                syn = [x for x in items if x.startswith('SYN_')]          # 가상 정보카드 정답은 SYN_REPEAT 배
                items += syn * max(0, int(os.environ.get('SYN_REPEAT', '1')) - 1)
            g, l = load_pair(items[rng.integers(len(items))])
            if g is None or l is None: continue
            for _ in range(self.per_img):
                gi, li = sample_crop(g, l, rng)
                yield torch.from_numpy(gi)[None].float() / 255, torch.from_numpy(li).long()


def loss_fn(o, y, w):
    ce = F.cross_entropy(o, y, weight=w, ignore_index=255)
    p = o.softmax(1); v = (y != 255)
    yy = torch.where(v, y, torch.zeros_like(y))
    oh = F.one_hot(yy, 4).permute(0, 3, 1, 2).float() * v[:, None]
    p = p * v[:, None]
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ce + 0.5 * dice[1:].mean(), ce.item()


def pad32(x):
    H, W = x.shape[-2:]; ph, pw = (-H) % 32, (-W) % 32
    return F.pad(x, (0, pw, 0, ph), value=1.0), H, W


@torch.no_grad()
def validate(m, items, bf16=True):
    m.eval(); dev = next(m.parameters()).device
    I = np.zeros(4); U = np.zeros(4)
    for it in items:
        g, l = load_pair(it['stem'])
        ys, xs = np.nonzero(l != 255)
        if not len(ys): continue
        y0, y1, x0, x1 = max(0, ys.min() - 16), ys.max() + 17, max(0, xs.min() - 16), xs.max() + 17
        g, l = g[y0:y1, x0:x1], l[y0:y1, x0:x1]
        x, H, W = pad32(torch.from_numpy(g)[None, None].float() / 255)
        x = x.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16 if dev.type == 'cuda' else torch.bfloat16, enabled=bf16):
            o = m(x.contiguous(memory_format=torch.channels_last))
        p = o.float()[0, :, :H, :W].argmax(0).cpu().numpy()
        v = l != 255
        for c in range(1, 4):
            a, b = (p == c) & v, (l == c)
            I[c] += (a & b).sum(); U[c] += (a | b).sum()
    m.train()
    iou = I / np.maximum(U, 1)
    return dict(wall=round(float(iou[1]), 4), window=round(float(iou[2]), 4), door=round(float(iou[3]), 4), mean=round(float(iou[1:].mean()), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=110)
    ap.add_argument('--out', default='models/ckpt')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val_every', type=float, default=12, help='분')
    ap.add_argument('--nval', type=int, default=50)
    ap.add_argument('--resume', default='')
    ap.add_argument('--warm', default='', help='가중치만 불러오고 새 일정(학습률 cosine·warmup)으로 계속 학습')
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda (Colab GPU)')
    ap.add_argument('--width', type=float, default=1.0, help='채널 배율(1.0 = 기본 1.12M 파라미터, 2.0 ≈ 4.4M)')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(0)
    dev = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) or a.device == 'cuda' else 'cpu')
    C = tuple(int(round(c * a.width / 4) * 4) for c in (12, 24, 40, 64, 96, 128))
    print('device', dev, 'channels', C, flush=True)
    m = StrNet(c=C).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=1e-4)
    t_off = 0.0
    if a.warm:
        m.load_state_dict(torch.load(a.warm, map_location=dev, weights_only=False)['model'])
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); t_off = ck.get('elapsed', 0)
    w = torch.tensor([1.0, 1.5, 3.0, 3.0]).to(dev)
    dl = DataLoader(Crops(), batch_size=a.bs, num_workers=a.workers, prefetch_factor=4, pin_memory=dev.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    budget = a.minutes * 60; t0 = time.time() - t_off; tv = time.time(); step = 0; best = -1; log = []
    run_ce = []
    for x, y in dl:
        el = time.time() - t0
        if el > budget: break
        fr = el / budget
        lr = a.lr * min(1.0, (step + 1) / 150) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * fr)))
        for gp in opt.param_groups: gp['lr'] = lr
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.float16 if dev.type == 'cuda' else torch.bfloat16):
            o = m(x.contiguous(memory_format=torch.channels_last))
        loss, ce = loss_fn(o.float(), y, w)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1; run_ce.append(loss.item())
        if step % 50 == 0:
            print(f'{time.strftime("%H:%M:%S")} step {step} el {el / 60:.1f}m lr {lr:.2e} loss {np.mean(run_ce):.4f}', flush=True); run_ce = []
        if time.time() - tv > a.val_every * 60 or el + 5 > budget:
            vi = D.cached('val')[:a.nval]
            r = validate(m, vi); r.update(step=step, min=round(el / 60, 1), nval=len(vi), ntrain=len(D.cached('train')))
            log.append(r); print('VAL', json.dumps(r), flush=True)
            ck = dict(model=m.state_dict(), opt=opt.state_dict(), elapsed=el, step=step, val=r, c=C)
            torch.save(ck, os.path.join(a.out, 'last.pt'))
            if r['mean'] > best: best = r['mean']; torch.save(ck, os.path.join(a.out, 'best.pt'))
            json.dump(log, open(os.path.join(a.out, 'val_log.json'), 'w'), indent=1)
            tv = time.time()
    vi = D.cached('val')[:a.nval]
    r = validate(m, vi); r.update(step=step, min=round((time.time() - t0) / 60, 1), nval=len(vi), final=True); log.append(r)
    print('VAL', json.dumps(r), flush=True)
    ck = dict(model=m.state_dict(), opt=opt.state_dict(), elapsed=time.time() - t0, step=step, val=r, c=C)
    torch.save(ck, os.path.join(a.out, 'last.pt'))
    if r['mean'] > best: torch.save(ck, os.path.join(a.out, 'best.pt'))
    json.dump(log, open(os.path.join(a.out, 'val_log.json'), 'w'), indent=1)
