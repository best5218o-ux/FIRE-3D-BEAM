# -*- coding: utf-8 -*-
"""스캔본(글자층 없는 이미지 PDF) 정보카드 시험 세트 — 가상 표준양식 카드(data_std 등)를 쪽마다 그림으로 바꾸고 스캔 열화를 입힌다.
python -m synth.scan data_std data_scan_dev [--seed 1]      /  python -m synth.scan data_std_hold data_scan_hold --seed 2   (--gray 0.2: 흑백 스캔 20%)

쪽마다: 150~200 dpi 래스터화 → 크기 ±3% · 회전 ±1.5° · 이동 → 종이색(회색·누런 바탕, 약한 명암 기울기) → 흐림 → 잡음 → JPEG(품질 55~85)
→ 쪽 크기 그대로 JPEG 한 장만 들어간 PDF(글자층 없음). gt.json·daejang.json·site.json 은 그대로 복사(정답·보조 입력 동일).
열화 값은 scan.json 에 기록(정답 아님, 재현용)."""
import json, os, shutil, sys
import numpy as np
import cv2
import pymupdf


GRAY_P = 0.0


def degrade(img, rng, prm):
    """img: RGB uint8(쪽 전체) → 열화된 RGB uint8"""
    H, W = img.shape[:2]
    s, ang = prm['scale'], prm['rot']
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
    M[0, 2] += prm['dx'] * W; M[1, 2] += prm['dy'] * H
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)).astype(np.float32)
    # 종이색 + 명암 기울기(스캐너 조명 불균일)
    bg = np.array(prm['paper'], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shade = 1.0 - prm['shade'] * ((xx / W) * np.cos(prm['shade_dir']) + (yy / H) * np.sin(prm['shade_dir']))
    out = out / 255.0 * bg[None, None, :] * shade[..., None]
    # 잉크 대비 약화(검은 글자가 짙은 회색으로)
    out = prm['ink'] + out * (1 - prm['ink'] / 255.0)
    if prm['blur'] > 0: out = cv2.GaussianBlur(out, (0, 0), prm['blur'])
    out += rng.normal(0, prm['noise'], out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def make_scan(src_pdf, dst_pdf, rng):
    doc = pymupdf.open(src_pdf)
    dpi = int(rng.choice([150, 175, 200]))
    gray = bool(rng.random() < GRAY_P)                    # 흑백(회색조) 스캔 비율(기본 0 — 색 기호 도면 인식은 OCR 과 별개 문제)
    out = pymupdf.open()
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy()
        pb = float(rng.uniform(222, 248))
        prm = dict(scale=float(rng.uniform(0.97, 1.03)), rot=float(rng.uniform(-1.5, 1.5)), dx=float(rng.uniform(-0.01, 0.01)), dy=float(rng.uniform(-0.01, 0.01)),
                   paper=[pb + float(rng.uniform(0, 6)), pb + float(rng.uniform(0, 4)), pb - float(rng.uniform(0, 10))],
                   shade=float(rng.uniform(0, 0.08)), shade_dir=float(rng.uniform(0, 6.28)), ink=float(rng.uniform(10, 45)),
                   blur=float(rng.uniform(0.3, 1.1)), noise=float(rng.uniform(2, 7)), jpeg=int(rng.integers(55, 86)))
        d = degrade(img, rng, prm)
        if gray: d = np.repeat(cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(d, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, prm['jpeg']])
        pg = out.new_page(width=page.rect.width, height=page.rect.height)
        pg.insert_image(pg.rect, stream=buf.tobytes())
        pages.append(prm)
    out.save(dst_pdf, garbage=3, deflate=True)
    return dict(dpi=dpi, gray=gray, pages=pages)


def main(src, dst, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        d = os.path.join(src, name)
        if not os.path.exists(os.path.join(d, 'card.pdf')): continue
        o = os.path.join(dst, name); os.makedirs(o, exist_ok=True)
        meta = make_scan(os.path.join(d, 'card.pdf'), os.path.join(o, 'card.pdf'), rng)
        for f in os.listdir(d):
            if f != 'card.pdf' and f.endswith('.json'): shutil.copy2(os.path.join(d, f), o)
        meta['src'] = os.path.join(src, name, 'card.pdf')
        json.dump(meta, open(os.path.join(o, 'scan.json'), 'w'), ensure_ascii=False, indent=1)
        print(name, meta['dpi'], 'gray' if meta['gray'] else 'color', f"{os.path.getsize(os.path.join(o, 'card.pdf')) / 1e6:.1f}MB", flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 1
    if '--gray' in sys.argv: GRAY_P = float(sys.argv[sys.argv.index('--gray') + 1])
    main(a[0], a[1], seed)
