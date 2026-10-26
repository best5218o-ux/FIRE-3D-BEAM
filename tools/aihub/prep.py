"""AI허브 표본 zip → 축소 PNG(긴 변 2048) + meta.json(원본 크기). python prep.py zip outdir"""
import sys, zipfile, io, json, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
zp, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
mp = os.path.join(out, 'meta.json'); meta = json.load(open(mp)) if os.path.exists(mp) else {}
L = 2048
with zipfile.ZipFile(zp) as z:
    for n in z.namelist():
        b = os.path.basename(n)
        if not b.upper().endswith('.PNG') or b in meta: continue
        im = Image.open(io.BytesIO(z.read(n))); W, H = im.size
        im = im.convert('L'); f = L / max(W, H); im = im.resize((round(W * f), round(H * f)), Image.LANCZOS)
        im.save(os.path.join(out, b), optimize=False)
        meta[b] = dict(w=W, h=H, f=f)
json.dump(meta, open(mp, 'w'))
print(out, len(meta))
