# -*- coding: utf-8 -*-
"""체크포인트 → models/str_seg.onnx (+ .json 메타). fp32, 입력 [1,1,H,W] (H,W 가변, 32의 배수), 출력 로짓 [1,4,H,W].
  python -m strseg.export models/ckpt/best.pt models/str_seg.onnx"""
import os, sys, json, time
import numpy as np, torch
from strseg.model import StrNet


def export(ck_path, out):
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    m = StrNet(c=tuple(ck['c'])) if ck.get('c') else StrNet(); m.load_state_dict(ck['model']); m.eval()
    x = torch.rand(1, 1, 256, 320)
    torch.onnx.export(m, x, out, input_names=['image'], output_names=['logits'], opset_version=17,
                      dynamic_axes={'image': {2: 'H', 3: 'W'}, 'logits': {2: 'H', 3: 'W'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    x2 = torch.rand(1, 1, 416, 640)
    with torch.no_grad(): ref = m(x2).numpy()
    got = s.run(None, {'image': x2.numpy()})[0]
    err = float(np.abs(ref - got).max())
    meta = dict(classes=['bg', 'wall', 'window', 'door'], input='gray 0..1 (white=1), [1,1,H,W], H,W multiple of 32',
                work_dpi=round(300 * 0.2476, 2), note='AI허브 건축 도면 데이터 STR 라벨로 학습(로컬 전용 자료). 작업 해상도 ≈74dpi (1/100 도면 1 m ≈ 29 px)',
                params=int(sum(p.numel() for p in m.parameters())), ckpt=os.path.basename(ck_path), step=ck.get('step'),
                train_minutes=round(ck.get('elapsed', 0) / 60, 1), dev_val=ck.get('val'), onnx_vs_torch_maxabs=err,
                size_mb=round(os.path.getsize(out) / 1e6, 2), created=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(meta, open(os.path.splitext(out)[0] + '.json', 'w'), ensure_ascii=False, indent=1)
    return meta


if __name__ == '__main__':
    print(json.dumps(export(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
