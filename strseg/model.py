# -*- coding: utf-8 -*-
"""작은 U-Net (CPU용, ≈1M 파라미터). 입력 1ch 회색조(0..1, 흰 바탕=1) → 4클래스 로짓(배경·벽·창·문), 입력과 같은 해상도.
입력 크기는 32의 배수로 맞춰 넣을 것(래퍼가 패딩)."""
import torch, torch.nn as nn, torch.nn.functional as F


def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.a = cbr(i, o, s); self.b = cbr(o, o)
    def forward(self, x): return self.b(self.a(x))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.a = cbr(i + skip, o); self.b = cbr(o, o)
    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False)
        return self.b(self.a(torch.cat([x, s], 1)))


class StrNet(nn.Module):
    def __init__(self, c=(12, 24, 40, 64, 96, 128), ncls=4):
        super().__init__()
        self.stem = cbr(1, c[0])                                  # 1/1
        self.e1 = Block(c[0], c[1], 2)                            # 1/2
        self.e2 = Block(c[1], c[2], 2)                            # 1/4
        self.e3 = Block(c[2], c[3], 2)                            # 1/8
        self.e4 = Block(c[3], c[4], 2)                            # 1/16
        self.e5 = nn.Sequential(Block(c[4], c[5], 2), cbr(c[5], c[5], d=2))   # 1/32 + 넓은 문맥
        self.d4 = Up(c[5], c[4], c[4])
        self.d3 = Up(c[4], c[3], c[3])
        self.d2 = Up(c[3], c[2], c[2])
        self.d1 = Up(c[2], c[1], c[1])
        self.d0 = nn.Sequential(cbr(c[1] + c[0], 16), nn.Conv2d(16, ncls, 1))

    def forward(self, x):
        x = (x - 0.8) / 0.3
        s0 = self.stem(x); s1 = self.e1(s0); s2 = self.e2(s1); s3 = self.e3(s2); s4 = self.e4(s3); s5 = self.e5(s4)
        y = self.d4(s5, s4); y = self.d3(y, s3); y = self.d2(y, s2); y = self.d1(y, s1)
        y = F.interpolate(y, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        return self.d0(torch.cat([y, s0], 1))


if __name__ == '__main__':
    import time
    torch.set_num_threads(2)
    m = StrNet(); print('params', sum(p.numel() for p in m.parameters()))
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for S, B in ((384, 8), (320, 8)):
        x = torch.rand(B, 1, S, S); y = torch.randint(0, 4, (B, S, S))
        for i in range(3):
            t = time.time(); l = F.cross_entropy(m(x), y); opt.zero_grad(); l.backward(); opt.step()
            print(S, B, f'{time.time() - t:.2f}s')
    m.eval()
    with torch.no_grad():
        x = torch.rand(1, 1, 896, 1248); t = time.time(); m(x); print('infer 896x1248', f'{time.time() - t:.2f}s')
