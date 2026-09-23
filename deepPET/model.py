"""DeepPET, the convolutional encoder-decoder of Haggstrom et al. 2019 (Fig. 1).

Input `(B, 1, 288, 371)`, a precorrected sinogram in SUV.mm; output
`(B, 1, grid, grid)` in SUV.

  encoder  3x3 conv + BN + ReLU; a stride-2 conv opens each stage
           32 @ 288x371 -> 64 @ 144x186 -> 128 @ 72x93 -> 256 @ 36x47
           -> 512 / 1024 / 512 @ 18x24
  decoder  bilinear upsampling to fixed sizes (x~1.7 per step, as the paper),
           then 3x3 conv + BN + ReLU, halving the channels
           26 -> 44 -> 75 -> 128 (-> 256 for `grid=256`), 256 -> 128 -> 64 -> 32 (-> 16)
  head     one 3x3 conv to a single channel, linear

31 conv layers at `grid=128`, the paper's count. How many convs each stage
holds is not in the paper's text; three per stage (six in the bottleneck)
reaches its 31. BN momentum 0.2 as in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER = ((32, 3), (64, 3), (128, 3), (256, 3))

BOTTLENECK = (512, 512, 1024, 1024, 1024, 512)

DECODER_SIZES = {128: (26, 44, 75, 128), 256: (26, 44, 75, 128, 256)}

DECODER_CH = (256, 128, 64, 32, 16)


def cbr(cin: int, cout: int, stride: int = 1, bn_momentum: float = 0.2) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout, momentum=bn_momentum),
                         nn.ReLU(inplace=True))


class DeepPET(nn.Module):

    def __init__(self, grid: int = 128, bn_momentum: float = 0.2):
        super().__init__()
        if grid not in DECODER_SIZES:
            raise ValueError(f"grid must be one of {tuple(DECODER_SIZES)}")
        self.grid = grid
        layers, c = [], 1
        for s, (w, n) in enumerate(ENCODER):
            for j in range(n):
                layers.append(cbr(c, w, 2 if (j == 0 and s > 0) else 1, bn_momentum))
                c = w
        for j, w in enumerate(BOTTLENECK):
            layers.append(cbr(c, w, 2 if j == 0 else 1, bn_momentum))
            c = w
        self.encoder = nn.Sequential(*layers)

        self.sizes = DECODER_SIZES[grid]
        self.decoder = nn.ModuleList()
        for w in DECODER_CH[:len(self.sizes)]:
            self.decoder.append(nn.Sequential(cbr(c, w, 1, bn_momentum),
                                              cbr(w, w, 1, bn_momentum),
                                              cbr(w, w, 1, bn_momentum)))
            c = w
        self.head = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for size, block in zip(self.sizes, self.decoder):
            h = block(F.interpolate(h, size=(size, size), mode="bilinear",
                                    align_corners=False))
        return self.head(h)

    def n_conv(self) -> int:
        return sum(isinstance(m, nn.Conv2d) for m in self.modules())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
