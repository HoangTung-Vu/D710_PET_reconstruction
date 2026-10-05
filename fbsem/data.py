from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from utils.paths import case as get_case
from utils.scanner import XY

from . import bed as B


def sim_cases(root, sets, sim: str = "sim_an_s1", names=None,
              need_label: bool = True) -> list:
    root = Path(root)
    out = []
    for s in sets:
        for d in sorted(root.glob(f"{s}_*")):
            if names and d.name not in names:
                continue
            if not (d / sim).is_dir():
                continue
            C = get_case(sim, str(d))
            beds = [n for n in C.decoded_beds() if B.ready(C, n, need_label)]
            if beds:
                out.append((C, beds))
    return out


def epoch_samples(cases, beds_per_case: int, rng) -> list:
    out = []
    for C, beds in cases:
        k = min(beds_per_case, len(beds))
        for n in rng.choice(beds, size=k, replace=False):
            out.append((C, int(n)))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def prefetch(samples, n_sub: int, psf, cache, with_label: bool = True,
             xy: int = XY):
    def load(C, n):
        return B.load_arrays(C, n, n_sub, psf, cache, with_label, xy)

    with ThreadPoolExecutor(max_workers=1) as ex:
        nxt = None
        for i, (C, n) in enumerate(samples):
            cur = nxt or ex.submit(load, C, n)
            nxt = (ex.submit(load, *samples[i + 1])
                   if i + 1 < len(samples) else None)
            yield cur.result()


def body_mask(xt: np.ndarray, frac: float = 0.05) -> np.ndarray:
    return xt > frac * np.percentile(xt, 99.9)
