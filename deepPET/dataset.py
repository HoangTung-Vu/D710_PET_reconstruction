"""The prepared slices, and a torch Dataset that simulates a sinogram per item.

Nothing is cached: every `__getitem__` projects, draws noise and precorrects
(~15-35 ms at 128, ~35-70 ms at 256), so training sees a new noise realisation,
count level and augmentation each epoch. Validation and test items are
deterministic: their generator is seeded by `(seed, index)`.

DataLoader workers are started with `spawn`, not `fork`. parallelproj runs on
OpenMP, and libgomp is not fork-safe once the parent has used it -- a forked
worker can hang in its first projection. `OMP_NUM_THREADS` is set before the
workers start so that `workers x threads` does not oversubscribe the CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .simulate import COUNT_RANGE, MODES, augment, downsample, simulate


def default_data() -> Path:
    from utils.paths import out_root

    return out_root() / "deeppet" / "data"


class SliceStore:
    """The `prepare` output: one memmapped `(n, 256, 256)` pair per study."""

    def __init__(self, root, studies=None):
        self.root = Path(os.path.expanduser(str(root)))
        idx = self.root / "index.json"
        if not idx.exists():
            raise SystemExit(f"error: no {idx}\n  run: python -m deepPET.prepare --data <dataset>")
        rows = {r["sid"]: r for r in json.loads(idx.read_text())}
        self.studies = list(studies) if studies is not None else sorted(rows)
        missing = [s for s in self.studies if s not in rows]
        if missing:
            raise SystemExit(f"error: {len(missing)} studies of the split are not in "
                             f"{idx}, e.g. {missing[:3]}")
        self.n = [rows[s]["n"] for s in self.studies]
        self._mm = {}

    def split(self, name: str) -> list[str]:
        return json.loads((self.root / "split.json").read_text())[name]

    def _arrays(self, k: int):
        if k not in self._mm:
            base = self.root / "slices" / self.studies[k]
            self._mm[k] = (np.load(f"{base}_suv.npy", mmap_mode="r"),
                           np.load(f"{base}_mu.npy", mmap_mode="r"))
        return self._mm[k]

    def get(self, k: int, i: int, grid: int):
        """`(suv, mu)` of slice `i` of study `k`, float32 on `grid`."""
        suv, mu = self._arrays(k)
        return (downsample(np.asarray(suv[i], np.float32), grid),
                downsample(np.asarray(mu[i], np.float32), grid))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mm"] = {}
        return d


class SinoDataset:
    """`(x_in (1, 288, 371), target (1, grid, grid), counts)` per slice of a split.

    `train=True`: fresh randomness and augmentation per call. Otherwise the
    generator is seeded by `(seed, index)`, and `counts` fixes the count level
    (else it is drawn from `count_range`, deterministically).
    """

    def __init__(self, root, split: str, grid: int = 128, mode: str = "paper",
                 train: bool = False, count_range=COUNT_RANGE, counts=None,
                 limit_studies=None, max_items=None, seed: int = 0,
                 eff=None, return_raw: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        studies = SliceStore(root).split(split)
        if limit_studies:
            studies = studies[:limit_studies]
        self.store = SliceStore(root, studies)
        self.items = [(k, i) for k, n in enumerate(self.store.n) for i in range(n)]
        if max_items and len(self.items) > max_items:
            pick = np.random.default_rng(seed).choice(len(self.items), max_items, replace=False)
            self.items = [self.items[j] for j in sorted(pick)]
        self.grid, self.mode, self.train = grid, mode, train
        self.count_range, self.counts, self.seed = count_range, counts, seed
        self.eff, self.return_raw = eff, return_raw
        self._scanner = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def scanner(self):
        if self._scanner is None:
            from .scanner2d import Scanner2D

            self._scanner = Scanner2D(self.grid)
        return self._scanner

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_scanner"] = None
        return d

    def sample(self, j: int):
        """The numpy item: `(x_in, target, info)`."""
        k, i = self.items[j]
        rng = np.random.default_rng() if self.train else \
            np.random.default_rng([self.seed, j])
        suv, mu = self.store.get(k, i, self.grid)
        if self.train:
            suv, mu = augment(suv, mu, rng, self.grid)
        x, t, info = simulate(suv, mu, self.scanner, rng, self.mode, counts=self.counts,
                              count_range=self.count_range, eff=self.eff,
                              return_raw=self.return_raw)
        info["study"], info["slice"] = self.store.studies[k], i
        return x, t, info

    def __getitem__(self, j: int):
        import torch

        x, t, info = self.sample(j)
        return (torch.from_numpy(x)[None], torch.from_numpy(t)[None],
                torch.tensor(info["counts"], dtype=torch.float32))


def loader(ds: SinoDataset, batch: int, workers: int, shuffle: bool,
           sim_threads: int = 1, drop_last: bool = False):
    """A DataLoader over `ds` with spawned workers of `sim_threads` OpenMP threads each."""
    import torch

    kw = {}
    if workers > 0:
        os.environ["OMP_NUM_THREADS"] = str(sim_threads)
        kw = {"multiprocessing_context": "spawn", "persistent_workers": True,
              "prefetch_factor": 4}
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available(), **kw)
