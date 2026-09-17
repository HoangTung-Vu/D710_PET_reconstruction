"""The single place that knows where run-time output goes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_BED_RE = re.compile(r"bed(\d+)$")


class NoOutputRoot(SystemExit):
    """Raised when there is nowhere to write."""

    def __init__(self) -> None:
        super().__init__(
            "error: no idea where to write the output.\n"
            "  set   export D710_OUT=~/UET/d710_out\n"
            "  or pass  --out <directory>\n"
            "  (deliberately no default: output must never land inside the source tree)")


def out_root(explicit: str | os.PathLike | None = None) -> Path:
    """The output root: `--out`, then `$D710_OUT`, then an error."""
    p = explicit or os.environ.get("D710_OUT")
    if not p:
        raise NoOutputRoot()
    return Path(os.path.expanduser(str(p))).resolve()


class Case:
    """One exam and all of its directories."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root / name

    def __repr__(self) -> str:
        return f"Case({self.name!r}, {self.root})"

    @property
    def decoded(self) -> Path:
        return self.root / "decoded"

    @property
    def vendor(self) -> Path:
        return self.root / "vendor"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def raw_sim(self) -> Path:
        return self.root / "raw_simulation"

    def raw_sim_bed(self, n: int) -> Path:
        return self.raw_sim / f"bed{n}"

    @property
    def recon(self) -> Path:
        return self.root / "recon.npz"

    @property
    def recon_lm(self) -> Path:
        return self.root / "recon_lm.npz"

    def vendor_bed(self, n: int) -> Path:
        return self.vendor / f"bed{n}"

    def work_bed(self, n: int) -> Path:
        return self.work / f"bed{n}"

    def prompt(self, n: int) -> Path:
        return self.decoded / f"bed{n}.hs"

    def header(self, n: int) -> dict:
        with open(self.decoded / f"bed{n}.json") as f:
            return json.load(f)

    def mkdirs(self) -> Case:
        for d in (self.decoded, self.vendor, self.work, self.export,
                  self.scratch, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def decoded_beds(self) -> list[int]:
        out = []
        for hs in self.decoded.glob("bed*.hs"):
            m = _BED_RE.match(hs.stem)
            if m and hs.with_suffix(".json").exists():
                out.append(int(m.group(1)))
        return sorted(out)

    def beds(self, terms=("background", "normdt")) -> list[int]:
        return [n for n in self.decoded_beds()
                if all((self.work_bed(n) / f"{t}.hs").exists() for t in terms)]


def case(name: str, out: str | os.PathLike | None = None) -> Case:
    """Case `name` under `--out` or `$D710_OUT`."""
    return Case(name, out_root(out))


def cases(out: str | os.PathLike | None = None) -> list[Case]:
    """Every case already present under the output root."""
    root = out_root(out)
    if not root.is_dir():
        return []
    return [Case(d.name, root) for d in sorted(root.iterdir())
            if d.is_dir() and (d / "decoded").is_dir()]
