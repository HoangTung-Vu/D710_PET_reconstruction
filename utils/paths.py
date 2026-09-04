"""Where everything generated at run time goes — and the ONLY place that knows it.

The code tree (`D710/`) holds code only. Every file produced at run time goes
under a root the operator names:

    $D710_OUT/<case>/
        decoded/       bed<n>.{hs,s,json,singles.npy,...}   <- d710 decode
        vendor/bed<n>/ {randoms,scatter,normdt,norm_only}.f32, prompts.u16, ...
        work/bed<n>/   {randoms,scatter,background,normdt,norm_only,attn}.{hs,s}
        export/        <case>_bqml.nii.gz, <case>_suvbw.nii.gz, dicom/
        scratch/       SIRF's tmp_*.hs/.s — safe to delete at any time
        logs/
        recon.npz      the stitched volume (count/voxel), the osem -> export bridge

**There is no default root.** Missing both `--out` and `$D710_OUT` is an error,
not a fallback to the code directory — that fallback is exactly what this
refactor removed.

Cases are nested (`<case>/vendor/bed<n>`) rather than flat (`<case>_bed<n>`):
deleting a case is `rm -rf <case>`, and case names no longer have to avoid `_`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: Root of the code tree — the `D710/` directory. Derived from `__file__`, so it
#: is correct no matter where the process was started.
ROOT = Path(__file__).resolve().parent.parent

_BED_RE = re.compile(r"bed(\d+)$")


class NoOutputRoot(SystemExit):
    """Nowhere to write. A `SystemExit` so scripts die cleanly, without a traceback."""

    def __init__(self) -> None:
        super().__init__(
            "error: no idea where to write the output.\n"
            "  set   export D710_OUT=~/UET/d710_out\n"
            "  or pass  --out <directory>\n"
            "  (deliberately no default: output must never land inside the source tree)")


def out_root(explicit: str | os.PathLike | None = None) -> Path:
    """`--out` > `$D710_OUT` > error."""
    p = explicit or os.environ.get("D710_OUT")
    if not p:
        raise NoOutputRoot()
    return Path(os.path.expanduser(str(p))).resolve()


class Case:
    """One exam and all of its directories. Paths only — it never touches the disk."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root / name

    def __repr__(self) -> str:
        return f"Case({self.name!r}, {self.root})"

    # -------------------------------------------------------- directories
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
    def recon(self) -> Path:
        """The stitched volume, count/voxel referred back to the injection time.

        It sits at the case root rather than in `export/`: it is the **input** to
        the export step, not one of its products. `d710 osem` writes it,
        `d710 export` reads it.
        """
        return self.root / "recon.npz"

    @property
    def recon_lm(self) -> Path:
        """The same thing from the list-mode path (`d710 lm recon`)."""
        return self.root / "recon_lm.npz"

    def vendor_bed(self, n: int) -> Path:
        return self.vendor / f"bed{n}"

    def work_bed(self, n: int) -> Path:
        return self.work / f"bed{n}"

    def prompt(self, n: int) -> Path:
        """Bed n's raw-prompt `.hs` — also the header template for every other term."""
        return self.decoded / f"bed{n}.hs"

    def header(self, n: int) -> dict:
        """Bed n's RDF header sidecar, written by `d710 decode`."""
        with open(self.decoded / f"bed{n}.json") as f:
            return json.load(f)

    def mkdirs(self) -> Case:
        for d in (self.decoded, self.vendor, self.work, self.export,
                  self.scratch, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # --------------------------------------------------- discovered on disk
    def decoded_beds(self) -> list[int]:
        """Beds that have been through step 1 (they have a `bed<n>.hs` and a sidecar)."""
        out = []
        for hs in self.decoded.glob("bed*.hs"):
            m = _BED_RE.match(hs.stem)
            if m and hs.with_suffix(".json").exists():
                out.append(int(m.group(1)))
        return sorted(out)

    def beds(self, terms=("background", "normdt")) -> list[int]:
        """Beds that have been through all three steps — the ones OSEM can eat.

        Discovered from disk, never typed by hand: a bed that failed at step 2
        still has its step-1 `bed<n>.hs`, and including it means stitching in a
        bed with no background term.
        """
        return [n for n in self.decoded_beds()
                if all((self.work_bed(n) / f"{t}.hs").exists() for t in terms)]


def case(name: str, out: str | os.PathLike | None = None) -> Case:
    """Case `name` under `--out` / `$D710_OUT`.

        from utils.paths import case
        C = case("ped")
        C.prompt(4), C.work_bed(4), C.export
    """
    return Case(name, out_root(out))


def cases(out: str | os.PathLike | None = None) -> list[Case]:
    """Every case already present under the output root."""
    root = out_root(out)
    if not root.is_dir():
        return []
    return [Case(d.name, root) for d in sorted(root.iterdir())
            if d.is_dir() and (d / "decoded").is_dir()]
