"""Two SIRF conditions that must hold before any `AcquisitionData` is touched."""

from __future__ import annotations

import os
import sys

from .paths import ROOT, Case

_redirector = None


def anchor_sys_path() -> None:
    """Make every relative `sys.path` entry absolute, and put `D710/` on the path."""
    root = str(ROOT)
    sys.path[:] = [os.path.abspath(p) if p in ("", ".") else p for p in sys.path]
    if root not in sys.path:
        sys.path.insert(0, root)


def setup(case: Case, quiet: bool = True):
    """Enter `<case>/scratch` and silence STIR's INFO output."""
    global _redirector

    anchor_sys_path()
    case.scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(case.scratch)

    if quiet and _redirector is None:
        import sirf.STIR as pet

        _redirector = pet.MessageRedirector("info.txt", "warn.txt", "err.txt")
    return case.scratch


def clear_scratch(case: Case) -> int:
    """Delete `tmp_*` from the scratch directory."""
    n = 0
    if not case.scratch.is_dir():
        return 0
    for p in case.scratch.glob("tmp_*"):
        p.unlink()
        n += 1
    return n
