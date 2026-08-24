"""Two SIRF traps that must be handled **before** touching any `AcquisitionData`.

Both showed up in real runs; neither is precautionary:

1. **`tmp_*.hs`/`.s`, 231 MB per pair.** SIRF writes them into the process's
   **current working directory** — not next to the source file — and every
   `get_uniform_copy` creates a pair, kept until the object is collected. The
   first run left 926 MB sitting next to the notebook. Hence: `chdir` into
   `<case>/scratch`.
2. **The `MessageRedirector` reference must be KEPT.** Drop it and it is
   collected immediately, after which STIR dumps thousands of INFO lines to
   stdout. `setup()` holds it for you.

And a third one that `chdir` creates by itself: the `''` entry in `sys.path`
(inserted by both Jupyter and `python script.py`) is resolved **at import time**,
so after the `chdir` it points into `scratch/` and `import osem` breaks halfway
through the notebook. `setup()` anchors it to an absolute path before changing
directory.
"""

from __future__ import annotations

import os
import sys

from .paths import ROOT, Case

#: Keeps the `MessageRedirector` alive for the life of the process. That is its
#: entire reason for existing.
_redirector = None


def anchor_sys_path() -> None:
    """Make every relative `sys.path` entry absolute, and ensure `D710/` is on it.

    Call before any `chdir`. Without it, `import utils` / `import osem` after a
    `chdir` go looking inside `scratch/`.
    """
    root = str(ROOT)
    sys.path[:] = [os.path.abspath(p) if p in ("", ".") else p for p in sys.path]
    if root not in sys.path:
        sys.path.insert(0, root)


def setup(case: Case, quiet: bool = True):
    """Enter `<case>/scratch` and silence STIR's INFO output. Returns the scratch dir.

    Idempotent: calling it again in the same process creates no further
    redirector.
    """
    global _redirector

    anchor_sys_path()
    case.scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(case.scratch)

    if quiet and _redirector is None:
        import sirf.STIR as pet

        # These three land in scratch alongside the tmp_*.hs — same fate.
        _redirector = pet.MessageRedirector("info.txt", "warn.txt", "err.txt")
    return case.scratch


def clear_scratch(case: Case) -> int:
    """Delete `tmp_*` in scratch. Returns the number of files removed.

    Safe between runs, NOT safe while any `AcquisitionData` is alive: SIRF still
    holds those files open, and deleting them empties the object.
    """
    n = 0
    if not case.scratch.is_dir():
        return 0
    for p in case.scratch.glob("tmp_*"):
        p.unlink()
        n += 1
    return n
