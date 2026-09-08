"""A `kornia_rs` that does nothing -- for CPUs without AVX2.

NOT on any path by default, and it must stay that way: on a normal machine it
would shadow the real extension.  Opt in per machine by putting this directory
on PYTHONPATH (see the README).

WHY.  `pytomography.utils.spatial` does `from kornia.geometry.transform import
rotate` at import time, so `import kornia` is unavoidable for `d710 lm` -- and
`kornia/__init__.py` imports `kornia.io`, which imports `kornia_rs`, a compiled
Rust extension.  The published `kornia_rs` wheels use AVX2/FMA, so on a
pre-Haswell CPU (Xeon E5-2600 v1/v2, i.e. any HP Z820) that import is
`Illegal instruction (core dumped)` -- SIGILL, no traceback.

Nothing here needs it.  `kornia_rs` is image FILE I/O only (JPEG/PNG/TIFF
readers and writers, `kornia/io/io.py`); the PET pipeline never loads an image
file through kornia, it reads sinograms and event tables with numpy.  So the
import has to succeed and the module has to never be called -- which is exactly
what this is.

Module-level `__getattr__` (PEP 562) means the failure is LOUD if that
assumption is ever wrong: any attribute access raises with the real fix in the
message, instead of silently returning something wrong.
"""

_MSG = (
    "kornia_rs is stubbed out on this machine (tools/stubs/kornia_rs.py): the "
    "published wheel uses AVX2 instructions this CPU does not have, and "
    "importing the real one dies with SIGILL.\n"
    "Reaching this means something is genuinely doing image file I/O through "
    "kornia, which the PET pipeline was not supposed to do.  Build the real "
    "extension for this CPU instead:\n"
    "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n"
    "    . \"$HOME/.cargo/env\"\n"
    "    RUSTFLAGS='-C target-cpu=sandybridge' \\\n"
    "        pip install --no-binary kornia-rs kornia-rs==0.1.14\n"
    "and take this directory off PYTHONPATH."
)

__version__ = "0.0.0+stub"


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise RuntimeError("%s\n(attribute asked for: kornia_rs.%s)" % (_MSG, name))
