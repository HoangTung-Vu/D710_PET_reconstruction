"""An inert `kornia_rs`, for CPUs without AVX2."""

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
