"""Shared fixtures.

The import path is **not** set here: it is declared in `pytest.ini`
(`pythonpath = . vendor`), so nothing in this file touches `sys.path`.
`utils` and `osem` are ordinary packages; `vendor/` stays a directory of
scripts because that is how the pipeline actually runs them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


@pytest.fixture(scope="session", autouse=True)
def _scratch(tmp_path_factory):
    """Run from a scratch directory.

    SIRF writes `tmp_<n>_<stamp>.hs`/`.s` into the **current** directory, one
    pair per `get_uniform_copy`, and never cleans them up -- the same trap
    `utils.sirf_env.setup()` sidesteps by chdir'ing into `<case>/scratch`.
    Every path the tests use is absolute, so moving the cwd costs nothing.
    """
    old = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def stir():
    """The `stir` module, or a skip.

    STIR only resolves once `conda activate petct_reconstruction` has run --
    the bare interpreter path is not enough, because the loader needs the
    environment's `LD_LIBRARY_PATH`.
    """
    try:
        import stir
    except ImportError as e:  # pragma: no cover - depends on the shell
        pytest.skip(f"STIR unavailable ({e}); activate petct_reconstruction")
    return stir


@pytest.fixture(scope="session")
def sirf():
    """`sirf.STIR`, or a skip."""
    try:
        import sirf.STIR as pet
    except ImportError as e:  # pragma: no cover - depends on the shell
        pytest.skip(f"SIRF unavailable ({e}); activate petct_reconstruction")
    return pet


@pytest.fixture(scope="session")
def mini_hs(tmp_path_factory):
    """A scaled-down span-2 acquisition on disk; returns the `.hs` path."""
    import synth_hs

    d = tmp_path_factory.mktemp("mini")
    return synth_hs.write(str(d / "mini"))


@pytest.fixture(scope="session")
def mini_info(stir, mini_hs):
    """`(proj_data, info)` for the miniature scanner.

    STIR's `get_proj_data_info()` hands back a borrowed pointer, so the
    `ProjData` has to stay alive for as long as the info is used -- hence the
    pair, and hence the session scope.
    """
    pd = stir.ProjData.read_from_file(mini_hs)
    return pd, pd.get_proj_data_info()


@pytest.fixture(scope="session")
def bed24(sirf, tmp_path_factory):
    """`(acq_data, image)` for a 24-ring miniature -- 47 image planes, as a bed.

    `attenuation.mu_image` insists on the real 47-plane bed grid, so the
    scanner keeps its 24 rings and shrinks in the two axes that grid does not
    care about: 48 detectors (24 views) and 9 tangential bins.  553 planes
    survive, at 1/1400th of the memory.
    """
    import synth_hs

    d = tmp_path_factory.mktemp("bed24")
    hs = synth_hs.write(str(d / "bed24"), num_rings=24, num_det=48, num_tang=9)
    sirf.MessageRedirector("info.txt", "warn.txt", "err.txt")
    ad = sirf.AcquisitionData(hs)
    return ad, ad.create_uniform_image(1.0, 32)


@pytest.fixture(scope="session")
def ct_dir(tmp_path_factory):
    """A CT series long enough to hold a whole bed: 60 slices at 3.27 mm."""
    import synth_ct

    return synth_ct.series(tmp_path_factory.mktemp("ct") / "s2", n_slices=60)


