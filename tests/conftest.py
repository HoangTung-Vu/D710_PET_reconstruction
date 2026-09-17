"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


@pytest.fixture(scope="session", autouse=True)
def _scratch(tmp_path_factory):
    old = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def stir():
    """The `stir` module, or a skip."""
    try:
        import stir
    except ImportError as e:
        pytest.skip(f"STIR unavailable ({e}); activate petct_reconstruction")
    return stir


@pytest.fixture(scope="session")
def sirf():
    """`sirf.STIR`, or a skip."""
    try:
        import sirf.STIR as pet
    except ImportError as e:
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
    """`(proj_data, info)` for the miniature scanner."""
    pd = stir.ProjData.read_from_file(mini_hs)
    return pd, pd.get_proj_data_info()


@pytest.fixture(scope="session")
def bed24(sirf, tmp_path_factory):
    """`(acq_data, image)` for a 24-ring miniature scanner: 47 image planes, as a bed."""
    import synth_hs

    d = tmp_path_factory.mktemp("bed24")
    hs = synth_hs.write(str(d / "bed24"), num_rings=24, num_det=48, num_tang=9)
    sirf.MessageRedirector("info.txt", "warn.txt", "err.txt")
    ad = sirf.AcquisitionData(hs)
    return ad, ad.create_uniform_image(1.0, 32)


@pytest.fixture(scope="session")
def ct_dir(tmp_path_factory):
    """A CT series long enough to cover a whole bed: 60 slices at 3.27 mm."""
    import synth_ct

    return synth_ct.series(tmp_path_factory.mktemp("ct") / "s2", n_slices=60)
