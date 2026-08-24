"""The contract `osem_pipeline.ipynb` runs under -- both halves of it.

1. **What the notebook assumes about SIRF**, checked against SIRF, on the
   miniature scanner where a whole acquisition model fits in a second.  Each
   one is a place where being wrong produces a plausible-looking image rather
   than an error.
2. **That the notebook holds no code of its own.**  This project has already
   been bitten once by a copy of `utils/` living inside a notebook and the two
   drifting apart.  The rule is enforced by machine here rather than promised
   in a README: a code cell may import, set parameters, make a call and draw a
   figure -- it may not define a `def` or a `class`, and it may not run long.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest

import interfile
from conftest import ROOT

#: A cell that is longer than this is doing work that belongs in a module.
MAX_STATEMENTS = 15

NOTEBOOK = ROOT / "osem_pipeline.ipynb"


def code_cells():
    """`(index, source, parsed module)` for every code cell, skipping magics.

    A cell starting with `%` or `!` is IPython syntax, not Python; there are
    none today and the notebook is better off without them, but they should
    make this test skip that cell rather than error.
    """
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(ln.lstrip().startswith(("%", "!")) for ln in src.splitlines()):
            continue
        out.append((i, src, ast.parse(src)))
    return out


def test_the_notebook_defines_no_functions_or_classes():
    """Anything worth a `def` is worth a module, where it can be tested.

    The drift this prevents is not hypothetical: `utils/` was once copied into
    a notebook and the two copies diverged, so the notebook and the pipeline
    stopped computing the same thing while both kept running.
    """
    offenders = []
    for i, _src, tree in code_cells():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                offenders.append(f"cell {i}: {type(node).__name__} {node.name!r}")
    assert not offenders, (
        "the notebook defines code of its own:\n  " + "\n  ".join(offenders)
        + "\nMove it into `utils/` (not algorithm-specific) or `osem/` "
          "(algorithm), and call it from the cell.")


def test_no_notebook_cell_runs_long():
    """A long cell is a module that has not been written yet.

    Counted as top-level statements, so a `for` loop over the beds is one
    statement no matter how many beds there are -- what this catches is a cell
    that has quietly grown into a script.
    """
    long_cells = [(i, len(tree.body)) for i, _s, tree in code_cells()
                  if len(tree.body) > MAX_STATEMENTS]
    assert not long_cells, (
        "cells over %d top-level statements: %s" % (MAX_STATEMENTS, long_cells))


@pytest.fixture
def model(sirf, bed24):
    """`(acq_data, image, fresh acquisition-model factory)`."""
    ad, img = bed24

    def make(sensitivity=None, background=None):
        am = sirf.AcquisitionModelUsingRayTracingMatrix()
        am.set_num_tangential_LORs(5)
        if sensitivity is not None:
            # Before `set_up`, so STIR folds it into the sensitivity image --
            # that is what makes the correction quantitative instead of a
            # re-weighting.
            am.set_acquisition_sensitivity(
                sirf.AcquisitionSensitivityModel(sensitivity))
        if background is not None:
            am.set_background_term(background)
        am.set_up(ad, img)
        return am

    return ad, img, make


def uniform(ad, value):
    a = ad.get_uniform_copy(0)
    a.fill(np.full(ad.as_array().shape, value, dtype=np.float32))
    return a


def test_the_forward_model_is_s_times_gx_plus_b(model):
    """`y = S(Gx) + b`, with `b` outside `S`.

    Randoms and scatter are already in the measured count domain, so they must
    NOT be scaled by the sensitivity.  If SIRF ever put the background inside
    `S`, every reconstruction here would under-subtract by the norm.
    """
    ad, img, make = model
    x = img.get_uniform_copy(1.0)

    plain = make().forward(x).as_array()
    s, b = 0.5, 3.0
    got = make(sensitivity=uniform(ad, s), background=uniform(ad, b)) \
        .forward(x).as_array()

    assert got == pytest.approx(s * plain + b, rel=1e-4, abs=1e-5)


def test_the_sensitivity_multiplies_rather_than_divides(model):
    """`AcquisitionSensitivityModel` treats its input as **bin efficiencies**.

    This is the direction that decides whether `normdt` is divided out or
    multiplied in.  A sensitivity < 1 has to make the forward projection
    smaller.
    """
    ad, img, make = model
    x = img.get_uniform_copy(1.0)
    plain = make().forward(x).as_array()
    dim = make(sensitivity=uniform(ad, 0.5)).forward(x).as_array()
    assert dim.sum() < plain.sum()
    assert dim == pytest.approx(0.5 * plain, rel=1e-4, abs=1e-5)


def test_stir_canonicalises_the_plane_order_on_read(sirf, bed24, tmp_path):
    """A SIRF-written term and a decoded one line up in `as_array()`.

    They do **not** line up on disk: `to_stir.py` clones the decoder's
    `segment, axial, view, tangential` header with segments in `0, +1, -1, ...`
    order, while `AcquisitionData.write` emits `segment, view, axial,
    tangential` with segments ascending `-11 .. +11`.  The notebook multiplies
    `normdt` by the cached `attn` as plain numpy arrays, so if `as_array()`
    followed the file instead of STIR's own order the two would be silently
    mismatched by a segment permutation.
    """
    ad, _img = bed24
    n_planes = ad.as_array().shape[1]
    stamp = np.zeros(ad.as_array().shape, dtype=np.float32)
    stamp += np.arange(n_planes, dtype=np.float32)[None, :, None, None]

    src = ad.get_uniform_copy(0)
    src.fill(stamp)
    out = str(tmp_path / "stamped.hs")
    src.write(out)

    assert (sirf.AcquisitionData(out).as_array() == stamp).all()

    # ... and the file really is in the other order, so the check above is not
    # a tautology.
    written = interfile.keys(out)
    assert written["matrix axis label [3]"].lower() == "view"
    ascending = [int(n) for n in
                 written["minimum ring difference per segment"]
                 .strip("{} ").split(",")]
    assert ascending == sorted(ascending)
    assert ascending[0] < 0 < ascending[-1]


def test_a_uniform_copy_lands_in_the_current_directory(sirf, bed24, tmp_path,
                                                       monkeypatch):
    """The 231 MB-a-time trap the notebook chdir's around.

    `get_uniform_copy` writes a `tmp_*.hs`/`.s` pair into the **process's**
    working directory -- not next to the source file -- and keeps it for as
    long as the returned object lives.  A notebook that holds six beds' worth
    of terms therefore holds six beds' worth of scratch files, which is where
    the first run's 926 MB beside the notebook came from.
    """
    ad, _img = bed24
    monkeypatch.chdir(tmp_path)
    keep = ad.get_uniform_copy(0)
    assert list(tmp_path.glob("tmp_*.s")), "SIRF stopped writing scratch files"
    del keep                                    # released only on collection


def test_decay_to_injection_uses_the_frame_average(model):
    """The factor the notebook applies before stitching beds.

    `f = exp(-lambda*dt) * (1 - exp(-lambda*T)) / (lambda*T)`.  The second
    factor is the mean activity **during** the frame; dropping it (using the
    instantaneous activity at frame start) biases every bed by ~0.5 % here and
    much more on a long frame, and the bias differs per bed, so it survives as
    an axial gradient rather than cancelling into `K`.
    """
    half_life, duration = 6586.2002, 90.0
    lam = np.log(2) / half_life

    def factor(dt_s, T):
        return 1.0 / (np.exp(-lam * dt_s) * (1 - np.exp(-lam * T)) / (lam * T))

    # Two beds 91 s apart differ by exactly one decay step, and the later bed
    # needs the LARGER factor because less activity is left.
    assert factor(3600.0, duration) / factor(3600.0 - 91.0, duration) == \
        pytest.approx(np.exp(lam * 91.0), rel=1e-9)
    assert factor(3600.0, duration) > factor(3509.0, duration)
    # The frame-average term is a pure function of T, and always > 1.
    assert factor(0.0, duration) == pytest.approx(
        lam * duration / (1 - np.exp(-lam * duration)), rel=1e-12)
    assert factor(0.0, duration) > 1.0


def test_bed_stitching_indices_are_exact_plane_offsets():
    """124.26 mm of table travel is exactly 38 planes, so 9 planes overlap.

    The stitch rounds `(z - z0) / PLANE_MM` to an integer.  That is only safe
    because the step really is an integer number of planes; a half-plane offset
    would put two beds' voxels in the same slot with a 1.6 mm axial error.
    """
    from utils import geometry

    step_mm = 124.26
    planes = step_mm / geometry.PLANE_MM
    assert planes == pytest.approx(round(planes), abs=0.02)
    assert round(planes) == 38
    assert 47 - round(planes) == 9
