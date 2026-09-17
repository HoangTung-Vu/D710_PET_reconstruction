"""The assumptions `osem/` makes about SIRF, checked against SIRF."""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest

import interfile
from conftest import ROOT

MAX_STATEMENTS = 15

NOTEBOOK = ROOT / "osem_pipeline.ipynb"


def code_cells():
    """`(index, source, parsed module)` for every code cell, skipping magics."""
    if not NOTEBOOK.exists():
        pytest.skip(f"no {NOTEBOOK.name}; nothing to hold to the contract")
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
    ad, img, make = model
    x = img.get_uniform_copy(1.0)

    plain = make().forward(x).as_array()
    s, b = 0.5, 3.0
    got = make(sensitivity=uniform(ad, s), background=uniform(ad, b)) \
        .forward(x).as_array()

    assert got == pytest.approx(s * plain + b, rel=1e-4, abs=1e-5)


def test_the_sensitivity_multiplies_rather_than_divides(model):
    ad, img, make = model
    x = img.get_uniform_copy(1.0)
    plain = make().forward(x).as_array()
    dim = make(sensitivity=uniform(ad, 0.5)).forward(x).as_array()
    assert dim.sum() < plain.sum()
    assert dim == pytest.approx(0.5 * plain, rel=1e-4, abs=1e-5)


def test_the_sensitivity_deadline_is_the_reconstructors_set_up(sirf, bed24):
    ad, img = bed24
    half = uniform(ad, 0.5)

    def subset_sensitivity(when):
        am = sirf.AcquisitionModelUsingRayTracingMatrix()
        am.set_num_tangential_LORs(1)
        if when == "before":
            am.set_acquisition_sensitivity(sirf.AcquisitionSensitivityModel(half))
        am.set_up(ad, img)
        if when == "after":
            am.set_acquisition_sensitivity(sirf.AcquisitionSensitivityModel(half))
        obj = sirf.make_Poisson_loglikelihood(ad, acq_model=am)
        obj.set_num_subsets(1)
        rec = sirf.OSMAPOSLReconstructor()
        rec.set_objective_function(obj)
        rec.set_num_subiterations(1)
        rec.set_input(ad)
        rec.set_up(img)
        return float(obj.get_subset_sensitivity(0).as_array().sum())

    none, before, after = (subset_sensitivity(w)
                           for w in ("none", "before", "after"))
    assert before == pytest.approx(after, rel=1e-6), (
        "am.set_up is not the deadline; if this ever fails, SIRF has started "
        "reading the sensitivity at set_up and osem/recon.py's order is load-"
        "bearing after all")
    assert before == pytest.approx(0.5 * none, rel=1e-4)


def test_stir_canonicalises_the_plane_order_on_read(sirf, bed24, tmp_path):
    ad, _img = bed24
    n_planes = ad.as_array().shape[1]
    stamp = np.zeros(ad.as_array().shape, dtype=np.float32)
    stamp += np.arange(n_planes, dtype=np.float32)[None, :, None, None]

    src = ad.get_uniform_copy(0)
    src.fill(stamp)
    out = str(tmp_path / "stamped.hs")
    src.write(out)

    assert (sirf.AcquisitionData(out).as_array() == stamp).all()

    written = interfile.keys(out)
    assert written["matrix axis label [3]"].lower() == "view"
    ascending = [int(n) for n in
                 written["minimum ring difference per segment"]
                 .strip("{} ").split(",")]
    assert ascending == sorted(ascending)
    assert ascending[0] < 0 < ascending[-1]


def test_a_uniform_copy_lands_in_the_current_directory(sirf, bed24, tmp_path,
                                                       monkeypatch):
    ad, _img = bed24
    monkeypatch.chdir(tmp_path)
    keep = ad.get_uniform_copy(0)
    assert list(tmp_path.glob("tmp_*.s")), "SIRF stopped writing scratch files"
    del keep


def test_decay_to_injection_uses_the_frame_average(model):
    half_life, duration = 6586.2002, 90.0
    lam = np.log(2) / half_life

    def factor(dt_s, T):
        return 1.0 / (np.exp(-lam * dt_s) * (1 - np.exp(-lam * T)) / (lam * T))

    assert factor(3600.0, duration) / factor(3600.0 - 91.0, duration) == \
        pytest.approx(np.exp(lam * 91.0), rel=1e-9)
    assert factor(3600.0, duration) > factor(3509.0, duration)
    assert factor(0.0, duration) == pytest.approx(
        lam * duration / (1 - np.exp(-lam * duration)), rel=1e-12)
    assert factor(0.0, duration) > 1.0


def test_bed_stitching_indices_are_exact_plane_offsets():
    from utils import geometry

    step_mm = 124.26
    planes = step_mm / geometry.PLANE_MM
    assert planes == pytest.approx(round(planes), abs=0.02)
    assert round(planes) == 38
    assert 47 - round(planes) == 9
