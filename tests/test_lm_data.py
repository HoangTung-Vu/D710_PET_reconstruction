"""The list-mode path on real data. Skips cleanly when `$D710_OUT` has none.

The check that matters is the first one: histogramming the decoded events back
through `lm.geom.BinMap` has to reproduce `decoded/bed<n>.s` **bit for bit**.
The vendor decoder produced that file by a completely separate path, so this is
an independent proof of the bin map, not a self-check -- the same standard
`vendor/to_stir.py` holds itself to.
"""

from __future__ import annotations

import numpy as np
import pytest
from cases import decoded_beds

from lm import events as ev
from lm import geom

#: Reading every term of a bed costs ~1 GB and a few seconds, so the per-event
#: lookups are checked on one bed rather than all of them.
LOOKUP_BEDS = 1


def _with_listmode():
    out = []
    for b in decoded_beds():
        from utils.paths import case as get_case

        npy = get_case(b["case"]).decoded / f"bed{b['bed']}.lm.npy"
        if npy.exists():
            out.append({**b, "npy": str(npy)})
    if not out:
        return [pytest.param(None, marks=pytest.mark.skip(
            reason="no decoded bed<n>.lm.npy under $D710_OUT; run "
                   "`d710 decode --listmode --format npy`"), id="no-listmode")]
    return [pytest.param(b, id=f"{b['case']}-bed{b['bed']}") for b in out]


LM_BEDS = _with_listmode()


@pytest.fixture(scope="module")
def _binmaps():
    return {}


def _binmap(cache, hs):
    if hs not in cache:
        cache[hs] = geom.BinMap(hs)
    return cache[hs]


@pytest.mark.parametrize("bed", LM_BEDS)
def test_histogram_reproduces_the_decoded_sinogram(bed, _binmaps):
    """Verification 6: the per-event bin map round-trips, bit-exactly."""
    binmap = _binmap(_binmaps, bed["hs"])
    e = ev.load(bed["npy"])
    ref = np.fromfile(bed["hs"].replace(".hs", ".s"), "<i2")
    n_tof = ref.size // binmap.n_bin
    assert ref.size == n_tof * binmap.n_bin, "the .s is not a whole number of bins"

    h, dropped = ev.histogram(e, binmap, n_tof)
    assert int(h.sum(dtype=np.int64)) == len(e) - dropped
    assert np.array_equal(h.reshape(-1), ref), (
        f"{int((h.reshape(-1) != ref).sum()):,} bins differ")


@pytest.mark.parametrize("bed", LM_BEDS)
def test_event_count_matches_the_header(bed):
    e = ev.load(bed["npy"])
    assert len(e) == bed["hdr"]["prompts"]


@pytest.mark.parametrize("bed", LM_BEDS[:LOOKUP_BEDS])
def test_per_event_terms_are_usable(bed, _binmaps):
    """Weights positive, additive non-negative, both finite -- OSEM divides by them."""
    from lm import terms
    from utils.paths import case as get_case

    C = get_case(bed["case"])
    if not (C.work_bed(bed["bed"]) / "attn.hs").exists():
        pytest.skip("no attn.hs yet; run `d710 attn --case <n>`")

    binmap = _binmap(_binmaps, bed["hs"])
    e = ev.load(bed["npy"], mmap=False)[:2_000_000]
    keep, w, add = terms.event_terms(C, bed["bed"], e, binmap, n_tof=1)

    assert keep.sum() > 0.99 * len(e)
    assert np.isfinite(w).all() and (w > 0).all()
    assert np.isfinite(add).all() and (add >= 0).all()

    # The identity the additive term is defined by: PyTomography's model is
    # `y = Hx + a` with the weight only in the sensitivity image, so `a` is the
    # background DIVIDED by the weight. Multiply it back and the per-LOR
    # background of that event's bin has to come out.
    b = ev.bins(e, binmap)[keep]
    bg = (terms.read(C, bed["bed"], "randoms", binmap)
          + terms.read(C, bed["bed"], "scatter", binmap))[b]
    assert np.allclose(add * w, bg, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("bed", LM_BEDS[:LOOKUP_BEDS])
def test_sensitivity_matches_the_sinogram_it_came_from(bed, _binmaps):
    """Summed over LORs, the per-LOR weights give back the per-bin sinogram."""
    from lm import terms
    from utils.paths import case as get_case

    C = get_case(bed["case"])
    if not (C.work_bed(bed["bed"]) / "attn.hs").exists():
        pytest.skip("no attn.hs yet; run `d710 attn --case <n>`")

    binmap = _binmap(_binmaps, bed["hs"])
    ids, w = terms.sensitivity(C, bed["bed"], binmap)
    assert len(ids) == int(binmap.mult.sum()) * binmap.n_view * binmap.n_tang

    per_bin = terms.read(C, bed["bed"], "normdt", binmap, per_lor=False) \
        * terms.read(C, bed["bed"], "attn", binmap)
    assert float(w.sum(dtype=np.float64)) == pytest.approx(
        float(per_bin.sum(dtype=np.float64)), rel=1e-4)
