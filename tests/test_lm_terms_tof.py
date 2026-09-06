"""The TOF frame `lm.terms` looks scatter up in.

There are two TOF frames and they differ for exactly the events recorded against
their bin's direction (`lm/events.py`). The scatter sinogram and GE's TOF weights
live in the **sinogram bin's** frame, so an event has to be mirrored into that
frame before it can index them -- `events.tof_index`, not `geom.tof_to_stir`.

Using the event's own frame here was wrong for ~9 % of a real bed's events and
completely invisible: the counts, the file sizes and every invariant in
`test_pipeline_data.py` are identical either way. Only the image differs.

Built on the miniature scanner, so this runs without SIRF, without PyTomography
and without a decoded exam.
"""

from __future__ import annotations

import numpy as np
import pytest

from lm import events as ev
from lm import geom
from lm import terms


@pytest.fixture(scope="module")
def binmap(mini_hs):
    return geom.BinMap(mini_hs)


class FakeCase:
    """The two attributes `lm.terms.read` uses, pointing at a tmp directory."""

    def __init__(self, work):
        self.name = "synthetic"
        self._work = work

    def work_bed(self, bed):
        return self._work


def write_terms(work, binmap, hs, **arrays):
    """Write each named term as a header cloned from `hs` plus a `<f4` array."""
    work.mkdir(parents=True, exist_ok=True)
    head = [ln for ln in open(hs).read().splitlines()]
    for name, a in arrays.items():
        data = work / f"{name}.s"
        np.asarray(a, "<f4").ravel().tofile(data)
        out = []
        for ln in head:
            low = ln.lower()
            if "name of data file" in low:
                ln = f"name of data file := {data.name}"
            elif "number format" in low:
                ln = "!number format := float"
            elif "number of bytes per pixel" in low:
                ln = "!number of bytes per pixel := 4"
            out.append(ln)
        (work / f"{name}.hs").write_text("\n".join(out) + "\n")


def swap_split(binmap, n_xtal):
    """Two crystal pairs on one bin: one recorded with the bin's direction, one against.

    `BinMap.flat(..., with_swap=True)` reports which; the test needs one of each
    or it cannot tell the two frames apart.
    """
    a = np.arange(n_xtal, dtype=np.uint16)
    b = (a + n_xtal // 2) % n_xtal
    flat, swap = binmap.flat(a, b, with_swap=True)
    ok = flat >= 0
    yes = np.flatnonzero(ok & swap)
    no = np.flatnonzero(ok & ~swap)
    if not yes.size or not no.size:
        pytest.skip("the miniature scanner gives no mixed-direction pairs here")
    return int(yes[0]), int(no[0])


def test_the_two_frames_disagree_for_events_recorded_against_the_bin(binmap):
    """The premise. Without this the rest of the file proves nothing."""
    n_tof, n_xtal = 11, binmap.vt.shape[0]
    i, j = swap_split(binmap, n_xtal)
    a = np.arange(n_xtal, dtype=np.uint16)
    b = (a + n_xtal // 2) % n_xtal
    e = np.zeros(n_xtal, dtype=[("xtal_a", "<u2"), ("xtal_b", "<u2"),
                                ("tof_bin", "i1"), ("t_ms", "<u4")])
    e["xtal_a"], e["xtal_b"] = a, b
    e["tof_bin"] = 7                       # any bin off the centre

    own = geom.tof_to_stir(np.asarray(e["tof_bin"]), n_tof)
    sino = ev.tof_index(e, binmap, n_tof)
    assert sino[i] == own[i], "a with-direction event is not mirrored"
    assert sino[j] == (n_tof - 1) - own[j], "an against-direction event is mirrored"
    assert sino[i] != sino[j]


def test_event_terms_reads_scatter_in_the_sinogram_frame(binmap, mini_hs, tmp_path):
    """A delta TOF profile must land on the events whose SINOGRAM index matches.

    The profile is all of the scatter in one TOF bin and none in any other, so
    an event picks up scatter if and only if its index into that profile is the
    delta's. Indexed in the wrong frame, the against-direction events pick up
    scatter at the mirrored bin instead -- which is the bug this pins.
    """
    n_tof, n_xtal = 11, binmap.vt.shape[0]
    work = tmp_path / "bed1"
    write_terms(work, binmap, mini_hs,
                normdt=np.ones(binmap.n_bin),
                attn=np.ones(binmap.n_bin),
                randoms=np.zeros(binmap.n_bin),      # isolate scatter
                scatter=np.ones(binmap.n_bin))
    case = FakeCase(work)

    a = np.arange(n_xtal, dtype=np.uint16)
    b = (a + n_xtal // 2) % n_xtal
    e = np.zeros(n_xtal, dtype=[("xtal_a", "<u2"), ("xtal_b", "<u2"),
                                ("tof_bin", "i1"), ("t_ms", "<u4")])
    e["xtal_a"], e["xtal_b"] = a, b
    e["tof_bin"] = 7

    sino_t = ev.tof_index(e, binmap, n_tof)
    delta_bin = int(sino_t[swap_split(binmap, n_xtal)[0]])
    profile = np.zeros(n_tof, np.float32)
    profile[delta_bin] = 1.0

    keep, w, add = terms.event_terms(case, 1, e, binmap, n_tof,
                                     tof_scatter=profile)
    got_scatter = add > 0                       # randoms are zero, so add == scatter

    want = sino_t[keep] == delta_bin
    assert np.array_equal(got_scatter, want), (
        "scatter landed on the events whose EVENT-frame TOF index matches the "
        "delta, not the sinogram-frame one -- lm/terms.py is indexing the "
        "vendor weights in the wrong frame")

    # And prove the wrong frame really would have given a different answer here,
    # so this test cannot quietly stop discriminating.
    own_t = geom.tof_to_stir(np.asarray(e["tof_bin"]), n_tof)[keep]
    assert not np.array_equal(own_t == delta_bin, want)
