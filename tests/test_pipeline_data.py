"""The invariants that only hold on real data, one test per bed.

Everything here skips when `$D710_OUT/<exam>/decoded/` and
`$D710_OUT/<exam>/work/bed<n>/` have not been built -- they are patient-derived
and live outside the source tree entirely.  Build them with
`d710 exam --raw <...> --ct <...> --case <exam>`.

These are the checks `osem_pipeline.ipynb` prints in its invariant cell, moved
somewhere that fails instead of printing.  Comparisons are made **per plane**,
never per bin: the raw sinogram runs at ~0.06 count/bin, so `p < r` holds in
about 82 % of bins on Poisson noise alone and a per-bin assertion means
nothing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import interfile
from cases import VENDOR_TERMS, bed_params, decoded_beds

CLONED_KEYS = ("name of data file", "number format", "number of bytes per pixel")

#: Keys that describe the DATA's timing axis, which the correction terms do not
#: have. `to_stir.strip_tof` removes exactly these; the scanner block's own TOF
#: keys stay, because those describe the hardware.
TOF_DATA_KEYS = ("matrix axis label [5]", "matrix size [5]", "tof mashing factor")

_cache: dict[str, np.ndarray] = {}


def planes(hs):
    """Per-plane sums, computed once per file for the whole session."""
    if hs not in _cache:
        _cache[hs] = interfile.per_plane(hs)
    return _cache[hs]


def term(bed, name):
    return os.path.join(bed["terms"], name + ".hs")


@pytest.fixture(params=bed_params())
def bed(request):
    return request.param


# ------------------------------------------------------------- the decode

def test_prompt_sum_equals_the_rdf_header(bed):
    """The count identity, end to end: nothing is lost in the transpose."""
    total = planes(bed["hs"]).sum()
    assert round(total) == bed["hdr"]["prompts"]


def test_the_bed_used_the_norm_its_own_header_declares(bed):
    """Never the vendor selftest fallback, and never another exam's norm.

    `estimate.py` falls back to GE's selftest normalisation when it cannot
    resolve the one the exam declares.  That warns on stderr and then produces
    a perfectly ordinary-looking `normdt.f32` describing **GE's test scanner**,
    which nothing downstream can detect.  The sidecar records what was actually
    loaded, so the check is cheap.
    """
    with open(os.path.join(bed["terms"], "to_stir.json")) as f:
        est = json.load(f).get("estimate", {})
    assert est, "to_stir.json has no estimate sidecar"
    assert est["norm_source"] == "resolved from norm_cal_uid", est["norm_source"]
    assert "selftest" not in est["norm"], est["norm"]
    # The console path the exam's cal record names, reached inside its own drop.
    assert est["norm"].endswith("/SINO0001"), est["norm"]


def test_every_bed_of_an_exam_used_the_same_norm():
    """One acquisition, one normalisation -- a per-bed difference is a bug."""
    beds = decoded_beds()
    if not beds:
        pytest.skip("no decoded bed with vendor terms")
    by_case = {}
    for b in beds:
        with open(os.path.join(b["terms"], "to_stir.json")) as f:
            by_case.setdefault(b["case"], set()).add(
                json.load(f).get("estimate", {}).get("norm"))
    for case, norms in by_case.items():
        assert len(norms) == 1, f"{case} used {len(norms)} different norms: {norms}"


def test_the_bin_mapping_was_proved_on_this_bed(bed):
    with open(os.path.join(bed["terms"], "to_stir.json")) as f:
        meta = json.load(f)
    assert meta["verified"]["bit_exact_vs_decoded"] is True
    assert meta["verified"]["prompts"] == bed["hdr"]["prompts"]
    assert meta["wcc_applied"] is False


# ------------------------------------------------------- one geometry only

def test_every_term_shares_the_prompts_geometry(bed):
    """Same LOR geometry. The timing axis is deliberately not shared -- see
    `test_terms_are_non_tof_even_when_the_prompts_are_not`."""
    want = interfile.shape(bed["hs"])[1:]
    for name in VENDOR_TERMS:
        assert interfile.shape(term(bed, name))[1:] == want, name


def test_term_headers_are_clones_of_the_prompt_header(bed):
    """Same ExamInfo by construction -- the fix that retired `same_bins()`.

    A term whose header was generated rather than cloned drifts on the energy
    window, and STIR throws `BinNormalisation set-up with different ExamInfo`
    only much later, inside `make_Poisson_loglikelihood`.

    The one licensed difference is the timing axis. Every correction term is
    non-TOF whatever the prompts are -- norm, dead time, attenuation and randoms
    do not depend on arrival time, and the scatter's time axis travels
    separately in `scatter_tof.npy` -- so `to_stir.strip_tof` takes the axis-5
    keys back out. The scanner block keeps its TOF description either way, which
    is what the ExamInfo is actually built from.
    """
    src = interfile.keys(bed["hs"])
    for name in VENDOR_TERMS:
        got = interfile.keys(term(bed, name))
        assert set(src) - set(got) <= set(TOF_DATA_KEYS), name
        assert set(got) - set(src) == set(), name
        differing = {k for k in set(src) & set(got) if src[k] != got[k]}
        assert differing <= set(CLONED_KEYS) | {"number of dimensions"}, \
            f"{name}: also changed {differing}"


def test_terms_are_non_tof_even_when_the_prompts_are_not(bed):
    """The other half of the contract above, stated as its own fact."""
    for name in VENDOR_TERMS:
        assert interfile.shape(term(bed, name))[0] == 1, \
            f"{name} has a timing axis; every correction term must be per LOR"


def test_a_tof_estimate_leaves_the_scatter_time_axis_beside_the_terms(bed):
    """`scatter_tof.npy` exists exactly when the estimate ran with reconMethod 3.

    Not "when the prompts are TOF": the two are independent settings, and a TOF
    decode paired with a non-TOF estimate is a real configuration -- OSEM then
    falls back to a measured profile and says so. What must not happen is
    `to_stir.json` claiming a TOF estimate with no weights on disk.
    """
    meta = json.load(open(os.path.join(bed["terms"], "to_stir.json")))
    claimed = bool(meta.get("estimate", {}).get("tof_scatter"))
    present = os.path.exists(os.path.join(bed["terms"], "scatter_tof.npy"))
    assert present == claimed, (
        f"to_stir.json says tof_scatter={claimed} but scatter_tof.npy "
        f"{'exists' if present else 'is missing'}")


def test_the_terms_are_float32_and_the_prompts_are_not(bed):
    assert interfile.dtype(bed["hs"]) == "<i2"
    for name in VENDOR_TERMS:
        assert interfile.dtype(term(bed, name)) == "<f4", name


# ----------------------------------------------------- the count-domain terms

def test_no_plane_has_more_randoms_than_prompts(bed):
    p, r = planes(bed["hs"]), planes(term(bed, "randoms"))
    bad = np.flatnonzero(p < r)
    assert bad.size == 0, f"{bad.size}/{p.size} planes with Sum(p) < Sum(r)"


#: The outermost planes of a segment, where the scatter estimate may overshoot.
EDGE_PLANES = 4


def test_no_interior_plane_has_a_negative_true_rate(bed):
    """`Sum(s) <= Sum(p - r)` everywhere the acquisition is well sampled.

    A negative true rate is physically impossible, so this is the amplitude
    check on scatter that no picture of the sinogram can give.  The old
    self-coded estimator broke it on 4.52 % of planes; the vendor kernel breaks
    it on none, once the segment-edge planes are excluded -- see the next test
    for those.
    """
    p = planes(bed["hs"])
    r = planes(term(bed, "randoms"))
    s = planes(term(bed, "scatter"))
    interior = interfile.edge_distance(bed["hs"]) >= EDGE_PLANES
    bad = np.flatnonzero((s > p - r) & interior)
    assert bad.size == 0, (
        f"{bad.size}/{interior.sum()} interior planes with a negative true "
        f"rate, worst s/(p-r) = {(s[bad] / (p - r)[bad]).max():.4f}")


def test_edge_plane_scatter_overshoot_stays_negligible(bed):
    """The outermost planes of a segment may overshoot, but only just.

    Measured 2026-08-24: all six pediatric beds are clean everywhere, and NEMA
    bed 2 overshoots on 11 of 553 planes -- every one of them among the
    outermost four planes of a segment, by at most 5.5 % of that plane's own
    scatter and 0.04 % of the bed's.  Those planes gather the fewest ring pairs
    in the acquisition, so the SSS tail fit has the least to work with there.

    The bound is what makes this a check rather than an excuse: a scatter scale
    that is genuinely wrong overshoots across whole segments, not in a
    four-plane fringe.
    """
    p = planes(bed["hs"])
    r = planes(term(bed, "randoms"))
    s = planes(term(bed, "scatter"))
    excess = np.clip(s - (p - r), 0.0, None)
    assert excess.sum() / s.sum() < 1e-3, (
        f"scatter exceeds the true rate by {100 * excess.sum() / s.sum():.3f} % "
        "of the bed's scatter -- that is a scale error, not an edge effect")
    over = np.flatnonzero(excess > 0)
    if over.size:
        assert interfile.edge_distance(bed["hs"])[over].max() < EDGE_PLANES, \
            "an interior plane overshoots; the fringe explanation does not hold"


def test_background_is_randoms_plus_scatter(bed):
    r = planes(term(bed, "randoms"))
    s = planes(term(bed, "scatter"))
    b = planes(term(bed, "background"))
    assert b == pytest.approx(r + s, rel=1e-5)


def test_randoms_agree_with_the_delayed_channel(bed):
    """GE's randoms kernel vs the scanner's own delay counter -- two paths."""
    ratio = planes(term(bed, "randoms")).sum() / bed["hdr"]["delays"]
    assert 0.97 < ratio < 1.02, f"Sum(R)/delays = {ratio:.5f}"


def test_the_scatter_fraction_is_physical(bed):
    p = planes(bed["hs"]).sum()
    r = planes(term(bed, "randoms")).sum()
    s = planes(term(bed, "scatter")).sum()
    sf = s / (p - r)
    assert 0.15 < sf < 0.45, f"S/(T+S) = {sf:.4f}"


# ------------------------------------------------------ the sensitivity term

def test_dead_time_is_a_livetime_fraction(bed):
    """`normdt / norm_only` < 1, and it is not a constant.

    This is the direction check for the whole sensitivity term: a *correction*
    factor would be > 1 and would rise with the count rate; a sensitivity has
    to fall.  Divide the data by it, never multiply.
    """
    nd = interfile.load(term(bed, "normdt"))[0]
    no = interfile.load(term(bed, "norm_only"))[0]
    mid = nd.shape[0] // 4                       # a well-populated direct plane
    a, b = np.asarray(nd[mid], np.float64), np.asarray(no[mid], np.float64)
    live = a[b > 0] / b[b > 0]
    assert live.size
    assert 0.80 < np.median(live) < 1.0, f"livetime {np.median(live):.4f}"


def test_normalisation_is_positive_where_the_prompts_are(bed):
    nd = interfile.load(term(bed, "normdt"))[0]
    p = interfile.load(bed["hs"])[0]
    mid = nd.shape[0] // 4
    hit = np.asarray(p[mid]) > 0
    assert hit.any()
    assert (np.asarray(nd[mid])[hit] > 0).all(), "a live bin with zero sensitivity"


# ------------------------------------------------------------------ span 2

def test_span_2_doubles_the_odd_planes_of_every_term(bed):
    """Including `normdt` -- which is why the multiplicity must not be re-applied.

    Segment 0 covers ring difference -1..+1, so its odd axial positions gather
    two ring pairs.  Prompts, randoms, scatter **and the sensitivity** all carry
    that factor, so `y = S(Gx) + b` balances with a projector that fires one
    LOR per bin.  Multiplying `geometry.ring_pair_multiplicity()` in on top
    would square it.
    """
    n0 = interfile.axial_sizes(bed["hs"])[0]
    for name in (None,) + VENDOR_TERMS:
        hs = bed["hs"] if name is None else term(bed, name)
        v = planes(hs)[:n0]
        ratio = v[1::2].mean() / v[0::2].mean()
        assert 1.90 < ratio < 2.10, \
            f"{name or 'prompts'}: odd/even = {ratio:.3f}, expected ~2"


def test_oblique_segments_carry_one_ring_pair(bed):
    """Only segment 0 is doubled: the ripple must stop at plane 47."""
    sizes = interfile.axial_sizes(bed["hs"])
    p = planes(bed["hs"])
    seg1 = p[sizes[0]:sizes[0] + sizes[1]]
    ratio = seg1[1::2].mean() / seg1[0::2].mean()
    assert 0.85 < ratio < 1.15, f"segment +1 odd/even = {ratio:.3f}, expected ~1"


# ------------------------------------------------------------- attenuation

def test_cached_attenuation_factors_are_survival_probabilities(bed, sirf):
    """`attn.hs` is written by the notebook; skip the beds it has not reached.

    It goes through SIRF rather than the memmap because SIRF wrote it, in its
    own segment-major layout -- see `interfile._check_layout`.
    """
    hs = term(bed, "attn")
    if not os.path.exists(hs):
        pytest.skip("attn.hs not cached for this bed")
    a = sirf.AcquisitionData(hs).as_array()
    # Non-TOF like every other correction term: attenuation is the survival
    # probability of the photon PAIR along the LOR and does not depend on when
    # either photon arrived. STIR enforces it too -- see `utils/attn.py`.
    assert a.shape == (1,) + interfile.shape(bed["hs"])[1:]
    v = a[0, a.shape[1] // 4].astype(np.float64)
    assert v.min() > 0.0
    assert v.max() <= 1.0 + 1e-5
    assert np.median(v) < 1.0                    # something actually attenuates


def test_the_ct_belongs_to_the_same_exam(bed):
    """`FrameOfReferenceUID == sop_instance_uid` is an identity, not a guess.

    `11082026/` holds images from two different exams, so a directory sitting
    next to the raw one proves nothing.
    """
    from utils import attenuation

    ct = os.environ.get("D710_CT")
    if not ct:
        pytest.skip("set D710_CT to the CT series directory to check the pairing")
    assert (attenuation.load(ct).meta["frame_of_reference_uid"]
            == bed["hdr"]["sop_instance_uid"])
