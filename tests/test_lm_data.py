"""The list-mode path on real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from cases import decoded_beds

from lm import events as ev
from lm import geom

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
    binmap = _binmap(_binmaps, bed["hs"])
    e = ev.load(bed["npy"])
    ref = np.fromfile(bed["hs"].replace(".hs", ".s"), "<i2")
    n_tof = ref.size // binmap.n_bin
    assert ref.size == n_tof * binmap.n_bin, "the .s is not a whole number of bins"

    h, dropped = ev.histogram(e, binmap, n_tof)
    assert int(h.sum(dtype=np.int64)) == len(e) - dropped
    assert np.array_equal(h.reshape(-1), ref), (
        f"{int((h.reshape(-1) != ref).sum()):,} bins differ")


def _lm_sidecar(bed) -> dict | None:
    p = Path(bed["npy"]).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else None


@pytest.mark.parametrize("bed", LM_BEDS)
def test_event_count_matches_the_header(bed):
    e = ev.load(bed["npy"])
    want = bed["hdr"]["prompts"]
    if len(e) == want:
        return

    side = _lm_sidecar(bed)
    hint = ""
    if side and len(e) == side["stats"]["coincidences"] - side["stats"]["preroll"]:
        hint = (f"\n  short by exactly the {side['stats']['preroll']:,} pre-roll "
                f"events, and the sidecar says keep_preroll="
                f"{side['keep_preroll']}."
                f"\n  This bed's header DOES count them, so it must be decoded "
                f"again: the policy is per bed, not per exam."
                f"\n  re-run: d710 decode --raw <SINO dir> --case "
                f"{bed['case']} --bed {bed['bed']} --force")
    assert len(e) == want, (
        f"{bed['case']} bed {bed['bed']}: {len(e):,} events against "
        f"{want:,} header prompts ({len(e) - want:+,}){hint}")


@pytest.mark.parametrize("bed", LM_BEDS)
def test_the_preroll_policy_the_sidecar_records_is_the_one_it_applied(bed):
    side = _lm_sidecar(bed)
    if side is None:
        pytest.skip("event table predates the .lm.json sidecar")
    st = side["stats"]
    want = st["coincidences"] - (0 if side["keep_preroll"] else st["preroll"])
    assert side["events"] == want, (
        f"sidecar says keep_preroll={side['keep_preroll']} and "
        f"{st['coincidences']:,} coincidences with {st['preroll']:,} pre-roll, "
        f"which is {want:,} events, but it recorded {side['events']:,}")
    assert len(ev.load(bed["npy"])) == side["events"]


@pytest.mark.parametrize("bed", LM_BEDS[:LOOKUP_BEDS])
def test_per_event_terms_are_usable(bed, _binmaps):
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

    b = ev.bins(e, binmap)[keep]
    bg = (terms.read(C, bed["bed"], "randoms", binmap)
          + terms.read(C, bed["bed"], "scatter", binmap))[b]
    assert np.allclose(add * w, bg, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize("bed", LM_BEDS[:LOOKUP_BEDS])
def test_sensitivity_matches_the_sinogram_it_came_from(bed, _binmaps):
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
