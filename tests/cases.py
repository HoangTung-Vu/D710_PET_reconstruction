"""Discovery of the decoded exams present on disk."""

from __future__ import annotations

import os

import pytest

from utils.paths import NoOutputRoot, cases

VENDOR_TERMS = ("randoms", "scatter", "background", "normdt", "norm_only")


def _beds(terms) -> list[dict]:
    want = os.environ.get("D710_CASE")
    try:
        found = cases()
    except NoOutputRoot:
        return []

    out = []
    for c in found:
        if want and c.name != want:
            continue
        for n in (c.beds(terms=terms) if terms else c.decoded_beds()):
            out.append({"case": c.name, "bed": n,
                        "hs": str(c.prompt(n)),
                        "terms": str(c.work_bed(n)),
                        "vendor": str(c.vendor_bed(n)),
                        "hdr": c.header(n)})
    return out


def decoded_beds() -> list[dict]:
    """Every bed with a decoded prompt and a full set of vendor terms."""
    return _beds(VENDOR_TERMS)


def beds_without_vendor_terms() -> list[dict]:
    """Every decoded bed, whether or not `d710 estimate` has run on it.

    The bin-map checks need the prompts and the events and nothing else, and
    they are the only place the ring pairing is pinned against GE's own data.
    Making them wait for the vendor kernel kept them skipped on every machine
    that had merely decoded a case.
    """
    return _beds(None)


def bed_params():
    """`decoded_beds()` as pytest parameters, with a visible skip when empty."""
    beds = decoded_beds()
    if not beds:
        return [pytest.param(None, marks=pytest.mark.skip(
            reason="no decoded bed with vendor terms under $D710_OUT; "
                   "run `d710 exam` first"),
            id="no-data")]
    return [pytest.param(b, id=f"{b['case']}-bed{b['bed']}") for b in beds]
