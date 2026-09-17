"""Discovery of the decoded exams present on disk."""

from __future__ import annotations

import os

import pytest

from utils.paths import NoOutputRoot, cases

VENDOR_TERMS = ("randoms", "scatter", "background", "normdt", "norm_only")


def decoded_beds() -> list[dict]:
    """Every bed with a decoded prompt and a full set of vendor terms."""
    want = os.environ.get("D710_CASE")
    try:
        found = cases()
    except NoOutputRoot:
        return []

    out = []
    for c in found:
        if want and c.name != want:
            continue
        for n in c.beds(terms=VENDOR_TERMS):
            out.append({"case": c.name, "bed": n,
                        "hs": str(c.prompt(n)),
                        "terms": str(c.work_bed(n)),
                        "vendor": str(c.vendor_bed(n)),
                        "hdr": c.header(n)})
    return out


def bed_params():
    """`decoded_beds()` as pytest parameters, with a visible skip when empty."""
    beds = decoded_beds()
    if not beds:
        return [pytest.param(None, marks=pytest.mark.skip(
            reason="no decoded bed with vendor terms under $D710_OUT; "
                   "run `d710 exam` first"),
            id="no-data")]
    return [pytest.param(b, id=f"{b['case']}-bed{b['bed']}") for b in beds]
