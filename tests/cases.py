"""Finding the decoded exams on disk.

The decoded beds and the vendor terms are patient-derived and live outside the
source tree entirely, under `$D710_OUT`.  So the data-backed tests discover
what exists instead of assuming it -- and skip cleanly when `$D710_OUT` is
unset, which is what happens on any machine that has the code but not the data.

Kept out of `conftest.py` so the test modules can import it by name without
depending on how pytest happens to have loaded the conftest.
"""

from __future__ import annotations

import os

import pytest

from utils.paths import NoOutputRoot, cases

#: Terms `to_stir.py` writes.  `attn` is added later by `utils.attn`.
VENDOR_TERMS = ("randoms", "scatter", "background", "normdt", "norm_only")


def decoded_beds() -> list[dict]:
    """Every bed with a decoded prompt **and** a full set of vendor terms.

    `$D710_CASE` narrows to one exam; without it every exam under `$D710_OUT`
    is collected.
    """
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
    """`decoded_beds()` as pytest params, with a visible skip when empty."""
    beds = decoded_beds()
    if not beds:
        return [pytest.param(None, marks=pytest.mark.skip(
            reason="no decoded bed with vendor terms under $D710_OUT; "
                   "run `d710 exam` first"),
            id="no-data")]
    return [pytest.param(b, id=f"{b['case']}-bed{b['bed']}") for b in beds]
