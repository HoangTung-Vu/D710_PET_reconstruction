"""Check the TOF axis from the DATA ITSELF, without calling any vendor function.

    python3 tools/tof_profile.py view1.npy [view2.npy ...] [--save prof.npy]

The input is the output of `gerdf decode <SINO> --view N -o v.npy` **without**
`--collapse-tof`, i.e. `(radial, tof, plane)` uint8.  A few views are enough;
one view of bed 1 has ~37 thousand counts, five views ~240 thousand.

## The three questions it answers, and why it can

All three rest on one geometric fact: **the further a LOR is from the centre,
the less of the patient it passes through**.  So slicing the sinogram by radius
is enough to separate three components without any model:

* large `|u|` (beyond ~280 mm) — the LOR misses the body entirely.  The counts
  there are almost purely **randoms**.  Randoms are accidental coincidences with
  no time correlation, so their TOF profile **must be flat**.  Measuring a CoV
  right at the Poisson floor confirms `randoms / n_tof`.  Measuring a clearly
  higher CoV refutes it.
* the tail ring in `|u|` (~140-210 mm) — outside the body but still reached by
  scatter.  Subtract the flat randoms background measured above and what remains
  **is the scatter TOF profile**, measured directly.  This is exactly the region
  GE itself uses to scale scatter (`CalcSinoTails`,
  `SCAT_TAILFIT_ANGLE_WINDOW`).
* the whole view — the total profile must be **one continuous hump**.  If it
  turns into a comb or into two peaks, the TOF axis is interleaved (STIR's own
  order is interleaved), and then `gerdf --tof-mash` mashing adjacent bins is
  mashing together time points that are not adjacent.

None of the three needs `GetScatterViewDataTof`.  They are measurements, and
they measure what that function is supposed to produce -- so they are also the
yardstick to GRADE it against once it can be called.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

BIN_MM = 2.1306          # Default bin size, matches interfile.DEFAULT_BIN_SIZE_CM
BIN_PS = 89.2459         # coincTimingPrecision of this scanner

#: Beyond this radius, treat the counts as randoms only. 280 mm is not a round
#: number picked for looks: over 200-280 mm the measurement below still sees a
#: CoV twice the Poisson floor, i.e. scatter still reaches that far.
R_ONLY_MM = 280.0

#: The ring the scatter profile is taken from: outside this child's body, but not
#: yet out in the randoms-only region.
TAIL_LO_MM, TAIL_HI_MM = 140.0, 210.0


def load(paths: list[str]) -> np.ndarray:
    """Sum several views. int64 because uint8 overflows on the second view already."""
    a = None
    for p in paths:
        v = np.load(p)
        if v.ndim != 3:
            raise SystemExit(
                f"error: {p} has {v.ndim} dimensions, need 3 = (radial, tof, plane).\n"
                "  Produce it with: gerdf decode <SINO> --view N -o v.npy\n"
                "  (WITHOUT --collapse-tof, otherwise the TOF axis is already summed away)")
        a = v.astype(np.int64) if a is None else a + v
    return a


def radial_mm(nr: int) -> np.ndarray:
    return (np.arange(nr) - (nr - 1) / 2) * BIN_MM


def report(A: np.ndarray, out=print) -> dict:
    nr, nt, _ = A.shape
    u = np.abs(radial_mm(nr))
    prof = lambda m: A[m].sum(axis=(0, 2)).astype(float)   # noqa: E731
    res: dict = {}

    out(f"{A.sum():,} counts, {nt} TOF bins, bin width {BIN_PS:g} ps\n")

    # -- 1. is the TOF axis monotonic in time ------------------------------
    p = prof(u < 200)
    sm = np.convolve(p, np.ones(9) / 9, mode="valid")
    turns = int((np.diff(np.sign(np.diff(sm))) != 0).sum())
    half = np.where(p >= p.max() / 2)[0]
    contiguous = bool(np.all(np.diff(half) == 1))
    res["monotonic"] = turns <= 1 and contiguous
    out("1. TOF AXIS")
    out(f"   peak bin {p.argmax()}/{nt}   max/min {p.max() / max(p.min(), 1):.2f}   "
        f"turning points (smoothed) {turns}")
    out(f"   half maximum: bins {half[0]}..{half[-1]} = "
        f"{(half[-1] - half[0] + 1) * BIN_PS:.0f} ps, contiguous {contiguous}")
    out(f"   -> {'MONOTONIC: mashing adjacent bins is correct' if res['monotonic'] else 'NOT single-peaked -- the axis may be INTERLEAVED, DO NOT use --tof-mash'}\n")

    # -- 2. are the randoms flat -------------------------------------------
    out("2. RANDOMS (LORs missing the body)")
    res["randoms_flat"] = None
    for lo, hi in [(200.0, R_ONLY_MM), (R_ONLY_MM, 1e9)]:
        m = (u >= lo) & (u < hi)
        if m.sum() == 0:
            continue
        q = prof(m)
        cov, poisson = q.std() / q.mean(), 1 / np.sqrt(q.mean())
        flat = cov < 1.5 * poisson
        out(f"   |u| {lo:.0f}-{min(hi, u.max()):.0f} mm: {int(q.sum()):>8,} counts   "
            f"CoV {cov:.4f}   Poisson floor {poisson:.4f}   "
            f"-> {'FLAT' if flat else 'not flat yet (scatter still present)'}")
        if hi > 1e8:
            res["randoms_flat"] = flat
            res["randoms_cov"], res["randoms_poisson"] = float(cov), float(poisson)
    out(f"   -> randoms/n_tof {'CONFIRMED by measurement' if res['randoms_flat'] else 'NOT confirmed'}\n")

    # -- 3. the scatter TOF profile ----------------------------------------
    m_tail = (u >= TAIL_LO_MM) & (u < TAIL_HI_MM)
    m_far = u >= R_ONLY_MM
    q_tail, q_far = prof(m_tail), prof(m_far)
    # Randoms are flat, so their level in the tail ring scales with the NUMBER of LORs.
    r_level = q_far.mean() / m_far.sum() * m_tail.sum()
    scat = np.clip(q_tail - r_level, 0, None)
    res["scatter_profile"] = scat / scat.sum() if scat.sum() else scat
    out("3. SCATTER (tail ring minus the randoms background)")
    out(f"   ring {TAIL_LO_MM:.0f}-{TAIL_HI_MM:.0f} mm: {int(q_tail.sum()):,} counts, "
        f"randoms background {r_level * nt:,.0f} ({100 * r_level * nt / q_tail.sum():.0f}%)")
    out("   " + " ".join(f"{v:4.2f}" for v in (scat / scat.max() if scat.max() else scat)))
    out(f"   peak bin {int(scat.argmax())}   max/mean {scat.max() / scat.mean():.2f}   "
        f"CoV {scat.std() / scat.mean():.3f}")

    # -- 4. the cost of spreading it flat ----------------------------------
    b_flat = r_level + scat.sum() / nt
    neg = int((q_tail < b_flat).sum())
    res["flat_negative_bins"] = neg
    res["flat_peak_error"] = float(scat.max() / scat.mean())
    out(f"\n4. IF THE SCATTER WERE SPREAD FLAT")
    out(f"   off by {scat.max() / scat.mean():.1f}x at the peak, and it puts scatter into bins that have none")
    out(f"   {neg}/{nt} bins ({100 * neg / nt:.0f}%) have prompts < background "
        f"-> NEGATIVE true rate")
    out(f"   -> {'ACCEPTABLE' if neg == 0 else 'NOT acceptable; see TOF_PLAN.md §3'}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views", nargs="+", help=".npy from `gerdf decode --view N`")
    ap.add_argument("--save", help="write the scatter TOF profile (normalised to sum = 1)")
    args = ap.parse_args(argv)

    res = report(load(args.views))
    if args.save:
        np.save(args.save, res["scatter_profile"])
        print(f"\nwrote {args.save}  ({res['scatter_profile'].size} bins, sum = 1)")
    return 0 if res["monotonic"] and res["randoms_flat"] else 1


if __name__ == "__main__":
    sys.exit(main())
