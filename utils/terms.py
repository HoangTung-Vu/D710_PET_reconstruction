"""Load the terms of one bed from Interfile, and summarise them.

The tree `y = S·(G x) + b` reads off disk as follows:

| term | file | meaning |
|---|---|---|
| `prompts` | `decoded/bed<n>.hs` | raw prompts, counts |
| `randoms` | `work/bed<n>/randoms.hs` | GE kernel |
| `scatter` | `work/bed<n>/scatter.hs` | GE's SSS |
| `background` | `work/bed<n>/background.hs` | `= randoms + scatter`, i.e. `b` |
| `normdt` | `work/bed<n>/normdt.hs` | norm × dead time — a **SENSITIVITY** |
| `norm_only` | `work/bed<n>/norm_only.hs` | norm alone; `deadtime = normdt/norm_only` |
| `attn` | `work/bed<n>/attn.hs` | built from CT in `utils.attn`, cached |

`normdt` is a sensitivity, not a correction factor: **dividing** the data by it
is what corrects, and `AcquisitionSensitivityModel` multiplies it into the
forward projection. See the docstring of `vendor/to_stir.py` for the two
measurements that pin that direction down.

One bed is ~6 × 231 MB. `load()` returns both the SIRF objects and the numpy
arrays and the caller must `del` them when done — holding all six beds at once
is ~9 GB.
"""

from __future__ import annotations

import json

import numpy as np

#: Terms in the count domain — adding them up is meaningful.
COUNT_TERMS = ("prompts", "randoms", "scatter", "background", "trues")

#: Terms that are (dimensionless) factors — averaging is meaningful, summing is not.
FACTOR_TERMS = ("norm_only", "deadtime", "normdt", "attenuation", "sensitivity")

ALL_TERMS = COUNT_TERMS + FACTOR_TERMS

#: Number of direct planes (segment 0) in the 553-plane sinogram.
NSEG0 = 47

#: What `to_stir.py` writes out. `attn` is added later by `utils.attn`.
ON_DISK = ("randoms", "scatter", "background", "normdt", "norm_only")


def total(a) -> float:
    """Sum in float64.

    Summing 60.7 million float32 bins drifts by a few counts — enough to make a
    comparison of the total against the header fail for no good reason.
    """
    return float(np.asarray(a).sum(dtype=np.float64))


def load(case, n: int, af=None):
    """Every term of bed `n`, as `(SIRF objs, dict of numpy arrays)`.

    `af` (attenuation factors, a numpy array) is optional: given one, this adds
    `attenuation` and `sensitivity = normdt × attenuation`; without it those two
    keys are absent.

    **Remember to `del` both when done.**
    """
    import sirf.STIR as pet

    work = case.work_bed(n)
    objs = {"prompts": pet.AcquisitionData(str(case.prompt(n)))}
    for name in ON_DISK:
        objs[name] = pet.AcquisitionData(str(work / f"{name}.hs"))

    A = {k: v.as_array() for k, v in objs.items()}
    A["trues"] = A["prompts"] - A["background"]   # negative per-bin values are normal (noise)
    A["deadtime"] = A["normdt"] / A["norm_only"]  # live fraction, < 1
    if af is not None:
        A["attenuation"] = af
        # S = norm × dead time × attenuation. All three are SENSITIVITIES, so they
        # MULTIPLY together; `normdt` already bundles the first two factors.
        A["sensitivity"] = A["normdt"] * af
    return objs, A


def ct_dir(case, n: int) -> str:
    """The CT series that `d710 estimate` used for this bed.

    Taken from the sidecar rather than asked again: attenuation and scatter
    **must** be built from the same CT, and that is the CT GE's kernel saw.
    """
    ct = meta(case, n).get("estimate", {}).get("ct")
    if not ct:
        raise SystemExit("error: sidecar của bed %d không ghi CT nào" % n)
    return ct


def bed_table(case, beds, out=print) -> None:
    """Acquisition summary table, one row per bed.

    The two patient cases carry PHI: name / patient ID / date of birth are
    deliberately **not** printed.
    """
    out(f"ca {case.name!r}: {len(beds)} bed  ->  {beds}\n")
    out(f"{'bed':>4} {'table mm':>10} {'prompts':>13} {'delays':>13} "
        f"{'giây':>5} {'kcps':>8} {'R/P':>6}")
    for n in beds:
        h = case.header(n)
        dur = h["frame_duration_ms"] / 1000
        out(f"{n:>4} {h['table_position_mm']:>10.2f} {h['prompts']:>13,} "
            f"{h['delays']:>13,} {dur:>5.0f} {h['prompts'] / dur / 1e3:>8.1f} "
            f"{h['delays'] / h['prompts']:>6.3f}")
    h0 = case.header(beds[0])
    out(f"\n{h0['dose_mbq']} MBq, {h0['patient_weight_kg']} kg, "
        f"{h0['radiopharmaceutical']}")


def collect(case, beds, af: dict, out=print):
    """One pass over ALL beds. Returns `(proj, stats, planes)`.

    Each bed is loaded → what is needed is extracted → RAM is handed straight
    back, so peak memory is **one** bed (~2.5 GB) rather than six (~15 GB).
    Slices are kept for plotting along with a few summary numbers; the full
    arrays are dropped — the reconstruction step reloads them.
    """
    from . import plots

    proj, stats, planes = {}, {}, {}
    for n in beds:
        objs, A = load(case, n, af=af[n])
        planes[n] = plots.busiest_plane(A["prompts"])
        for t in ALL_TERMS:
            proj[n, t] = plots.slices(A[t][0], planes[n])
        stats[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                    for t in ALL_TERMS}
        del A, objs
        out(f"bed {n}: xong  (plane vẽ = {planes[n]})")

    out(f"\n{'bed':>4} " + "".join(f"{t:>13}" for t in COUNT_TERMS)
        + f"{'scat.frac':>11}{'livetime':>10}")
    for n in beds:
        s = stats[n]
        sf = s["scatter"] / (s["prompts"] - s["randoms"])
        out(f"{n:>4} " + "".join(f"{s[t]:>13,.0f}" for t in COUNT_TERMS)
            + f"{sf:>11.4f}{s['deadtime']:>10.4f}")
    return proj, stats, planes


def invariant_table(case, beds, proj: dict, stats: dict, out=print) -> list:
    """Print the four invariants for every bed, return the list of bad beds.

    Aggregate per plane before comparing — the raw sinogram runs below 1
    count/bin, so `p < r` holds at a great many bins purely from Poisson noise.
    """
    out(f"{'bed':>4} {'Σp<Σr':>8} {'Σs>Σ(p−r)':>11} {'ΣR/delays':>11} "
        f"{'S/(T+S)':>9} {'livetime':>9} {'kcps':>8} {'bit-exact':>10}")
    bad = []
    for n in beds:
        per_plane = {t: proj[n, t]["per_plane"]
                     for t in ("prompts", "randoms", "scatter")}
        v = invariants(case, n, per_plane, stats[n])
        out(f"{n:>4} {v['frac_p_lt_r']:>7.2f}% {v['frac_s_gt_t']:>10.2f}% "
            f"{v['randoms_over_delays']:>11.5f} {v['scatter_fraction']:>9.4f} "
            f"{v['livetime']:>9.5f} {v['kcps']:>8.1f} {str(v['bit_exact']):>10}")
        if v["frac_p_lt_r"] or v["frac_s_gt_t"] or not v["bit_exact"]:
            bad.append(n)

    out("\n1. Σp ≥ Σr và 2. Σs ≤ Σ(p−r): phải là 0 % ở mọi plane — "
        "true rate âm là bất khả.")
    out("3. ΣR/delays ~0,99: randoms của GE so với delay máy đếm, hai đường độc lập.")
    out("5. livetime GIẢM khi tốc độ đếm TĂNG — đó là dấu hiệu nó là độ nhạy,")
    out("   không phải hệ số hiệu chỉnh. Nó KHÔNG phải hằng số, đừng so với một số cố định.")
    out("7. WCC: chưa áp ở đâu -> ảnh là count/voxel, CHƯA phải Bq/mL.")
    out(f"\n{'MỌI BED ĐẠT' if not bad else f'BED CÓ VẤN ĐỀ: {bad}'}")
    return bad


def meta(case, n: int) -> dict:
    """The `to_stir.json` of bed `n` — including the nested `estimate.json`.

    The key that matters most: `verified.bit_exact_vs_decoded`, set by
    `to_stir.py` after it proves the bin mapping on this bed's own data.
    """
    with open(case.work_bed(n) / "to_stir.json") as f:
        return json.load(f)


def summarise(case, beds, A_by_bed: dict) -> dict:
    """One row of numbers per bed: totals for count terms, means for factors."""
    out = {}
    for n in beds:
        A = A_by_bed[n]
        out[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                  for t in ALL_TERMS if t in A}
    return out


def invariants(case, n: int, per_plane: dict, stats: dict) -> dict:
    """The four invariants that must hold on real data, aggregated **per plane**.

    Aggregated per plane, not per bin: the raw sinogram runs at ~0.06 count/bin,
    so `p < r` holds at ~82 % of bins purely from Poisson noise and a per-bin
    assertion says nothing.

    * `frac_p_lt_r`  — % of planes with Σp < Σr. Must be 0: a negative true rate
      is impossible.
    * `frac_s_gt_t`  — % of planes with Σs > Σ(p−r). Must be 0 (NEMA has an
      exception at the segment edges, see `tests/test_pipeline_data.py`).
    * `randoms_over_delays` — GE's randoms against the scanner's delayed counts,
      two independent paths; ~0.99.
    * `bit_exact` — `to_stir.py` proved the bin mapping when it wrote the file.
    """
    h = case.header(n)
    P, R, S = per_plane["prompts"], per_plane["randoms"], per_plane["scatter"]
    s = stats
    return {
        "bed": n,
        "frac_p_lt_r": 100.0 * float((P < R).mean()),
        "frac_s_gt_t": 100.0 * float((S > P - R).mean()),
        "randoms_over_delays": s["randoms"] / h["delays"],
        "scatter_fraction": s["scatter"] / (s["prompts"] - s["randoms"]),
        "livetime": s["deadtime"],
        "kcps": h["prompts"] / (h["frame_duration_ms"] / 1000.0) / 1e3,
        "bit_exact": bool(meta(case, n)["verified"]["bit_exact_vs_decoded"]),
    }
