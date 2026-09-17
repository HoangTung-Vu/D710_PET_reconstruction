"""Loading one bed's reconstruction terms from Interfile, and summarising them."""

from __future__ import annotations

import json

import numpy as np

from .scanner import NSEG0

COUNT_TERMS = ("prompts", "randoms", "scatter", "background", "trues")

FACTOR_TERMS = ("norm_only", "deadtime", "normdt", "attenuation", "sensitivity")

ALL_TERMS = COUNT_TERMS + FACTOR_TERMS

ON_DISK = ("randoms", "scatter", "background", "normdt", "norm_only")

TOF_AXIS_KEY = "; TOF axis"


def total(a) -> float:
    """Sum in float64."""
    return float(np.asarray(a).sum(dtype=np.float64))


RECON_TERMS = ("sensitivity",)


def load(case, n: int, af=None, tof_scatter=None, lean: bool = False):
    """Every term of bed `n`, as `(SIRF objects, dict of numpy arrays)`."""
    import sirf.STIR as pet

    work = case.work_bed(n)
    objs = {"prompts": pet.AcquisitionData(str(case.prompt(n)))}
    for name in ON_DISK:
        objs[name] = pet.AcquisitionData(str(work / f"{name}.hs"))

    A = {k: v.as_array() for k, v in objs.items()}
    n_tof = A["prompts"].shape[0]
    if n_tof > 1:
        check_tof_axis(case, n)

    if not lean:
        A["trues"] = (A["prompts"].sum(axis=0, keepdims=True) if n_tof > 1
                      else A["prompts"]) - A["background"]
        A["deadtime"] = A["normdt"] / A["norm_only"]
    if af is not None:
        A["attenuation"] = af
        A["sensitivity"] = A["normdt"] * af

    if n_tof > 1:
        expand_to_tof(case, n, objs, A, n_tof, tof_scatter)

    if lean:
        for k in [k for k in A if k not in RECON_TERMS]:
            del A[k]
    return objs, A


def check_tof_axis(case, n: int) -> None:
    """Refuse TOF prompts written before the TOF axis was corrected."""
    hs = case.prompt(n)
    if TOF_AXIS_KEY in hs.read_text():
        return
    raise SystemExit(
        "error: %s has a TOF axis but does not declare which way it runs, so it\n"
        "  was decoded before 2026-08-29 and its TOF bins are MIRRORED with\n"
        "  respect to STIR's signed timing positions.  Reconstructing it would\n"
        "  put the activity on the wrong half of every LOR -- worse than no TOF.\n"
        "  Nothing else detects this: the counts and every invariant are\n"
        "  identical either way.\n"
        "  Decode the case again (the decoder now reverses the axis):\n"
        "      d710 decode --raw <SINO dir> --case %s --tof --force\n"
        "      d710 tostir --case %s\n"
        "  or reconstruct without TOF:  d710 decode ... --no-tof" % (hs, case.name, case.name))


TOF_PROFILE_FILE = "scatter_tof_profile.npy"


def measured_tof_weights(case, n: int, n_tof: int):
    """`(w, note)` from `work/bed<n>/scatter_tof_profile.npy`, or `None`."""
    import numpy as np

    path = case.work_bed(n) / TOF_PROFILE_FILE
    if not path.exists():
        return None
    w = np.load(path).astype(np.float64).ravel()
    if w.size % n_tof:
        raise SystemExit(
            "error: %s has %d TOF bins, which does not divide into the %d bins "
            "of the prompts.\n  delete it, or rebuild the case with a --tof-mash "
            "that divides %d." % (path, w.size, n_tof, w.size))
    if w.min() < 0 or not w.sum():
        raise SystemExit(f"error: {path} must be non-negative and sum to > 0")
    n_full = w.size
    w = w.reshape(n_tof, n_full // n_tof).sum(axis=1)
    w = w / w.sum()
    return w, (f"measured on the full-count source, {n_full} bins -> {n_tof}, "
               f"peak bin {int(w.argmax())}, max/mean {w.max() * n_tof:.2f}")


def vendor_tof_weights(case, n: int, n_tof: int, n_view: int, n_tang: int):
    """GE's own TOF distribution for this bed's scatter, or `None`."""
    import numpy as np

    path = case.work_bed(n) / "scatter_tof.npy"
    if not path.exists():
        return None
    try:
        axis = (meta(case, n).get("scatter_tof") or {}).get("tof_axis")
    except OSError:
        axis = "stir"
    if axis != "stir":
        raise SystemExit(
            "error: %s was written before 2026-08-29 and its TOF axis runs GE's\n"
            "  way round, mirrored against the prompts.  Re-run:\n"
            "      d710 tostir --case %s --bed %d\n"
            "  (cheap -- it only re-reads the vendor .f32 that is already there)"
            % (path, case.name, n))
    w = np.load(path).astype(np.float64)
    n_full, got_views, ds_nu = w.shape
    if got_views != n_view:
        raise SystemExit("error: %s has %d views, the sinogram has %d"
                         % (path, got_views, n_view))
    if n_full % n_tof:
        raise SystemExit(
            "error: %s has %d TOF bins, which does not divide into the %d bins "
            "of the prompts.\n"
            "  The estimate and the decode disagree about TOF mashing; re-run\n"
            "  `d710 decode` with a --tof-mash that divides %d."
            % (path, n_full, n_tof, n_full))
    mash = n_full // n_tof
    w = w.reshape(n_tof, mash, n_view, ds_nu).sum(axis=1)

    w = w @ _upsample_matrix(ds_nu, n_tang)

    tot = w.sum(axis=0)
    dead = tot <= 0
    w = np.where(dead[None], 0.0, w / np.where(dead, 1.0, tot)[None])

    live = ~dead
    if not live.any():
        raise SystemExit(
            "error: %s is empty -- every LOR has zero scatter in every TOF "
            "bin.\n  That means the estimate ran without reconMethod = 3; "
            "re-run `d710 estimate` for this bed." % path)

    prof = w.sum(axis=(1, 2))
    prof = prof / prof.sum()
    mu = (w * np.arange(n_tof)[:, None, None]).sum(axis=0)
    note = ("GE's own (reconMethod 3), mash %d, peak bin %d, max/mean %.2f, "
            "centroid %.1f-%.1f over (view, u), %.1f%% of LORs covered"
            % (mash, int(prof.argmax()), prof.max() * n_tof,
               mu[live].min(), mu[live].max(), 100.0 * live.mean()))
    return w.astype(np.float32), note


def _upsample_matrix(n_src: int, n_dst: int):
    import numpy as np

    src = (np.arange(n_dst) + 0.5) * n_src / n_dst - 0.5
    j0 = np.clip(np.floor(src).astype(int), 0, n_src - 1)
    j1 = np.clip(j0 + 1, 0, n_src - 1)
    frac = np.clip(src - j0, 0.0, 1.0)
    M = np.zeros((n_src, n_dst))
    np.add.at(M, (j0, np.arange(n_dst)), 1.0 - frac)
    np.add.at(M, (j1, np.arange(n_dst)), frac)
    return M


def scatter_tof_profile(A, n_tof: int):
    """Measure the scatter's TOF profile from this bed."""
    import numpy as np

    P = A["prompts"].sum(axis=(1, 2)).astype(np.float64)
    R = A["randoms"][0].sum(axis=(0, 1)).astype(np.float64)
    S = A["scatter"][0].sum(axis=(0, 1)).astype(np.float64)
    T = np.clip(P.sum(axis=0) - R - S, 0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(S > 0, T / S, np.inf)
    for thresh in (0.2, 0.35, 0.5, 0.8):
        tail = (ts < thresh) & (S > S.max() * 1e-3)
        if tail.sum() >= 8 and P[:, tail].sum() >= 5000:
            break
    else:
        raise SystemExit(
            "error: no tail ring with enough counts to measure a scatter TOF "
            "profile.\n"
            "  measure it elsewhere and pass it in:\n"
            "    tools/tof_profile.py v*.npy --save prof.npy\n"
            "    d710 osem --case <n> --tof-scatter prof.npy\n"
            "  or re-estimate this bed so GE supplies its own:\n"
            "    d710 estimate --raw ... --ct ... --case <n> --bed <n>")

    prof = P[:, tail].sum(axis=1) - R[tail].sum() / n_tof
    prof = np.clip(prof, 0, None)
    if not prof.sum():
        raise SystemExit("error: the tail ring holds nothing but randoms; "
                         "no profile can be measured from it")
    w = prof / prof.sum()
    note = (f"measured in place, {int(tail.sum())} u bins (T/S<{thresh:g}), "
            f"{int(P[:, tail].sum()):,} counts, peak bin {int(w.argmax())}, "
            f"max/mean {w.max() * n_tof:.2f}")
    return w, note


def expand_to_tof(case, n: int, objs, A, n_tof: int, tof_scatter=None) -> None:
    """Bring `S` and `b` onto the grid of the TOF prompts."""
    import numpy as np

    missing = [k for k in ("prompts", "randoms", "scatter", "background")
               if k not in A]
    if missing:
        raise SystemExit(
            "error: expanding to TOF needs %s in A, separately -- randoms are "
            "flat in TOF and scatter is not, so a pre-summed `background` has "
            "already destroyed the distinction this function exists to make."
            % ", ".join(missing))

    if "sensitivity" in A:
        A["sensitivity"] = np.repeat(A["sensitivity"], n_tof, axis=0)

    if tof_scatter is not None:
        w = np.asarray(tof_scatter, dtype=np.float64).ravel()
        if w.min() < 0 or not w.sum():
            raise SystemExit("error: a scatter profile must be non-negative "
                             "and sum to more than zero")
        if w.size != n_tof:
            if w.size % n_tof:
                raise SystemExit(
                    "error: the supplied scatter profile has %d bins, which "
                    "does not divide into the %d bins of the prompts"
                    % (w.size, n_tof))
            w = w.reshape(n_tof, w.size // n_tof).sum(axis=1)
        w = w / w.sum()
        note = "supplied profile"
    else:
        _, n_view, n_tang = A["scatter"].shape[1:]
        got = (vendor_tof_weights(case, n, n_tof, n_view, n_tang)
               or measured_tof_weights(case, n, n_tof))
        w, note = got if got else scatter_tof_profile(A, n_tof)
    print(f"  TOF: {n_tof} bins -- S repeated (exact), randoms/{n_tof} "
          f"(confirmed by measurement), scatter: {note}")

    shape = (n_tof,) + A["background"].shape[1:]
    bg = np.empty(shape, dtype=np.float32)
    rnd = A["randoms"][0] / n_tof
    sct = A["scatter"][0]
    for t in range(n_tof):
        bg[t] = rnd + sct * w[t]

    obj = objs["prompts"].get_uniform_copy(0)
    obj.fill(bg)
    objs["background"] = obj
    A["background"] = bg


def ct_dir(case, n: int) -> str:
    """The CT series `d710 estimate` used for this bed."""
    ct = meta(case, n).get("estimate", {}).get("ct")
    if not ct:
        raise SystemExit("error: the sidecar of bed %d records no CT" % n)
    return ct


def bed_table(case, beds, out=print) -> None:
    """Acquisition summary table, one row per bed."""
    out(f"case {case.name!r}: {len(beds)} beds  ->  {beds}\n")
    out(f"{'bed':>4} {'table mm':>10} {'prompts':>13} {'delays':>13} "
        f"{'sec':>5} {'kcps':>8} {'R/P':>6}")
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
    """One pass over every bed."""
    from . import plots

    proj, stats, planes = {}, {}, {}
    for n in beds:
        objs, A = load(case, n, af=af[n])

        def lor(t, _A=A):
            a = _A[t]
            return a.sum(axis=0) if a.shape[0] > 1 else a[0]

        planes[n] = plots.busiest_plane(lor("prompts")[None])
        for t in ALL_TERMS:
            proj[n, t] = plots.slices(lor(t), planes[n])
        stats[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                    for t in ALL_TERMS}
        del A, objs
        out(f"bed {n}: done  (plotted plane = {planes[n]})")

    out(f"\n{'bed':>4} " + "".join(f"{t:>13}" for t in COUNT_TERMS)
        + f"{'scat.frac':>11}{'livetime':>10}")
    for n in beds:
        s = stats[n]
        sf = s["scatter"] / (s["prompts"] - s["randoms"])
        out(f"{n:>4} " + "".join(f"{s[t]:>13,.0f}" for t in COUNT_TERMS)
            + f"{sf:>11.4f}{s['deadtime']:>10.4f}")
    return proj, stats, planes


def invariant_table(case, beds, proj: dict, stats: dict, out=print) -> list:
    """Print the four invariants for every bed and return the list of bad beds."""
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

    out("\n1. Σp ≥ Σr and 2. Σs ≤ Σ(p−r): must be 0 % on every plane — "
        "a negative true rate is impossible.")
    out("3. ΣR/delays ~0.99: GE's randoms against the delays the scanner counted, two independent routes.")
    out("5. livetime DROPS as the count rate RISES — that is the sign it is a sensitivity,")
    out("   not a correction factor. It is NOT a constant, do not compare it against a fixed number.")
    out("7. WCC: not applied anywhere yet -> the image is count/voxel, NOT Bq/mL yet.")
    out(f"\n{'ALL BEDS PASS' if not bad else f'BEDS WITH PROBLEMS: {bad}'}")
    return bad


def meta(case, n: int) -> dict:
    """The `to_stir.json` of bed `n`, including the nested `estimate.json`."""
    with open(case.work_bed(n) / "to_stir.json") as f:
        return json.load(f)


def summarise(case, beds, A_by_bed: dict) -> dict:
    """One row per bed: totals for the count terms, means for the factors."""
    out = {}
    for n in beds:
        A = A_by_bed[n]
        out[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                  for t in ALL_TERMS if t in A}
    return out


def invariants(case, n: int, per_plane: dict, stats: dict) -> dict:
    """The four invariants that must hold on real data, aggregated per plane."""
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
