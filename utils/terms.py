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

TOF prompts need `b` and `S` on the same 5-D grid; `expand_to_tof` below builds
them, and `work/bed<n>/scatter_tof.npy` — GE's own TOF distribution for the
scatter — is what makes that more than an approximation.

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

from .scanner import NSEG0  # noqa: F401

#: Terms in the count domain — adding them up is meaningful.
COUNT_TERMS = ("prompts", "randoms", "scatter", "background", "trues")

#: Terms that are (dimensionless) factors — averaging is meaningful, summing is not.
FACTOR_TERMS = ("norm_only", "deadtime", "normdt", "attenuation", "sensitivity")

ALL_TERMS = COUNT_TERMS + FACTOR_TERMS

#: What `to_stir.py` writes out. `attn` is added later by `utils.attn`.
ON_DISK = ("randoms", "scatter", "background", "normdt", "norm_only")

#: The comment line `gerdf.interfile` stamps into every TOF prompts header
#: (`custom_tool/gerdf/interfile.py:TOF_AXIS_KEY` -- spelled out rather than
#: imported, because `gerdf` lives inside `d710:full` and this module runs on the
#: host). Its presence is the ONLY thing that distinguishes prompts whose TOF
#: axis runs STIR's way from prompts that run GE's, and the difference is not
#: cosmetic: STIR's timing positions are signed displacements ALONG the LOR, so
#: a mirrored axis puts the activity on the wrong half of every one of them.
#: Counts, file size and every invariant in `invariant_table` are identical
#: either way -- this key is the check.
TOF_AXIS_KEY = "; TOF axis"


def total(a) -> float:
    """Sum in float64.

    Summing 60.7 million float32 bins drifts by a few counts — enough to make a
    comparison of the total against the header fail for no good reason.
    """
    return float(np.asarray(a).sum(dtype=np.float64))


#: Keys `osem.recon` actually reads out of `A`. Everything else in there is a
#: diagnostic for `collect` / `invariant_table`, and at 11 TOF bins each TOF-sized
#: one is another 2.7 GB — see `load(lean=True)`.
RECON_TERMS = ("sensitivity",)


def load(case, n: int, af=None, tof_scatter=None, lean: bool = False):
    """Every term of bed `n`, as `(SIRF objs, dict of numpy arrays)`.

    `af` (attenuation factors, a numpy array) is optional: given one, this adds
    `attenuation` and `sensitivity = normdt × attenuation`; without it those two
    keys are absent.

    `tof_scatter` overrides how the scatter is spread over the TOF axis; it is
    ignored when the prompts are not TOF. See `expand_to_tof` for what the
    default is and why there are three of them.

    `lean` drops every numpy array the reconstruction does not read, as soon as
    it is no longer needed. **With TOF prompts that is the difference between
    running and exhausting the machine**: at mash 5 the numpy `prompts`,
    `background` and `sensitivity` are 2.7 GB each, and their SIRF counterparts
    are another copy, so a full `load` peaks near 16 GB for one bed. `lean`
    keeps the peak to roughly the three SIRF objects. The diagnostics
    (`trues`, `deadtime`, the per-LOR terms) are still built first, because
    `expand_to_tof` needs `randoms` and `scatter` separately — they are just not
    held afterwards.

    **Remember to `del` both when done.**
    """
    import sirf.STIR as pet

    work = case.work_bed(n)
    objs = {"prompts": pet.AcquisitionData(str(case.prompt(n)))}
    for name in ON_DISK:
        objs[name] = pet.AcquisitionData(str(work / f"{name}.hs"))

    A = {k: v.as_array() for k, v in objs.items()}
    n_tof = A["prompts"].shape[0]                 # as_array is (tof, axial, view, tang)
    if n_tof > 1:
        check_tof_axis(case, n)

    if not lean:
        # Non-TOF `trues` even when the prompts are TOF: it is a DIAGNOSTIC, and
        # one per LOR is what anybody reading it wants. Summing the prompts over
        # TOF is also the honest comparison, because `background` really is
        # non-TOF.
        A["trues"] = (A["prompts"].sum(axis=0, keepdims=True) if n_tof > 1
                      else A["prompts"]) - A["background"]  # negatives are noise
        A["deadtime"] = A["normdt"] / A["norm_only"]  # live fraction, < 1
    if af is not None:
        A["attenuation"] = af
        # S = norm × dead time × attenuation. All three are SENSITIVITIES, so they
        # MULTIPLY together; `normdt` already bundles the first two factors.
        A["sensitivity"] = A["normdt"] * af

    if n_tof > 1:
        expand_to_tof(case, n, objs, A, n_tof, tof_scatter)

    if lean:
        # `objs["background"]` and `objs["prompts"]` carry everything the model
        # needs; these numpy copies are pure duplication from here on.
        for k in [k for k in A if k not in RECON_TERMS]:
            del A[k]
    return objs, A


def check_tof_axis(case, n: int) -> None:
    """Refuse TOF prompts written before the axis was turned the right way round.

    Until 2026-08-29 the decoder wrote GE's TOF bin order straight through,
    while STIR reads index 0 as the **most negative** timing position and turns
    it into a signed displacement along the LOR (`ProjDataInfo::get_k`: timing
    position -2 is -294.31 mm on this scanner, +2 is +294.31 mm). The two
    conventions are mirror images, so every TOF reconstruction made before then
    placed activity on the wrong half of every LOR and came out worse than the
    same bed with no TOF at all.

    Nothing else can catch it. The prompts and the scatter travel down the same
    axis, so they agree with each other whichever way it points; the counts, the
    file size, `Sum(p) >= Sum(r)` and every invariant in `invariant_table` are
    the same in both. Only the image differs. That is why the fix came with a
    stamp in the header rather than only a code change: an old `.s` on disk is
    otherwise indistinguishable from a new one.

    `tools/tof_direction.py` is the measurement this stamp stands in for.
    """
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


def vendor_tof_weights(case, n: int, n_tof: int, n_view: int, n_tang: int):
    """GE's own TOF distribution for this bed's scatter, or `None`.

    Returns `(w, note)` with `w` of shape `(n_tof, n_view, n_tang)` summing to 1
    along TOF everywhere the scatter is non-zero — a **shape**, not a sinogram.
    The amplitude stays in `A["scatter"]`, which is why `b` keeps exactly the
    total it had before the TOF axis was added.

    `work/bed<n>/scatter_tof.npy` is written by `vendor/to_stir.py` from
    `scatter_tof.f32`, which `estimate.py` gets by running GE's scatter model
    with `reconMethod = 3`. That one job field is what sets
    `CScatterFully3dModel::m_bTOFDim`, and `CreateTaskList` then queues
    `MSCAT_CALC_SCAT_ESTIMATE_TOF` and friends alongside the ordinary SSS
    iterations. So this is not a reimplementation of anything: it is the vendor's
    single-scatter simulation with its time axis kept.

    Two axes have to be brought onto the data's grid, and both are cheap:

    * **TOF** — the vendor works at the RDF's full 55 bins while the decoded
      prompts are mashed (`d710` defaults to mash 5 → 11 bins). Adjacent bins are
      summed, exactly the grouping `gerdf.cli.mash_tof` applied to the prompts,
      so the two axes stay aligned bin for bin. Both arrive already reversed
      into STIR's timing-position order — `to_stir.py` for these weights,
      `gerdf.cli._tof_to_stir` for the prompts — and reversing commutes with the
      mashing because the mash factor divides the bin count, so bin `t` here is
      bin `t` there whichever mash is in force.
    * **tangential** — GE's SSS runs on `ds_nu = 43` downsampled bins. They are
      interpolated up to the full 381 with cell-centred linear weights. The
      profile is smooth in `u` (the centroid moves by 0.5 bins across the whole
      radial range once views are averaged), so the interpolation is not where
      the error lives.

    Dropped: the two length-4 axes of the vendor buffer, summed away in
    `to_stir.py`. Measured on ped bed 1 that costs a median 0.45 TOF bins of
    centroid — 40 ps, 6 mm — against the 10 bins of real variation across
    `(view, u)` that this function keeps.
    """
    import numpy as np

    path = case.work_bed(n) / "scatter_tof.npy"
    if not path.exists():
        return None
    # Same mirror, same reason as `check_tof_axis`, other term. The prompts are
    # stamped in their header; these weights are stamped in `to_stir.json`. Both
    # reversals were made together and neither is meaningful alone -- half a fix
    # subtracts the scatter from the wrong half of every LOR.
    try:
        axis = (meta(case, n).get("scatter_tof") or {}).get("tof_axis")
    except OSError:
        axis = "stir"      # weights supplied by hand, with no sidecar to check
    if axis != "stir":
        raise SystemExit(
            "error: %s was written before 2026-08-29 and its TOF axis runs GE's\n"
            "  way round, mirrored against the prompts.  Re-run:\n"
            "      d710 tostir --case %s --bed %d\n"
            "  (cheap -- it only re-reads the vendor .f32 that is already there)"
            % (path, case.name, n))
    w = np.load(path).astype(np.float64)          # (tof_full, view, ds_nu)
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

    # ds_nu -> n_tang, as one (ds_nu, n_tang) matrix applied with a matmul rather
    # than n_tof x n_view calls to np.interp.
    w = w @ _upsample_matrix(ds_nu, n_tang)

    # Normalise per LOR. Where GE's coarse grid has no scatter at all there is
    # nothing to distribute, and A["scatter"] is ~0 there too, so the column is
    # left at zero instead of being filled with a made-up profile.
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
    """`(n_src, n_dst)` linear-interpolation weights, cell centres aligned.

    Destination cell `i` covers `[i, i+1) · n_src/n_dst` of the source, so its
    centre lands at `(i + 0.5)·n_src/n_dst − 0.5` in source index units, and the
    two neighbours either side of that share the weight.

    Cell-centred rather than endpoint-aligned (`linspace(0, n_src-1, n_dst)`)
    because the two grids sample the same physical extent at different rates,
    not the same index range. Endpoint alignment would pin source cell 0 to
    destination cell 0 and stretch everything between, shifting the profile by
    up to half a coarse cell. Measured on ped bed 1 the two choices differ by at
    most 0.18 TOF bins of centroid and 0.24 % in L1, so this is about being right
    for a reason rather than about a visible difference.
    """
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
    """Measure the scatter's TOF profile from THIS bed. `(w, note)`, `w.sum() == 1`.

    The fallback for beds estimated before `reconMethod = 3` existed, or with
    `--no-tof`. It needs nothing from the vendor and no geometry constant. The
    lever is geometry: **the further a LOR is from the centre the less of the
    patient it crosses**, so cutting the sinogram by tangential bin separates the
    three components. And there is more to work with here than that — `randoms`
    and `scatter` are already on disk in exact non-TOF form, so the tail ring is
    chosen by **their own measurement** rather than by a millimetre threshold
    guessed for one body size:

        T(u) = Σ prompts(u) − R(u) − S(u)        estimated true rate, per bin u

    The tail is the set of `u` where `T/S` is small, i.e. where scatter dominates
    trues. There

        profile(t) ∝ Σ_u prompts(t, u) − Σ_u R(u)/n_tof

    because randoms are flat in TOF — measured, `CoV 0.0404` against a Poisson
    floor of `0.0393` on LORs that miss the patient entirely
    (`tools/tof_profile.py`).

    This is the same region GE fits its scatter tails in:
    `CScatterFully3dModel::CalcSinoTails`, `SCAT_TAILFIT_ANGLE_WINDOW 31`.

    **Two limits, stated up front.** It is one global profile for every LOR,
    while the real distribution moves with view and radius — by about 10 TOF bins
    on ped bed 1, which is why `vendor_tof_weights` is preferred whenever it is
    available. And the tail ring still holds the `T/S` fraction of trues, and
    trues are narrower than scatter, so the measured profile comes out slightly
    too narrow.
    """
    import numpy as np

    # Collapse the axial and view axes; keep only (tof, tangential). The profile
    # is a function of TOF, and u is the axis used to SELECT the region.
    P = A["prompts"].sum(axis=(1, 2)).astype(np.float64)     # (n_tof, n_tang)
    R = A["randoms"][0].sum(axis=(0, 1)).astype(np.float64)  # (n_tang,)
    S = A["scatter"][0].sum(axis=(0, 1)).astype(np.float64)
    T = np.clip(P.sum(axis=0) - R - S, 0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(S > 0, T / S, np.inf)
    # Widen until the tail holds enough counts: a thin bed has few bins with a
    # small T/S.
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
    """Make `S` and `b` match TOF prompts. Mutates `objs` and `A` in place.

    GE's prep stage keeps its *sinograms* non-TOF — `vendor/README.md` §3a shows
    `RawPrompts 288 view × 421386 B = 2 B × 381 × 553`, no TOF axis — and that is
    deliberate rather than a gap: norm, dead time, attenuation and randoms do not
    depend on arrival time at all, so one copy is the whole answer and GE spends
    no memory on 55 identical ones. Scatter is the exception, and prep produces
    its time distribution too, compactly, in `m_pScatterTOF`.

    The three terms are **not** the same problem and must not share a code path:

    `S` (norm × dead time × attenuation) — **repeated**, and that is exact, not
    an approximation. `CNorm3d::ApplyNormalization` and
    `CDeadtime3d::ApplyDeadtime` have no `...Tof` sibling anywhere in
    `pet_recon`, and `OclMultAttnNormDt` — the kernel that multiplies
    attn × norm × dt *inside* the TOF loop — takes `COsemTofValues` while the
    data it multiplies is non-TOF. Norm is the efficiency of a crystal PAIR and
    dead time the live fraction of a block: neither knows when the photon
    arrived. STIR agrees on the arithmetic: `ProjMatrixByBin::get_tof_value` is
    `0.5·(erf(d2) − erf(d1))`, which telescopes to 1 over the window, so
    `Σ_t S·(G_t x) = S·(G x)` — the same model as the non-TOF run.

    `randoms` — **divided by n_tof**, also exact. 55 × 89.2459 ps = 4.909 ns is
    exactly the coincidence window, and accidental coincidences carry no timing
    correlation, so they really are flat across it. `GetRandomsViewData(view,
    out)` has no TOF-parameterised sibling either.

    `scatter` — **shaped**, from one of three sources, in this order:

    1. `tof_scatter`, if the caller passed one — one global profile, mashed to
       fit if it arrives at the RDF's full bin count, for comparing against a run
       made some other way;
    2. `work/bed<n>/scatter_tof.npy` — GE's own distribution, one profile per
       `(view, tangential)`. This is the right answer and the default;
    3. `scatter_tof_profile` — one profile measured from this bed's own tail
       ring, for beds estimated before the vendor path existed.

    Flat is not on that list, and the difference is not cosmetic. Measured on ped
    bed 1 at mash 5: with GE's distribution **0 of 11** TOF bins have a negative
    total true rate; spread flat, 69 % of them do. A single global profile sits
    between the two — it is 20.6 % away from GE's in L1, because the real
    distribution's centroid moves about 10 bins (890 ps, 134 mm) across
    `(view, u)` and one profile cannot follow it.

    `b = randoms + scatter` arrives already summed on disk, so splitting them
    needs `A["randoms"]`, which `load` still has in its non-TOF form.
    """
    import numpy as np

    # `load` always supplies these (prompts plus everything in ON_DISK); a direct
    # caller may not. randoms and scatter must arrive SEPARATE: randoms is flat in
    # TOF and scatter is not, so `background` pre-summed has already destroyed the
    # one distinction this function exists to make.
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

    # `w` is either (n_tof,) -- one profile for every LOR -- or
    # (n_tof, n_view, n_tang), one per LOR. Both broadcast against a
    # (n_axial, n_view, n_tang) scatter slice, so the assembly below is shared.
    if tof_scatter is not None:
        w = np.asarray(tof_scatter, dtype=np.float64).ravel()
        if w.min() < 0 or not w.sum():
            raise SystemExit("error: a scatter profile must be non-negative "
                             "and sum to more than zero")
        # `tools/tof_profile.py --save` measures at the RDF's full 55 bins,
        # because it reads a raw decoded view. Mash it the same way the prompts
        # and the vendor weights are mashed rather than making the operator do
        # the arithmetic -- getting it wrong here is silent.
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
        got = vendor_tof_weights(case, n, n_tof, n_view, n_tang)
        w, note = got if got else scatter_tof_profile(A, n_tof)
    print(f"  TOF: {n_tof} bins -- S repeated (exact), randoms/{n_tof} "
          f"(confirmed by measurement), scatter: {note}")

    shape = (n_tof,) + A["background"].shape[1:]
    bg = np.empty(shape, dtype=np.float32)
    rnd = A["randoms"][0] / n_tof
    sct = A["scatter"][0]
    for t in range(n_tof):
        # w[t] is a scalar for a global profile and (n_view, n_tang) for the
        # vendor's, which broadcasts over the axial axis of `sct`.
        bg[t] = rnd + sct * w[t]

    # The background has to become a SIRF object, not just an array: OSEM takes
    # it through `set_background_term`, and a non-TOF one against a TOF model is
    # rejected by `set_up`. Built from the prompts so the geometry is theirs by
    # construction rather than by a header that has to be kept in step.
    obj = objs["prompts"].get_uniform_copy(0)
    obj.fill(bg)
    objs["background"] = obj
    A["background"] = bg

    # randoms/scatter are left non-TOF on purpose: nothing in the model reads
    # them (only `background` does), they are what the diagnostics in this
    # module report per LOR, and a TOF copy of each is another 1.2 GB at mash 11.


def ct_dir(case, n: int) -> str:
    """The CT series that `d710 estimate` used for this bed.

    Taken from the sidecar rather than asked again: attenuation and scatter
    **must** be built from the same CT, and that is the CT GE's kernel saw.
    """
    ct = meta(case, n).get("estimate", {}).get("ct")
    if not ct:
        raise SystemExit("error: the sidecar of bed %d records no CT" % n)
    return ct


def bed_table(case, beds, out=print) -> None:
    """Acquisition summary table, one row per bed.

    The two patient cases carry PHI: name / patient ID / date of birth are
    deliberately **not** printed.
    """
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

        # Every plot and every invariant below is PER LOR, so anything carrying a
        # TOF axis is summed over it first. `A[t][0]` would otherwise silently
        # take TOF bin 0 alone and compare one fifty-fifth of the prompts against
        # the whole of randoms -- invariant 1 (Σp ≥ Σr) would fail on data that is
        # perfectly fine. `total()` is unaffected either way, but go through the
        # same collapse so the table and the plots describe the same array.
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

    out("\n1. Σp ≥ Σr and 2. Σs ≤ Σ(p−r): must be 0 % on every plane — "
        "a negative true rate is impossible.")
    out("3. ΣR/delays ~0.99: GE's randoms against the delays the scanner counted, two independent routes.")
    out("5. livetime DROPS as the count rate RISES — that is the sign it is a sensitivity,")
    out("   not a correction factor. It is NOT a constant, do not compare it against a fixed number.")
    out("7. WCC: not applied anywhere yet -> the image is count/voxel, NOT Bq/mL yet.")
    out(f"\n{'ALL BEDS PASS' if not bad else f'BEDS WITH PROBLEMS: {bad}'}")
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
