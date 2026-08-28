"""OSEM for one bed: build `y = S·(G x) + b`, then iterate.

Three inputs, **not interchangeable**:

| | file | how it is attached |
|---|---|---|
| `y` raw prompts | `decoded/bed<n>.hs` | `recon.set_input` |
| `S` | `work/bed<n>/normdt.hs` × af | `set_acquisition_sensitivity` **before** `set_up` |
| `b` | `work/bed<n>/background.hs` | `set_background_term` |

`S` must be attached **before** `set_up` so that STIR folds it into the
sensitivity image — that is what makes the correction quantitative rather than a
mere reweighting. `b` **bypasses** `S` because randoms and scatter already live
in the measured count domain; `tests/test_notebook_contract.py` rebuilds exactly
that identity on a miniature scanner.

⚠ **Do not multiply `geometry.ring_pair_multiplicity()` in here.** GE's `normdt`
already carries the span-2 multiplicity; multiplying again squares it (4× at odd
bins). See that function's docstring.

API sources, nothing invented — the official SIRF examples in
`$CONDA_PREFIX/dlevel/build/sources/SIRF/examples/Python/PET/`:
`osem_reconstruction.py` (`make_Poisson_loglikelihood` + `OSMAPOSLReconstructor`),
`get_multiplicative_sinogram.py` (`AcquisitionSensitivityModel`),
`listmode_reconstruction.py` (sensitivity + background together). No example
combines **all four** terms on a real sinogram; that combination happens here.
"""

from __future__ import annotations

import time

import numpy as np

from utils import terms

#: 288 views, so the number of subsets must divide 288.
N_SUBSETS = 12
N_ITERATIONS = 3

#: The default of 1 LOR/bin is too coarse for this geometry.
#:
#: It is also the single biggest cost knob in the whole reconstruction: the ray
#: tracer traces this many rays per bin, so 5 -> 1 is a ~5x speedup for a
#: coarser transaxial model. That matters much more with TOF, where each traced
#: LOR is also spread over `n_tof` bins. Use `--lors 1` for a smoke test, not
#: for a result.
TANGENTIAL_LORS = 5

#: Number of transverse voxels. STIR pins the voxel size at 2.1306 mm regardless
#: of this value, so it sets the **FOV**, not the resolution.
XY = 328


def image_grid(case, bed: int, xy: int = XY):
    """The image grid shared by every bed — same scanner, same geometry.

    Returns `(acq_template, image_template)`. `acq_template` must be kept alive:
    it is the source of the ExamInfo for everything built from it.
    """
    import sirf.STIR as pet

    y0 = pet.AcquisitionData(str(case.prompt(bed)))
    x0 = y0.create_uniform_image(1.0, xy)
    # `attenuation.mu_image` requires exactly a bed's 47-plane = 2·24 − 1 grid.
    if x0.as_array().shape[0] != terms.NSEG0:
        raise SystemExit("error: image grid has %d planes, must be %d"
                         % (x0.as_array().shape[0], terms.NSEG0))
    return y0, x0


def acquisition_model(objs, sensitivity, image, tangential_lors=TANGENTIAL_LORS,
                      projector: str = "auto"):
    """`y = S·(G x) + b`, built in the order STIR requires.

    `projector` picks `G`:

    * `"ray"` — `AcquisitionModelUsingRayTracingMatrix`, the projector every
      non-TOF result in this project was made with.
    * `"parallelproj"` — matrix-free, and TOF-aware natively.
    * `"auto"` (default) — `ray` without TOF, `parallelproj` with it.

    **Why TOF must not use the ray-tracing matrix here.**
    `ProjMatrixByBinUsingRayTracing` caches computed matrix rows, and its own
    header says that is the fast choice "IF your system does not start
    swapping". Without TOF the cache is bounded by the symmetries STIR exploits.
    With TOF those symmetries no longer collapse bins onto each other — every
    timing bin of a LOR is a separate row — so the cache grows until the machine
    dies: measured here, one bed at 5 TOF bins reached 30 GB and had to be
    killed, against ~8 GB of actual sinogram data.

    `matrix.enable_cache(False)` fixes the memory but leaves the ray tracer
    recomputing every row on every subiteration. Parallelproj avoids the
    question entirely: it never forms a matrix, and TOF is part of its kernel
    rather than a multiplier on the number of rows.
    """
    import sirf.STIR as pet

    n_tof = int(objs["prompts"].dimensions()[0])
    if projector == "auto":
        projector = "parallelproj" if n_tof > 1 else "ray"

    if projector == "parallelproj":
        am = pet.AcquisitionModelUsingParallelproj()
    elif projector == "ray":
        am = pet.AcquisitionModelUsingRayTracingMatrix()
        am.set_num_tangential_LORs(tangential_lors)
        if n_tof > 1:
            # Asked for explicitly, so honour it -- but not with the cache on.
            am.get_matrix().enable_cache(False)
    else:
        raise SystemExit("error: --projector must be auto, ray or parallelproj "
                         "(got %r)" % projector)

    # BEFORE set_up: that is what makes S go into the sensitivity image.
    am.set_acquisition_sensitivity(pet.AcquisitionSensitivityModel(sensitivity))
    am.set_background_term(objs["background"])
    am.set_up(objs["prompts"], image)
    return am


def reconstruct(case, bed: int, af, image, n_sub: int = N_SUBSETS,
                n_it: int = N_ITERATIONS, xy: int = XY, tof_scatter=None,
                tangential_lors: int = TANGENTIAL_LORS,
                projector: str = "auto"):
    """OSEM for one bed. Returns `(image, sensitivity)`, both `(47, xy, xy)`.

    `sensitivity` is **STIR's own sensitivity image** — the denominator OSEM
    divides by on each iteration, i.e. the backprojection of `S` over all bins.
    It already includes norm, dead time, attenuation AND how the projector really
    samples LORs, so using it as the bed-stitching weight is a *measurement*
    rather than a *geometric assumption*. Getting it costs nothing extra:
    `set_up` has already computed it.

    The bed is reloaded from disk and its RAM handed back when done, so running
    six beds does not mean holding six sets of sinograms in memory (~2.5 GB
    instead of ~15 GB).
    """
    import sirf.STIR as pet

    # lean=True: the reconstruction reads only `sensitivity` out of `A`, and with
    # TOF prompts every other entry is a multi-GB duplicate of a SIRF object that
    # is already loaded. Without it one bed at mash 5 peaks near 16 GB.
    objs, A = terms.load(case, bed, af=af, tof_scatter=tof_scatter, lean=True)

    S = objs["prompts"].get_uniform_copy(0)
    S.fill(A["sensitivity"])
    del A                       # numpy arrays no longer needed; S holds a copy

    am = acquisition_model(objs, S, image, tangential_lors, projector)
    print(f"    projector {am.__class__.__name__}", flush=True)

    obj = pet.make_Poisson_loglikelihood(objs["prompts"], acq_model=am)
    obj.set_num_subsets(n_sub)
    obj.set_up(image)
    # Summed over all subsets -> the full sensitivity of this bed.
    sens = sum(obj.get_subset_sensitivity(s).as_array() for s in range(n_sub))

    rec = pet.OSMAPOSLReconstructor()
    rec.set_objective_function(obj)
    rec.set_num_subsets(n_sub)
    rec.set_num_subiterations(n_sub * n_it)
    rec.set_input(objs["prompts"])
    rec.set_up(image)           # computes the sensitivity image — the slowest step
    rec.set_current_estimate(image)
    # Report each subiteration. With TOF a bed runs for tens of minutes, and a
    # run that prints nothing until it finishes is indistinguishable from one
    # that has hung — which is exactly how the first TOF bed was misread.
    n = rec.get_num_subiterations()
    t0 = time.time()
    for k in range(n):
        rec.update_current_estimate()
        el = time.time() - t0
        print(f"    subiter {k + 1:3d}/{n}  {el:6.0f} s elapsed, "
              f"~{el / (k + 1) * (n - k - 1):6.0f} s left", flush=True)

    out = rec.get_current_estimate().as_array().copy()
    del objs
    return out, sens.astype(np.float32)


def reconstruct_all(case, beds, af: dict, image, out=print, **kw):
    """`reconstruct` for every bed, timed. Returns `(img, sens)`, two dicts.

    The `sens edge/centre` value printed per bed is worth a glance: it
    shows how far sensitivity drops off at the end of a bed, i.e. how weak the
    overlap region is when stitching.
    """
    img, sens = {}, {}
    for n in beds:
        t0 = time.time()
        img[n], sens[n] = reconstruct(case, n, af[n], image, **kw)
        sp = sens[n].mean(axis=(1, 2))
        out(f"bed {n}: {time.time() - t0:5.0f} s   max {img[n].max():9.4g}   "
            f"mean {img[n].mean():9.4g}   sens edge/centre {sp[0] / sp.max():.4f}")
    return img, sens
