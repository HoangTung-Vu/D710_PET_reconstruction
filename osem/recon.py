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

import numpy as np

from utils import terms

#: 288 views, so the number of subsets must divide 288.
N_SUBSETS = 12
N_ITERATIONS = 3

#: The default of 1 LOR/bin is too coarse for this geometry.
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
        raise SystemExit("error: lưới ảnh có %d plane, phải là %d"
                         % (x0.as_array().shape[0], terms.NSEG0))
    return y0, x0


def acquisition_model(objs, sensitivity, image, tangential_lors=TANGENTIAL_LORS):
    """`y = S·(G x) + b`, built in the order STIR requires."""
    import sirf.STIR as pet

    am = pet.AcquisitionModelUsingRayTracingMatrix()
    am.set_num_tangential_LORs(tangential_lors)
    # BEFORE set_up: that is what makes S go into the sensitivity image.
    am.set_acquisition_sensitivity(pet.AcquisitionSensitivityModel(sensitivity))
    am.set_background_term(objs["background"])
    am.set_up(objs["prompts"], image)
    return am


def reconstruct(case, bed: int, af, image, n_sub: int = N_SUBSETS,
                n_it: int = N_ITERATIONS, xy: int = XY):
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

    objs, A = terms.load(case, bed, af=af)

    S = objs["prompts"].get_uniform_copy(0)
    S.fill(A["sensitivity"])
    del A                       # numpy arrays no longer needed; S holds a copy

    am = acquisition_model(objs, S, image)

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
    for _ in range(rec.get_num_subiterations()):
        rec.update_current_estimate()

    out = rec.get_current_estimate().as_array().copy()
    del objs
    return out, sens.astype(np.float32)


def reconstruct_all(case, beds, af: dict, image, out=print, **kw):
    """`reconstruct` for every bed, timed. Returns `(img, sens)`, two dicts.

    The `sens rìa/giữa` (edge/centre) value printed per bed is worth a glance: it
    shows how far sensitivity drops off at the end of a bed, i.e. how weak the
    overlap region is when stitching.
    """
    import time

    img, sens = {}, {}
    for n in beds:
        t0 = time.time()
        img[n], sens[n] = reconstruct(case, n, af[n], image, **kw)
        sp = sens[n].mean(axis=(1, 2))
        out(f"bed {n}: {time.time() - t0:5.0f} s   max {img[n].max():9.4g}   "
            f"mean {img[n].mean():9.4g}   sens rìa/giữa {sp[0] / sp.max():.4f}")
    return img, sens
