"""OSEM for one bed: assembly of `y = S(Gx) + b`, then iteration."""

from __future__ import annotations

import time

import numpy as np

from utils import scanner, terms
from utils.scanner import (N_ITERATIONS, N_SUBSETS,
                           TANGENTIAL_LORS, XY)


def image_grid(case, bed: int, xy: int = XY):
    """`(acq_template, image_template)`; the acquisition template carries the ExamInfo and must be kept alive."""
    import sirf.STIR as pet

    y0 = pet.AcquisitionData(str(case.prompt(bed)))
    x0 = scanner.sirf_grid(y0, xy)
    if x0.as_array().shape[0] != terms.NSEG0:
        raise SystemExit("error: image grid has %d planes, must be %d"
                         % (x0.as_array().shape[0], terms.NSEG0))
    return y0, x0


def acquisition_model(objs, sensitivity, image, tangential_lors=TANGENTIAL_LORS,
                      projector: str = "auto"):
    """`y = S(Gx) + b`, assembled in the order STIR requires."""
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
            am.get_matrix().enable_cache(False)
    else:
        raise SystemExit("error: --projector must be auto, ray or parallelproj "
                         "(got %r)" % projector)

    am.set_acquisition_sensitivity(pet.AcquisitionSensitivityModel(sensitivity))
    am.set_background_term(objs["background"])
    am.set_up(objs["prompts"], image)
    return am


def reconstruct(case, bed: int, af, image, n_sub: int = N_SUBSETS,
                n_it: int = N_ITERATIONS, xy: int = XY, tof_scatter=None,
                tangential_lors: int = TANGENTIAL_LORS,
                projector: str = "auto"):
    """Reconstruct one bed with OSEM."""
    import sirf.STIR as pet

    objs, A = terms.load(case, bed, af=af, tof_scatter=tof_scatter, lean=True)

    S = objs["prompts"].get_uniform_copy(0)
    S.fill(A["sensitivity"])
    del A

    am = acquisition_model(objs, S, image, tangential_lors, projector)
    print(f"    projector {am.__class__.__name__}", flush=True)

    obj = pet.make_Poisson_loglikelihood(objs["prompts"], acq_model=am)
    obj.set_num_subsets(n_sub)

    rec = pet.OSMAPOSLReconstructor()
    rec.set_objective_function(obj)
    rec.set_num_subsets(n_sub)
    rec.set_num_subiterations(n_sub * n_it)
    rec.set_input(objs["prompts"])

    rec.set_up(image)
    sens = sum(obj.get_subset_sensitivity(s).as_array() for s in range(n_sub))

    n_tang = int(objs["prompts"].dimensions()[3])
    mask = scanner.fov_mask(image.as_array().shape[-1], n_tang)
    x0 = image.clone()
    x0.fill(image.as_array() * mask)
    rec.set_current_estimate(x0)

    n = rec.get_num_subiterations()
    t0 = time.time()
    for k in range(n):
        rec.update_current_estimate()
        el = time.time() - t0
        print(f"    subiter {k + 1:3d}/{n}  {el:6.0f} s elapsed, "
              f"~{el / (k + 1) * (n - k - 1):6.0f} s left", flush=True)

    out = rec.get_current_estimate().as_array().copy()
    del objs
    return out * mask, (sens * mask).astype(np.float32)


def bed_key(case, n: int, image, af, kw: dict) -> str:
    """Fingerprint of everything that changes bed `n`'s reconstruction."""
    import hashlib

    import numpy as np

    hs = case.prompt(n)
    st = hs.stat()
    parts = [f"prompts={hs.name}:{st.st_size}:{int(st.st_mtime)}",
             f"grid={tuple(image.as_array().shape)}",
             f"af={np.shape(af)}:{float(np.asarray(af, np.float64).sum()):.12e}"]
    for k in sorted(kw):
        v = kw[k]
        parts.append(f"{k}=" + (f"sum:{float(np.asarray(v, np.float64).sum()):.12e}"
                                if isinstance(v, np.ndarray) else repr(v)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def reconstruct_all(case, beds, af: dict, image, out=print, resume: bool = False,
                    **kw):
    """Reconstruct every bed, with timings."""
    import numpy as np

    img, sens = {}, {}
    for n in beds:
        p = case.work_bed(n) / "osem.npz"
        key = bed_key(case, n, image, af[n], kw)
        if resume and p.exists():
            z = np.load(p, allow_pickle=False)
            if "key" in z.files and str(z["key"]) == key:
                img[n], sens[n] = z["img"], z["sens"]
                out(f"bed {n}: reused {p.name} "
                    f"(took {float(z['seconds']):.0f} s when it was made)")
                continue
            out(f"bed {n}: {p.name} was made with different settings "
                f"-- reconstructing again")

        t0 = time.time()
        img[n], sens[n] = reconstruct(case, n, af[n], image, **kw)
        el = time.time() - t0
        sp = sens[n].mean(axis=(1, 2))
        np.savez_compressed(p, img=img[n], sens=sens[n], key=key, bed=n,
                            seconds=el)
        out(f"bed {n}: {el:5.0f} s   max {img[n].max():9.4g}   "
            f"mean {img[n].mean():9.4g}   sens edge/centre {sp[0] / sp.max():.4f}"
            f"   -> {p.name}")
    return img, sens
