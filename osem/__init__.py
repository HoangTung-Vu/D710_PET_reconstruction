"""OSEM — the algorithm, and nothing else.

* `recon` — `y = S·(G x) + b` for **one** bed, plus that bed's sensitivity image
* `stitch` — decay-correct and stitch the beds, weighted by the sensitivity image

Everything that is not the algorithm (paths, CT → attenuation, reading terms,
quantification, export, plotting) lives in `utils/` and is shared. A later
algorithm — FBP, MLEM, deep prior — gets its own package alongside this one and
reuses that same `utils/`.

    from utils.paths import case
    from utils import sirf_env, attn
    from osem import recon, stitch

    C = case("ped"); sirf_env.setup(C)
    beds = C.beds()
    y0, x0 = recon.image_grid(C, beds[0])
    at = attn.Attenuation(C, ct_dir, x0, y0)
    img = {n: recon.reconstruct(C, n, at.af(n), x0) for n in beds}
"""

from . import recon, stitch  # noqa: F401
