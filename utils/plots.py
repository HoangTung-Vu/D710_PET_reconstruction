"""Figures for the notebook. No computation here — it only plots what was computed.

It lives in `utils/` rather than `osem/` because nothing about it is
OSEM-specific: a different reconstruction (FBP, MLEM) would want exactly these
figures.

One convention across every image: **clip at a percentile, not at the max**. A
handful of outlier bins turn the whole image black under `vmax = max`, and that
is the easiest way to look at a correct sinogram and think it is broken.
"""

from __future__ import annotations

import numpy as np

from .terms import COUNT_TERMS, FACTOR_TERMS, NSEG0


def slices(a, plane: int) -> dict:
    """The four slices needed to look at one term, from a `(553, 288, 381)` array.

    Keeping just these instead of the full array is why the notebook can run six
    beds in ~2.5 GB instead of ~15 GB.
    """
    return {"sino": a[plane].copy(),                       # (view, tangential)
            "axial": a.sum(axis=1, dtype=np.float64),      # (plane, tangential)
            "prof": a[plane].mean(axis=0),                 # 1D transverse profile
            "per_plane": a.sum(axis=(1, 2), dtype=np.float64)}


def busiest_plane(prompts) -> int:
    """The direct plane with the most counts — plotting an empty plane proves nothing."""
    return int(np.argmax(prompts[0, :NSEG0].sum(axis=(1, 2))))


def sinogram_grid(proj, plane, names, title, cmap, unit, subtitle=""):
    """Two rows per term: the transverse sinogram at `plane`, and the view-summed image.

    The bottom row spans all 553 planes; the dashed white line marks the end of
    segment 0 (plane 47). That row shows the michelogram layout at a glance — and
    equally shows a wrong plane mapping at a glance.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, len(names), figsize=(3.5 * len(names), 7.4),
                           squeeze=False)
    for j, t in enumerate(names):
        p = proj[t]
        for i, (img, lab) in enumerate(((p["sino"], f"plane {plane}"),
                                        (p["axial"], "summed over views"))):
            lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
            im = ax[i, j].imshow(img, cmap=cmap, aspect="auto",
                                 vmin=lo, vmax=max(hi, lo + 1e-9))
            ax[i, j].set_title(f"{t}\n{lab}", fontsize=9)
            plt.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.02)
        ax[1, j].axhline(NSEG0 - 0.5, color="w", lw=0.9, ls="--")
        ax[1, j].set_xlabel("tangential (381)")
    ax[0, 0].set_ylabel("view (288)")
    ax[1, 0].set_ylabel("plane (553)")
    fig.suptitle(f"{title} — {subtitle}   [{unit}]", fontsize=12)
    fig.tight_layout()
    return fig


def profiles(proj, planes, beds, case_name=""):
    """2D images show shape; comparing MAGNITUDE requires the 1D profiles.

    Row 1: one bed at a time, the count-domain terms overlaid.
    Row 2: the factors for that same bed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, len(beds), figsize=(3.2 * len(beds), 8), squeeze=False)
    for j, n in enumerate(beds):
        for t in COUNT_TERMS:
            ax[0, j].plot(proj[n, t]["prof"], lw=1.0, label=t)
        ax[0, j].axhline(0, color="k", lw=0.6)
        ax[0, j].set_title(f"bed {n} — transaxial, plane {planes[n]}", fontsize=9)
        ax[0, j].set_xlabel("tangential (381)")

        for t in FACTOR_TERMS:
            ax[1, j].plot(proj[n, t]["prof"], lw=1.0, label=t)
        ax[1, j].axhline(1.0, color="k", lw=0.6, ls=":")
        ax[1, j].set_title(f"bed {n} — factors", fontsize=9)
        ax[1, j].set_xlabel("tangential (381)")

    ax[0, 0].set_ylabel("count/bin (mean over views)")
    ax[1, 0].set_ylabel("factor")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle(f"Transaxial profiles per bed — {case_name}", fontsize=12)
    fig.tight_layout()
    return fig


def per_plane(proj, beds):
    """Per-plane totals, every bed on the same axes — an outlying bed shows up immediately."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(COUNT_TERMS),
                           figsize=(3.4 * len(COUNT_TERMS), 3.6), squeeze=False)
    ax = ax[0]
    for k, t in enumerate(COUNT_TERMS):
        for n in beds:
            ax[k].plot(proj[n, t]["per_plane"], lw=1.0, label=f"bed {n}")
        ax[k].axvline(NSEG0 - 0.5, color="k", lw=0.8, ls="--")
        ax[k].set_title(t, fontsize=10)
        ax[k].set_xlabel("plane (553)")
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("count/plane")
    ax[0].legend(fontsize=7)
    fig.suptitle("Sum per plane, all beds (line = end of segment 0)", fontsize=12)
    fig.tight_layout()
    return fig


def beds(img, headers, title=""):
    """Per bed: the mid-bed transaxial slice plus that bed's own coronal view."""
    import matplotlib.pyplot as plt

    ns = sorted(img)
    fig, ax = plt.subplots(2, len(ns), figsize=(2.9 * len(ns), 6.6), squeeze=False)
    for j, n in enumerate(ns):
        a = img[n]
        hi = np.percentile(a, 99.9)
        ax[0, j].imshow(a[a.shape[0] // 2], cmap="hot", vmin=0, vmax=hi)
        ax[0, j].set_title(f"bed {n}\ntransaxial", fontsize=9)
        ax[1, j].imshow(a[:, a.shape[1] // 2], cmap="hot", aspect="auto",
                        vmin=0, vmax=hi)
        ax[1, j].set_title(f"table {headers[n]['table_position_mm']:.0f} mm\ncoronal",
                           fontsize=9)
        for i in (0, 1):
            ax[i, j].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def whole_body(vol, vox, plane_mm, title=""):
    """Coronal + sagittal MIPs plus the per-plane total of the stitched volume.

    DICOM's z axis increases towards the head while `imshow` draws row 0 at the
    top, so it is flipped to put the head up, the way such images are normally
    read.
    """
    import matplotlib.pyplot as plt

    hi = np.percentile(vol, 99.9)
    fig, ax = plt.subplots(1, 3, figsize=(13, 7))
    ax[0].imshow(vol.max(axis=1)[::-1], cmap="hot", vmin=0, vmax=hi,
                 aspect=plane_mm / vox[1])
    ax[0].set_title("MIP coronal")
    ax[1].imshow(vol.max(axis=2)[::-1], cmap="hot", vmin=0, vmax=hi,
                 aspect=plane_mm / vox[2])
    ax[1].set_title("MIP sagittal")
    ax[2].plot(vol.sum(axis=(1, 2)), np.arange(vol.shape[0]))
    ax[2].set(title="sum per plane", xlabel="count", ylabel="plane (increasing z)")
    ax[2].grid(alpha=0.3)
    for a in ax[:2]:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def suv(suv_bw, mask, vox, plane_mm, K, case_name=""):
    """A clinically windowed 0–5 MIP, a p99.9 MIP, and the SUV distribution inside the body."""
    import matplotlib.pyplot as plt

    mip = suv_bw.max(axis=1)[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))
    for a, vmax, t in ((ax[0], 5.0, "MIP SUVbw — window 0-5 (clinical reading)"),
                       (ax[1], float(np.percentile(suv_bw, 99.9)),
                        "MIP SUVbw — p99.9")):
        im = a.imshow(mip, cmap="hot", vmin=0, vmax=vmax, aspect=plane_mm / vox[1])
        a.set_title(t, fontsize=10)
        a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.03, pad=0.02)
    ax[2].hist(suv_bw[mask].ravel(), bins=120, range=(0, 6), log=True)
    ax[2].set(title="SUVbw distribution inside the body", xlabel="SUVbw",
              ylabel="voxel count (log)")
    ax[2].grid(alpha=0.3)
    fig.suptitle(f"SUV — {case_name}   (K = {K:,.0f}, STILL AN ESTIMATE)", fontsize=12)
    fig.tight_layout()
    return fig
