"""Hình cho notebook. Không có tính toán nào ở đây — chỉ vẽ cái đã tính.

Nằm ở `utils/` chứ không ở `osem/` vì không có gì mang tính OSEM: một
reconstruction khác (FBP, MLEM) vẫn muốn đúng những hình này.

Quy ước chung cho mọi ảnh: **cắt ở phân vị, không ở max**. Vài bin ngoại lai
làm cả ảnh thành đen nếu để `vmax = max`, và đó là cách dễ nhất để nhìn một
sinogram đúng mà tưởng là sai.
"""

from __future__ import annotations

import numpy as np

from .terms import COUNT_TERMS, FACTOR_TERMS, NSEG0


def slices(a, plane: int) -> dict:
    """Bốn lát cắt đủ để nhìn một số hạng, từ mảng `(553, 288, 381)`.

    Giữ lại từng này thay vì cả mảng là lý do notebook chạy được sáu bed trong
    ~2,5 GB thay vì ~15 GB.
    """
    return {"sino": a[plane].copy(),                       # (view, tangential)
            "axial": a.sum(axis=1, dtype=np.float64),      # (plane, tangential)
            "prof": a[plane].mean(axis=0),                 # cắt ngang 1D
            "per_plane": a.sum(axis=(1, 2), dtype=np.float64)}


def busiest_plane(prompts) -> int:
    """Plane trực tiếp nhiều count nhất — vẽ plane rỗng thì không kết luận gì."""
    return int(np.argmax(prompts[0, :NSEG0].sum(axis=(1, 2))))


def sinogram_grid(proj, plane, names, title, cmap, unit, subtitle=""):
    """Hai hàng cho mỗi số hạng: sinogram ngang tại `plane`, và ảnh gộp theo view.

    Hàng dưới đủ cả 553 plane; vạch đứt trắng là ranh giới cuối segment 0
    (plane 47). Nhìn hàng này là thấy ngay cách xếp michelogram — và thấy ngay
    nếu ánh xạ plane sai.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, len(names), figsize=(3.5 * len(names), 7.4),
                           squeeze=False)
    for j, t in enumerate(names):
        p = proj[t]
        for i, (img, lab) in enumerate(((p["sino"], f"plane {plane}"),
                                        (p["axial"], "gộp theo view"))):
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
    """Ảnh 2D cho thấy hình dạng; muốn so ĐỘ LỚN thì phải xem lát cắt 1D.

    Hàng 1: từng bed một, các số hạng miền count chồng lên nhau.
    Hàng 2: các hệ số của cùng bed đó.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, len(beds), figsize=(3.2 * len(beds), 8), squeeze=False)
    for j, n in enumerate(beds):
        for t in COUNT_TERMS:
            ax[0, j].plot(proj[n, t]["prof"], lw=1.0, label=t)
        ax[0, j].axhline(0, color="k", lw=0.6)
        ax[0, j].set_title(f"bed {n} — cắt ngang, plane {planes[n]}", fontsize=9)
        ax[0, j].set_xlabel("tangential (381)")

        for t in FACTOR_TERMS:
            ax[1, j].plot(proj[n, t]["prof"], lw=1.0, label=t)
        ax[1, j].axhline(1.0, color="k", lw=0.6, ls=":")
        ax[1, j].set_title(f"bed {n} — hệ số", fontsize=9)
        ax[1, j].set_xlabel("tangential (381)")

    ax[0, 0].set_ylabel("count/bin (trung bình theo view)")
    ax[1, 0].set_ylabel("hệ số")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.suptitle(f"Cắt ngang từng bed — {case_name}", fontsize=12)
    fig.tight_layout()
    return fig


def per_plane(proj, beds):
    """Tổng theo plane, mọi bed trên cùng một trục — bed nào lệch là thấy ngay."""
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
    fig.suptitle("Tổng theo plane, mọi bed (vạch = hết segment 0)", fontsize=12)
    fig.tight_layout()
    return fig


def beds(img, headers, title=""):
    """Từng bed: lát ngang giữa bed + coronal của riêng bed đó."""
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
    """MIP coronal + sagittal + tổng theo plane của khối đã ghép.

    Trục z của DICOM tăng về phía đầu, mà `imshow` vẽ hàng 0 ở trên, nên lật lại
    để đầu ở trên như phim thường đọc.
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
    ax[2].set(title="tổng theo plane", xlabel="count", ylabel="plane (z tăng dần)")
    ax[2].grid(alpha=0.3)
    for a in ax[:2]:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def suv(suv_bw, mask, vox, plane_mm, K, case_name=""):
    """MIP cửa sổ lâm sàng 0–5, MIP p99.9, và phân bố SUV trong thân."""
    import matplotlib.pyplot as plt

    mip = suv_bw.max(axis=1)[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))
    for a, vmax, t in ((ax[0], 5.0, "MIP SUVbw — cửa sổ 0–5 (đọc lâm sàng)"),
                       (ax[1], float(np.percentile(suv_bw, 99.9)),
                        "MIP SUVbw — p99.9")):
        im = a.imshow(mip, cmap="hot", vmin=0, vmax=vmax, aspect=plane_mm / vox[1])
        a.set_title(t, fontsize=10)
        a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.03, pad=0.02)
    ax[2].hist(suv_bw[mask].ravel(), bins=120, range=(0, 6), log=True)
    ax[2].set(title="phân bố SUVbw trong thân", xlabel="SUVbw",
              ylabel="số voxel (log)")
    ax[2].grid(alpha=0.3)
    fig.suptitle(f"SUV — {case_name}   (K = {K:,.0f}, CÒN LÀ SUY ĐOÁN)", fontsize=12)
    fig.tight_layout()
    return fig
