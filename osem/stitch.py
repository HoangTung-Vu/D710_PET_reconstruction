"""Hiệu chỉnh phân rã rồi ghép các bed theo trục.

### Phân rã phải làm TRƯỚC khi ghép

Sáu bed chụp cách nhau 91 s, bed 6 muộn hơn bed 1 tới 458 s. Ghép thẳng là dán
sáu thời điểm khác nhau vào một khối → gradient giả dọc trục. Quy hết về **thời
điểm tiêm**, đúng chuẩn `DecayCorrection = START` của DICOM:

    f = exp(−λ·Δt) · (1 − exp(−λ·T)) / (λ·T)
        exp(−λ·Δt)          phân rã từ lúc tiêm tới lúc bed bắt đầu
        (1−exp(−λT))/(λT)   hoạt độ trung bình TRONG khung T, không phải tức thời

Bỏ thừa số thứ hai (dùng hoạt độ tức thời lúc bed bắt đầu) lệch ~0,5 % ở đây và
nhiều hơn hẳn với khung dài — và lệch **khác nhau theo bed**, nên nó sống sót
thành gradient trục chứ không tan vào hằng số `K`.

### Vùng chồng là chỗ CẢ HAI bed yếu nhất — nên phải đánh trọng số

Bước bàn 124,26 mm = **đúng 38 plane**, bed dài 47 plane → chồng 9 plane. Nhưng
9 plane đó là plane 38–46 của bed dưới và plane 0–8 của bed trên: **hai đầu yếu
nhất gặp nhau**. Trọng số là `SENS[n]`, sensitivity image của chính STIR — mẫu
số OSEM chia vào mỗi vòng lặp, nên đã gồm norm, dead time, suy giảm và cách
projector lấy mẫu LOR thật, và là ảnh **3D**, tức trọng số theo TỪNG VOXEL. Với
ML Poisson `Var(x̂) ≈ x / sens`, nên trọng số nghịch đảo phương sai chính là
`w ∝ sens`.

Chỉ số plane làm tròn được là vì bước bàn thật sự là số nguyên lần plane; lệch
nửa plane sẽ dồn hai bed vào cùng ô với sai số trục 1,6 mm.
`tests/test_notebook_contract.py` chốt đúng điều đó.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from utils.geometry import PLANE_MM


def injection_epoch(hdr) -> float:
    """Thời điểm tiêm, epoch UTC. Header RDF ghi giờ UTC (DICOM ghi giờ địa phương)."""
    t = dt.datetime.strptime(hdr["radiopharm_start_datetime"][:14], "%Y%m%d%H%M%S")
    return t.replace(tzinfo=dt.timezone.utc).timestamp()


def decay_factor(hdr, t_inj: float) -> float:
    """Hệ số NHÂN đưa ảnh của một bed về hoạt độ tại thời điểm tiêm."""
    lam = np.log(2) / hdr["half_life_s"]
    dt_s = hdr["bed_start_time"] - t_inj
    T = hdr["frame_duration_ms"] / 1000.0
    return 1.0 / (np.exp(-lam * dt_s) * (1 - np.exp(-lam * T)) / (lam * T))


def plane_index(case, beds):
    """`(idx, z0, nz)` — mỗi bed ánh xạ vào những plane nào của khối chung.

    `z = table_position + i·PLANE_MM`, **đúng công thức `attenuation.mu_image`
    dùng để cắt CT cho bed đó**, nên hình học hai bên nhất quán theo cấu tạo chứ
    không phải nhờ trùng hợp.
    """
    nz_bed = int(round(2 * 24 - 1))          # 47 plane một bed
    zs = {n: case.header(n)["table_position_mm"] + np.arange(nz_bed) * PLANE_MM
          for n in beds}
    z0 = min(z[0] for z in zs.values())
    idx = {n: np.rint((zs[n] - z0) / PLANE_MM).astype(int) for n in beds}
    return idx, float(z0), int(max(i[-1] for i in idx.values()) + 1)


def stitch(case, beds, img: dict, sens: dict, verbose: bool = True):
    """Ghép các bed thành một khối toàn thân. Trả `(vol, z0, factors)`.

    `vol` là count/voxel **đã quy về thời điểm tiêm**.
    """
    t_inj = injection_epoch(case.header(beds[0]))
    factors = {n: decay_factor(case.header(n), t_inj) for n in beds}

    if verbose:
        up = (case.header(beds[0])["bed_start_time"] - t_inj) / 60
        inj = dt.datetime.fromtimestamp(t_inj, dt.timezone.utc)
        print(f"tiêm {inj:%Y-%m-%d %H:%M:%S} UTC   "
              f"uptake tới bed {beds[0]}: {up:.1f} phút")
        for n in beds:
            print(f"  bed {n}: × {factors[n]:.4f}")

    idx, z0, nz = plane_index(case, beds)
    shape = (nz,) + img[beds[0]].shape[1:]
    num = np.zeros(shape, dtype=np.float64)
    den = np.zeros_like(num)
    for n in beds:
        w = sens[n].astype(np.float64)
        num[idx[n]] += img[n] * factors[n] * w
        den[idx[n]] += w

    ok = den > 0
    vol = np.zeros_like(num)
    vol[ok] = num[ok] / den[ok]
    vol = vol.astype(np.float32)

    if verbose:
        print(f"\ntoàn thân: {vol.shape}  "
              f"z {z0:.1f} .. {z0 + (nz - 1) * PLANE_MM:.1f} mm")
    return vol, z0, factors


def overlap_report(case, beds, img: dict, factors: dict, out=print):
    """KIỂM chỗ ghép bằng SỐ, đừng chỉ nhìn.

    Ô "tương quan" so hai bản tái tạo **độc lập** của *cùng một đoạn cơ thể*.
    Đặt sai chiều trục hay lệch một bed thì nó sụp ngay, trong khi ảnh ghép vẫn
    trông như một cơ thể.
    """
    idx, _z0, _nz = plane_index(case, beds)
    out(f"\n{'cặp bed':>10} {'plane chồng':>12} {'tương quan':>11} {'tỉ lệ biên độ':>14}")
    rows = []
    for a, b in zip(beds, beds[1:]):
        common = np.intersect1d(idx[a], idx[b])
        if not common.size:
            out(f"{f'{a}-{b}':>10} {0:>12}   (không chồng)")
            continue
        va = img[a][np.searchsorted(idx[a], common)].ravel() * factors[a]
        vb = img[b][np.searchsorted(idx[b], common)].ravel() * factors[b]
        corr = float(np.corrcoef(va, vb)[0, 1])
        ratio = float(vb.sum() / max(va.sum(), 1e-9))
        out(f"{f'{a}-{b}':>10} {common.size:>12} {corr:>11.4f} {ratio:>14.3f}")
        rows.append({"pair": (a, b), "planes": int(common.size),
                     "corr": corr, "ratio": ratio})
    return rows
