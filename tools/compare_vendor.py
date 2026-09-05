#!/usr/bin/env python3
"""So sánh `recon.npz` với ảnh PET BQML của chính GE, và đo `K`.

    python3 -m tools.compare_vendor --case chuong \
        --vendor ~/UET/Handson_PET_CT_Reconstruction/data/cases/20260806_FDG26080604_ok/dicom/PT_s012_PET_WB_3D_AC

`K` là (Bq/mL) trên (count/voxel) — đúng đơn vị mà `d710 export --K` cần.

Cách đo: lấy mẫu thể tích của pipeline lên ĐÚNG lưới của vendor (cùng hệ toạ
độ LPS máy quét), rồi khớp một hệ số duy nhất qua gốc toạ độ. Không căn ảnh,
không xoay — nếu hai ảnh lệch nhau về hình học thì tương quan sẽ tụt và đó
chính là tín hiệu cần thấy, chứ không phải thứ nên "sửa" bằng cách căn lại.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.attenuation import to_radiological
from utils.export import grid_origin
from utils.paths import case as get_case


def read_vendor(d):
    """Đọc một series PET DICOM -> (vol Bq/mL [z,y,x], z[], x0, y0, px, py, meta)."""
    import pydicom

    sl = []
    for fn in os.listdir(d):
        ds = pydicom.dcmread(os.path.join(d, fn))
        sl.append(ds)
    if not sl:
        raise SystemExit(f"error: không có file DICOM nào trong {d}")
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    units = str(getattr(sl[0], "Units", "?"))
    if units != "BQML":
        print(f"  ⚠ Units = {units!r}, không phải BQML — K đo ra sẽ sai đơn vị")

    vol = np.stack([s.pixel_array.astype(np.float32)
                    * float(getattr(s, "RescaleSlope", 1.0))
                    + float(getattr(s, "RescaleIntercept", 0.0)) for s in sl])
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl], dtype=np.float64)
    ipp = sl[0].ImagePositionPatient
    px, py = (float(v) for v in sl[0].PixelSpacing)
    meta = {
        "units": units,
        "recon": str(getattr(sl[0], "ReconstructionMethod", "?")),
        "corrections": list(getattr(sl[0], "CorrectedImage", [])),
        "n": len(sl), "shape": tuple(vol.shape),
    }
    return vol, z, float(ipp[0]), float(ipp[1]), px, py, meta


def resample_to(vol, z0, vox, zt, x0t, y0t, pxt, pyt, nzt, nyt, nxt):
    """Lấy mẫu vol (STIR order, count/voxel) lên lưới đích, trả về DICOM order."""
    from scipy.ndimage import map_coordinates

    d = np.ascontiguousarray(to_radiological(vol))          # (z, y, x) DICOM order
    nz, ny, nx = d.shape
    vz, vy, vx = vox                                        # mm
    x0s, y0s = grid_origin(nx, ny, vx, vy)

    # toạ độ đích (mm, LPS) -> chỉ số nguồn (voxel)
    zt_ = (zt - z0) / vz
    yt_ = ((y0t + np.arange(nyt) * pyt) - y0s) / vy
    xt_ = ((x0t + np.arange(nxt) * pxt) - x0s) / vx
    gz, gy, gx = np.meshgrid(zt_, yt_, xt_, indexing="ij")
    out = map_coordinates(d, np.stack([gz, gy, gx]), order=1,
                          mode="constant", cval=0.0)
    inside = ((gz >= 0) & (gz <= nz - 1) & (gy >= 0) & (gy <= ny - 1)
              & (gx >= 0) & (gx <= nx - 1))
    return out.astype(np.float32), inside


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_vendor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--vendor", required=True, help="thư mục series PET BQML của GE")
    ap.add_argument("--out", help="gốc đầu ra; mặc định $D710_OUT")
    ap.add_argument("--thresh", type=float, default=1000.0,
                    help="ngưỡng Bq/mL để lấy voxel vào phép khớp (mặc định 1000)")
    ap.add_argument("--save", help="ghi thể tích đã căn lưới ra .npz để xem sau")
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    if not C.recon.exists():
        raise SystemExit(f"error: chưa có {C.recon} — chạy `d710 osem --case {C.name}` trước")

    z = np.load(C.recon, allow_pickle=False)
    vol, z0, vox = z["vol"], float(z["z0"]), [float(v) for v in z["vox"]]
    beds = [int(b) for b in z["beds"]]
    print(f"pipeline : {vol.shape} (z,y,x)  voxel {vox} mm  z0 {z0:.2f}  beds {beds}")

    ven, zt, x0t, y0t, pxt, pyt, meta = read_vendor(args.vendor)
    print(f"vendor   : {meta['shape']} {meta['units']}  {meta['recon']}  "
          f"z {zt[0]:.1f}..{zt[-1]:.1f}")
    print(f"           corrections: {','.join(meta['corrections'])}")

    ours, inside = resample_to(vol, z0, vox, zt, x0t, y0t, pxt, pyt, *ven.shape)

    ov = inside.mean()
    print(f"\nchồng lấn hình học: {ov*100:.1f} % voxel của vendor nằm trong FOV pipeline")
    # Chạy một phần bed thì chồng lấn thấp là ĐÚNG, không phải lỗi -- nên mốc
    # cảnh báo phải tính theo số bed đã dựng, chứ không phải theo toàn ảnh.
    span_ours = vol.shape[0] * vox[0]
    span_ven = float(zt[-1] - zt[0]) + 1e-9
    if ov < 0.5 * min(1.0, span_ours / span_ven) * 0.8:
        print(f"  ⚠ thấp hơn nhiều so với mức chờ đợi cho {len(beds)} bed "
              f"({span_ours:.0f} mm / {span_ven:.0f} mm) — nghi lệch bed/trục, "
              "đừng tin K bên dưới")
    elif ov < 0.5:
        print(f"  (bình thường: mới dựng {len(beds)} bed, phủ {span_ours:.0f} mm "
              f"trên {span_ven:.0f} mm của vendor)")

    m = inside & (ven > args.thresh) & (ours > 0)
    n = int(m.sum())
    if n < 1000:
        raise SystemExit(f"error: chỉ {n} voxel qua ngưỡng — hạ --thresh")
    v, o = ven[m].astype(np.float64), ours[m].astype(np.float64)

    k_ls = float((v * o).sum() / (o * o).sum())     # bình phương tối thiểu qua gốc
    k_med = float(np.median(v / o))                 # trung vị tỉ số, chịu nhiễu tốt
    k_tot = float(v.sum() / o.sum())                # tỉ số tổng hoạt độ
    r = float(np.corrcoef(v, o)[0, 1])

    print(f"\nvoxel dùng để khớp: {n:,} (Bq/mL > {args.thresh:g})")
    print(f"  tương quan r          = {r:.4f}")
    print(f"  K (bình phương tối thiểu) = {k_ls:12,.1f}")
    print(f"  K (trung vị tỉ số)        = {k_med:12,.1f}")
    print(f"  K (tổng hoạt độ)          = {k_tot:12,.1f}")
    spread = (max(k_ls, k_med, k_tot) / min(k_ls, k_med, k_tot) - 1) * 100
    print(f"  ba ước lượng lệch nhau {spread:.1f} %")
    if spread > 20:
        print("  ⚠ lệch > 20 % ⇒ sai lệch KHÔNG phải một hệ số toàn cục;"
              " đừng chốt K, đi tìm thành phần phụ thuộc không gian trước")

    resid = v / (o * k_ls)
    print(f"\n  vendor/(pipeline·K): trung vị {np.median(resid):.3f}  "
          f"p05 {np.percentile(resid,5):.3f}  p95 {np.percentile(resid,95):.3f}")
    print(f"\n  dùng:  d710 export --case {C.name} --K {k_ls:.1f}")

    if args.save:
        np.savez_compressed(args.save, vendor=ven, pipeline=ours,
                            inside=inside, z=zt, K_ls=k_ls)
        print(f"  đã ghi {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
