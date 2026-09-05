#!/usr/bin/env python3
"""Một series PET DICOM (BQML) -> một file NIfTI SUVbw.

    python3 -m tools.dicom_suv --all              # ảnh của GE, mọi ca
    python3 -m tools.dicom_suv --case fdg26081008
    python3 -m tools.dicom_suv --dicom <thư mục PT> --out suv.nii.gz

Dùng để so ảnh bằng **SUV** thay vì Bq/mL. SUV bỏ đi hai thứ khác nhau giữa các
ca — liều tiêm và cân nặng — nên hai ca mới đặt cạnh nhau được, và thang SUV là
thang người đọc phim thực sự nhìn.

    SUVbw = C(Bq/mL) / (liều còn lại lúc quét / cân nặng)

**Đồng hồ: chỉ trừ trong CÙNG một file, không bao giờ trộn hai nguồn.** Header
RDF ghi giờ UTC còn DICOM ghi giờ địa phương (lệch +7 h ở đây), nên lấy giờ quét
từ file này rồi trừ giờ tiêm từ file kia là sai đúng 7 h — tức là SUV sai ~1,5
lần. Cả `SeriesTime` lẫn `RadiopharmaceuticalStartTime` đều đọc từ chính series
đang xử lý, nên chênh lệch luôn đúng dù series đó của ai.

**Mốc thời gian.** `DecayCorrection = START` nghĩa là giá trị điểm ảnh đã quy về
lúc BẮT ĐẦU series, nên mẫu số phải là liều còn lại **ở đúng lúc đó**. Kiểm trên
dữ liệu: `SeriesTime` của GE = 11:53:16, còn giờ bắt đầu bed 1 mà pipeline tính
ra từ RDF là 11:53:18 — lệch 2 s. Vì `d710 export` ghi `SeriesTime` theo đúng
quy ước ấy, một mã này chạy được cho cả ảnh của GE lẫn ảnh của mình.

Chạy trên `export/dicom` của chính mình thì phải ra đúng
`<case>_suvbw.nii.gz` mà `d710 export` đã ghi — `--ours` làm việc đó, và đó là
phép thử rằng file DICOM giao cho bác sĩ đọc SUV ra đúng như mình tính.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.paths import out_root
from utils.quant import suv_bw

#: Ngoài dải này thì gần như chắc chắn sai đồng hồ, không phải ca lạ.
UPTAKE_MIN_OK = (5.0, 300.0)


def read_series(d: str):
    """`(vol Bq/mL [z,y,x], meta)` — đọc, sắp theo z, áp rescale."""
    import pydicom

    files = sorted(glob.glob(os.path.join(d, "*")))
    sl = []
    for f in files:
        if os.path.isdir(f):
            continue
        try:
            sl.append(pydicom.dcmread(f))
        except Exception:
            continue                      # DICOMDIR, README, ... bỏ qua
    if not sl:
        raise SystemExit(f"error: không đọc được file DICOM nào trong {d}")
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    d0 = sl[0]

    units = str(getattr(d0, "Units", "?"))
    if units != "BQML":
        raise SystemExit(
            f"error: Units = {units!r}, không phải BQML — không tính SUV được.\n"
            f"  (ảnh CNTS chưa nhân hệ số hiệu chuẩn thì không có đơn vị hoạt độ)")

    vol = np.stack([s.pixel_array.astype(np.float32)
                    * float(getattr(s, "RescaleSlope", 1.0))
                    + float(getattr(s, "RescaleIntercept", 0.0)) for s in sl])

    iop = [float(v) for v in getattr(d0, "ImageOrientationPatient",
                                     [1, 0, 0, 0, 1, 0])]
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-6):
        raise SystemExit(f"error: series không phải axial chuẩn (IOP {iop}) — "
                         "affine bên dưới chỉ đúng cho axial")

    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    dz = float(np.median(np.diff(z))) if len(z) > 1 else \
        float(getattr(d0, "SliceThickness", 1.0))
    if len(z) > 2 and not np.allclose(np.diff(z), dz, atol=1e-3):
        print(f"  ⚠ khoảng cách lát không đều (dz {np.diff(z).min():.4f}.."
              f"{np.diff(z).max():.4f}) — affine dùng trung vị {dz:.4f}")

    py, px = (float(v) for v in d0.PixelSpacing)      # [hàng, cột] = [y, x]
    return vol, {
        "ds": d0, "n": len(sl), "shape": vol.shape,
        "x0": float(d0.ImagePositionPatient[0]),
        "y0": float(d0.ImagePositionPatient[1]),
        "z0": float(z[0]), "px": px, "py": py, "dz": dz,
        "decay": str(getattr(d0, "DecayCorrection", "?")),
        "recon": str(getattr(d0, "ReconstructionMethod", "?")),
        "desc": str(getattr(d0, "SeriesDescription", "?")),
    }


def _dt(date: str, time: str) -> dt.datetime:
    return dt.datetime.strptime(date + time[:6].ljust(6, "0"), "%Y%m%d%H%M%S")


def scan_and_injection(ds, which: str = "series"):
    """`(giờ quét, giờ tiêm, T½ s, liều Bq, cân nặng kg)` — TẤT CẢ từ một file."""
    date = str(getattr(ds, "SeriesDate", "") or getattr(ds, "AcquisitionDate", ""))
    tmap = {"series": "SeriesTime", "acquisition": "AcquisitionTime",
            "content": "ContentTime"}
    tag = tmap[which]
    tstr = str(getattr(ds, tag, "") or "")
    if not date or not tstr:
        raise SystemExit(f"error: series thiếu {tag}/SeriesDate — "
                         f"thử --time acquisition")
    scan = _dt(date, tstr)

    try:
        rp = ds.RadiopharmaceuticalInformationSequence[0]
    except Exception:
        raise SystemExit("error: không có RadiopharmaceuticalInformationSequence "
                         "— thiếu liều và giờ tiêm, không tính SUV được")
    if getattr(rp, "RadiopharmaceuticalStartDateTime", None):
        inj = dt.datetime.strptime(
            str(rp.RadiopharmaceuticalStartDateTime)[:14], "%Y%m%d%H%M%S")
    else:
        inj = _dt(date, str(rp.RadiopharmaceuticalStartTime))
        # Tiêm trước nửa đêm, quét sau: chỉ xảy ra khi mượn SeriesDate.
        if inj > scan:
            inj -= dt.timedelta(days=1)

    dose = float(rp.RadionuclideTotalDose)              # Bq
    half = float(rp.RadionuclideHalfLife)               # s
    w = float(getattr(ds, "PatientWeight", 0) or 0)
    if w <= 0:
        raise SystemExit("error: PatientWeight rỗng — SUVbw cần cân nặng")
    return scan, inj, half, dose, w


def to_suv(vol, ds, which: str = "series", out=print):
    """Bq/mL -> SUVbw. Trả về `(suv, thông tin)`."""
    scan, inj, half, dose, w = scan_and_injection(ds, which)
    uptake = (scan - inj).total_seconds()
    decay_tag = str(getattr(ds, "DecayCorrection", "?"))

    if decay_tag == "START":
        # Điểm ảnh đã quy về lúc bắt đầu quét -> mẫu số là liều CÒN LẠI lúc đó.
        dose_ref = dose * 2 ** (-uptake / half)
    elif decay_tag == "ADMIN":
        # Đã quy về lúc tiêm -> mẫu số là toàn bộ liều.
        dose_ref = dose
    else:
        raise SystemExit(
            f"error: DecayCorrection = {decay_tag!r}. SUV chỉ có nghĩa trên ảnh"
            " đã hiệu chỉnh phân rã; ảnh NONE thì phải tái tạo lại.")

    out(f"  giờ quét {scan:%Y-%m-%d %H:%M:%S}   tiêm {inj:%H:%M:%S}   "
        f"hấp thu {uptake / 60:.1f} min")
    out(f"  liều {dose / 1e6:.1f} MBq -> còn {dose_ref / 1e6:.1f} MBq lúc quét"
        f"   ({w:.0f} kg, T½ {half:.1f} s, DecayCorrection={decay_tag})")
    if not UPTAKE_MIN_OK[0] <= uptake / 60 <= UPTAKE_MIN_OK[1]:
        out(f"  ⚠ hấp thu {uptake / 60:.1f} min nằm ngoài "
            f"{UPTAKE_MIN_OK[0]:.0f}-{UPTAKE_MIN_OK[1]:.0f} min — gần như chắc"
            " chắn lệch đồng hồ, ĐỪNG tin SUV bên dưới")

    suv = suv_bw(vol, dose_ref, w)
    return suv.astype(np.float32), {
        "uptake_min": uptake / 60, "dose_mbq": dose / 1e6,
        "dose_ref_mbq": dose_ref / 1e6, "weight_kg": w,
        "half_life_s": half, "decay_correction": decay_tag,
        "scan_time": scan.isoformat(), "injection_time": inj.isoformat(),
    }


def write_nifti(data_zyx, path, m):
    """Ghi `.nii.gz`. Affine RAS dựng từ ImagePositionPatient THẬT của series.

    Cùng quy ước với `utils/export.write_nifti` (LPS -> RAS bằng cách đổi dấu x,
    y), nên ảnh của GE và ảnh của mình nằm chung một không gian và chồng lên
    nhau được mà không cần căn gì thêm.
    """
    import nibabel as nib

    data = np.transpose(np.ascontiguousarray(data_zyx), (2, 1, 0))   # (x, y, z)
    affine = np.array([[-m["px"], 0.0, 0.0, -m["x0"]],
                       [0.0, -m["py"], 0.0, -m["y0"]],
                       [0.0, 0.0, m["dz"], m["z0"]],
                       [0.0, 0.0, 0.0, 1.0]])
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    img.header.set_xyzt_units("mm")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    nib.save(img, path)
    return path


def convert(dicom_dir: str, out_path: str, which: str = "series") -> dict:
    print(f"\n{dicom_dir}")
    vol, m = read_series(dicom_dir)
    print(f"  {m['n']} lát {m['shape'][1]}x{m['shape'][2]} @ "
          f"{m['px']:.4f} mm, dz {m['dz']:.4f}   '{m['desc']}'  {m['recon']}")
    suv, info = to_suv(vol, m["ds"], which)

    body = suv > 0.02 * np.percentile(suv, 99.9)      # ngưỡng tương đối
    print(f"  SUVbw  trung vị thân {np.median(suv[body]):.3f}   "
          f"p95 {np.percentile(suv[body], 95):.2f}   max {suv.max():.1f}")
    p = write_nifti(suv, out_path, m)
    print(f"  -> {p}")
    info.update({"dicom": os.path.abspath(dicom_dir), "nifti": os.path.abspath(p),
                 "shape": list(m["shape"]), "series_description": m["desc"],
                 "reconstruction": m["recon"],
                 "suv_body_median": float(np.median(suv[body])),
                 "suv_max": float(suv.max())})
    return info


def vendor_dir(root, case: str) -> str:
    """Series BQML của GE cho một ca, lấy từ sidecar `tools.compare_vendor` để lại."""
    for name in ("calib_sino.json", "calib_lm.json"):
        p = root / case / name
        if p.exists():
            with open(p) as f:
                v = json.load(f).get("vendor")
            if v and os.path.isdir(v):
                return v
    raise SystemExit(
        f"error: không biết ảnh GE của {case} nằm đâu — thiếu "
        f"{root / case}/calib_sino.json.\n"
        f"  chỉ ra thẳng bằng --dicom <thư mục PT_s012...>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dicom_suv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dicom", help="một series PET DICOM bất kỳ")
    g.add_argument("--case", help="ảnh GE của ca này (đường dẫn lấy từ calib_*.json)")
    g.add_argument("--all", action="store_true", help="ảnh GE của mọi ca trong $D710_OUT")
    ap.add_argument("--out", help="file .nii.gz ra; mặc định "
                                  "<case>/export/<case>_ge_suvbw.nii.gz")
    ap.add_argument("--out-root", help="gốc đầu ra; mặc định $D710_OUT")
    ap.add_argument("--ours", action="store_true",
                    help="chuyển cả export/dicom và export/dicom_lm của mình — "
                         "phải ra đúng <case>_suvbw.nii.gz mà d710 export đã ghi")
    ap.add_argument("--time", choices=("series", "acquisition", "content"),
                    default="series",
                    help="tag nào là mốc quy đổi phân rã (mặc định SeriesTime, "
                         "đúng cho GE và cho ảnh của mình)")
    args = ap.parse_args(argv)
    if args.out and (args.all or args.ours):
        # Một --out cho nhiều series thì series sau đè series trước, im lặng.
        ap.error("--out chỉ dùng cho MỘT series; bỏ nó đi để mỗi ca tự đặt tên")

    if args.dicom:
        out = args.out or os.path.join(os.path.dirname(
            os.path.abspath(args.dicom)), "suvbw.nii.gz")
        convert(args.dicom, out, args.time)
        return 0

    root = out_root(args.out_root)
    if args.all:
        cases = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob(str(root / "*" / "calib_sino.json")))
        if not cases:
            raise SystemExit(f"error: không ca nào dưới {root} có calib_sino.json")
    else:
        cases = [args.case]

    made = []
    for c in cases:
        exp = root / c / "export"
        made.append(convert(vendor_dir(root, c),
                            args.out or str(exp / f"{c}_ge_suvbw.nii.gz"),
                            args.time))
        if args.ours:
            for tag, sub in (("", "dicom"), ("_lm", "dicom_lm")):
                d = exp / sub
                if d.is_dir():
                    made.append(convert(str(d),
                                        str(exp / f"{c}{tag}_suvbw_fromdicom.nii.gz"),
                                        args.time))

    print(f"\n{len(made)} file SUV. So sánh trong cùng một thư mục export/:")
    for c in cases:
        print(f"  {c}:")
        for f in sorted(glob.glob(str(root / c / "export" / "*suvbw*.nii.gz"))):
            print(f"    {os.path.basename(f)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
