"""Xuất khối PET đã tái tạo ra NIfTI và DICOM.

Khối vào là mảng STIR `(z, y, x)`; trục z tăng theo `table_position`, tức tăng
về phía đầu — cùng chiều với z của DICOM.

**Một phép lật duy nhất, và nó không tuỳ chọn.** `utils/attenuation.py`
dựng mu-map theo thứ tự DICOM rồi gọi `to_radiological()` (lật trục y) trước khi
đổ vào ảnh STIR. Nên trục y của ảnh STIR **ngược** với y của bệnh nhân trong
DICOM, và muốn xuất ra DICOM/NIfTI thì phải lật lại. Hàm đó tự nghịch đảo, nên ở
đây dùng đúng nó chứ không viết lại phép lật.

Trục x thì không lật: lưới ngang của `mu_image` lấy thẳng toạ độ DICOM.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np

from .attenuation import to_radiological

#: SOP Class của PET Image Storage.
PET_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.128"

#: Gốc UID riêng, để UID sinh ra không đụng của hãng.
UID_ROOT = "1.2.826.0.1.3680043.10.1338"


def to_dicom_order(vol: np.ndarray) -> np.ndarray:
    """Ảnh STIR `(z, y, x)` -> thứ tự bệnh nhân của DICOM (chỉ lật y).

    Dùng đúng `to_radiological` mà `attenuation.mu_image` đã dùng, chứ không
    viết lại phép lật: hàm đó tự nghịch đảo, nên gọi nó lần nữa là hoàn tác.
    """
    return np.ascontiguousarray(to_radiological(vol))


def grid_origin(nx: int, ny: int, vx: float, vy: float) -> tuple[float, float]:
    """Toạ độ DICOM (x, y) của voxel [0, 0] trong một lát.

    Lưới ngang của `attenuation.mu_image` là `(arange(n) - n//2) * v`, tâm máy
    ở (0, 0) — nên gốc là `-(n//2)*v`. Hai trục hỏi riêng: lưới hiện tại vuông,
    nhưng lấy một cỡ dùng cho cả hai thì lưới không vuông sẽ lệch gốc mà không
    báo gì.
    """
    return -(nx // 2) * vx, -(ny // 2) * vy


def write_nifti(vol, path, vx, vy, vz, z0):
    """Ghi `.nii.gz`. Affine theo RAS (NIfTI), đổi từ LPS bằng cách đảo dấu x, y."""
    import nibabel as nib

    d = to_dicom_order(vol)                       # (z, y, x), thứ tự DICOM
    x0, y0 = grid_origin(d.shape[2], d.shape[1], vx, vy)
    data = np.transpose(d, (2, 1, 0))             # nibabel muốn (i, j, k) = (x, y, z)
    affine = np.array([[-vx, 0.0, 0.0, -x0],
                       [0.0, -vy, 0.0, -y0],
                       [0.0, 0.0, vz, z0],
                       [0.0, 0.0, 0.0, 1.0]])
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    img.header.set_xyzt_units("mm")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    nib.save(img, path)
    return path


def _dcm_dt(epoch: float) -> tuple[str, str]:
    t = dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    return t.strftime("%Y%m%d"), t.strftime("%H%M%S.%f")[:13]


def write_dicom(vol, out_dir, hdr, vx, vy, vz, z0, series_desc="OSEM SIRF BQML",
                units="BQML", series_number=901):
    """Ghi một series PET DICOM, một file mỗi lát.

    Ghi đủ những tag mà một máy đọc ảnh cần để **tự tính SUV**: `Units = BQML`,
    liều và thời điểm tiêm, cân nặng, `DecayCorrection`. Thiếu bất kỳ cái nào là
    viewer sẽ hiện Bq/mL thô hoặc từ chối tính SUV.

    `FrameOfReferenceUID` lấy đúng của exam (`sop_instance_uid` trong header
    RDF), nên ảnh PET này **tự khớp toạ độ với CT** trong mọi viewer — chính là
    đồng nhất thức mà notebook đã kiểm ở cell suy giảm.
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    d = to_dicom_order(vol)
    nz, ny, nx = d.shape
    x0, y0 = grid_origin(nx, ny, vx, vy)
    os.makedirs(out_dir, exist_ok=True)

    # PET lưu int16 + RescaleSlope; giữ 4 chữ số có nghĩa ở đỉnh là quá đủ.
    # Đỉnh phải lấy trên phần HỮU HẠN: `nanmax` bỏ qua NaN nhưng KHÔNG bỏ ±inf,
    # mà một voxel inf thì slope = inf -> RescaleSlope ghi "inf" (sai chuẩn DS)
    # và mọi pixel chia cho inf thành 0 — mất trắng cả series mà không báo gì.
    finite = d[np.isfinite(d)]
    peak = float(finite.max()) if finite.size else 0.0
    slope = (peak / 32000.0) if peak > 0 else 1.0

    series_uid = generate_uid(prefix=UID_ROOT + ".")
    study_uid = hdr.get("study_instance_uid") or generate_uid(prefix=UID_ROOT + ".")
    acq_date, acq_time = _dcm_dt(hdr["bed_start_time"])
    inj = dt.datetime.strptime(hdr["radiopharm_start_datetime"][:14], "%Y%m%d%H%M%S")

    paths = []
    for i in range(nz):
        fm = Dataset()
        fm.MediaStorageSOPClassUID = PET_SOP_CLASS
        fm.MediaStorageSOPInstanceUID = generate_uid(prefix=UID_ROOT + ".")
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        # Cú pháp truyền đã khai trong file_meta; `is_little_endian` /
        # `is_implicit_VR` là API cũ, pydicom 4 bỏ hẳn.
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)

        ds.SOPClassUID = PET_SOP_CLASS
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.Modality = "PT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = hdr["sop_instance_uid"]
        ds.SeriesNumber = series_number
        ds.InstanceNumber = i + 1
        ds.SeriesDescription = series_desc
        ds.Manufacturer = hdr.get("manufacturer", "")
        ds.ManufacturerModelName = hdr.get("model_name", "")
        ds.InstitutionName = hdr.get("institution", "")
        ds.StudyDescription = hdr.get("study_description", "")
        ds.AccessionNumber = hdr.get("accession_number", "")

        ds.PatientName = hdr.get("patient_name", "")
        ds.PatientID = hdr.get("patient_id", "")
        ds.PatientBirthDate = hdr.get("patient_birth_date", "")
        ds.PatientSex = ""
        ds.PatientWeight = float(hdr["patient_weight_kg"])
        ds.PatientSize = float(hdr.get("patient_height_m", 0) or 0)

        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = ds.ContentDate = acq_date
        ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = ds.ContentTime = acq_time

        ds.Rows, ds.Columns = ny, nx
        ds.PixelSpacing = [f"{vy:.6f}", f"{vx:.6f}"]
        ds.SliceThickness = f"{vz:.6f}"
        ds.SpacingBetweenSlices = f"{vz:.6f}"
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImagePositionPatient = [f"{x0:.4f}", f"{y0:.4f}", f"{z0 + i * vz:.4f}"]
        ds.SliceLocation = f"{z0 + i * vz:.4f}"

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = f"{slope:.10g}"
        ds.RescaleIntercept = "0"
        ds.Units = units
        ds.DecayCorrection = "START"       # đã hiệu chỉnh phân rã về lúc tiêm
        ds.CorrectedImage = ["DECY", "ATTN", "SCAT", "DTIM", "NORM", "RAN"]
        ds.SeriesType = ["STATIC", "IMAGE"]
        ds.ImageType = ["DERIVED", "PRIMARY"]
        ds.ActualFrameDuration = int(hdr["frame_duration_ms"])
        ds.RadiopharmaceuticalStartTime = inj.strftime("%H%M%S.00")

        rp = Dataset()
        rp.Radiopharmaceutical = hdr.get("radiopharmaceutical", "FDG")
        rp.RadionuclideTotalDose = float(
            (hdr["dose_mbq"] - hdr.get("residual_dose_mbq", 0.0)) * 1e6)   # Bq
        rp.RadionuclideHalfLife = float(hdr["half_life_s"])
        rp.RadionuclidePositronFraction = float(hdr.get("positron_fraction", 0.967))
        rp.RadiopharmaceuticalStartTime = inj.strftime("%H%M%S.00")
        rp.RadiopharmaceuticalStartDateTime = inj.strftime("%Y%m%d%H%M%S.00")
        ds.RadiopharmaceuticalInformationSequence = [rp]

        sl = np.nan_to_num(d[i], nan=0.0, posinf=0.0, neginf=0.0) / slope
        ds.PixelData = np.clip(np.rint(sl), -32768, 32767).astype("<i2").tobytes()

        p = os.path.join(out_dir, f"PET_{i + 1:04d}.dcm")
        ds.save_as(p, enforce_file_format=True)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# `d710 export` — áp K rồi ghi ra đĩa.  Đọc `<ca>/recon.npz` mà `d710 osem`
# để lại, nên không phải tái tạo lại gì.
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    """`python3 -m utils.export --case ped --format nifti`."""
    import argparse

    from . import quant
    from .paths import case as get_case

    ap = argparse.ArgumentParser(
        prog="export", description=main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", help="gốc đầu ra; mặc định $D710_OUT")
    ap.add_argument("--format", choices=("nifti", "dicom", "both"), default="both")
    ap.add_argument("--K", type=float,
                    help="(Bq/mL)/(count/voxel). Mặc định: WCC của chính exam; "
                         "không có thì lấy CHẶN TRÊN theo liều.")
    args = ap.parse_args(argv)

    C = get_case(args.case, args.out)
    if not C.recon.exists():
        raise SystemExit("error: chưa có %s -- chạy `d710 osem --case %s` trước"
                         % (C.recon, C.name))

    z = np.load(C.recon, allow_pickle=False)
    vol, z0, vox = z["vol"], float(z["z0"]), [float(v) for v in z["vox"]]
    beds = [int(b) for b in z["beds"]]
    hdr = C.header(beds[0])

    # --- K ---------------------------------------------------------------
    K = args.K
    if K is None:
        f = quant.wcc_activity_factor(C, beds[0])
        if f is not None:
            K = quant.k_from_wcc(f)
            print(f"  hrActivityFactor = {f:.6f}   (× {quant.WCC_UNIT_SCALE:g}"
                  f" = {K:,.2f})")
    k_dose = quant.k_from_dose(vol, vox, quant.dose_bq(hdr))
    print(f"K chặn trên (100 % liều trong FOV) = {k_dose:,.1f}")
    if K is None:
        K = k_dose
        print("không có WCC -> lấy mốc liều")

    r = quant.report(vol, K, hdr, vox)

    # --- ghi --------------------------------------------------------------
    C.export.mkdir(parents=True, exist_ok=True)
    if args.format in ("nifti", "both"):
        for name, arr in (("bqml", r["bqml"]), ("suvbw", r["suv"])):
            p = write_nifti(arr, str(C.export / f"{C.name}_{name}.nii.gz"),
                            vox[2], vox[1], vox[0], z0)
            print(f"ghi {p}")
    if args.format in ("dicom", "both"):
        # DICOM cố ý CHỈ có Bq/mL: viewer tự tính SUV từ Units=BQML + liều +
        # cân nặng, và người đọc đổi được bw/lbm/bsa.  Ghim SUV vào DICOM là
        # khoá cứng một lựa chọn.
        n_it, n_sub = int(z["n_iterations"]), int(z["n_subsets"])
        paths = write_dicom(r["bqml"], str(C.export / "dicom"), hdr,
                            vox[2], vox[1], vox[0], z0,
                            series_desc=f"OSEM SIRF {n_it}x{n_sub} BQML")
        print(f"ghi {len(paths)} file DICOM -> {C.export / 'dicom'}")
        print("   Units=BQML + liều + cân nặng + DecayCorrection=START -> "
              "viewer tự tính SUV;\n   FrameOfReferenceUID = của exam -> "
              "tự chồng khít CT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
