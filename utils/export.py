"""Export the reconstructed PET volume to NIfTI and DICOM.

The input volume is a STIR array `(z, y, x)`; the z axis increases with
`table_position`, i.e. towards the head — the same direction as DICOM's z.

**There is exactly one flip, and it is not optional.** `utils/attenuation.py`
builds the mu-map in DICOM order and then calls `to_radiological()` (which flips
the y axis) before pouring it into the STIR image. So the y axis of the STIR
image is **reversed** with respect to the patient y of DICOM, and exporting back
to DICOM/NIfTI has to flip it again. That function is its own inverse, so this
module calls it rather than re-implementing the flip.

The x axis is not flipped: the transverse grid of `mu_image` takes DICOM
coordinates directly.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np

from .attenuation import to_radiological

#: SOP Class of PET Image Storage.
PET_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.128"

#: Our own UID root, so generated UIDs never collide with the vendor's.
UID_ROOT = "1.2.826.0.1.3680043.10.1338"


def to_dicom_order(vol: np.ndarray) -> np.ndarray:
    """STIR image `(z, y, x)` -> DICOM patient order (flips y only).

    Calls the very `to_radiological` that `attenuation.mu_image` used rather than
    re-implementing the flip: that function is its own inverse, so calling it
    again undoes it.
    """
    return np.ascontiguousarray(to_radiological(vol))


def grid_origin(nx: int, ny: int, vx: float, vy: float) -> tuple[float, float]:
    """DICOM (x, y) coordinate of voxel [0, 0] within a slice.

    The transverse grid of `attenuation.mu_image` is `(arange(n) - n//2) * v`,
    with the scanner centre at (0, 0) — so the origin is `-(n//2)*v`. The two
    axes are asked for separately: the current grid is square, but reusing one
    size for both would silently shift the origin on a non-square grid.
    """
    return -(nx // 2) * vx, -(ny // 2) * vy


def write_nifti(vol, path, vx, vy, vz, z0):
    """Write `.nii.gz`. The affine is RAS (NIfTI), converted from LPS by negating x, y."""
    import nibabel as nib

    d = to_dicom_order(vol)                       # (z, y, x), DICOM order
    x0, y0 = grid_origin(d.shape[2], d.shape[1], vx, vy)
    data = np.transpose(d, (2, 1, 0))             # nibabel wants (i, j, k) = (x, y, z)
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
    """Write one PET DICOM series, one file per slice.

    Writes every tag a viewer needs in order to **compute SUV itself**:
    `Units = BQML`, the dose and injection time, the weight, `DecayCorrection`.
    Miss any one of them and the viewer shows raw Bq/mL or refuses to compute SUV.

    `FrameOfReferenceUID` is taken from the exam itself (`sop_instance_uid` in
    the RDF header), so this PET image **lines up with the CT automatically** in
    any viewer — exactly the identity the notebook checked in the attenuation
    cell.
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    d = to_dicom_order(vol)
    nz, ny, nx = d.shape
    x0, y0 = grid_origin(nx, ny, vx, vy)
    os.makedirs(out_dir, exist_ok=True)

    # PET stores int16 + RescaleSlope; 4 significant digits at the peak is plenty.
    # The peak must be taken over the FINITE part: `nanmax` skips NaN but NOT
    # ±inf, and a single inf voxel makes slope = inf -> RescaleSlope written as
    # "inf" (invalid DS) with every pixel divided by inf becoming 0 — the whole
    # series lost, silently.
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
        # The transfer syntax is declared in file_meta; `is_little_endian` /
        # `is_implicit_VR` are the old API, dropped entirely in pydicom 4.
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
        ds.DecayCorrection = "START"       # decay-corrected back to injection time
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
# `d710 export` — apply K, then write to disk.  Reads the `<case>/recon.npz`
# that `d710 osem` left behind, so nothing has to be reconstructed again.
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
                    help="(Bq/mL)/(count/voxel). Mặc định: K riêng của export "
                         "($D710_K hoặc quant.K_EXPORT); không có thì WCC của "
                         "chính exam; không có nữa thì CHẶN TRÊN theo liều.")
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
    if K is not None:
        print(f"K từ --K = {K:,.2f}")
    if K is None:
        K = quant.k_export()
        if K is not None:
            print(f"K riêng của export = {K:,.2f}"
                  + ("   (D710_K)" if os.environ.get("D710_K") else
                     "   (quant.K_EXPORT)"))
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

    # --- write -------------------------------------------------------------
    C.export.mkdir(parents=True, exist_ok=True)
    if args.format in ("nifti", "both"):
        for name, arr in (("bqml", r["bqml"]), ("suvbw", r["suv"])):
            p = write_nifti(arr, str(C.export / f"{C.name}_{name}.nii.gz"),
                            vox[2], vox[1], vox[0], z0)
            print(f"ghi {p}")
    if args.format in ("dicom", "both"):
        # The DICOM deliberately carries ONLY Bq/mL: the viewer computes SUV
        # itself from Units=BQML + dose + weight, and the reader can switch
        # between bw/lbm/bsa.  Baking SUV into the DICOM would hard-code one
        # choice.
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
