#!/usr/bin/env python3
import argparse
import os

import nibabel as nib
import numpy as np
import pydicom

AIR_HU = -1024.0


def read_ct(src):
    ds = [pydicom.dcmread(f) for f in
          (os.path.join(src, n) for n in sorted(os.listdir(src)))
          if os.path.isfile(f)]
    ds = [d for d in ds if getattr(d, "Modality", None) == "CT"]
    if not ds:
        raise SystemExit(f"không có lát CT nào trong {src}")
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    iop = [float(v) for v in ds[0].ImageOrientationPatient]
    assert np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-6), f"CT nghiêng: {iop}"

    hu = np.stack([d.pixel_array * float(d.RescaleSlope)
                   + float(d.RescaleIntercept) for d in ds]).astype(np.float32)
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    dz = float(np.median(np.diff(z)))
    assert np.allclose(np.diff(z), dz, atol=1e-3), "thiếu lát / bước z không đều"

    py, px = (float(v) for v in ds[0].PixelSpacing)
    x0, y0 = (float(v) for v in ds[0].ImagePositionPatient[:2])
    affine = np.array([[-px, 0, 0, -x0],
                       [0, -py, 0, -y0],
                       [0, 0, dz, z[0]],
                       [0, 0, 0, 1.0]])
    print(f"{len(ds)} lát {hu.shape[1]}x{hu.shape[2]} @ {px:.4f} mm, dz {dz:.4f}"
          f"   z {z[0]:.1f}..{z[-1]:.1f}   HU {hu.min():.0f}..{hu.max():.0f}")
    return np.transpose(hu, (2, 1, 0)), affine


def resample_like(vol, affine, ref_affine, ref_shape):
    from scipy.ndimage import map_coordinates

    shape = tuple(int(n) for n in ref_shape[:3])
    m = np.linalg.inv(np.asarray(affine, np.float64)) @ np.asarray(ref_affine, np.float64)
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    ij = np.stack([i.ravel(), j.ravel()]).astype(np.float64)
    hi = np.asarray(vol.shape[:3], np.float64)[:, None] - 0.5
    out = np.empty(shape, np.float32)
    for k in range(shape[2]):
        c = m[:3, :2] @ ij + (m[:3, 2] * k + m[:3, 3])[:, None]
        v = map_coordinates(vol, c, order=1, mode="nearest")
        v[np.any((c < -0.5) | (c > hi), axis=0)] = AIR_HU
        out[:, :, k] = v.reshape(shape[:2])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ct_nifti",
        description="CT DICOM series to a NIfTI volume in Hounsfield units")
    ap.add_argument("src", help="thư mục series CT DICOM")
    ap.add_argument("dst", nargs="?",
                    help="file .nii.gz ra; mặc định <thư mục cha của src>/ct.nii.gz")
    ap.add_argument("--like", metavar="NIFTI",
                    help="resample CT lên lưới của ảnh này (vd. PET <case>_ge_suvbw.nii.gz): "
                         "nội suy tuyến tính, ngoài CT là không khí -1024 HU")
    a = ap.parse_args(argv)

    vol, affine = read_ct(a.src)
    if a.like:
        ref = nib.load(a.like)
        vol = resample_like(vol, affine, ref.affine, ref.shape)
        affine = ref.affine

    dst = a.dst or os.path.join(os.path.dirname(os.path.abspath(a.src)), "ct.nii.gz")
    img = nib.Nifti1Image(np.asarray(vol, np.float32), affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, dst)
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
