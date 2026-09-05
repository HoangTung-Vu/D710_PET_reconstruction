#!/usr/bin/env python3
"""CT DICOM -> NIfTI HU. Ảnh gốc của máy quét, không cắt, không lấy mẫu lại.

    python3 tools/ct_nifti.py <thư mục CT> [ct.nii.gz]

Affine cùng quy ước với `utils/export.write_nifti` (LPS -> RAS, đổi dấu x và y),
nên chồng khít lên các file SUV trong `export/`.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pydicom


def main(src, dst=None):
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
    # NIfTI chỉ diễn tả được bước z đều: thiếu lát thì khoảng cách sẽ sai lặng lẽ.
    assert np.allclose(np.diff(z), dz, atol=1e-3), "thiếu lát / bước z không đều"

    py, px = (float(v) for v in ds[0].PixelSpacing)
    x0, y0 = (float(v) for v in ds[0].ImagePositionPatient[:2])
    affine = np.array([[-px, 0, 0, -x0],
                       [0, -py, 0, -y0],
                       [0, 0, dz, z[0]],
                       [0, 0, 0, 1.0]])

    dst = dst or os.path.join(os.path.dirname(os.path.abspath(src)), "ct.nii.gz")
    img = nib.Nifti1Image(np.transpose(hu, (2, 1, 0)), affine)   # (x, y, z)
    img.header.set_xyzt_units("mm")
    nib.save(img, dst)
    print(f"{len(ds)} lát {hu.shape[1]}x{hu.shape[2]} @ {px:.4f} mm, dz {dz:.4f}"
          f"   z {z[0]:.1f}..{z[-1]:.1f}   HU {hu.min():.0f}..{hu.max():.0f}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
