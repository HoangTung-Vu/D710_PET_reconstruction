"""A synthetic CT series, because the real ones carry PHI.

`background/attenuation.load` only reads eight tags, so this writes exactly
those and nothing else.  The phantom is a water cylinder in air with a denser
off-centre insert, which is what the orientation tests need: a volume that is
**not** symmetric in y tells a y-flip apart from the identity, and the real
NEMA phantom cannot.
"""

from __future__ import annotations

import os

import numpy as np

FRAME_OF_REFERENCE = "1.2.826.0.1.3680043.10.1338.99.1"
DEFAULT_DZ = 3.2699997
DEFAULT_PIXEL_MM = 1.3672


def volume(n_slices=20, n=64, pixel_mm=DEFAULT_PIXEL_MM):
    """HU volume `[slice, row, col]` -- air, water cylinder, bone insert."""
    c = (np.arange(n) - n // 2) * pixel_mm
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(xx, yy)
    hu = np.full((n, n), -1000.0, dtype=np.float32)
    hu[r < 0.30 * n * pixel_mm] = 0.0                       # water
    # An insert at +y only: the discriminator for a y-flip.
    hu[(np.hypot(xx, yy - 0.15 * n * pixel_mm) < 0.06 * n * pixel_mm)] = 800.0
    vol = np.repeat(hu[None], n_slices, axis=0)
    # A z-dependent step, so a z shift is visible too.
    vol[: n_slices // 2] += 50.0
    return vol


def series(path, n_slices=20, n=64, pixel_mm=DEFAULT_PIXEL_MM, dz=DEFAULT_DZ,
           z0=-100.0, kvp=120.0, hu=None, drop=()):
    """Write the series into `path`; returns the directory as a str.

    `drop` removes slice indices after the geometry is laid out, which is how
    a partial export (the failure `attenuation.load` refuses) is simulated.
    """
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    path = str(path)
    os.makedirs(path, exist_ok=True)
    vol = volume(n_slices, n, pixel_mm) if hu is None else np.asarray(hu, np.float32)
    n_slices, ny, nx = vol.shape
    x0 = y0 = -(n // 2) * pixel_mm
    series_uid = generate_uid()

    for i in range(n_slices):
        if i in drop:
            continue
        fm = Dataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = "synthetic"
        ds.FrameOfReferenceUID = FRAME_OF_REFERENCE
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImagePositionPatient = [f"{x0:.6f}", f"{y0:.6f}", f"{z0 + i * dz:.6f}"]
        ds.PixelSpacing = [f"{pixel_mm:.6f}", f"{pixel_mm:.6f}"]
        ds.SliceThickness = f"{dz:.6f}"
        ds.KVP = float(kvp)
        ds.Rows, ds.Columns = ny, nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope = "1"
        ds.RescaleIntercept = "-1024"
        ds.PixelData = np.rint(vol[i] + 1024).astype("<i2").tobytes()
        ds.save_as(os.path.join(path, f"CT_{i + 1:04d}.dcm"),
                   enforce_file_format=True)
    return path
