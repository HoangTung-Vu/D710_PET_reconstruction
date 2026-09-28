from __future__ import annotations

import gzip
import math
import struct

import numpy as np

DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def _raw(path) -> bytes:
    with open(path, "rb") as f:
        head = f.read(2)
    opener = gzip.open if head == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        return f.read()


def _endian(buf: bytes) -> str:
    for e in "<>":
        n = struct.unpack_from(e + "i", buf, 0)[0]
        if n == 348:
            return e
        if n == 540:
            raise SystemExit("error: NIfTI-2 is not supported; save the CT as NIfTI-1")
    raise SystemExit("error: not a NIfTI file (sizeof_hdr is neither 348 nor 540)")


def quaternion_affine(b, c, d, qfac, zooms, offset) -> np.ndarray:
    a = math.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
    r = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]])
    z = np.array(zooms, dtype=np.float64)
    z[2] *= qfac
    aff = np.eye(4)
    aff[:3, :3] = r * z
    aff[:3, 3] = offset
    return aff


def read(path):
    buf = _raw(path)
    if len(buf) < 348:
        raise SystemExit(f"error: {path} is too short to be a NIfTI file")
    e = _endian(buf)
    magic = buf[344:348]
    if magic != b"n+1\0":
        raise SystemExit(f"error: {path} has magic {magic!r}; only single-file "
                         f"NIfTI-1 (.nii / .nii.gz) is supported")
    dim = struct.unpack_from(e + "8h", buf, 40)
    code = struct.unpack_from(e + "h", buf, 70)[0]
    pixdim = struct.unpack_from(e + "8f", buf, 76)
    vox_offset = int(struct.unpack_from(e + "f", buf, 108)[0])
    slope, inter = struct.unpack_from(e + "2f", buf, 112)
    qform_code, sform_code = struct.unpack_from(e + "2h", buf, 252)
    qb, qc, qd, qx, qy, qz = struct.unpack_from(e + "6f", buf, 256)
    srow = struct.unpack_from(e + "12f", buf, 280)

    nd = dim[0]
    if not 3 <= nd <= 7 or any(n != 1 for n in dim[4:nd + 1]):
        raise SystemExit(f"error: {path} is {nd}-D with dim {dim[1:nd + 1]}; "
                         f"a CT must be a single 3-D volume")
    if code not in DTYPES:
        raise SystemExit(f"error: {path} has unsupported NIfTI datatype {code}")
    shape = tuple(int(n) for n in dim[1:4])
    count = shape[0] * shape[1] * shape[2]
    dt = np.dtype(DTYPES[code]).newbyteorder(e)
    if len(buf) < vox_offset + count * dt.itemsize:
        raise SystemExit(f"error: {path} is truncated")
    arr = np.frombuffer(buf, dt, count, vox_offset).reshape(shape, order="F")
    if slope != 0.0 and not math.isnan(slope):
        arr = arr * np.float32(slope) + np.float32(0.0 if math.isnan(inter) else inter)

    if sform_code > 0:
        aff = np.eye(4)
        aff[:3] = np.array(srow, dtype=np.float64).reshape(3, 4)
    elif qform_code > 0:
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        aff = quaternion_affine(qb, qc, qd, qfac, pixdim[1:4], (qx, qy, qz))
    else:
        raise SystemExit(f"error: {path} has neither an sform nor a qform, so "
                         f"its position in the scanner is unknown")
    return np.asarray(arr), aff
