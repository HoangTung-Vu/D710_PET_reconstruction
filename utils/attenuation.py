"""CT -> mu-map trên lưới ảnh của bed. Đây là **đầu vào** của scatter, không phải đầu ra.

Scatter cần biết vật chất nằm ở đâu; module này chỉ dựng đúng chừng đó, không
làm hiệu chỉnh suy giảm cho phần tái tạo.
"""

from __future__ import annotations

import glob
import os

import numpy as np

#: Khoảng cách plane PET, và số plane trong một bed (= 2·num_rings − 1).
PLANE_MM = 3.2699997
PLANES_PER_BED = 47

# Carney bilinear HU -> mu(511 keV), 1/mm.
MU_WATER_511 = 0.0096
MU_BONE_511 = 0.0172
CARNEY_B = {80: 0.681, 100: 0.755, 120: 0.837, 140: 1.0}


def hu_to_mu(hu: np.ndarray, kvp: float = 120.0) -> np.ndarray:
    """Carney bilinear HU -> mu(511 keV), 1/mm."""
    b = CARNEY_B.get(int(round(kvp)), 0.837)
    hu = np.asarray(hu, dtype=np.float32)
    soft = MU_WATER_511 * (1.0 + hu / 1000.0)
    bone = MU_WATER_511 + hu * (MU_BONE_511 - MU_WATER_511) / (1000.0 * b)
    return np.clip(np.where(hu <= 0, soft, bone), 0.0, None).astype(np.float32)


def to_radiological(arr: np.ndarray) -> np.ndarray:
    """Lật trục y của STIR sang y bệnh nhân của DICOM; là nghịch đảo của chính nó."""
    return np.flip(np.asarray(arr), axis=1)


class CTAC:
    """Một series CT: khối HU ``[slice, row, col]`` theo thứ tự DICOM, kèm hình học."""

    def __init__(self, hu, z, x0, y0, pixel_mm, kvp, meta):
        self.hu, self.z, self.x0, self.y0 = hu, z, x0, y0
        self.pixel_mm, self.kvp, self.meta = pixel_mm, kvp, meta

    @property
    def dz(self) -> float:
        return float(np.diff(self.z).mean())

    def describe(self) -> str:
        return (f"CT {self.meta['series_description']}: {self.hu.shape[0]} slice "
                f"{self.hu.shape[1]}x{self.hu.shape[2]} @ {self.pixel_mm:.4f} mm, "
                f"{self.kvp:.0f} kVp, z {self.z[0]:.2f}..{self.z[-1]:.2f} "
                f"bước {self.dz:.4f} mm")


def load(path: str) -> CTAC:
    """Đọc một thư mục series CT."""
    import pydicom

    ds = []
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        if not os.path.isfile(f):
            continue
        try:
            d = pydicom.dcmread(f)
        except Exception:
            continue
        if getattr(d, "Modality", None) == "CT":
            ds.append(d)
    if not ds:
        raise SystemExit(f"lỗi: không có instance CT nào dưới {path}")
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    iop = [float(v) for v in ds[0].ImageOrientationPatient]
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-6):
        raise SystemExit(f"lỗi: {path} bị xiên (IOP {iop}); resample trước đã")

    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    step = np.round(np.diff(z), 3)
    # Một bản export thiếu slice sẽ được nội suy bắc cầu qua lỗ hổng thành một
    # mu-map sai nhưng trông hợp lý.  Chẩn đoán, đừng lấy trung bình.
    if len(step) and step.std() > 0.05:
        modal = float(np.bincount((step * 100).astype(int)).argmax()) / 100
        gaps = step[np.abs(step - modal) > 0.01]
        raise SystemExit(
            f"lỗi: {path} là bản export thiếu, không phải series thưa đều.\n"
            f"  {len(ds)} slice, bước {modal:.2f} mm ở {len(step) - len(gaps)}/"
            f"{len(step)} khoảng, {len(gaps)} lỗ hổng tới {gaps.max():.1f} mm.\n"
            f"  Một bed cần {PLANES_PER_BED * PLANE_MM:.1f} mm liên tục.")

    hu = np.stack([d.pixel_array * float(getattr(d, "RescaleSlope", 1))
                   + float(getattr(d, "RescaleIntercept", 0)) for d in ds])
    return CTAC(hu=hu.astype(np.float32), z=z,
                x0=float(ds[0].ImagePositionPatient[0]),
                y0=float(ds[0].ImagePositionPatient[1]),
                pixel_mm=float(ds[0].PixelSpacing[0]),
                kvp=float(getattr(ds[0], "KVP", 120.0) or 120.0),
                meta={"path": path,
                      "series_description": str(getattr(ds[0], "SeriesDescription", "?")),
                      "frame_of_reference_uid":
                          str(getattr(ds[0], "FrameOfReferenceUID", "")),
                      "num_slices": len(ds)})


def mu_image(ct: CTAC, table_position_mm: float, template, edge_tol_planes: float = 1.5):
    """SIRF ``ImageData`` chứa mu-map của bed, 1/cm, trên lưới của ``template``.

    ``edge_tol_planes`` cho phép bed thò ra khỏi CT vài plane ở hai đầu. Bed đầu
    và bed cuối của một ca gần như luôn thò ra vài mm — ca nhi bed 1 nằm ở
    -767.7 mm còn CT chỉ bắt đầu từ -765.4 mm — và từ chối cả bed vì 2 mm đó thì
    mất hẳn một bed. Phần thò ra được **kẹp về lát CT ngoài cùng** (lặp lại nó),
    chứ không phải điền không khí: chỗ đó vẫn còn thân người, coi là không khí sẽ
    hiệu chỉnh suy giảm thiếu. Quá ``edge_tol_planes`` thì vẫn báo lỗi.
    """
    from scipy.ndimage import map_coordinates

    shape = tuple(int(s) for s in template.shape)
    if shape[0] != PLANES_PER_BED or shape[1] != shape[2]:
        raise SystemExit(f"lỗi: lưới ảnh {shape} không phải (47, xy, xy)")
    vz, vy, vx = (float(v) for v in template.voxel_sizes())
    if abs(vy - vx) > 1e-3:
        raise SystemExit(f"lỗi: voxel ngang không đẳng hướng {vy} × {vx}")

    # Theo trục: plane PET thứ p nằm ở table_position + p·PLANE_MM.
    zc = table_position_mm + np.arange(PLANES_PER_BED) * PLANE_MM
    gz = (zc - ct.z[0]) / ct.dz
    # Phần thò ra khỏi lát CT ngoài cùng, đo bằng mm rồi mới đổi sang PLANE PET.
    # Nửa lát CT đầu tiên vẫn nội suy được nên không tính. Dung sai phải là một
    # KHOẢNG CÁCH, không phải số lát: đo bằng chỉ số lát thì CT 1,25 mm bị siết
    # chặt gấp 2,6 lần CT 3,27 mm cho cùng một bed.
    out_mm = max(ct.z[0] - zc.min(), zc.max() - ct.z[-1], 0.0)
    over = max(out_mm - ct.dz / 2, 0.0) / PLANE_MM
    if over > edge_tol_planes:
        raise SystemExit(
            f"lỗi: bed ở {table_position_mm:.2f} mm cần CT z "
            f"{zc[0]:.1f}..{zc[-1]:.1f} mm, series chỉ phủ "
            f"{ct.z[0]:.1f}..{ct.z[-1]:.1f} mm "
            f"(thò ra {out_mm:.1f} mm = {over:.2f} plane > cho phép "
            f"{edge_tol_planes})")
    if over > 0:
        # Báo phần thò ra THẬT (mm), không phải phần vượt quá dung sai.
        print(f"  cảnh báo: bed {table_position_mm:.2f} mm thò khỏi CT "
              f"{out_mm:.1f} mm; kẹp về lát CT ngoài cùng")
        gz = np.clip(gz, 0.0, len(ct.z) - 1.0)

    # Theo mặt cắt: lưới PET tâm ở chỉ số xy//2, trục máy ở DICOM (x, y) = (0, 0).
    xy = shape[1]
    c = (np.arange(xy) - xy // 2) * vy
    g = np.meshgrid(gz, (c - ct.y0) / ct.pixel_mm, (c - ct.x0) / ct.pixel_mm,
                    indexing="ij")
    hu = map_coordinates(ct.hu, [x.ravel() for x in g], order=1,
                         mode="constant", cval=-1000.0).reshape(PLANES_PER_BED, xy, xy)
    mu = hu_to_mu(hu, ct.kvp) * 10.0                   # 1/mm -> 1/cm cho STIR

    out = template.get_uniform_copy(0)
    out.fill(np.ascontiguousarray(to_radiological(mu), dtype=np.float32))
    return out


def factors(ad, mu_img):
    """``(af, acf)`` — xác suất sống sót và nghịch đảo của nó, dạng AcquisitionData."""
    import sirf.STIR as pet

    return pet.AcquisitionSensitivityModel.compute_attenuation_factors(ad, mu_img)
