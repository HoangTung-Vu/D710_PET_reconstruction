"""The simulation's inputs on one bed's grid: activity in Bq/mL and CT in HU.

GATE and parallelproj both read what this module writes, so the two methods
can differ in physics but never in where the patient is.

The grid is the reconstruction's own -- `XY` x `XY` at `DR_MM`, planes at
`PLANE_MM` -- extended by `margin_mm` beyond both ends of the bed so that
activity outside the axial field of view still feeds scatter and randoms in
GATE. Plane `p` of the bed sits at patient z `table_position_mm + p * PLANE_MM`
and the result is flipped to the image's row order, exactly as
`utils.attenuation.mu_map` does it; that orientation was verified on NEMA.

The world frame is the one `utils.geometry.crystal_positions(stir_frame=True)`
uses: x and y centred on the gantry axis, z = 0 at the middle of the bed. The
extended grid is symmetric about that point, so its centre is the world origin
-- which is where GATE puts the centre of an image volume and of a voxel source.

Activity may be given three ways (`UNITS`):

  bqml      the image is Bq/mL already
  suv       SUVbw, inverted with the dose, injection time, scan time and weight
            from `decoded/bed<n>.json` -- the inverse of `tools/dicom_suv.py`
            for an image whose DecayCorrection is START
  relative  any map proportional to activity, plus one global activity
            `A` at a stated time; the map is scaled so that it holds `A`
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.attenuation import hu_to_mu, resample_to_bed, to_radiological
from utils.scanner import DR_MM, NSEG0, PLANE_MM, XY

UNITS = ("suv", "bqml", "relative")

VOXEL_XYZ = (DR_MM, DR_MM, PLANE_MM)

VOXEL_ML = DR_MM * DR_MM * PLANE_MM / 1000.0

DEFAULT_MARGIN_MM = 900.0
"""Enough to take in the whole 896 mm GE whole-body image from any bed. Measured
on fdg26081008 bed 1 (GATE, 20 ms runs): with 200 mm of margin the singles were
0.53 of the real ones and the prompts 0.80; with the whole image, 0.88 and
0.98. What is still missing is activity the image never held (the legs below
bed 1: the deficit is largest at ring 0)."""


@dataclass
class Volume:
    """One input series as `[slice, row, col]` in DICOM (LPS) order."""

    data: np.ndarray
    z: np.ndarray
    x0: float
    y0: float
    pixel_mm: float
    source: str
    meta: dict = field(default_factory=dict)

    @property
    def voxel_ml(self) -> float:
        return self.pixel_mm ** 2 * float(np.diff(self.z).mean()) / 1000.0

    def describe(self) -> str:
        return (f"{self.source}: {self.data.shape[0]}x{self.data.shape[1]}x"
                f"{self.data.shape[2]} @ {self.pixel_mm:.4f} mm, "
                f"z {self.z[0]:.2f}..{self.z[-1]:.2f} step "
                f"{float(np.diff(self.z).mean()):.4f} mm")


def load_volume(path, modality: str) -> Volume:
    """A NIfTI file or a DICOM folder, as a `Volume`. `modality` is CT or PT."""
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        raise SystemExit(f"error: no {p}")
    return _load_dicom(p, modality) if p.is_dir() else _load_nifti(p)


def _load_nifti(path: Path) -> Volume:
    """NIfTI in RAS, as `tools/ct_nifti.py` and `tools/dicom_suv.py` write it."""
    import nibabel as nib

    img = nib.load(str(path))
    a = np.asarray(img.get_fdata(dtype=np.float32))
    if a.ndim != 3:
        raise SystemExit(f"error: {path} is {a.ndim}-D, not a 3-D volume")
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ img.affine
    rot = lps[:3, :3]
    if not np.allclose(rot, np.diag(np.diag(rot)), atol=1e-6):
        raise SystemExit(f"error: {path} is oblique (affine {img.affine.tolist()}); "
                         "only axis-aligned volumes are supported")
    step = np.diag(rot).astype(np.float64)
    org = lps[:3, 3].astype(np.float64)
    for ax in range(3):
        if step[ax] < 0:
            a = np.flip(a, ax)
            org[ax] += step[ax] * (a.shape[ax] - 1)
            step[ax] = -step[ax]
    if abs(step[0] - step[1]) > 1e-4:
        raise SystemExit(f"error: {path} has non-square pixels {step[:2]}")
    data = np.ascontiguousarray(a.transpose(2, 1, 0), dtype=np.float32)
    z = org[2] + np.arange(data.shape[0]) * step[2]
    return Volume(data, z, float(org[0]), float(org[1]), float(step[0]),
                  f"NIfTI {path.name}", {"path": str(path)})


def _load_dicom(path: Path, modality: str) -> Volume:
    if modality.upper() == "CT":
        from utils.attenuation import load

        ct = load(str(path))
        return Volume(ct.hu, np.asarray(ct.z, np.float64), ct.x0, ct.y0,
                      ct.pixel_mm, f"DICOM {ct.meta['series_description']}",
                      {"path": str(path), "kvp": ct.kvp})

    from tools.dicom_suv import read_series

    vol, m = read_series(str(path))
    if abs(m["px"] - m["py"]) > 1e-4:
        raise SystemExit(f"error: {path} has non-square pixels")
    z = m["z0"] + np.arange(vol.shape[0]) * m["dz"]
    return Volume(np.asarray(vol, np.float32), z, m["x0"], m["y0"], m["px"],
                  f"DICOM {m['desc']} (BQML)",
                  {"path": str(path), "decay": m["decay"]})


def utc_epoch(stamp: str) -> float:
    """`YYYYmmddHHMMSS[.ff]` read as UTC, the RDF header's clock."""
    t = dt.datetime.strptime(str(stamp)[:14], "%Y%m%d%H%M%S")
    return t.replace(tzinfo=dt.timezone.utc).timestamp()


def exam_timing(case) -> dict:
    """Dose, weight, isotope and clock of the exam, from its bed sidecars.

    `t_scan` is the start of the first bed. GE decay-corrects the image to it
    (DecayCorrection START), so the Bq/mL in the image are activities then; it
    equals the DICOM SeriesTime (local, +7 h) on fdg26081008.

    The dose is net of the residual in the syringe: the DICOM
    RadionuclideTotalDose, and so every SUV made from it, is `dose_mbq -
    residual_dose_mbq` (281.2 = 284.9 - 3.7 MBq on fdg26081008). Using the
    gross dose puts every activity 1.3 % high.
    """
    beds = case.decoded_beds()
    if not beds:
        raise SystemExit(f"error: {case.decoded} has no bed<n>.json")
    hdrs = {n: case.header(n) for n in beds}
    h = hdrs[beds[0]]
    return {
        "t_scan": float(min(x["bed_start_time"] for x in hdrs.values())),
        "t_inj": utc_epoch(h["radiopharm_start_datetime"]),
        "bed_start": {n: float(x["bed_start_time"]) for n, x in hdrs.items()},
        "dose_bq": (float(h["dose_mbq"])
                    - float(h.get("residual_dose_mbq", 0.0) or 0.0)) * 1e6,
        "weight_kg": float(h["patient_weight_kg"]),
        "half_life_s": float(h["half_life_s"]),
        "positron_fraction": float(h.get("positron_fraction", 1.0)),
    }


def decay(dt_s: float, half_life_s: float) -> float:
    return 2.0 ** (-dt_s / half_life_s)


def frame_integral_s(frame_s: float, half_life_s: float) -> float:
    """`∫_0^T 2^(-t/T½) dt`, the decays per Bq present at the start of the frame."""
    lam = math.log(2.0) / half_life_s
    return (1.0 - math.exp(-lam * frame_s)) / lam


def parse_activity(spec: str) -> tuple[float, float]:
    """`MBq@YYYYmmddHHMMSS` (UTC) to `(Bq, epoch)`."""
    try:
        a, t = spec.split("@")
        return float(a) * 1e6, utc_epoch(t)
    except ValueError:
        raise SystemExit(f"error: --activity {spec!r} is not MBq@YYYYmmddHHMMSS")


def bqml_scale(pet: Volume, units: str, timing: dict, activity=None):
    """`(factor, info)` turning the input's values into Bq/mL at `t_scan`."""
    if units not in UNITS:
        raise SystemExit(f"error: --pet-units must be one of {UNITS}")
    info = {"units": units}
    pos = np.clip(pet.data, 0.0, None)
    if units == "bqml":
        if activity is not None:
            raise SystemExit("error: --activity applies to --pet-units relative only")
        f = 1.0
    elif units == "suv":
        if activity is not None:
            raise SystemExit("error: --activity applies to --pet-units relative only")
        uptake = timing["t_scan"] - timing["t_inj"]
        if not 0 < uptake < 6 * 3600:
            raise SystemExit(
                f"error: uptake {uptake / 60:.1f} min between injection and scan "
                "start -- a clock problem (UTC header vs local DICOM?)")
        dose_ref = timing["dose_bq"] * decay(uptake, timing["half_life_s"])
        f = dose_ref / (timing["weight_kg"] * 1000.0)
        info.update(uptake_min=uptake / 60, dose_ref_bq=dose_ref,
                    weight_kg=timing["weight_kg"],
                    fraction_of_dose_in_image=float(
                        pos.sum(dtype=np.float64) * pet.voxel_ml
                        / (timing["weight_kg"] * 1000.0)))
    else:
        if activity is None:
            raise SystemExit("error: --pet-units relative needs --activity MBq@time")
        a_bq, t_a = activity
        a_scan = a_bq * decay(timing["t_scan"] - t_a, timing["half_life_s"])
        total = pos.sum(dtype=np.float64) * pet.voxel_ml
        if total <= 0:
            raise SystemExit("error: the PET map sums to zero")
        f = a_scan / total
        info.update(activity_bq=a_bq, activity_epoch=t_a, activity_at_scan_bq=a_scan)
    info["bqml_per_unit"] = float(f)
    info["image_total_bq_at_scan"] = float(pos.sum(dtype=np.float64) * pet.voxel_ml * f)
    return float(f), info


def grid_planes(margin_mm: float) -> tuple[int, int]:
    """`(first_plane, n_planes)` of the bed grid extended by `margin_mm` each side."""
    m = int(round(max(margin_mm, 0.0) / PLANE_MM))
    return -m, NSEG0 + 2 * m


def world_origin(shape_zyx) -> np.ndarray:
    """World `(x, y, z)` of voxel `[0, 0, 0]` of a centred `(z, y, x)` grid."""
    n = np.asarray(shape_zyx[::-1], np.float64)
    return ((-n / 2 + 0.5) * np.asarray(VOXEL_XYZ)).astype(np.float32)


def on_bed_grid(vol: Volume, table_position_mm: float, margin_mm: float,
                cval: float, what: str) -> np.ndarray:
    """`vol` on the extended bed grid, `(n_planes, XY, XY)` in the image's order."""
    first, n = grid_planes(margin_mm)
    a = resample_to_bed(vol.data, vol.z, vol.x0, vol.y0, vol.pixel_mm,
                        table_position_mm, XY, DR_MM, first_plane=first,
                        n_planes=n, cval=cval, what=what, clamp_edges=True)
    return np.ascontiguousarray(to_radiological(a), dtype=np.float32)


def bed_planes(arr, margin_mm: float) -> np.ndarray:
    """The bed's own 47 planes out of an extended-grid array."""
    first, _ = grid_planes(margin_mm)
    return arr[-first:-first + NSEG0]


def write_mhd(path, arr_zyx, spacing_xyz=VOXEL_XYZ, origin_xyz=None) -> Path:
    """A MetaImage pair `<stem>.mhd` + `<stem>.raw`, float32, identity direction."""
    path = Path(path)
    a = np.ascontiguousarray(arr_zyx, dtype="<f4")
    if origin_xyz is None:
        origin_xyz = world_origin(a.shape)
    raw = path.with_suffix(".raw")
    a.tofile(raw)
    nz, ny, nx = a.shape
    path.write_text(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\n"
        "BinaryDataByteOrderMSB = False\nCompressedData = False\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        f"Offset = {' '.join(f'{v:.6f}' for v in origin_xyz)}\n"
        "CenterOfRotation = 0 0 0\nAnatomicalOrientation = RAI\n"
        f"ElementSpacing = {' '.join(f'{v:.7f}' for v in spacing_xyz)}\n"
        f"DimSize = {nx} {ny} {nz}\nElementType = MET_FLOAT\n"
        f"ElementDataFile = {raw.name}\n")
    return path


def read_mhd(path) -> np.ndarray:
    """The `(z, y, x)` float32 array of a MetaImage written by `write_mhd`."""
    path = Path(path)
    k = dict(ln.split(" = ", 1) for ln in path.read_text().splitlines() if " = " in ln)
    nx, ny, nz = (int(v) for v in k["DimSize"].split())
    return np.fromfile(path.parent / k["ElementDataFile"], "<f4").reshape(nz, ny, nx)


def directory(sim_root: Path, bed: int) -> Path:
    return sim_root / "phantom" / f"bed{bed}"


def build(case, bed: int, ct_path, pet_path, pet_units: str, out_dir: Path,
          activity=None, margin_mm: float = DEFAULT_MARGIN_MM,
          kvp: float | None = None, out=print) -> dict:
    """Write `act_bqml.mhd`, `ct_hu.mhd`, `mu_bed.npy` and `phantom.json` for one bed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = case.header(bed)
    timing = exam_timing(case)
    tp = float(hdr["table_position_mm"])

    ct = load_volume(ct_path, "CT")
    pet = load_volume(pet_path, "PT")
    out(f"  CT  {ct.describe()}")
    out(f"  PET {pet.describe()}")
    kvp = float(kvp or ct.meta.get("kvp") or 120.0)

    f, info = bqml_scale(pet, pet_units, timing, activity)
    hu = on_bed_grid(ct, tp, margin_mm, -1000.0, "CT")
    act = np.clip(on_bed_grid(pet, tp, margin_mm, 0.0, "PET"), 0.0, None) * np.float32(f)

    frame_s = float(hdr["frame_duration_ms"]) / 1000.0
    t_bed = timing["bed_start"][bed]
    k_bed = decay(t_bed - timing["t_scan"], timing["half_life_s"])
    pf = timing["positron_fraction"]
    a_grid = float(act.sum(dtype=np.float64) * VOXEL_ML)
    a_fov = float(bed_planes(act, margin_mm).sum(dtype=np.float64) * VOXEL_ML)

    write_mhd(out_dir / "act_bqml.mhd", act)
    write_mhd(out_dir / "ct_hu.mhd", hu)
    mu = hu_to_mu(bed_planes(hu, margin_mm), kvp)
    np.save(out_dir / "mu_bed.npy", mu.astype(np.float32))

    first, n = grid_planes(margin_mm)
    meta = {
        "case": case.name, "bed": bed, "table_position_mm": tp,
        "ct": str(ct_path), "pet": str(pet_path), "ct_kvp": kvp,
        "first_plane": first, "n_planes": n, "margin_mm": margin_mm,
        "shape_zyx": list(act.shape), "voxel_xyz_mm": list(VOXEL_XYZ),
        "origin_xyz_mm": world_origin(act.shape).tolist(),
        "activity": info, "timing": {k: v for k, v in timing.items()
                                     if k != "bed_start"},
        "t_bed": t_bed, "frame_s": frame_s,
        "bed_start_ticks": int(hdr.get("bed_start_ticks", 0)),
        "decay_scan_to_bed": k_bed,
        "grid_activity_at_scan_bq": a_grid,
        "fov_activity_at_scan_bq": a_fov,
        "grid_positron_activity_at_bed_start_bq": a_grid * k_bed * pf,
        "frame_integral_s": frame_integral_s(frame_s, timing["half_life_s"]),
    }
    (out_dir / "phantom.json").write_text(json.dumps(meta, indent=2))
    out(f"  activity: {info['bqml_per_unit']:.6g} Bq/mL per unit; grid "
        f"{a_grid / 1e6:.1f} MBq at scan start ({a_fov / 1e6:.1f} MBq in the "
        f"bed's FOV), {a_grid * k_bed * pf / 1e6:.1f} MBq of positrons at bed start")
    if "fraction_of_dose_in_image" in info:
        out(f"  the image holds {100 * info['fraction_of_dose_in_image']:.1f} % of "
            f"the decayed dose (uptake {info['uptake_min']:.1f} min)")
    return meta


def load(out_dir: Path) -> dict:
    p = Path(out_dir) / "phantom.json"
    if not p.exists():
        raise SystemExit(f"error: no {p}\n  run: d710 simulate phantom ...")
    return json.loads(p.read_text())


def decays_per_voxel(out_dir: Path, meta: dict) -> np.ndarray:
    """Positron decays per voxel over the frame, on the bed's own 47 planes."""
    act = bed_planes(read_mhd(Path(out_dir) / "act_bqml.mhd"), meta["margin_mm"])
    k = (meta["decay_scan_to_bed"] * meta["timing"]["positron_fraction"]
         * meta["frame_integral_s"] * VOXEL_ML)
    return (act * np.float32(k)).astype(np.float32)
