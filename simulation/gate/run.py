"""One GATE run for one bed: the D710 around the CT, with the PET as the source.

Run as its own process (`python -m simulation.gate.run config.json`); the
`d710 simulate gate` command writes the config and launches it, once for a
short run that also writes the singles and once for the full frame, which
writes coincidences only.

Physics and digitizer, and where each number comes from:

  source      VoxelSource over the PET in Bq/mL, `back_to_back` 511 keV pairs
              with accolinearity (or `e+` on the F-18 spectrum), activity =
              positrons in the grid at the bed's start, half-life F-18
  patient     Image volume from the CT, HounsfieldUnit_to_material with the
              Schneider 2000 tables shipped in opengate/data
  readout     energy-weighted centroid per block, discretised to the crystal
  energy      InverseSquare blur, 12 % FWHM at 511 keV (D690 paper: 10-20 %)
  dead time   off by default (see DEADTIME_NS); 300 ns paralysable per
              block in the D690 paper
  window      425-650 keV (D690 paper, and `energy_window_*_kev` in bed.json)
  timing      Gaussian, 675/sqrt(2) ps FWHM per single, so 675 ps per pair
              (`TIMING_PS`, GE's coincidence timing resolution)
  coincidence window 2.4545 ns: the real singles.log gives randoms as
              `4.909 ns * S_i * S_j`, and a window `w` gives `2 w S_i S_j`
  delays      the same sorter with its window offset by 500 ns (D690 paper)
  scatter     truth flag: Compton / Rayleigh steps in the patient, counted per
              photon and inherited by the electrons it sets moving in a crystal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from utils.scanner import TIMING_PS

from .. import phantom as ph
from .geometry import ShieldSpec, add_d710, patient_half_width_mm

WINDOW_NS = 4.909 / 2.0

DELAY_OFFSET_NS = 500.0

ENERGY_WINDOW_KEV = (425.0, 650.0)

DEADTIME_NS = 0.0
"""Off: opengate 10.1.1's DigitizerDeadTimeActor loses the end of every run --
measured on bed 1, a 20 ms run stopped at 15.97 ms (20 % of its singles gone)
and a 1 s run at 986 ms, while without the actor both run to the end. The D690
paper's 300 ns per block would cost only ~0.4 % at 13 kcps per block anyway.
Pass --deadtime-ns 300 to turn it back on."""

ENERGY_RESOLUTION = 0.12

SCATTER_ATTRS = {"PhantomCompton": "compt", "PhantomRayleigh": "Rayl"}

PROMPTS, DELAYS, SINGLES = "Prompts", "Delays", "Singles"


@dataclass
class RunConfig:
    phantom_dir: str
    out_dir: str
    seconds: float
    t0: float = 0.0
    threads: int = 16
    seed: int = 1
    write_singles: bool = False
    positron: bool = False
    energy_resolution: float = ENERGY_RESOLUTION
    deadtime_ns: float = DEADTIME_NS
    window_ns: float = WINDOW_NS
    delay_offset_ns: float = DELAY_OFFSET_NS
    multiples_policy: str = "TakeWinnerIfAllAreGoods"
    physics_list: str = "G4EmStandardPhysics_option3"
    shield: dict = field(default_factory=lambda: asdict(ShieldSpec()))
    ct_step: tuple = (3, 3)
    pad_voxels: int = 3
    check_overlap: bool = False
    phantom_key: str = ""

    @classmethod
    def load(cls, path) -> "RunConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _box(mask2d, pad: int, lim: tuple[int, int]):
    """Index range `[lo, hi)` per axis of the true region of a 2-D mask, padded and clipped."""
    out = []
    for ax in (0, 1):
        idx = np.nonzero(mask2d.any(axis=1 - ax))[0]
        if idx.size == 0:
            raise SystemExit("error: the CT has no voxel above -900 HU on this bed")
        out.append((max(int(idx[0]) - pad, lim[0]), min(int(idx[-1]) + 1 + pad, lim[1])))
    return out


def _centre_mm(lo: int, hi: int, n: int, step_mm: float) -> float:
    """World coordinate of the centre of index range `[lo, hi)` on a centred `n` grid."""
    return ((lo + hi - 1) / 2.0 - (n - 1) / 2.0) * step_mm


def _block_mean(a, f: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = (s // k for s, k in zip(a.shape, f))
    a = a[:nz * f[0], :ny * f[1], :nx * f[2]]
    return a.reshape(nz, f[0], ny, f[1], nx, f[2]).mean(axis=(1, 3, 5))


def cropped_inputs(cfg: RunConfig, shield: ShieldSpec) -> dict:
    """Cut the phantom down to what GATE needs; write `ct.mhd` and `act.mhd`.

    Every voxel boundary a photon crosses is a Geant4 step -- opengate builds an
    image as nested replicas and does not merge equal materials -- so the CT is
    (1) cropped to the body and couch (HU > -900) plus `pad` voxels, which also
    keeps the box inside the bore, where it must stay since Geant4 volumes may
    not overlap, and (2) averaged over `ct_step` voxels for the geometry only:
    attenuation and scatter of 511 keV photons do not need 2 mm voxels. The
    activity keeps the reconstruction's voxels, cropped to the same box.
    What the crop cuts off is reported.
    """
    src, out = Path(cfg.phantom_dir), Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    act = ph.read_mhd(src / "act_bqml.mhd")
    hu = ph.read_mhd(src / "ct_hu.mhd")
    nz, ny, nx = hu.shape
    dx, _, dz = ph.VOXEL_XYZ

    half = patient_half_width_mm(shield)
    k = max(0, int(np.ceil((nx * dx / 2 - half) / dx)))
    (y0, y1), (x0, x1) = _box((hu > -900).any(axis=0), cfg.pad_voxels,
                              (k, nx - k))
    fz, fxy = cfg.ct_step

    def whole_blocks(lo, hi, lim):
        up = lo + -(-(hi - lo) // fxy) * fxy
        return up if up <= lim else lo + ((hi - lo) // fxy) * fxy

    y1 = whole_blocks(y0, y1, ny - k)
    x1 = whole_blocks(x0, x1, nx - k)
    z1 = (nz // fz) * fz

    a_c = act[:, y0:y1, x0:x1]
    h_c = _block_mean(hu[:z1, y0:y1, x0:x1], (fz, fxy, fxy))
    centre = [_centre_mm(x0, x1, nx, dx), _centre_mm(y0, y1, ny, dx), 0.0]
    ct_centre = [centre[0], centre[1], _centre_mm(0, z1, nz, dz)]

    corner = max(np.hypot(abs(c) + w / 2, abs(d) + v / 2) for c, d, w, v in
                 [(centre[0], centre[1], (x1 - x0) * dx, (y1 - y0) * dx)])
    if corner >= geometry_bore(shield):
        raise SystemExit(f"error: the image box reaches r = {corner:.1f} mm, "
                         f"inside the scanner (bore {geometry_bore(shield):.1f} mm)")

    ph.write_mhd(out / "act.mhd", a_c, origin_xyz=[0, 0, 0])
    ph.write_mhd(out / "ct.mhd", h_c, spacing_xyz=(dx * fxy, dx * fxy, dz * fz),
                 origin_xyz=[0, 0, 0])
    total = max(float(act.sum(dtype=np.float64)), 1e-30)
    body = hu > -500
    return {"box_yx": [[y0, y1], [x0, x1]], "ct_step": [fz, fxy],
            "act_centre_mm": centre, "ct_centre_mm": ct_centre,
            "ct_shape_zyx": list(h_c.shape), "shape_zyx": list(a_c.shape),
            "box_mm": [(x1 - x0) * dx, (y1 - y0) * dx, nz * dz],
            "activity_lost_to_crop": 1.0 - float(a_c.sum(dtype=np.float64)) / total,
            "body_voxels_lost_to_crop":
                1.0 - float(body[:, y0:y1, x0:x1].sum()) / max(float(body.sum()), 1.0),
            "activity_bqml_sum": float(a_c.sum(dtype=np.float64))}


def geometry_bore(shield: ShieldSpec) -> float:
    from .geometry import bore_radius_mm

    return bore_radius_mm(shield)


def build(cfg: RunConfig, meta: dict, crop: dict):
    """The opengate `Simulation` for one run."""
    import opengate as gate

    u = gate.g4_units
    mm, m, ns, keV, sec, Bq = u.mm, u.m, u.ns, u.keV, u.s, u.Bq
    shield = ShieldSpec(**cfg.shield)
    out = Path(cfg.out_dir)

    sim = gate.Simulation()
    sim.number_of_threads = cfg.threads
    sim.random_seed = cfg.seed
    sim.output_dir = str(out)
    sim.check_volumes_overlap = cfg.check_overlap
    sim.g4_verbose = False
    sim.visu = False
    sim.progress_bar = True

    z_extent = crop["box_mm"][2]
    sim.world.size = [2 * m, 2 * m, max(2 * m, (z_extent + 200) * mm)]
    sim.world.material = "G4_AIR"

    patient = sim.add_volume("Image", "patient")
    patient.image = str(out / "ct.mhd")
    patient.material = "G4_AIR"
    patient.translation = [v * mm for v in crop["ct_centre_mm"]]
    data = Path(gate.__file__).parent / "data"
    patient.voxel_materials, _ = gate.geometry.materials.HounsfieldUnit_to_material(
        sim, 0.05 * u.g_cm3, str(data / "Schneider2000MaterialsTable.txt"),
        str(data / "Schneider2000DensitiesTable.txt"))

    ring, block, crystal = add_d710(sim, "d710", shield)

    sim.physics_manager.physics_list_name = cfg.physics_list
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    reg = sim.physics_manager.add_region("crystals")
    reg.production_cuts.all = 0.1 * mm
    reg.associate_volume(crystal)

    pf_activity = meta["grid_positron_activity_at_bed_start_bq"]
    pf_activity *= crop["activity_bqml_sum"] / max(
        float(ph.read_mhd(Path(cfg.phantom_dir) / "act_bqml.mhd").sum(dtype=np.float64)), 1e-30)
    src = sim.add_source("VoxelSource", "pet")
    src.image = str(out / "act.mhd")
    src.position.translation = [v * mm for v in crop["act_centre_mm"]]
    src.direction.type = "iso"
    if cfg.positron:
        src.particle = "e+"
        src.energy.type = "F18"
    else:
        src.particle = "back_to_back"
        src.direction.accolinearity_flag = True
    src.activity = pf_activity / cfg.threads * Bq
    src.half_life = meta["timing"]["half_life_s"] * sec

    sim.add_actor("SimulationStatisticsActor", "Stats").output_filename = "stats.txt"

    for name, proc in SCATTER_ATTRS.items():
        aux = sim.activate_auxiliary_attribute("ProcessDefinedStepInVolumeAttribute", name)
        aux.process_name = proc
        aux.volume_name = patient.name
        aux.propagate_from_parent_track = True

    def digi(kind, name, inp, **kw):
        a = sim.add_actor(kind, name)
        a.attached_to = crystal.name
        a.authorize_repeated_volumes = True
        if inp is not None:
            a.input_digi_collection = inp
        for k, v in kw.items():
            setattr(a, k, v)
        a.write_to_disk = False
        return a

    hc = digi("DigitizerHitsCollectionActor", "Hits", None,
              attributes=["EventID", "PostPosition", "TotalEnergyDeposit",
                          "PreStepUniqueVolumeID", "GlobalTime", *SCATTER_ATTRS])
    rd = digi("DigitizerReadoutActor", "Readout", hc.name,
              group_volume=block.name, discretize_volume=crystal.name,
              policy="EnergyWeightedCentroidPosition")
    eb = digi("DigitizerBlurringActor", "EnergyBlur", rd.name,
              blur_attribute="TotalEnergyDeposit", blur_method="InverseSquare",
              blur_reference_value=511 * keV,
              blur_resolution=cfg.energy_resolution)
    last = eb.name
    if cfg.deadtime_ns > 0:
        last = digi("DigitizerDeadTimeActor", "DeadTime", eb.name,
                    dead_time=cfg.deadtime_ns * ns, policy="Paralyzable",
                    group_volume=block.name).name
    ew = digi("DigitizerEnergyWindowsActor", "EnergyWindow", last,
              channels=[{"name": "Window", "min": ENERGY_WINDOW_KEV[0] * keV,
                         "max": ENERGY_WINDOW_KEV[1] * keV}])
    sg = digi("DigitizerBlurringActor", SINGLES, "Window",
              blur_attribute="GlobalTime", blur_method="Gaussian",
              blur_fwhm=TIMING_PS / math.sqrt(2.0) * 1e-3 * ns)
    if cfg.write_singles:
        sg.write_to_disk = True
        sg.output_filename = "singles.root"

    for name, offset in ((PROMPTS, 0.0), (DELAYS, cfg.delay_offset_ns)):
        cc = sim.add_actor("CoincidenceSorterActor", name)
        cc.input_digi_collection = sg.name
        cc.window = cfg.window_ns * ns
        cc.offset = offset * ns
        cc.multiples_policy = cfg.multiples_policy
        cc.output_filename = "coinc.root"

    sim.run_timing_intervals = [[cfg.t0 * sec, (cfg.t0 + cfg.seconds) * sec]]
    return sim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="simulation.gate.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    a = ap.parse_args(argv)
    cfg = RunConfig.load(a.config)
    meta = ph.load(cfg.phantom_dir)
    crop = cropped_inputs(cfg, ShieldSpec(**cfg.shield))
    bx = crop["box_mm"]
    print(f"  image box {bx[0]:.0f} x {bx[1]:.0f} x {bx[2]:.0f} mm, CT "
          f"{crop['ct_shape_zyx']} (step {crop['ct_step']}), activity "
          f"{crop['shape_zyx']}; the crop loses "
          f"{100 * crop['activity_lost_to_crop']:.2f} % of the activity and "
          f"{100 * crop['body_voxels_lost_to_crop']:.2f} % of the body", flush=True)
    sim = build(cfg, meta, crop)
    t0 = time.time()
    sim.run()
    wall = time.time() - t0
    stats = sim.get_actor("Stats")
    n_events = int(stats.counts.events) if hasattr(stats, "counts") else None
    info = {"config": asdict(cfg), "crop": crop, "wall_s": wall,
            "events": n_events,
            "expected_decays": meta["grid_positron_activity_at_bed_start_bq"]
            * (1 - crop["activity_lost_to_crop"])
            * ph.decay(cfg.t0, meta["timing"]["half_life_s"])
            * ph.frame_integral_s(cfg.seconds, meta["timing"]["half_life_s"])}
    Path(cfg.out_dir, "run.json").write_text(json.dumps(info, indent=2))
    print(f"  {n_events} events in {wall:.0f} s wall "
          f"(expected {info['expected_decays']:.3g} decays)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
