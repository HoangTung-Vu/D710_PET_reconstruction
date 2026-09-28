"""The D710 as opengate volumes: 256 flat blocks of 9 x 6 LYSO crystals.

Where every crystal goes is decided in `simulation/crystals.py`, in plain
numpy, so the GE-id map can be tested without opengate. This module only
hands those placements to GATE.

The end shields are an assumption, not a measurement: the D710 has lead
shielding at both ends of the detector ring, but its dimensions are not in any
file we hold. `ShieldSpec` is therefore off by default and opt-in: with the
whole GE image as the source, the simulated singles are already 12 % BELOW
the real ones without any shield (and 24 % below with this one), because
the activity outside the image is missing. Tune it against the real singles
only once that activity is accounted for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import crystals as cr

RING_RMIN_MM = 404.0

RING_RMAX_MM = 432.0


@dataclass
class ShieldSpec:
    """Two lead annuli, one at each end of the detector ring."""

    enabled: bool = False
    rmin_mm: float = 380.0
    rmax_mm: float = RING_RMAX_MM
    thickness_mm: float = 30.0
    gap_mm: float = 2.0
    material: str = "G4_Pb"


def ring_half_length_mm() -> float:
    return cr.N_BLOCKS_Z * cr.BLOCK_SIZE_MM[2] / 2.0 + 1.0


def bore_radius_mm(shield: ShieldSpec) -> float:
    """The largest radius a patient volume may reach without overlapping the scanner."""
    return min(RING_RMIN_MM, shield.rmin_mm) if shield.enabled else RING_RMIN_MM


def add_materials(sim) -> None:
    """LYSO as the Philips model in `opengate.contrib.pet` defines it."""
    import opengate as gate

    db = sim.volume_manager.material_database
    if "LYSO" not in getattr(db, "material_builders", {}):
        db.add_material_nb_atoms("LYSO", ["Lu", "Y", "Si", "O"], [18, 2, 10, 50],
                                 7.1 * gate.g4_units.g_cm3)


def add_d710(sim, name: str = "d710", shield: ShieldSpec | None = None):
    """Add the ring (and the shields); returns `(ring, block, crystal)` volumes."""
    import opengate as gate
    from opengate.geometry.utility import get_grid_repetition

    mm = gate.g4_units.mm
    shield = shield or ShieldSpec()
    add_materials(sim)

    ring = sim.add_volume("Tubs", name)
    ring.rmin = RING_RMIN_MM * mm
    ring.rmax = RING_RMAX_MM * mm
    ring.dz = ring_half_length_mm() * mm
    ring.material = "G4_AIR"

    block = sim.add_volume("Box", f"{name}_block")
    block.mother = ring.name
    block.size = [v * mm for v in cr.BLOCK_SIZE_MM]
    block.material = "G4_AIR"
    tr, rot = cr.block_placements()
    block.translation = [[v * mm for v in t] for t in tr]
    block.rotation = rot

    crystal = sim.add_volume("Box", f"{name}_crystal")
    crystal.mother = block.name
    crystal.size = [v * mm for v in cr.CRYSTAL_MM]
    crystal.material = "LYSO"
    crystal.translation = get_grid_repetition(
        [1, cr.BLOCK_CRYSTALS[0], cr.BLOCK_CRYSTALS[1]],
        [0, cr.PITCH_T * mm, cr.PITCH_Z * mm])

    if shield.enabled:
        s = sim.add_volume("Tubs", f"{name}_shield")
        s.rmin = shield.rmin_mm * mm
        s.rmax = shield.rmax_mm * mm
        s.dz = shield.thickness_mm / 2.0 * mm
        s.material = shield.material
        zc = ring_half_length_mm() + shield.gap_mm + shield.thickness_mm / 2.0
        s.translation = [[0, 0, zc * mm], [0, 0, -zc * mm]]

    return ring, block, crystal


def patient_half_width_mm(shield: ShieldSpec) -> float:
    """Half the side of the largest square image box that fits inside the bore."""
    return (bore_radius_mm(shield) - 1.0) / math.sqrt(2.0)
