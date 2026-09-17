# lm

List-mode reconstruction. Events are passed directly to PyTomography, with no
intermediate file format and without SIRF or STIR.

```bash
conda activate petct_recon
d710 attn        --case ped
d710 lm check    --case ped --bed 1
d710 lm tofcheck --case ped --bed 1
d710 lm recon    --case ped --tof-bins 55
```

`d710 attn` runs once per case and builds the only term that is not GE's:
attenuation, by projecting the CT mu-map with parallelproj
(`utils/attn_proj.py`). It used to be the one step on this path that needed
SIRF and no longer is. `lm check` verifies the bin map bit for bit,
`lm tofcheck` determines the direction of the TOF axis, and `lm recon` writes
`recon_lm.npz`.

## Rationale

The sinogram path spends 2.2 hours per bed because STIR with parallelproj
projects the entire 303 M-bin TOF sinogram once per subset. List mode is a
single pass over the 18–87 M events that were actually detected, and it retains
the native 89.2459 ps bins instead of rebinning 55 into 5.

Measured on this CPU, paediatric bed 1, 18,759,294 events, grid 337, 2 × 24
iterations and subsets, PSF 6.4:

| | sensitivity image | OSEM 2×24 | wall clock |
|---|---|---|---|
| TOF, all 55 bins | 54 s | 62 s | 2 m 01 s |
| non-TOF | 56 s | 95 s | 2 m 39 s |

TOF is the faster of the two. Each event's ray is truncated to ±3σ about its own
TOF position and is therefore shorter than the full chord — the opposite of the
sinogram path, where a TOF axis multiplies the work. The note in `d710` that TOF
is disabled by default on account of its cost does not apply here.

## The runtime boundary

| | `d710:full` | host python (`petct_recon`) | `sirf-local:0.1` |
|---|---|---|---|
| commands | `decode estimate tostir exam` | `attn lm lowdose export` | `osem` only |
| driver | `./d710` or `./d710_apptainer` | `./d710` | `./d710_isolate_stir.sh` |

Nothing this path touches imports `sirf` or `stir`. Three changes were required
for that:

* `utils/interfile.py` parses the projdata header itself — segments, ring
  differences, view and tangential counts — instead of consulting
  `ProjDataInfo`. `tests/test_lm_geom.py` checks every number it returns
  against STIR's own whenever STIR happens to be importable.
* `utils/attn.py` writes `attn.hs` and `attn.s` with a header cloned from the
  prompts, so that the whole of `work/bed<n>/` shares one layout and
  `np.fromfile` suffices. SIRF's own writer places the view axis before the
  axial axis and stores segments ascending; `as_array()` conceals that,
  `np.fromfile` does not. `Header.require_plane_major()` refuses a file left
  over in the earlier layout rather than reading it incorrectly.
* `utils/attn_proj.py` computes the attenuation factors themselves, with
  `parallelproj.joseph3d_fwd` over the crystal positions, in the same world
  frame `PETLMSystemMatrix` uses. It takes the ring pairing from
  `BinMap.ring_pairs_by_plane()` rather than restating it, so `d710 lm check`
  proves the attenuation geometry at the same time as the bin map.

## The ring pairing

A plane merges several detector-ring pairs; which ring sits on `det1` decides
how the LOR tilts, and reversing it mirrors the segment axis (`+s` for `-s` at
the same axial index, segment 0 unchanged). `utils/binmap.py` states the
convention, records the measurement, and is the only place it is written down.

The measurement is `d710 lm check`: the events from GE's `LIST*.BLF`,
re-histogrammed through `BinMap.flat()`, against the prompts sinogram the same
decoder produced from GE's independent `SINO*`. Zero bins differ; reversing the
pairing makes 24-29 M bins differ, and puts segment `+s`'s counts exactly where
segment `-s`'s belong.

Since 2026-09-18 the decoder declares the ring differences with STIR's sign and
`BinMap` pairs the rings the same way. The two flips cancel, so nothing on this
path changed by a single bit -- the point of that change was `d710 osem`, which
reads its geometry from the header and was mirrored until then (r against GE
0.9602 -> 0.9728). **A case decoded before that date must be decoded again**, or
`lm check` fails.

Until 2026-09-18 `attn.hs` also used the other pairing, having been fixed
against SIRF's own factors -- circular, since every other term in
`work/bed<n>/` comes from GE. That one did change the numbers, though less than
its 1.7 % rms suggests: measured on one bed with everything else held fixed,
+0.21 % on the image. The mirror is nearly harmless because `af(+s)` and
`af(-s)` differ by only 0.2 % in segment 0 -- where it is exactly zero -- rising
to 2.4 % at `|s| = 11`, which carries 0.8 % of the counts. The two rays share a
midpoint and a tilt magnitude and differ only in tilt sign.
