# lm

List-mode reconstruction. Events are passed directly to PyTomography, with no
intermediate file format and without SIRF or STIR.

```bash
./d710_isolate_stir.sh attn --case ped

conda activate petct_recon
d710 lm check    --case ped --bed 1
d710 lm tofcheck --case ped --bed 1
d710 lm recon    --case ped --tof-bins 55
```

The first command runs once per case in the SIRF runtime, attenuation being the
only term SIRF builds. `lm check` verifies the bin map bit for bit, `lm tofcheck`
determines the direction of the TOF axis, and `lm recon` writes `recon_lm.npz`.

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

| | SIRF/STIR (`sirf-local:0.1`) | PyTomography (`petct_recon`) |
|---|---|---|
| commands | `decode estimate tostir attn osem export` | `lm`, `lowdose` |
| driver | `./d710_isolate_stir.sh` | `./d710` |

Nothing under `lm/` or `lowdose/` imports `sirf` or `stir`. Two changes were
required for that:

* `lm/interfile.py` parses the projdata header itself — segments, ring
  differences, view and tangential counts — instead of consulting
  `ProjDataInfo`. `tests/test_lm_geom.py` checks every number it returns
  against STIR's own whenever STIR happens to be importable.
* `utils/attn.py` writes `attn.hs` and `attn.s` with a header cloned from the
  prompts, so that the whole of `work/bed<n>/` shares one layout and
  `np.fromfile` suffices. SIRF's own writer places the view axis before the
  axial axis and stores segments ascending; `as_array()` conceals that,
  `np.fromfile` does not. `Header.require_plane_major()` refuses a file left
  over in the earlier layout rather than reading it incorrectly.

## Modules

### `geom.py` — crystal identifiers and sinogram bins

| | |
|---|---|
| `scanner_lut()` | `(13824, 3)` crystal centres in mm, serving as PyTomography's `scanner_LUT` and allowing the GATE-style `pet_scanner_info.txt` model to be skipped entirely |
| `tof_meta(n)` | `PETTOFMeta(n, n·c·lsb/2, c·550 ps/2)` |
| `tof_to_stir(bin, n)` | GE's signed `−27..+27` to a 0-based STIR timing position, rebinned to `n`. The reversal is `CListRecordGEHDF5::get_tof_bin() = −deltaTime`; rebinning commutes with it because the factor divides 55 |
| `BinMap` | built from the bed's own header |
| `BinMap.flat(a, b)` | `(N,)` flat index into `(plane, view, tang)`, `−1` outside the sinogram. `with_swap=True` additionally reports which events run against the bin's `(det1, det2)` direction |
| `BinMap.lor_table()` | every valid LOR as `(ids, bin)`; 63,203,328 rows, one per crystal pair |
| `BinMap.mult` | ring pairs per plane, which is 2 at the odd axial positions of segment 0 |

`BinMap` is the inverse of `utils.geometry.det_pair_map` and `crystal_to_det`,
built by filling a `(576, 576)` table from the forward map, so the two cannot
diverge. No part of the geometry is re-derived here.

### `events.py` — the event table

Provides `load`, `bins`, `histogram`, `detector_ids` and `tof_index`.

There are two TOF frames and they are not the same. TOF is a signed displacement
along a *directed* LOR:

* Into a **sinogram bin**, whose direction is the bin's own `(det1, det2)`, so
  the two orderings receive opposite TOF indices (`tof_index`, used by
  `histogram`). On paediatric bed 1, 89.8 % of events are recorded against the
  bin's direction and 10.2 % with it. Treating them alike leaves 3.0 M of 667 M
  bins wrong while the total counts still agree exactly; at 11 bins, the wrong
  global sign is 22.7 M bins out, so the two errors are easily distinguished.
* Into **PyTomography**, which receives `(xtal_a, xtal_b)` in the recorded
  order, so a single global `tof_sign` covers every event (`detector_ids`).

### `terms.py` — per-event weights and additive term

PyTomography's list-mode path requires a weight and a background *per event*,
whereas everything on disk is per *bin*.

| | |
|---|---|
| `read(case, bed, name, binmap)` | one term as a flat float32 array, via `np.fromfile` |
| `lor_sensitivity` | `normdt × attn` per LOR |
| `event_terms(...)` | `(keep, weights, additive)` |
| `sensitivity(...)` | `(ids, weights)` over all 63.2 M valid LORs |
| `scatter_tof_weights(...)` | GE's own `scatter_tof.npy`, otherwise measured from this bed's tail ring |

Two points are easily inverted and both are load-bearing:

* **Multiplicity.** `normdt`, `randoms` and `scatter` are per *bin* and already
  carry the span-2 factor of two at odd segment-0 planes. An event is one LOR,
  so those three are *divided* by `mult` — the exact inverse of the error that
  `utils.geometry.ring_pair_multiplicity` warns about. `attn` is not divided: it
  is the survival probability of one LOR, not a count.
* **Placement of the weight.** PyTomography models `y = Hx + a` with the weight
  present only in the sensitivity image `H̃ᵀw`, so the background must arrive
  *already divided* by the weight: `a = (randoms + scatter) / w`. Multiplying it
  back must recover the bin's background, which is what
  `tests/test_lm_data.py` asserts.

### `recon.py` — LM-OSEM and BSREM

`reconstruct()` returns a `(47, xy, xy)` volume in counts per voxel plus a
sensitivity image, the same shapes `osem/recon.py` produces, so `osem.stitch`
and `utils.export` accept them unchanged. `--beta > 0` substitutes BSREM with a
relative-difference prior for OSEM. `PETLMSystemMatrix` places `1e7` in voxels
its sensitivity never reaches; that is undone before the array is used as a
bed-stitching weight.

## Evidence

| claim | evidence |
|---|---|
| the bin map | histogramming paediatric beds 1–6 reproduces `decoded/bed<n>.s` bit for bit, with no events dropped |
| the TOF index in the bin frame | the same, at 5 and 11 TOF bins; no other combination of the two signs comes within 2.3 M bins |
| the TOF sign in the event frame | `d710 lm tofcheck`: a non-TOF reference image with both signs scored by Poisson log-likelihood. `+1` wins by 98 sd overall and by 28 sd on `\|tof_bin\| ≥ 8` |
| the header parser | every segment, ring difference and ring pair equals STIR's own (`tests/test_lm_geom.py`) |
| the per-event terms | `a × w` reproduces the bin's background exactly, and `Σ w` over LORs equals `Σ normdt × attn` over bins |
| the image frame | `tools/lm_frame.py` on paediatric bed 1: 0° rotation, angular-profile correlation +0.994, and no transpose |

## `lm` compared with `osem` on the same events

Both reconstructed from the same acquisition at 2 × 24, with PSF disabled on
both so that the projectors are compared rather than the resolution models:

| | |
|---|---|
| rotation between the frames | 0°, angular profiles correlated at +0.994 |
| axial orientation | no flip (profile correlation +0.83 against −0.29 flipped) |
| voxel correlation, body mask | +0.69 raw, +0.83 once both are smoothed to a common resolution |

The scale row of this measurement has been withdrawn. It was made at
`--xy 256`, reported `lm / osem` = 0.57 overall with a ratio flat at about 0.60
along the axis, and interpreted that as a constant absorbed by `K`, concluding
that the list-mode path required its own `K`. It does not. The two runs used
different voxel sizes: `osem` ran in `sirf-local:0.1`, whose STIR fixes the FOV
at 718.01 mm and therefore made `--xy 256` mean 2.8047 mm voxels, while `lm`
hard-coded 2.1306 mm. `(2.1306/2.8047)² = 0.577` accounts for the whole offset.
Both paths now take the grid from `utils/scanner.py` (`XY = 337`,
`DR_MM = 2.1306`), which yields 2.130600 mm in either SIRF build, so one `K`
covers both. Re-measure the comparison on the current grid before quoting a
scale figure again.

Quantitative equivalence has not yet been demonstrated. The residual +0.83 is
consistent with the projector difference — STIR ray tracing with 5 tangential
LORs against parallelproj's single-ray Joseph — but that has not been separated
from a genuine modelling error. Measuring `K` on the NEMA case and repeating the
comparison there is the outstanding work.

## Known limitations

* **No GPU in this environment.** `parallelproj` here is CPU-only
  (`libparallelproj_c.so`, without `libparallelproj_cuda.so`).
  `parallelproj/backend.py` detects CUDA through `nvidia-smi` and switches with
  no code change, so a GPU machine requires an install rather than an edit.
* **The scatter TOF profile falls back** to the tail-ring measurement for beds
  estimated without `reconMethod 3`. That fallback is a single global profile,
  whereas GE's own moves by roughly 10 TOF bins across `(view, u)`. Re-estimate
  with `--tof` to obtain the real one.
* The two ring pairs of an odd segment-0 bin receive equal weight when the bin's
  terms are split. Nothing on disk indicates otherwise.

## The transaxial FOV is a disc and must be imposed

The grid is square and the scanner is round. At `XY = 337` the corners lie at
506 mm, while the outermost tangential bin reaches only
`(R + DOI)·sin(π·380/1152) = 356.7 mm`, checked against
`geometry.tangential_s_mm` to float32 precision. Beyond that radius no bin
crosses the voxel at all, so its sensitivity is not merely small but
meaningless: 2e−05 against a peak of 1.16e+04.

`PETLMSystemMatrix._get_object_initial` masks only the axial extent, so the
corners start at 1 and OSEM divides by that 2e−05. Measured on paediatric bed 1
at `--xy 337`, before the correction:

| | counts outside 357 mm | peak outside | peak inside |
|---|---|---|---|
| non-TOF | 34.4 % | 346.06 | 6.59 |
| TOF, 55 bins | 0.04 % | 0.34 | 6.86 |

OSEM conserves counts, so that 34 % was missing from the patient, and from `K`.
TOF barely shows the effect only because TOF localisation forbids placing
activity there; it is constrained rather than immune.

The correction is `scanner.fov_mask` applied to the *initial estimate* rather
than as a clip at the end: OSEM is multiplicative, so zero remains zero and the
counts land inside instead of being discarded afterwards. `osem/` starts from
the same masked image for the same reason. GE's own `PT` series are likewise a
round FOV in a square matrix.

### The axial mask cannot be borrowed either, because it costs plane 46

The other half of `_get_object_initial` is the axial cut, and until 2026-09-05
that half was still PyTomography's. It zeroes `[:, :, :ceil(zmin)]` and
`[:, :, floor(zmax):]` in units of planes, and this grid is laid exactly on the
ring extent — 47 planes of `PLANE_MM` against 24 rings of `2·PLANE_MM` — so
`zmax` is 46.0 to within a float32 ulp and `floor` removes plane 46 from the
object. It always does: adjusting `RING_PITCH_MM` only moves the problem from
one end to the other, because `ceil` spares plane 0 at 0.0 while `floor`
eliminates plane 46 at 46.0.

A plane that starts at zero remains zero, so image plane 46 of every bed
returned identically zero. `osem.stitch` then averaged that hard zero into the
seam with the weight `norm_BP` actually assigns it — the axial sensitivity is a
symmetric triangle worth 0.043 of a mid-bed plane on the axis at both plane 0
and plane 46 — so the seam plane was deflated by `0.380/(0.380+0.043)`, while
the counts the model could not place accumulated in plane 45 beside it. Measured
on `fdg26081901`, all seven beds, per-plane totals against GE's own VPFXS of the
same exam and detrended:

| local plane of the lower bed | 44 | 45 | 46 |
|---|---|---|---|
| list mode | −4.6 % | +13.7 % | −2.6 % |
| sinogram (STIR truncates nothing) | +0.5 % | −0.4 % | −1.8 % |

Smeared over three planes by the `[1, 4, 1]` post-filter, this is the bright
line visible at every bed junction in the coronal and sagittal views. The top
plane of the whole volume, which has no bed above it to dilute the zero, is the
clearest proof: deconvolving `recon_lm.npz` plane 274 back through the axial
post-filter gives exactly 0 with no negative voxels, where `recon.npz` gives
47 % negative.

`recon.axial_mask` retains the guard — a grid taller than the ring extent is
still cut at both ends — but rounds rather than truncating: a plane whose centre
lies within half a plane of the outermost ring is inside the FOV.
