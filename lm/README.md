# `lm/` — list-mode reconstruction

Events straight into PyTomography. No file format in between, no SIRF, no STIR.

```bash
# once per case, in the SIRF runtime (attenuation is the only term SIRF builds)
./d710_isolate_stir.sh attn --case ped

conda activate petct_reconstruction              # PyTomography runtime
d710 lm check    --case ped --bed 1              # the bin map, bit-exact
d710 lm tofcheck --case ped --bed 1              # which way the TOF axis runs
d710 lm recon    --case ped --tof-bins 55        # -> recon_lm.npz
```

## Why this exists

The sinogram path spends 2.2 h/bed because STIR-parallelproj projects the whole
303 M-bin TOF sinogram once per subset. List mode is one pass over the 18–87 M
events that were actually detected, and it keeps the native 89.2459 ps bins
instead of mashing 55 → 5.

Measured on this CPU, ped bed 1, 18,759,294 events, grid 337, 2 × 24, PSF 6.4:

| | sensitivity image | OSEM 2×24 | wall |
|---|---|---|---|
| **TOF, all 55 bins** | 54 s | **62 s** | **2 m 01 s** |
| non-TOF | 56 s | 95 s | 2 m 39 s |

**TOF is the faster one.** Each event's ray is cut to ±3σ about its own TOF
position, so it is shorter than the full chord — the opposite of the sinogram
path, where a TOF axis multiplies the work. Nothing about `d710`'s "TOF is off
by default because of the time cost" applies here.

## Two runtimes, and the boundary between them

| | SIRF / STIR (`sirf-local:0.1`) | PyTomography (`petct_reconstruction`) |
|---|---|---|
| commands | `decode estimate tostir attn osem export` | `lm`, `lowdose` |
| driver | `./d710_isolate_stir.sh` | `./d710` |

Nothing under `lm/` or `lowdose/` imports `sirf` or `stir`. Two things had to
change for that:

* **`lm/interfile.py`** parses the projdata header itself — segments, ring
  differences, view/tangential counts — instead of asking `ProjDataInfo`.
  `tests/test_lm_geom.py` checks every number it returns against STIR's own,
  whenever STIR happens to be importable.
* **`utils/attn.py`** now writes `attn.hs`/`.s` with a header cloned from the
  prompts, so the whole of `work/bed<n>/` is in one layout and `np.fromfile` is
  enough. SIRF's own writer puts view before axial and stores segments
  ascending; `as_array()` hides that, `np.fromfile` does not.
  `Header.require_plane_major()` refuses a file left over in the old layout
  rather than reading it wrong.

## The modules

### `geom.py` — crystal ids ↔ sinogram bins

| | |
|---|---|
| `scanner_lut()` | `(13824, 3)` crystal centres in mm — PyTomography's `scanner_LUT`, and what lets it skip the GATE-style `pet_scanner_info.txt` model entirely |
| `tof_meta(n)` | `PETTOFMeta(n, n·c·lsb/2, c·550 ps/2)` |
| `tof_to_stir(bin, n)` | GE's signed `-27..+27` → 0-based STIR timing position, mashed to `n`. The reversal is `CListRecordGEHDF5::get_tof_bin() = -deltaTime`; mashing commutes with it because the factor divides 55 |
| `BinMap` | built from the bed's own header |
| `BinMap.flat(a, b)` | `(N,)` flat index into `(plane, view, tang)`, `-1` outside the sinogram. `with_swap=True` also returns which events run against the bin's `(det1, det2)` direction |
| `BinMap.lor_table()` | every valid LOR as `(ids, bin)` — 63,203,328 of them, one row per crystal pair |
| `BinMap.mult` | ring pairs per plane; 2 at the odd axial positions of segment 0 |

`BinMap` is the **inverse** of `utils.geometry.det_pair_map` and
`crystal_to_det`, built by filling a `(576, 576)` table from the forward map, so
the two cannot drift apart. Nothing about the geometry is re-derived here.

### `events.py` — the event table

`load`, `bins`, `histogram`, `detector_ids`, `tof_index`.

**Two TOF frames, and they are not the same.** TOF is a signed displacement
along a *directed* LOR:

* into a **sinogram bin**, whose direction is the bin's own `(det1, det2)`, so
  the two orderings get opposite TOF indices (`tof_index`, used by
  `histogram`). On ped bed 1, 89.8 % of events are recorded against the bin's
  direction and 10.2 % with it. Treating them alike leaves 3.0 M of 667 M bins
  wrong while the total counts still match perfectly — and at 11 bins the wrong
  global sign is 22.7 M bins out, so the two are easy to tell apart;
* into **PyTomography**, which gets `(xtal_a, xtal_b)` in the recorded order, so
  one global `tof_sign` covers every event (`detector_ids`).

### `terms.py` — the per-event weights and additive term

This is what was missing before: PyTomography's list-mode path needs a weight
and a background *per event*, and everything on disk is per *bin*.

| | |
|---|---|
| `read(case, bed, name, binmap)` | one term as a flat float32 array, `np.fromfile` |
| `lor_sensitivity` | `normdt × attn` per LOR |
| `event_terms(...)` | `(keep, weights, additive)` |
| `sensitivity(...)` | `(ids, weights)` over all 63.2 M valid LORs |
| `scatter_tof_weights(...)` | GE's own `scatter_tof.npy`, else measured from this bed's tail ring |

Two things are easy to get backwards and both are load-bearing:

* **Multiplicity.** `normdt`, `randoms` and `scatter` are per *bin* and already
  carry the span-2 factor of two at odd segment-0 planes. An event is one LOR,
  so those three are **divided** by `mult` — the exact inverse of the mistake
  `utils.geometry.ring_pair_multiplicity` warns about. `attn` is not divided: it
  is a survival probability of one LOR, not a count.
* **Where the weight goes.** PyTomography models `y = Hx + a` with the weight
  only in the sensitivity image `H̃ᵀw`, so the background must arrive **already
  divided** by the weight: `a = (randoms + scatter) / w`. Multiply it back and
  you must recover the bin's background — `tests/test_lm_data.py` asserts exactly
  that.

### `recon.py` — LM-OSEM / BSREM

`reconstruct()` returns `(47, xy, xy)` count/voxel plus a sensitivity image, the
same shapes `osem/recon.py` produces, so `osem.stitch` and `utils.export` take
them unchanged. `--beta > 0` switches OSEM for BSREM with a relative-difference
prior. `PETLMSystemMatrix` parks `1e7` in voxels its sensitivity never reaches;
that is undone before the array is used as a bed-stitching weight.

## What is proven, and how

| claim | evidence |
|---|---|
| the bin map | histogramming ped beds 1–6 reproduces `decoded/bed<n>.s` **bit for bit**, 0 events dropped |
| the TOF index in the bin frame | same, at 5 and 11 TOF bins (`pedtof5`, `pedtof`) — and no other combination of the two signs comes within 2.3 M bins |
| the TOF sign in the event frame | `d710 lm tofcheck`: non-TOF reference image, both signs scored by Poisson log-likelihood. `+1` by 98 sd overall and 28 sd on `\|tof_bin\| ≥ 8` |
| the header parser | every segment, ring difference and ring pair equals STIR's own (`tests/test_lm_geom.py`) |
| the per-event terms | `a × w` reproduces the bin's background exactly; `Σ w` over LORs equals `Σ normdt × attn` over bins |
| the image frame | `tools/lm_frame.py` on ped bed 1: **0° rotation, angular-profile corr +0.994**, and no transpose |

## `lm` against `osem`, ped bed 1, same events

Both reconstructed from the same acquisition, 2 × 24, PSF off on both so the
projectors are compared and not the resolution models:

| | |
|---|---|
| rotation between the frames | **0°**, angular profiles corr **+0.994** |
| axial | no flip (profile corr +0.83 against −0.29 flipped) |
| voxel corr, body mask | +0.69 raw, **+0.83** once both are smoothed to a common resolution |

⚠ **The measurement above was made at `--xy 256` and its scale row has been
withdrawn.** It reported `lm / osem` = 0.57 total and a ratio flat at ~0.60 down
the axis, and read that as a constant `K` absorbs — so the list-mode path was
said to need its own `K`. It does not. The two runs were on **different voxel
sizes**: `osem` ran in `sirf-local:0.1`, whose STIR pins the FOV at 718.01 mm and
made `--xy 256` mean 2.8047 mm voxels, while `lm` hard-coded 2.1306 mm.
`(2.1306/2.8047)² = 0.577` — the whole offset. Both paths now take the grid from
`utils/scanner.py` (`XY = 337`, `DR_MM = 2.1306`), which lands on 2.130600 mm in
either SIRF build, so **one `K` covers both**. Re-measure the comparison on the
new grid before quoting a scale number again.

What is **not** yet demonstrated is quantitative equivalence. The residual +0.83
is consistent with the projector difference — STIR ray-tracing with 5 tangential
LORs against parallelproj's single-ray Joseph — but that has not been separated
from a real modelling error. Measuring `K` on the NEMA case, and repeating the
comparison there, is the outstanding work.

## Known limits

* **No GPU here.** `parallelproj` in this env is CPU-only (`libparallelproj_c.so`,
  no `libparallelproj_cuda.so`). `parallelproj/backend.py` detects CUDA through
  `nvidia-smi` and switches with no code change, so a GPU box needs an install,
  not an edit.
* **The scatter TOF profile falls back** to the tail-ring measurement for beds
  estimated without `reconMethod 3`. It is one global profile; GE's own moves by
  ~10 TOF bins across `(view, u)`. Re-estimate with `--tof` for the real one.
* The two ring pairs of an odd segment-0 bin are given **equal** weight when the
  bin's terms are split. There is nothing on disk that says otherwise.

## The transaxial FOV is a disc, and it has to be imposed

The grid is square, the scanner is round. At `XY = 337` the corners sit at
506 mm while the outermost tangential bin reaches only
`(R + DOI)·sin(π·380/1152) = 356.7 mm` — checked against
`geometry.tangential_s_mm` to float32. Past that radius **no bin crosses the
voxel at all**, so its sensitivity is not small but meaningless: 2e-05 against a
peak of 1.16e+04.

`PETLMSystemMatrix._get_object_initial` masks only the **axial** extent, so the
corners start at 1 and OSEM divides by that 2e-05. Measured on ped bed 1 at
`--xy 337`, before the fix:

| | counts outside 357 mm | peak outside | peak inside |
|---|---|---|---|
| non-TOF | **34.4 %** | 346.06 | 6.59 |
| TOF 55 bins | 0.04 % | 0.34 | 6.86 |

OSEM conserves counts, so that 34 % was **missing from the patient** — and from
`K`. TOF barely shows it only because TOF localisation forbids putting activity
there; it is not immune, just constrained.

The fix is `scanner.fov_mask` applied to the **initial estimate**, not as a clip
at the end: OSEM is multiplicative, so zero stays zero and the counts land
inside instead of being thrown away afterwards. `osem/` starts from the same
masked image, for the same reason. GE's own `PT` series are a round FOV in a
square matrix too.
