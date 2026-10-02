# simulation

This package simulates D710 raw data, in list mode and as a sinogram, from a real exam's CT and its clinical PET. There are three methods:

* **GATE 10** (opengate) is the formal one: a Monte Carlo of the scanner around the patient's CT, with the PET as the source.
* **parallelproj** is its analytic twin, the forward model `y = S(Gx) + b` evaluated per crystal pair and TOF bin.
* **analytic** needs no Monte Carlo at all: parallelproj trues, single-scatter simulation and randoms from modelled singles, calibrated once on a real exam. Non-TOF, ~1.5 min per bed once its scatter and singles are cached. See the next section.

Each simulated bed is written as an ordinary case **inside** the real one, `<case>/sim_<method>_s<seed>/`, with the phantom and the caches in `<case>/sim_cache/`. `d710 lm check`, `d710 lm recon`, `d710 sino` and `d710 export` run on it unchanged, given the case directory as the output root: `--out $D710_OUT/<case> --case sim_an_s1`. `d710 simulate compare` then puts it next to the real raw data of the same bed.

```bash
conda activate petct_recon
export D710_OUT=~/UET/Handson_PET_CT_Reconstruction/d710_out
d710 simulate phantom --case fdg26081008 --bed 1              # CT + PET on the bed grid
d710 simulate gate    --case fdg26081008 --bed 1 --seconds 3  # ~28 min per simulated second
d710 simulate pp      --case fdg26081008 --bed 1 --seconds 3  # ~20 min per bed
d710 simulate compare --case fdg26081008 --bed 1
d710 lm check --out $D710_OUT/fdg26081008 --case sim_gate_s1 --bed 1  # bit-exact, like a real case
```

## Without GATE: `analytic`

```bash
d710 simulate calibrate --case fdg26081901          # once; writes analytic_calib.json + crystal_eff.npy (~50 min)
d710 simulate phantom   --case fdg26081901 --depostfilter-iters 5   # GE's post-filter undone first
d710 simulate analytic  --case fdg26081901          # -> fdg26081901/sim_an_s1; --label an_rl5_s1 -> sim_an_rl5_s1
d710 simulate check-terms --case fdg26081901        # simulated randoms/scatter vs GE's and vs the tails
d710 lm check --out $D710_OUT/fdg26081901 --case sim_an_s1 --bed 1   # bit-exact
d710 sino     --out $D710_OUT/fdg26081901 --case sim_an_s1           # non-TOF, like the simulation
d710 export   --out $D710_OUT/fdg26081901 --case sim_an_s1 --sino --format nifti

# a PET/CT with no raw data: a header-only case, <case>/sim_virtual/, built around it from a template exam
d710 simulate virtual  --case thyr_trainset_0002_20191016 --template fdg26081008 \
    --ct CT.nii.gz --pet SUV.nii.gz --pet-units suv
d710 simulate analytic --case thyr_trainset_0002_20191016        # finds sim_virtual/ by itself
# lympho2: the patient's own dose, weight and times instead of the template's
d710 simulate virtual  --case lympho2_fdg26031106 --template fdg26081008 --pet-units suv \
    --ct source/CT.nii.gz --pet export/lympho2_fdg26031106_ge_suvbw.nii.gz --exam source/manifest.json
```

Scatter (SSS) and the singles model run on the GPU whenever torch sees one: `--device auto` is the default, `cpu` or `cuda` forces it, and so does `D710_SIM_DEVICE`. On sslab's Z820, SSS on the CPU took 580-700 s of the ~900 s a bed needed; the GPU batch is sized from free memory (up to 256 scatter points, 8192 singles voxels). The result does not depend on the device or the batch beyond float rounding (`tests/test_simulation_analytic.py`), so the caches, which are keyed by the phantom and the parameters only, are shared between the two.

Per LOR `(a, b)` in bin `beta`: `y = kappa normdt AF P[x] + k_s normdt SSS + 2w I2 S_a S_b`.

| part | how | constant (fitted on beds 2, 3, 5, 6 of fdg26081901) |
|---|---|---|
| trues | `joseph3d_fwd` through the PET (decays per mm^3), blurred by GE's PSF (`--psf`, 4.87/4.87/4.45 mm), and the CT's mu | `kappa` |
| scatter | PyTomography's single-scatter kernel (Watson 2007), summed over points, activity and mu 24 planes beyond each end of the bed | `k_s`, weight `normdt` |
| singles | `eff_i (c_s G_i + c_0)`, `G_i` = attenuated solid angle of every voxel of the whole image | `c_s`, `c_0` (mean of fdg26081901 and fdg26081008), `crystal_eff.npy` |
| out-of-FOV photons | pass the lead end shields only through the patient port (350 mm), shield plane 30 mm beyond the crystals | chosen among three |

`decoded/bed<n>.singles.npy` is in GE's hardware order (modules of 2 x 4 blocks); `singles.measured()` puts it in crystal-id order. Without it, randoms from singles are 30-50 % wrong plane by plane.

Measured on fdg26081901 (held-out beds 1, 4, 7), simulated / real: prompts 0.969, 1.023, 0.978; randoms / GE's 0.949, 1.026, 1.082; scatter / GE's 0.971, 1.061, 0.956. The weakest part is the singles' axial shape on the first and last bed. Details, and every number: `.claude/audit/simulation-analytic/ANALYTIC.md`.

`c_0`, the singles every crystal sees whatever the image holds, does not carry over between patients. Refitted on four of the five raw FDG exams (2026-10-02, leave one exam out), it ranges 82-267 cps while `c_s` moves 2.55-3.14. `analytic_calib.json` holds `c_0` = 240, a round value chosen after that test. With it, the simulated prompts are within -9 to +1 % of the real ones per bed on the three larger patients, and up to +44 % on the leg beds of a 14 kg child. The randoms error changes the image noise, not its mean, because `d710 sino` subtracts the simulation's own expected randoms. Delays are `2 tau S^2` on 36 of the 37 beds of the five FDG exams, so the randoms are only as good as the singles. For a case with raw data, `--singles measured` stays exact. Details: `.claude/audit/simulation-analytic/singles_5case/RESULTS.md`.

The PET a simulation starts from is GE's image, which went through the PSF, GE's reconstruction and GE's 6.4 mm + [1,4,1] post-filter. Simulated from as it is, then reconstructed and post-filtered again, it ends at 0.76 of GE's lesion SUVmax (median of 10 lymphoma lesions); without our post-filter it is 0.99. `--depostfilter-iters N` on `phantom` and `virtual` undoes GE's post-filter first, with N Richardson-Lucy iterations of `osem.stitch.post_filter` itself on the PET's own grid, and `analytic` blurs the activity of the trues by the PSF the reconstruction models (`--psf`, default GE's; 0 turns it off). Scatter and singles are computed from the unblurred activity. The PET is resampled onto the bed grid with cubic splines (linear would blur back about 2.6 mm), and `phantom.json` records both choices, so the scatter and singles caches follow them. N is chosen against the real raw data of fdg26081008, not guessed; each iteration brings back some of the noise GE's filter had smoothed.

Every bed also writes `raw_simulation/bed<n>_x_true.npy`: the object before the PSF, on the bed grid, (z, y, x), in the units of `work/bed<n>/sino.npz` `img` (`kappa` times decays per voxel). Post-filtered, it is GE's image again, so it is the reference both for the reconstruction and for a network.

A virtual case borrows the template's bed 4 `normdt` (symlinked by absolute path: keep the template unpruned, and build the virtual case on the machine that runs it), its dose, weight and injection time for SUV -> Bq/mL, and its bed spacing; the beds are laid over the NIfTI's own z range, and an image spanning a whole number of beds (to a thousandth of a step) gets exactly that many. The template's identifying header fields are not copied. `--exam manifest.json` replaces the dose, weight, half-life, injection and scan-start times with the patient's own, read from a lympho2 manifest's `SUV` block; its local times are shifted to UTC by `--utc-offset-h` (7), and the decayed dose it recomputes must match the manifest's.

## One night of GATE: `overnight.sh`

```bash
export D710_OUT=~/UET/Handson_PET_CT_Reconstruction/d710_out
export D710_PYTHON=~/miniconda3/envs/petct_recon/bin/python
nohup D710/simulation/overnight.sh > ~/overnight_nohup.out 2>&1 &   # BEDS="7 6" SIM_SECONDS=9 HOURS=8
```

It simulates each bed in `$BEDS` to `$SIM_SECONDS` in 1 s chunks, and starts a chunk only if it can end before the deadline. It resumes where it stopped. Every time a bed completes, it converts, reconstructs with `d710 lm recon` (the simulation's own randoms and scatter), exports, and runs both comparisons:

* `d710 simulate compare`: raw data against the real exam
* `python -m simulation.compare_images`: the SUV image against the real exam thinned to the same time (`<case>_lowcount_time`), our full-dose image and GE's

The default beds are the head and neck (7, then 6). Bed 7 takes ~5 h here. It refuses to start if the case fails `d710 lm check`, i.e. was decoded before the 2026-09-18 ring-pairing fix.

## GATE on a host with an old glibc (the workstation)

`opengate-core` publishes only `manylinux_2_34` wheels and no source distribution, so on Ubuntu 20.04 (glibc 2.31) `pip install opengate` fails with "No matching distribution found for opengate-core". Geant4 then runs in a container while everything after it -- the conversion, `lm recon`, the comparisons -- stays on the host, which needs only numpy, uproot and PyTomography.

```bash
docker build -t d710:gate D710/simulation/gate      # where docker is; Debian 12, glibc 2.36
sif-convert d710:gate                               # as root -> /home/shared/apptainer/images/d710_gate.sif
echo D710_GATE_SIF=/home/shared/apptainer/images/d710_gate.sif >> D710/.env
./d710_apptainer doctor                             # reports the image and whether opengate imports
./d710_apptainer simulate gate --case fdg26081008 --bed 7 --seconds 9
```

To build the image on another machine instead: `docker save d710:gate | gzip > d710_gate.tar.gz`, copy it over, then `zcat d710_gate.tar.gz | docker load && sif-convert d710:gate`.

`d710_apptainer` sets `D710_GATE_RUNNER` to `apptainer exec --bind <repo> --bind $D710_OUT <sif> python`, which [`gate/driver.py`](gate/driver.py) uses in place of the host interpreter. Set that variable yourself to run Geant4 any other way. The image carries the 2.2 GB of Geant4 data, so it needs no network and nothing writable.

## Inputs

| input | default | also accepted |
|---|---|---|
| CT, HU | `<case>_ct.nii.gz` (in `export/` or `Pipeline reproduced/<case>/`) | a CT DICOM folder |
| PET | `export/<case>_ge_suvbw.nii.gz`, GE's clinical reconstruction in SUVbw | Bq/mL NIfTI, or the BQML DICOM folder |
| geometry | `utils/scanner.py`, `utils/geometry.py` | |
| norm × dead time | `work/bed<n>/normdt.hs` (GE's) | |
| clock, dose, weight | `decoded/bed<n>.json` | `--activity` |

NIfTI is read in the RAS convention that `tools/ct_nifti.py` and `tools/dicom_suv.py` write, and turned back into LPS. Both volumes are then resampled onto the bed grid by `utils.attenuation.resample_to_bed`. That is the resampler `mu_map` uses, split out of it without any change in behaviour (`tests/test_simulation.py`), so both volumes land in the orientation that was verified on NEMA.

### An SUV map plus one global activity is enough

SUVbw is Bq/mL times a single constant, so the map already has the shape of the activity. GATE's voxel source uses only that shape. As its documentation puts it, the image "is internally normalized such that the sum of all pixels values is 1", while the total is set by `activity`. There are three input modes (`--pet-units`):

| mode | Bq/mL | needs |
|---|---|---|
| `bqml` | the image | — |
| `suv` | `SUV · D · 2^(−(t_scan − t_inj)/T½) / W` | `bed<n>.json` only |
| `relative` | `A · 2^(−(t_scan − t_A)/T½) · map / Σ(map·V)` | `--activity MBq@YYYYmmddHHMMSS` (UTC) |

**The dose is net of the residual.** The DICOM `RadionuclideTotalDose`, and so every SUV made from it, is `dose_mbq − residual_dose_mbq` (281.2 = 284.9 − 3.7 MBq on fdg26081008). With that, the SUV NIfTI inverts to GE's BQML DICOM **exactly**: the ratio is 1.00000 over every voxel. The gross dose puts every activity 1.3 % high. `t_scan`, the first bed's `bed_start_time` in UTC, equals the DICOM `SeriesTime`, which is local time (UTC+7).

In `relative` mode, `A` is taken as the activity inside the image. On fdg26081008 the image holds 82 % of the decayed dose; the rest is excreted or lies outside the scanned range.

## The grid, and why it reaches so far

The grid is the reconstruction's own: 337 × 337 at 2.1306 mm, with planes at 3.27 mm. It extends 900 mm beyond each end of the bed, enough to take in the whole 896 mm WB image from any bed. The margin matters to GATE, measured on bed 1 with 20 ms runs:

| margin | end shields | singles / real | prompts / real | delays / real |
|---|---|---|---|---|
| 200 mm | assumed | 0.53 | 0.80 | 0.23 |
| 450 mm | assumed | 0.70 | 0.89 | 0.45 |
| 900 mm | assumed | 0.76 | 0.91 | 0.53 |
| **900 mm** | **none (default)** | **0.88** | **0.98** | **0.77** |

The trues do not move (~70 kcps): the margin feeds singles, and so randoms and some scatter. The singles that are still missing belong to activity the image never held. The deficit is largest at ring 0, next to the legs below bed 1. The end shields are therefore opt-in (`--shield`). The D710 has them, but their dimensions are not in any file we hold, and adding them only widens the gap until the missing activity is accounted for.

## Method 1: GATE (`gate/`)

Everything below is from the opengate 10.1.1 documentation or source. The detector numbers are from the GATE model of the GE D690 (PMC7875045), which has the D710's detector.

| part | setting | source |
|---|---|---|
| crystals | 13,824 LYSO, 4.2 × 6.3 × 25 mm, 9 × 6 per block, 64 blocks per ring, 4 blocks axially | D690 paper |
| placement | flat blocks, front face at `R_MM` = 405.1 mm, transaxial pitch `2R tan(π/64)/9` = 4.4225 mm | `crystals.py` |
| patient | `Image` volume + `HounsfieldUnit_to_material` (Schneider 2000 tables, `opengate/data`) | opengate docs |
| source | `VoxelSource`, `back_to_back` ("an alias for colinear gamma pairs of 511 keV") with accolinearity; `--positron` for `e+` on the F-18 spectrum | opengate docs |
| readout | `DigitizerReadoutActor`, `EnergyWeightedCentroidPosition` per block, discretised to the crystal | opengate docs |
| energy | `DigitizerBlurringActor`, InverseSquare, 12 % FWHM at 511 keV | D690 paper: 10–20 % |
| window | `DigitizerEnergyWindowsActor`, 425–650 keV | D690 paper, `bed<n>.json` |
| timing | Gaussian `GlobalTime` blur, 675/√2 ps per single | `TIMING_PS` |
| coincidences | online `CoincidenceSorterActor`, window 2.4545 ns, `TakeWinnerIfAllAreGoods` | see below |
| delays | the same sorter with `offset` = 500 ns ("for estimating the number of random coincidences") | opengate source, D690 paper |
| scatter truth | `ProcessDefinedStepInVolumeAttribute` for `compt` and `Rayl` in the patient, inherited by secondaries | opengate source |

**The coincidence window.** The real `bed<n>.singles.log` states that randoms are `R_ij = 4.909 ns * S_i * S_j`, and a window `w` gives `2 w S_i S_j`, so `w` = 2.4545 ns. It checks out: on a 50 ms run, the randoms flagged in the prompts, the delayed window and `2w ΣS_aS_b` gave 2,612, 2,483 and 2,702.

**Crystal ids.** `CrystalLookup` maps each digitised crystal centre to the nearest GE crystal id by azimuth and ring. The flat blocks move crystals by at most 0.12 of a pitch, and the map is a bijection (tested). The centroid of about 5 % of singles lands in a gap between crystals; GATE then writes the position (0, 0, 0), and those fall back to the crystal named in `PreStepUniqueVolumeID`.

**Speed.** Every voxel boundary is a Geant4 step, because opengate builds an image as nested replicas and merges nothing. The CT is therefore cropped to the body and couch, and averaged to 6.4 × 6.4 × 9.8 mm for the geometry only (`--ct-step`). That was 43 % faster than 4.3 mm with the same rates. The activity keeps its 2.13 mm voxels. With 16 threads the rate is 81,000 decays/s, so one simulated second takes ~28 min and the full 90 s frame ~42 h per bed. The frame is simulated in resumable 1 s chunks, each with its own seed and start time so that decay is right. A separate 0.5 s run writes the singles; writing them for the whole frame would take ~25 GB.

**The dead-time actor is off by default.** opengate 10.1.1's `DigitizerDeadTimeActor` loses the end of every run. A 20 ms run stopped at 15.97 ms, losing 20 % of its singles; without the actor it ran to the end. The 300 ns per block of the D690 paper would cost only ~0.4 % at these rates anyway. `--deadtime-ns 300` in the run config turns it back on.

**The norm.** GATE's crystals are all identical. GE's efficiency pattern is imposed by keeping a pair in bin `b` with probability `min(1, normdt_b / mean over views)`. That applies only the relative losses; the geometry is GATE's own.

## Method 2: parallelproj (`pp.py`)

For crystals `a`, `b` (sinogram bin `β`) and TOF bin `t`:

```
y = κ · normdt_β · exp(−∫μ) · P_t[x](a,b) + 2w S_a S_b / 55 + S_β φ(u, t)
```

* `P_t` is `joseph3d_fwd_tof_sino` over the positron decays in the frame. Its docstring defines the TOF kernel: "sigma of Gaussian TOF kernel in spatial units (same units as xstart)". Here σ = c·675 ps/2/2.355 = 43.0 mm, with 55 bins of 13.38 mm cut at 3 σ, and the LOR ends are `crystal_positions(stir_frame=True)`.
* `κ` is the only absolute constant a line integral cannot supply. It is fitted to GATE's trues on the same bed, never to the real data.
* The randoms use GATE's singles rates, and the scatter is GATE's scatter-flagged coincidences, smoothed, with their TOF shape `φ`.
* Each part is drawn from its own Poisson distribution, so every event carries a truth label.

A bed takes ~20 min: 576 ring pairs × 1.8 s for the TOF projection, measured.

**TOF sign.** `tof_bin > 0` means the annihilation was nearer `xtal_a`. parallelproj's positive bins run towards the LOR's end, which was measured, and PyTomography hands it `27 − tof_bin − 27`. `tests/test_simulation.py` projects a drawn event back through `lm.events.detector_ids` and `joseph3d_fwd_tof_lm` and gets the exact value it was drawn from.

## Output

```
$D710_OUT/<case>/sim_virtual/            a case without raw data: headers only, from the template
$D710_OUT/<case>/sim_cache/phantom/bed<n>/  act_bqml.mhd, ct_hu.mhd, mu_bed.npy, phantom.json
$D710_OUT/<case>/sim_cache/compare/      bed<n>.json, bed<n>.png, terms_<label>.json
$D710_OUT/<case>/sim_gate_s<seed>/       an ordinary case
$D710_OUT/<case>/sim_pp_s<seed>/         an ordinary case
$D710_OUT/<case>/sim_an_s<seed>/         an ordinary case; + raw_simulation/bed<n>_singles_rate.npy
$D710_OUT/<case>/sim_cache/{sss,singles}/  cached scatter and singles per bed, keyed by the phantom
$D710_OUT/<case>/sim_cache/calib/        report.json, profiles.npz, calib.png
    decoded/bed<n>.lm.npy                event table, the decoder's dtype
    decoded/bed<n>.{s,hs,json}           histogrammed by lm.events.histogram
    work/bed<n>/{normdt,norm_only}       GE's, copied
    work/bed<n>/{attn,randoms,scatter,background}   the truth used
    raw_simulation/bed<n>_truth.npz      per event: is_random / is_scatter, or label
    raw_simulation/gate/bed<n>/          GATE configs, logs, ROOT files, summary.npz
```

## Known limitations

* The PET input is a reconstruction, so it already carries GE's resolution and post-filter. The simulated data are smoother than a real acquisition of the true activity.
* Activity outside the PET image is missing, which is the singles deficit above. Lu-176 background and pile-up are not modelled either.
* The pp model is by construction an "inverse crime" against `d710 lm`. That makes it right for testing the pipeline and wrong for testing the scatter or randoms model.
