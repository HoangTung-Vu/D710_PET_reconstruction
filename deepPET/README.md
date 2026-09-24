# deepPET

This package is a pilot of **DeepPET** (Häggström et al., *Med. Image Anal.* 54, 2019) on the D710. DeepPET is a convolutional encoder–decoder: it reads one precorrected 2D sinogram and writes the image in one pass, without iterating. It is trained on sinograms simulated from reconstructed PET/CT NIfTI, and then run on real D710 beds.

## Why DeepPET, and not FastPET

- The paper simulated the **GE D710/D690**. Its sinogram, 288 views × 381 radial bins, has the same view and radial dimensions as our `bed<n>.s`.
- Its training data were made the way ours are: 79 real whole-body OSEM images plus CT, forward projected, with Poisson noise at 9 count levels. That gave 176,585 sinogram/image pairs, split by patient.
- It is 2D, so both simulation and training are cheap.
- FastPET (Whiteley et al. 2021) reads TOF histo-images: every event is placed at its most likely position along the LOR. Training it would need a TOF event stream per sample and a 3D network.

## Data flow

```
H108 NIfTI (CT HU, PET SUV)  --prepare-->  256² slices: SUV + mu      (once, ~16 GB)
          slice  --simulate (on the fly, per item)-->  x_in (288 x 371)  +  target (grid²)
          x_in   --DeepPET-->  SUV image
real bed: prompts, background, normdt, attn  --SSRB + precorrect-->  x_in  --DeepPET-->  SUV
```

## How the corrections are handled

The network only ever sees the **precorrected** sinogram, the paper's eq. 3:

```
x_in = (counts − γ) / (normdt · AF) / s          E[x_in] = P x   (line integrals, SUV·mm)
```

- The mean of every correction is removed before the network sees the data.
- Inside the simulation, a correction matters only through the noise it leaves behind.
- Negative bins are kept, because clipping them would bias the input.

| | simulation (`simulate.py`) | real bed (`real.py`) |
|---|---|---|
| attenuation | `hu_to_mu(CT)` → `AF = exp(−P μ)`; multiplied in before Poisson, divided out after | `÷ work/bed<n>/attn.s` |
| normdt | off by default: all crystals equal, no dead time | `÷ normdt.s`: a sensitivity, with the span-2 multiplicity already inside |
| randoms | `paper` mode: flat over the sinogram, 0.26–0.54 of prompts; the exact mean is subtracted | `− randoms.s` |
| scatter | `paper` mode: AF·Px blurred tangentially, 0.14–0.16 of prompts; the exact mean is subtracted | `− scatter.s` |
| | | `background.s` = randoms + scatter, subtracted once |

The randoms and scatter fractions come from `calib.json`, measured on the 7 beds of the adult exam fdg26081901. `--mode attn` drops randoms and scatter. `--mode pure` also drops attenuation, leaving only the forward model plus Poisson noise.

## Forward model: no PSF blur

The simulation projects the SUV as it is: `Px = P(SUV)`, `simulate.SIM_PSF_MM = 0`. The H108 targets are already reconstructed images, so they carry the scanner's recon resolution. Blurring them again before projecting would blur them twice. The 6.4 mm in GE's DICOM (`post_filt_parm`, `utils.scanner.POST_FILTER_FWHM_MM`) is GE's **post-filter**, not a PSF. It is still applied where it belongs, after OSEM (`osem2d`, `real --osem`).

The check was run on a real sinogram. We correlate a tangential band-pass (a difference of Gaussians, σ in bins) of the SSRB sinogram `x` with the same band of `P(GE ⊗ k)`, over planes 10–36 of fdg26081901. If a blur k were missing from the model, the correlation would peak at that k. It does not:

| bed | band σ (bins) | k = 0 | 3 | 4.7 | 6.4 | 8 | 10 mm |
|---|---|---|---|---|---|---|---|
| 7 | 0.7–2 | **0.3771** | 0.3762 | 0.3730 | 0.3680 | 0.3608 | 0.3487 |
| 7 | 1–3 | **0.6001** | 0.5998 | 0.5981 | 0.5947 | 0.5891 | 0.5787 |
| 4 | 0.7–2 | **0.1111** | 0.1104 | 0.1085 | 0.1061 | 0.1031 | 0.0988 |
| 4 | 1–3 | **0.2159** | 0.2154 | 0.2139 | 0.2119 | 0.2091 | 0.2046 |

In the coarser 2–6 band the differences shrink, as expected once the band is wider than the blur. Bed 4 still falls (0.5554 at 0, 0.5537 at 6.4). Bed 7 stays within 0.0004 up to 6.4 mm (0.8727 at 0, 0.8729 at 3–4.7, 0.8725 at 6.4). Runs trained before 2026-09-24 (`g128`, `g256`) saw the doubly blurred data. `g256_nopsf` is the retrain.

## From SUV to counts: `calib.json`

The NIfTI is in SUV, and the simulation needs counts. Two factors, measured once on a real adult D710 exam by `python -m deepPET.calibrate`, make the chain explicit:

```
SUV  × A  →  Bq/mL      A = net dose × decay(injection → scan start) / weight      = 4599 Bq/mL per SUV
Bq/mL × B →  counts     B = real rebinned trues / Σ AF · P(GE's Bq/mL image)      = 6.96×10⁻⁵ counts per Bq/mL·mm
                        s = A · B = 0.320 counts per SUV·mm, per 2D LOR (beds 0.294–0.397)
```

- **A** uses GE's own SUV convention: the net dose (injected minus residual) decayed to the scan start (`utils.quant.dose_bq`, `scan_start_factor`).
- **B** is the scanner's: sensitivity × the 90 s frame × what SSRB adds from the oblique segments × crystal efficiency and dead time, all in the simulation's 2D units. Measured on the child exam fdg26081008 it is 6.85×10⁻⁵, within 1.4 % of the adult's, which is what a property of the scanner should do.
- **The exam is fdg26081901** (77 y, 40 kg, 259 MBq net, 7 beds of 90 s). fdg26081008 is a child (11 y) and is left out. The other three FDG exams have had their sinograms pruned and could not be used.
- **H108 carries no dose or weight**, so every study is scaled by this one A: an assumption about the population. A 40 kg patient at 259 MBq is 6.5 MBq/kg, on the high side for adults.

The simulation then draws `trues = s · AF · P(SUV)` per bin, so a slice's counts follow its activity, as on the scanner: on H108 slices that gives 3.5×10⁵ – 1.7×10⁶ prompts (5th–95th percentile, median 6.6×10⁵), against 6.3×10⁵ – 1.2×10⁶ per plane on the real adult beds. `--count-scale 0.25` is a quarter of the injected dose: trues and scatter drop 4×, randoms 16× (they go as the singles squared), so the randoms fraction falls as it does on the scanner. It is not a shorter scan, where the randoms would drop 4× too. `--scale counts` restores the older behaviour, drawing the prompts per slice from `--count-min`..`--count-max`.

## Splits

`test` is `imagesTs` as the dataset gives it (47 studies). `imagesTr` is split by patient into `train` and `val` (`--val-frac 0.15`): 157 and 27 studies. Patients 0029 and 0033 have studies in both folders; their 2 `imagesTr` studies are left out so that no test patient is seen in training (`--keep-overlap` keeps them in train). `python -m deepPET.prepare --split-only` rewrites `split.json` without touching the slices.

**The CT is used for attenuation only.** The network input is the sinogram alone, and the target is the SUV slice. μ is made from HU at native resolution and then resampled, because μ is the quantity that gets line-integrated. On a real bed, the attenuation of oblique LORs is already inside `attn.s`.

**The bore.** Pixels, LORs and the loss are all limited to r ≤ 350 mm, the D710's bore radius:
- The tangential bins are cropped to 371 of 381 (bins 5..375). This is the paper's rule; its 269 bins would cut our FOV to r = 276 mm.
- SUV and μ are zeroed outside the circle before projection.
- The loss is masked to the same circle.

## Measured

| what | value |
|---|---|
| real prompts per direct plane, segment 0 only, 90 s bed | 3×10⁴ – 1.2×10⁵ (DeepPET fails below ~10⁵) |
| the same after SSRB of all 23 segments | 3×10⁵ – 1.2×10⁶ |
| real randoms / scatter per prompt, adult exam | 0.26–0.54 / 0.14–0.16 |
| calibrated `s`, adult exam | 0.320 counts per SUV·mm (beds 0.294–0.397) |
| precorrection bias, mean(x_in)/Px over 20 draws | 1.000 ± 0.0005 in every mode (draw noise) |
| one training sample, laptop, 16 threads | 45 ms at 128², 74 ms at 256² |
| `prepare` | ~0.7 s per study, activity conserved 1.000, ~250 KB per slice |
| model | 37.0 M params, 31 conv layers (34 at 256²) |
| training memory, batch 30, fp32, lower bound | ~11.7 GiB at 128², ~13.1 GiB at 256² |
| real bed 6, sinogram against P(GE), smoothed | r = 0.988 as is, 0.952 x-flipped, 0.840 y-flipped |
| real bed 6, our 2D OSEM against GE's image | r = 0.944, total 0.998 of GE's |

## Commands

Run everything from `D710/` in the `petct_recon` env:

```bash
export D710_OUT=~/UET/Handson_PET_CT_Reconstruction/d710_out
export PYTHONPATH=$PWD:$PWD/vendor:$PWD/tools/stubs

pytest tests/test_deeppet.py -q                               # 24 tests, ~5 s

python -m deepPET.calibrate                                  # only where the raw exam is; writes calib.json (committed)
python -m deepPET.prepare --data ~/Downloads/H108/H108_PETCT_Thyroid/Dataset4000_PETCT_H108
python -m deepPET.prepare --split-only                       # after changing --val-frac, no re-prepare
python -m deepPET.simulate --study 0002_20191016 --slice 170 --mode paper   # PNG, bias check
python -m deepPET.real --case fdg26081008 --bed 6            # geometry tripwire + OSEM 2D, no net
```

### Training (on sslab: Quadro RTX 8000, 48 GB)

```bash
python -m deepPET.train --name g128 --grid 128 --epochs 30 --workers 16 --amp   # run 1: the paper's grid
python -m deepPET.train --name g256 --grid 256 --epochs 30 --workers 16 --amp   # run 2: GE's clinical grid
python -m deepPET.train --name g256_nopsf --grid 256 --epochs 30 --workers 16 --amp   # run 3: run 2 without the double blur
python -m deepPET.train --name g128 --resume                                   # after an interruption
```

The default is the paper's full 100 epochs; `--epochs 30` is the pilot.

**The first epoch's log gives the real cost.** The `s/step` figure and the epoch time are printed. The CPU simulation can become the bottleneck: set `--workers` to about the number of cores, and keep `--sim-threads 1`.

### Evaluation

```bash
python -m deepPET.evaluate --name g128 --limit 300           # DeepPET vs OSEM 2D at x1, x0.25, x0.1 dose
python -m deepPET.real --case fdg26081008 --bed 6 --name g128
```

`real` reconstructs OSEM 2D the way GE does clinically (`--osem ge`, the default): 2 iterations × 24 subsets per slice, then GE's post-filter on the stack of slices — 6.4 mm transaxial and the axial [1, 4, 1] (`osem.stitch.post_filter`). On fdg26081901 bed 4 it matches GE's image at r = 0.983 (rRMSE 0.63), against r = 0.972 (rRMSE 0.81) for the paper's 5 × 16 with the transaxial filter only (`--osem paper`). `evaluate` keeps the paper's 5 × 16: its test slices are drawn one at a time, so there are no neighbours to filter axially.

Outputs go to `$D710_OUT/deeppet/{data,runs/<name>,real}/`.

## Known limits

- **One calibrated `s` for every slice.** Within a real bed, the rebinned counts fall about 30× from the centre plane to the edge planes, because fewer oblique segments land there. The simulation uses the bed average, so it never shows the network an edge plane's noise.
- **γ is exact in the simulation**, but on a real bed randoms and scatter are estimates. Their errors pass straight into the input. The paper has the same limitation.
- **SSRB ignores the axial blur of oblique segments**, which reach up to ring difference 23.
- **`s_real` is fitted to GE's image of the same bed**, one number per bed. That makes GE the answer key for the *scale* of the real result. Replace it with the real calibration (K, frame time, decay, SUV factor) before quoting SUVs.
- **The targets are GE OSEM images**, smoothed by GE's 6.4 mm post-filter. The network can reproduce that resolution, not exceed it. With an MSE loss it also comes out smooth, as the paper found; `--loss l1` is there to try.
- **The deepPET branch was cut from `main`, which does not have `simulation/`.** `nifti.load` therefore repeats that package's NIfTI reader. Keep the two in step.
