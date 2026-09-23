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
| randoms | `paper` mode: flat over the sinogram, 0.24–0.54 of prompts; the exact mean is subtracted | `− randoms.s` |
| scatter | `paper` mode: AF·Px blurred tangentially, 0.15–0.20 of prompts; the exact mean is subtracted | `− scatter.s` |
| | | `background.s` = randoms + scatter, subtracted once |

The randoms and scatter fractions were measured on the 7 beds of fdg26081008. `--mode attn` drops randoms and scatter. `--mode pure` also drops attenuation, leaving only the forward model plus Poisson noise.

**The CT is used for attenuation only.** The network input is the sinogram alone, and the target is the SUV slice. μ is made from HU at native resolution and then resampled, because μ is the quantity that gets line-integrated. On a real bed, the attenuation of oblique LORs is already inside `attn.s`.

**The bore.** Pixels, LORs and the loss are all limited to r ≤ 350 mm, the D710's bore radius:
- The tangential bins are cropped to 371 of 381 (bins 5..375). This is the paper's rule; its 269 bins would cut our FOV to r = 276 mm.
- SUV and μ are zeroed outside the circle before projection.
- The loss is masked to the same circle.

## Measured

| what | value |
|---|---|
| real prompts per direct plane, segment 0 only, 90 s bed | 3×10⁴ – 1.2×10⁵ (DeepPET fails below ~10⁵) |
| the same after SSRB of all 23 segments | 3×10⁵ – 1.2×10⁶, so training draws 10⁵ – 10⁷ |
| real randoms / scatter per prompt | 0.24–0.54 / 0.17–0.18 |
| precorrection bias, mean(x_in)/Px over 20 draws | 1.000 ± 0.0003 in every mode |
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

pytest tests/test_deeppet.py -q                               # 18 tests, ~4 s

python -m deepPET.prepare --data ~/Downloads/H108/H108_PETCT_Thyroid/Dataset4000_PETCT_H108
python -m deepPET.simulate --study 0002_20191016 --slice 170 --mode paper   # PNG, bias check
python -m deepPET.real --case fdg26081008 --bed 6            # geometry tripwire + OSEM 2D, no net
```

### Training (on sslab: Quadro RTX 8000, 48 GB)

```bash
python -m deepPET.train --name g128 --grid 128 --epochs 30 --workers 16 --amp   # run 1: the paper's grid
python -m deepPET.train --name g256 --grid 256 --epochs 30 --workers 16 --amp   # run 2: GE's clinical grid
python -m deepPET.train --name g128 --resume                                   # after an interruption
```

The default is the paper's full 100 epochs; `--epochs 30` is the pilot.

**The first epoch's log gives the real cost.** The `s/step` figure and the epoch time are printed. The CPU simulation can become the bottleneck: set `--workers` to about the number of cores, and keep `--sim-threads 1`.

### Evaluation

```bash
python -m deepPET.evaluate --name g128 --limit 300           # DeepPET vs OSEM 2D at 1e5/1e6/1e7 prompts
python -m deepPET.real --case fdg26081008 --bed 6 --name g128
```

Outputs go to `$D710_OUT/deeppet/{data,runs/<name>,real}/`.

## Known limits

- **γ is exact in the simulation**, but on a real bed randoms and scatter are estimates. Their errors pass straight into the input. The paper has the same limitation.
- **SSRB ignores the axial blur of oblique segments**, which reach up to ring difference 23.
- **`s_real` is fitted to GE's image of the same bed**, one number per bed. That makes GE the answer key for the *scale* of the real result. Replace it with the real calibration (K, frame time, decay, SUV factor) before quoting SUVs.
- **The targets are GE OSEM images**, smoothed by GE's 6.4 mm post-filter. The network can reproduce that resolution, not exceed it. With an MSE loss it also comes out smooth, as the paper found; `--loss l1` is there to try.
- **The deepPET branch was cut from `main`, which does not have `simulation/`.** `nifti.load` therefore repeats that package's NIfTI reader. Keep the two in step.
