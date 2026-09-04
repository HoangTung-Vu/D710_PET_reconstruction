# `lowdose/` — retrospective count reduction

Turn a full-dose exam into a lower-dose one by decimating its event stream. The
output is an **ordinary case**, not a special format, so `d710 osem`, `d710 lm`
and `d710 export` all run on it unchanged.

```bash
conda activate petct_reconstruction              # numpy only; no SIRF, no torch
d710 lowdose --case ped --drf 10                 # -> ped_drf10
d710 lowdose --case ped --drf 10 --mode randoms  # randoms go as f^2
d710 lowdose --case ped --split 2                # -> ped_r0, ped_r1  (Noise2Noise)
d710 lowdose --case ped --drf 4 --replicates     # -> ped_drf4_r0 .. _r3

./d710_isolate_stir.sh osem --case ped_drf10     # or: d710 lm recon --case ped_drf10
./d710_isolate_stir.sh export --case ped_drf10   # K x DRF applied automatically
./d710_isolate_stir.sh export --case ped_drf10 --lm   # ... from recon_lm.npz
```

Needs `decoded/bed<n>.lm.npy` (`d710 decode --listmode --format npy`) and the
`work/bed<n>` terms of the source case.

## Why event-level, and why it is legitimate

Thinning a Poisson variable gives a Poisson variable, so histogramming after
thinning and binomially thinning the histogram are the *same distribution*. One
decimator therefore serves both the sinogram path and the list-mode path, and
statistical equivalence is guaranteed by construction rather than by argument.

The method itself is the standard one — it was validated against real
dual-injection low-dose/standard-dose human scans (Schaefferkoetter 2019) and is
how the UDPET dataset was built.

## `thin.py` — the decimators

| | |
|---|---|
| `keep(e, f, "uniform", rng)` | keep each event with probability `f` |
| `keep(e, f, "randoms", rng, bins, rho)` | keep with `q_b = f(1−ρ_b) + f²ρ_b` |
| `rho_per_plane(p, r, binmap)` | `ρ_b = randoms/prompts`, estimated per plane |
| `split(n, k, rng)` | `k` disjoint, mutually independent replicates |

**Mode 1, `uniform`** — exactly a reduced-acquisition-time simulation, and a
*conservative* low-dose one: it scales trues and scatter correctly but leaves the
randoms fraction artificially high, so the emulated image is if anything slightly
worse than a real low-dose scan. This is the default and the primary result.
`f = 1` is the identity, bit for bit.

**Mode 2, `randoms`** — at dose fraction `f`, physics says `T → fT`, `S → fS`,
`R → f²R`. Thinning bin `b` with `q_b` gives exactly `Poisson(f(T+S) + f²R)`:
the right mean *and* the right Poisson variance. `q_b ≤ f ≤ 1` always, so it is
a valid probability for every `f`.

> **Caveat that belongs in the paper.** `ρ_b` is estimated **per plane**, not per
> bin: raw bins hold ~0.06 counts, so `ρ > 1` happens constantly from Poisson
> noise alone. And the per-ring randoms distribution this inherits is known to be
> ±20–40 % out, so Mode 2's *total* randoms scaling is right while its *spatial*
> redistribution carries that error — and at small `f` the tails go to `f²`,
> which is where the error lives. Report both modes; lead with Mode 1.

**Mode 4, `split`** — independent multinomial assignment of a Poisson total gives
exactly independent Poisson subsets, unlike repeated thinning whose realisations
overlap. `k = 2` is a Noise2Noise pair. The labels are drawn from a seeded RNG of
the event count alone, so replicate `k` and `k'` of the same bed are guaranteed
disjoint.

> **Correlation trap.** A thinned dataset is a *subset* of the full one, so
> "low-dose recon vs full-dose recon" carries a positive correlation bias. For
> clean noise measurements compare against the **complement** replicate, not the
> full scan.

## `write.py` — the case it writes

What changes, and only what physics says must:

| | |
|---|---|
| prompts | thinned event by event, then histogrammed |
| `scatter` | `× f` — linear in activity |
| `randoms` | `× f`, or `× f²` in randoms-aware mode |
| `background` | rebuilt as `randoms + scatter` (the halves no longer share a factor) |
| `normdt`, `norm_only`, `attn` | copied — sensitivity does not depend on dose |
| `scatter_tof.npy` | copied — it is a *shape*, not an amplitude |
| `to_stir.json` | **rewritten**, not copied (below) |
| `K` | `× 1/f`, recorded in `lowdose.json` |

`to_stir.json` has to travel — `utils.terms.ct_dir` and `d710_isolate_stir.sh`
read `estimate.ct` out of it to find the CT — but most of it measures the
*source*. Copied verbatim it made the derived case claim a bit-exactness proof
for prompts it does not have, and carry `stats` for terms that have since been
scaled; `tests/test_pipeline_data.py` catches exactly that. So `verified.prompts`
is rewritten, `stats` is dropped, and a `lowdose` stanza records what was scaled.

Headers are **cloned** from the source, never regenerated: a fresh header
desynchronises ExamInfo and STIR only complains much later, inside
`make_Poisson_loglikelihood`. Term scaling is element-wise, so the segment
layout never has to be known.

`lowdose.json` carries the dose fraction, the mode, the seed, the per-bed counts
and `k_scale = 1/f`. `utils.quant.lowdose_k_scale` reads it and
`utils/export.py` applies it, so SUV stays flat along the dose ladder without
anyone remembering a flag.

## `verify.py` — what runs after every write

| | |
|---|---|
| `binomial(src, dst, ...)` | per plane, `Σy'` within 3 sd of `Binomial(Σy, f)` |
| `invariants(dst, ...)` | `Σp ≥ Σr` and `Σs ≤ Σ(p−r)`, per plane, must be 0 |

Both aggregate **per plane**. The raw sinogram runs at ~0.06 count/bin, so
`p < r` is true at ~82 % of bins from Poisson noise alone and a per-bin
assertion says nothing. `--no-check` skips them.

## Mode 3 — dead time — is deliberately not implemented

Lower activity means less dead time and so higher live sensitivity. The measured
livetime across ped beds 1–6 runs 0.9569 → 0.9285 over 208 → 969 kcps, so the
correction is bounded by ~7 % and would be applied as
`normdt'(f) = normdt · livetime(f·kcps)/livetime(kcps)` — a fit to that 6-point
curve, no reverse engineering needed. It is a ~5 % quantitative refinement on top
of a simulator whose leading error is elsewhere, so it is written down here
rather than coded.

## Step 5 of the study plan — the realistic-background ablation

Everything above hands the low-dose reconstruction an **oracle** background,
estimated from full-count data. A real low-dose scan estimates randoms and
scatter from its own noisy data. That ablation is re-running randoms-from-singles
with singles scaled by `f` and the scatter estimate on the thinned sinogram; it
is expected to hurt, and the scatter fit's tail numerator is exactly where the
randoms model is weakest.
