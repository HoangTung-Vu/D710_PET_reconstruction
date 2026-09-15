# `lowdose/` — retrospective count reduction

Turn a full-dose exam into a lower-count or lower-dose one by decimating its
event stream. The output is an **ordinary case**, not a special format, so
`d710 osem`, `d710 lm` and `d710 export` all run on it unchanged.

```bash
conda activate petct_recon                          # numpy only; no SIRF, no torch
d710 lowdose --case ped --drf 10                    # -> ped_drf10    (low-count)
d710 lowdose --case ped --drf 10 --window time      # a literally shorter frame
d710 lowdose --case ped --drf 10 --mode low-dose    # -> ped_drf10_lowdose
d710 lowdose --case ped --split 2                   # -> ped_r0, ped_r1 (Noise2Noise)
d710 lowdose --case ped --drf 4 --replicates        # -> ped_drf4_r0 .. _r3

./d710_isolate_stir.sh osem --case ped_drf10        # or: d710 lm recon --case ped_drf10
./d710_isolate_stir.sh export --case ped_drf10      # K x DRF applied automatically
./d710_isolate_stir.sh export --case ped_drf10 --lm # ... from recon_lm.npz
```

Needs `decoded/bed<n>.lm.npy` (`d710 decode --listmode --format npy`) and the
`work/bed<n>` terms of the source case. `--sinogram binomial` drops the first
requirement.

## The two modes are two different experiments

| mode | simulates | keep probability | prompts | randoms | scatter |
|---|---|---|---|---|---|
| `low-count` | a **shorter scan** | `f` | `× f` | `× f` | `× f` |
| `low-dose` | **less activity** | `q_b = f(1−ρ_b) + f²ρ_b` | trues+scatter `× f`, randoms `× f²` | `× f²` | `× f` |

(`--window time` changes the `low-count` row: the factors become decay-weighted
and randoms stop matching trues — see below.)

A coincidence rate is linear in the live time, so halving the scan halves
everything alike. Halving the *injection* does not: a random is a coincidence
between two unrelated singles, and each single is linear in activity, so randoms
go as the square. That one exponent is the whole difference between the modes,
and `thin.randoms_power` is the only place it is decided — three call sites used
to carry their own copy of it.

**Mode 1, `low-count`** is the default and the primary result. It is a
reduced-acquisition-time simulation, and a *conservative* low-dose one: it scales
trues and scatter correctly but leaves the randoms fraction artificially high, so
the emulated image is if anything slightly worse than a real low-dose scan.
`f = 1` is the identity, bit for bit.

### Which short scan? `--window`

A uniform draw keeps a fraction `f` of every event **whatever time it arrived**,
so it reproduces the frame's *average* activity. A real short scan samples the
activity *at the moment it ran*. Those agree only when the window sits at the
frame's mid-time:

```
frame average = (1/T)∫λ(t)dt = λ(t_c)   ⟹   t_c ≈ T/2
```

So `--window uniform` (default) is **a short scan at the frame's mid-time**, and
`--window time` — keep the first `f` of the frame — is **a short scan at its
start**. They differ by the decay across half the frame.

**And in a time window the terms do not scale by `f`.** The activity decays
*during* the frame, so the retained window holds more than its share — and
randoms hold more still, because a random is a coincidence of two singles and its
rate falls as `A²`:

```
trues, scatter   rate ∝ A    →  ∫₀^τ A dt    →  (1−e^{−μτ}) / (1−e^{−μT})
randoms          rate ∝ A²   →  ∫₀^τ A² dt   →  (1−e^{−2μτ}) / (1−e^{−2μT})
```

**Randoms therefore carry a different exponent from trues even in low-count
mode** — not because of dose, but because of decay inside the window. F-18, 90 s
frame, `f = 0.1`: trues/scatter **×0.100427**, randoms **×0.100855**. Measured on
fdg26081008, all 7 beds kept **0.1004–0.1008**, and the predicted prompts factor
`(1−ρ)·lin + ρ·quad` = 0.10061 at ρ = 0.435.

How much it matters depends entirely on frame/half-life:

| isotope | T½ (s) | trues × | randoms × | randoms vs `f` |
|---|---|---|---|---|
| F-18 | 6586 | 0.10043 | 0.10085 | +0.9 % |
| Ga-68 | 4062 | 0.10069 | 0.10139 | +1.4 % |
| C-11 | 1222 | 0.10231 | 0.10466 | +4.7 % |
| O-15 | 122 | 0.12448 | 0.15171 | **+51.7 %** |

For F-18 it is under 1 %, which is why the literature's uniform thinning is not
wrong. For a short-lived tracer it is not optional.

**A uniform draw has none of this** — it keeps a fraction `f` of every event
regardless of arrival time, so every integral scales by exactly `f` and the frame
is unchanged. That exact self-consistency is its real advantage, along with being
a *controlled* experiment: a contiguous window is one moment, and if the patient
moved or the tracer redistributed you get a different scan rather than a
lower-count version of the same one.

`--window time` updates `frame_duration_ms` to the true `f·T`, which is all
`osem.stitch.decay_factor` needs — it divides by the mean activity over that
duration, and the window starts where the frame does, so `bed_start_time` is
untouched. `delays` follows the randoms factor, so `randoms/delays` stays ≈ 0.99.

> One residual, small and deliberately not chased: `quant.scan_start_factor`
> notes that the frame-duration term cancels against GE's, because GE applies the
> same mean-activity-over-frame correction. With a 9 s frame against GE's 90 s it
> no longer cancels exactly — a **0.42 %** offset, which `K` absorbs since `K` is
> fitted. It is well under the 3.5–5 % residual the dose ladder already shows.

`--window time` is restricted to `--mode low-count --sinogram derived` and refuses
`--replicates`: combining it with `low-dose` would reduce activity and duration at
once (a third set of factors nobody here has measured), it needs timestamps, and
time-separated replicates are not exchangeable.

**Mode 2, `low-dose`** thinning bin `b` with `q_b` gives exactly
`Poisson(f(T+S) + f²R)`: the right mean *and* the right Poisson variance.
`q_b ≤ f ≤ 1` always, so it is a valid probability for every `f`.

### `ρ` must be estimated per (plane, tangential bin) — `--rho`

That guarantee holds **only if `ρ_b` is the bin's own randoms fraction**, and `ρ`
cannot be estimated per bin: raw bins hold ~0.06 counts, so `ρ > 1` happens
constantly from Poisson noise alone. It therefore has to be aggregated, and the
axis it is aggregated over decides whether Mode 2 is right:

| `--rho` | cell | counts/cell, fdg26081008 bed 1 | tail error at `f = 0.1` |
|---|---|---|---|
| `tangential` (default) | (plane, u), summed over 288 views | ~70 | none by construction |
| `plane` | the whole plane | ~26,800 | **6.1× too many counts** |

`ρ` varies far more across `u` than down the axis — a tail LOR misses the patient
and is almost pure randoms (`ρ → 1`), a central one is mostly trues (`ρ ≈ 0.2`).
Hand every bin in a plane the plane mean (`ρ̄ = 0.435` on bed 1) and the tails keep
`q̄ = 0.061` where they should keep `f² = 0.010`, while the randoms sinogram there
was scaled by exactly `f²`. Data and model then disagree by 6× in the tails.

**That is not a theoretical concern; it broke the pipeline.** The scatter TOF
estimator picks its tail ring where `T/S` is small, with `T = ΣP − R − S`; the
un-subtracted randoms excess inflated `T` so that `min T/S` went `0.000 → 0.712`,
the ring collapsed from 165 tangential bins to 4, and `scatter_tof_weights`
refused. Past that it would also bias the reconstruction, whose background term
under-subtracts in the tails.

Note what did *not* catch it: `Σ q̄·y = f(T+S) + f²R` still holds **on aggregate**,
so the per-plane binomial check passed (10/3871 planes outside 3 sd) and the
invariants gave 0. The error is purely the redistribution *inside* each plane.
`--rho plane` is kept for a bed so thin that a `(plane, u)` cell cannot be
divided, and is wrong in the tails whenever it is used.

### `ρ` also depends on TOF, and that is modelled — `--tof-rho`

The same argument applies one axis further in. **Randoms are flat in TOF; trues
are not.** So inside a single LOR the randoms fraction climbs steeply away from
the TOF peak, and giving every TOF bin of that LOR one `q` repeats the mistake.
Measured on fdg26081008 bed 1, central LORs, `f = 0.1`:

| TOF bin | `|offset|` | % of counts | `ρ(t)` | correct `q` | non-TOF `q` | error |
|---|---|---|---|---|---|---|
| 27 | 0 (peak) | 5.10 % | 0.081 | 0.0927 | 0.0795 | 0.86× |
| 17 | 10 | 1.99 % | 0.208 | 0.0813 | 0.0795 | 0.98× |
| 12 | 15 | 0.63 % | 0.658 | 0.0408 | 0.0795 | 1.95× |
| 7 | 20 | 0.43 % | 0.962 | 0.0134 | 0.0795 | **5.92×** |
| 2 | 25 | 0.42 % | 0.996 | 0.0103 | 0.0795 | **7.71×** |

Over the whole bed that is **+10.6 % excess counts** (95,874 of 906,694), sitting
in the deep TOF tails as un-thinned randoms — while the *total* comes out only
0.7 % low, because the 12 % deficit at the peak nearly cancels it. So it is
invisible in any count total, and it reaches only the **list-mode TOF path**: the
sinogram is non-TOF and gets `E[y'_b] = f(T_b+S_b) + f²R_b` exactly either way.

`ρ` cannot be *estimated* per `(bin, TOF)` — that is ~0.001 counts a cell. It is
**modelled** instead, using the one fact that makes it possible (randoms flat in
TOF, measured at CoV 0.0404 against a Poisson floor of 0.0393):

```
ρ(b,t) = (R_b / n_tof) / y(b,t)  and  y(b,t) = y_b · φ(u,t)
      ⟹  ρ(b,t) = ρ_nonTOF(b) · (1/n_tof) / φ(u,t)
```

The non-TOF `ρ` already carries the axial and radial structure; the factor depends
only on `(u, t)`. `φ(u,t)` is the prompts' TOF profile at radial offset `u`,
measured on the **source** at `n_tang × n_tof` cells (~707 counts each), so it adds
no meaningful noise. The one approximation is that the TOF *shape* at a given `u`
does not depend on plane or view.

Checked against a direct measurement of `ρ(t)` on that bed — TOF bin 27 gives
0.081 both ways, bin 7 gives 0.962 both ways — and `tests/test_lowdose.py` asserts
the identity algebraically. `--tof-rho off` reproduces the older behaviour.

Two consequences worth knowing:

* **A flat `φ` is a no-op.** A bin that really is 100 % randoms *has* a flat TOF
  profile — the two are the same statement — so the factor is exactly 1 there and
  `q` stays `f²`.
* **The TOF peak does not reach `q = f`.** A flat randoms floor sits under the
  peak, so even there `ρ ≈ 0.14` rather than 0. Measured `q = 0.877 f`.

Because `q` now varies *inside* a `(plane, u)` cell, the binomial check can no
longer form its expectation from one probability per cell. **And it must not model
that expectation either** — the first attempt did, weighting `q` by the global
`φ(u,·)` within each `(plane, u)`, and oblique planes have broader TOF
distributions than the global average, so their events drew lower `q` than the
prediction assumed: **98 of 3871 planes outside the 3 sd band, every bed biased the
same way**. The data was correct; only the prediction was not.

The expectation has no model in it now. A thinned plane total is a sum of
independent Bernoulli draws, one per event, so

```
mu  = Σ q_i        var = Σ q_i (1 − q_i)        over the events in that plane
```

`thin.event_q` returns the `q_i` the draw is made against and `thin.expectation`
sums them; `verify.binomial` takes the pair as absolute counts. Re-measured on
fdg26081008: **11 of 3871 planes, 0.28 %**, residuals scattering both ways.

> **Caveat that still belongs in the paper.** The per-ring randoms distribution
> Mode 2 inherits is known to be ±20–40 % out, so its *total* randoms scaling is
> right while its *spatial* redistribution carries that error — and at small `f`
> the tails go to `f²`, which is where the error lives. Report both modes; lead
> with Mode 1.

**Mode 4, `--replicates`** — independent multinomial assignment of a Poisson
total gives exactly independent Poisson subsets, unlike repeated thinning whose
realisations overlap. `k = 2` is a Noise2Noise pair. The labels are drawn from a
seeded RNG of the event count alone, so replicate `k` and `k'` of the same bed
are guaranteed disjoint. It partitions the stream, which is uniform by
construction, so it is a `low-count` operation only.

> **Correlation trap.** A thinned dataset is a *subset* of the full one, so
> "low-dose recon vs full-dose recon" carries a positive correlation bias. For
> clean noise measurements compare against the **complement** replicate, not the
> full scan.

The old names `uniform` and `randoms` still work everywhere a mode is accepted,
including in the `mode` field of a `lowdose.json` already on disk.

## Both forms of the data are thinned

The exam exists as an event table *and* as a histogram, and both reach a
reconstructor — `d710 lm` reads `decoded/bed<n>.lm.npy`, `d710 osem` reads
`decoded/bed<n>.s`. Both are thinned, and by default from **one draw**:

* `--sinogram derived` (default) — thin the events, then histogram the survivors.
  The two paths then see literally the same thinned data, and thinning a Poisson
  variable gives a Poisson variable, so this is distributionally identical to
  thinning the histogram. One decimator serves both paths and the equivalence is
  guaranteed by construction rather than by argument.
* `--sinogram binomial` — draw `y'_b ~ Binomial(y_b, q_b)` straight off
  `decoded/bed<n>.s`. Same distribution; it is what lets the simulator run on a
  bed with **no event table at all**, which is otherwise a hard error. The result
  has no `bed<n>.lm.npy`, so `d710 lm` cannot run on it — this is the fallback,
  not a second opinion. Two independent draws would hand the two reconstructors
  different data.

`derived` is only correct if the event table reproduces the decoded sinogram in
the first place, so `verify.lm_matches_sinogram` checks that before any work,
per bed. `decode_in.sh` enforces the same thing at decode time from the
`.lm.json` sidecar; a thinned case has no sidecar, so it is measured here.
Measured on fdg26081008: exact on all seven beds, with no event falling outside
the sinogram.

The method itself is the standard one — it was validated against real
dual-injection low-dose/standard-dose human scans (Schaefferkoetter 2019) and is
how the UDPET dataset was built.

## `thin.py` — the decimators

| | |
|---|---|
| `canonical(mode)`, `randoms_power(mode)` | either spelling → the mode; `1` or `2` |
| `keep(e, f, "low-count", rng)` | keep each event with probability `f` |
| `keep(e, f, "low-dose", rng, bins, rho)` | keep with `q_b = f(1−ρ_b) + f²ρ_b` |
| `binomial_sinogram(y, q, rng)` | thin a histogram directly, one plane at a time |
| `rho_bins(P, R, binmap, axis)` | `ρ = randoms/prompts` at `tangential` or `plane` |
| `time_window(t_ms, f, frame_ms)` | keep the first `f` of the frame — no RNG |
| `decay_scales(T½, frame_ms, f)` | `(linear, quadratic)` factors for that window |
| `tof_rho_factor(φ, n_tof)` | `(1/n_tof)/φ(u,t)` — turns `ρ` TOF-resolved |
| `event_q(f, bins, ρ, tof)` | each event's keep probability, the thing actually drawn against |
| `expectation(q, plane, n_plane)` | `Σq` and `Σq(1−q)` per plane — the draw's exact mean |
| `split(n, k, rng)` | `k` disjoint, mutually independent replicates |

## `write.py` — the case it writes

**Everything the simulator generates goes in `<case>/raw_simulation/`**, and is
symlinked into `decoded/` and `work/bed<n>/` under its usual name:

```
d710_out/<case>/
  lowdose.json                       dose fraction, mode, seed, per-bed counts, k_scale
  raw_simulation/
    README.txt                       provenance, in plain text
    bed<n>.hs  bed<n>.s              thinned prompt sinogram
    bed<n>.lm.npy                    thinned event table — the same draw
    bed<n>/randoms.{hs,s}            × f, or × f² in low-dose mode
           scatter.{hs,s}            × f
           background.{hs,s}         randoms + scatter, rebuilt not scaled
  decoded/bed<n>.{hs,s,lm.npy}  ->  ../raw_simulation/...
  work/bed<n>/{randoms,scatter,background}.{hs,s}  ->  ../../raw_simulation/bed<n>/...
  work/bed<n>/{normdt,norm_only,attn}.{hs,s}, scatter_tof.npy   copies of the source
```

One directory therefore holds exactly what thinning changed, and nothing else —
which is also what tells a simulated case from a measured one at a glance. The
links are **relative**, so they resolve the same way on the host and inside
`d710:full` / `sirf-local:0.1`, which bind-mount the output root at its own path;
and they stay inside the case, so `rm -rf <case>` still takes everything. Links
rather than second copies because the thinned prompts are 116 MiB a bed and each
scaled term 232 MiB.

What changes, and only what physics says must:

| | |
|---|---|
| prompts | thinned event by event, then histogrammed |
| `scatter` | `× f` — linear in activity |
| `randoms` | `× f`, or `× f²` in low-dose mode |
| `background` | rebuilt as `randoms + scatter` (the halves no longer share a factor) |
| `delays` (sidecar) | `× f^power`, so `randoms/delays` stays ≈ 0.99 |
| `normdt`, `norm_only`, `attn` | copied — sensitivity does not depend on dose |
| `scatter_tof.npy` | copied — it is a *shape*, not an amplitude |
| `scatter_tof_profile.npy` | **measured on the full-count source** (below) |
| `to_stir.json` | **rewritten**, not copied (below) |
| `K` | `× 1/f`, recorded in `lowdose.json` |

### The scatter TOF profile is measured on the source, not on the thinned tails

When the source has no `scatter_tof.npy` — i.e. GE's scatter was estimated
non-TOF — the list-mode path measures the scatter's TOF profile from the bed's own
tail ring. A thinned bed is the wrong place to do that: the profile is a
*dose-independent shape*, but the tails it is measured from are exactly what
thinning destroys. Measured at DRF 10 on fdg26081008:

| bed | `lowcount` profile vs the source's | centroid shift |
|---|---|---|
| 1 | 8.8 % of peak | −0.33 TOF bins |
| 4 | 9.2 % of peak | −0.08 TOF bins |

— and in low-dose mode the tail ring disappeared entirely. So `write.carry_tof_profile`
measures it once on the **source's** events at the full 55 bins and writes
`work/bed<n>/scatter_tof_profile.npy`. `utils.terms.measured_tof_weights` picks it
up and mashes it to whatever `--tof-bins` is asked for, and both
`utils/terms.py` and `lm/terms.py` consult it between GE's own weights and the
in-place fallback. No flag to remember, and the thinned case stays self-sufficient.

It is an accuracy improvement, not a prerequisite: if the source has no measurable
tail ring either, the failure is reported in the build output and the recon falls
back to its own measurement.

`work/bed<n>/lm.npz` — the keyed cache `d710 lm recon` leaves behind — is
deliberately **not** copied. It holds prepared full-dose data, and a thinned case
that inherited it would reconstruct the source exam while every count on disk
said otherwise.

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
| `lm_matches_sinogram(src, beds)` | before thinning: `Σ bed<n>.s` = the event count |
| `binomial(src, dst, ...)` | per plane, `Σy'` within 3 sd of `Binomial(Σy, q)` |
| `invariants(dst, ...)` | `Σp ≥ Σr` and `Σs ≤ Σ(p−r)`, per plane, must be 0 |
| `plane_tang_sums(...)` | the `(plane, u)` aggregate the last two need |

All three report **per plane**. The raw sinogram runs at ~0.06 count/bin, so
`p < r` is true at ~82 % of bins from Poisson noise alone and a per-bin assertion
says nothing. `--no-check` skips them.

The binomial check must build its expectation at **whatever resolution `q` has**:
`Σ q_b y_b`, not `f · Σy`, the moment `q` varies inside the plane. So it reads the
shape of `q` — scalar, `(n_plane,)` or `(n_plane, n_tang)` — and uses
`plane_tang_sums` for the last. Passing a scalar `f` for a low-dose run fails
553/553 planes on perfectly good data.

**Know what these checks cannot see.** All three aggregate, and the `--rho plane`
defect above passed every one of them: the totals were right and only the
distribution inside each plane was wrong. An aggregate check is not a proof of
spatial correctness.

## Measured

ped bed 1, 2026-09-03:

| mode | kept | expected | planes outside 3 sd | invariants |
|---|---|---|---|---|
| low-count DRF 10 | 1,876,897 (0.1001) | 1,875,929 | 2/553 | 0 |
| low-dose DRF 10 | 1,158,231 (0.0617) | 1,158,247 | 4/553 | 0 |
| split 2 | 0.4999 / 0.5001 | — | — | — |

fdg26081008, all seven beds, DRF 10 — three cases end to end through OSEM,
list-mode OSEM, export and a `K` fit against GE's own image of the same exam:

| | low-count uniform | low-count `time` | low-dose |
|---|---|---|---|
| kept | 0.0999–0.1001 | 0.1004–0.1008 | 0.0528–0.0788 |
| randoms / scatter | 0.1000 / 0.1000 | **0.100855 / 0.100427** | **0.0100** / 0.1000 |
| `frame_duration_ms` | 90,000 | **9,000** | 90,000 |
| planes outside 3 sd | 11/3871 | 9/3871 | 11/3871 |
| invariants | 0 | 0 | 0 |
| `k_ls` sino (10× = 629,713) | 653,212 (1.0373) | 651,091 (1.0339) | 651,669 (1.0349) |
| `k_ls` lm (10× = 1,241,780) | 1,289,312 (1.0383) | 1,290,437 (1.0392) | 1,304,047 (1.0501) |

**`k_scale = 1/f` leaves a residual +3.4–5 %, and the choice of method moves `K`
by only ~0.3 %.** So that residual is the 2-iteration OSEM's nonlinearity in
counts, not the thinning scheme: uniform and `time` agree to 0.3 % (sinogram) and
0.1 % (list-mode), exactly as the decay analysis predicts for F-18. `r` falls
0.974 → 0.949 and 0.982 → 0.966, as 10× the noise predicts. ~12 GB a case.

## Mode 3 — dead time — is deliberately not implemented

Lower activity means less dead time and so higher live sensitivity. The measured
livetime across ped beds 1–6 runs 0.9569 → 0.9285 over 208 → 969 kcps, so the
correction is bounded by ~7 % and would be applied as
`normdt'(f) = normdt · livetime(f·kcps)/livetime(kcps)` — a fit to that 6-point
curve, no reverse engineering needed. It is a ~5 % quantitative refinement on top
of a simulator whose leading error is elsewhere, so it is written down here
rather than coded. Note it is a *`low-dose`* effect: a shorter scan of the same
patient has the same count rate and the same dead time.

## Step 5 of the study plan — the realistic-background ablation

Everything above hands the low-dose reconstruction an **oracle** background,
estimated from full-count data. A real low-dose scan estimates randoms and
scatter from its own noisy data. That ablation is re-running randoms-from-singles
with singles scaled by `f` and the scatter estimate on the thinned sinogram; it
is expected to hurt, and the scatter fit's tail numerator is exactly where the
randoms model is weakest.
