# lowdose

Retrospective count reduction. A full-dose exam is turned into a lower-count or
lower-dose one by decimating its event stream. The output is an ordinary case
rather than a special format, so `d710 osem`, `d710 lm` and `d710 export` all
run on it unchanged.

```bash
conda activate petct_recon
d710 lowdose --case ped --drf 10                    # -> ped_drf10    (low-count)
d710 lowdose --case ped --drf 10 --window time      # a literally shorter frame
d710 lowdose --case ped --drf 10 --mode low-dose    # -> ped_drf10_lowdose
d710 lowdose --case ped --split 2                   # -> ped_r0, ped_r1 (Noise2Noise)
d710 lowdose --case ped --drf 4 --replicates        # -> ped_drf4_r0 .. _r3

d710 attn      --case ped_drf10
d710 lm recon  --case ped_drf10
d710 export    --case ped_drf10 --lm    # K × DRF applied automatically
```

The runtime requires numpy only: no SIRF and no torch. The simulator needs
`decoded/bed<n>.lm.npy` (`d710 decode --listmode --format npy`) and the
`work/bed<n>` terms of the source case. `--sinogram binomial` removes the first
requirement.

## The two modes are two different experiments

| mode | simulates | keep probability | prompts | randoms | scatter |
|---|---|---|---|---|---|
| `low-count` | a shorter scan | `f` | `× f` | `× f` | `× f` |
| `low-dose` | less activity | `q_b = f(1−ρ_b) + f²ρ_b` | trues and scatter `× f`, randoms `× f²` | `× f²` | `× f` |

`--window time` modifies the `low-count` row: the factors become decay-weighted
and randoms cease to match trues, as described below.

A coincidence rate is linear in live time, so halving the scan halves everything
alike. Halving the *injection* does not: a random is a coincidence between two
unrelated singles, each single is linear in activity, and randoms therefore
scale as the square. That single exponent is the whole difference between the
modes, and `thin.randoms_power` is the only place it is decided; three call
sites previously carried their own copy of it.

### Mode 1, `low-count`

This is the default and the primary result. It is a reduced-acquisition-time
simulation and a *conservative* low-dose one: it scales trues and scatter
correctly but leaves the randoms fraction artificially high, so the emulated
image is if anything slightly worse than a real low-dose scan. `f = 1` is the
identity, bit for bit.

#### Which short scan: `--window`

A uniform draw keeps a fraction `f` of every event whatever time it arrived, so
it reproduces the frame's *average* activity. A real short scan samples the
activity at the moment it ran. The two agree only when the window sits at the
frame's mid-time:

```
frame average = (1/T)∫λ(t)dt = λ(t_c)   ⟹   t_c ≈ T/2
```

`--window uniform` (the default) is therefore a short scan at the frame's
mid-time, and `--window time` — keeping the first `f` of the frame — is a short
scan at its start. The two differ by the decay across half the frame.

In a time window the terms do not scale by `f`. The activity decays *during* the
frame, so the retained window holds more than its share, and randoms hold more
still, because a random is a coincidence of two singles and its rate falls
as `A²`:

```
trues, scatter   rate ∝ A    →  ∫₀^τ A dt    →  (1−e^{−μτ}) / (1−e^{−μT})
randoms          rate ∝ A²   →  ∫₀^τ A² dt   →  (1−e^{−2μτ}) / (1−e^{−2μT})
```

Randoms therefore carry a different exponent from trues even in low-count mode,
not because of dose but because of decay inside the window. For F-18 with a 90 s
frame at `f = 0.1`: trues and scatter ×0.100427, randoms ×0.100855. Measured on
fdg26081008, all seven beds kept 0.1004–0.1008, and the predicted prompts factor
`(1−ρ)·lin + ρ·quad` is 0.10061 at ρ = 0.435.

The magnitude depends entirely on the ratio of frame length to half-life:

| isotope | T½ (s) | trues × | randoms × | randoms against `f` |
|---|---|---|---|---|
| F-18 | 6586 | 0.10043 | 0.10085 | +0.9 % |
| Ga-68 | 4062 | 0.10069 | 0.10139 | +1.4 % |
| C-11 | 1222 | 0.10231 | 0.10466 | +4.7 % |
| O-15 | 122 | 0.12448 | 0.15171 | +51.7 % |

For F-18 it is below 1 %, which is why the uniform thinning used in the
literature is not wrong. For a short-lived tracer it is not optional.

A uniform draw exhibits none of this: it keeps a fraction `f` of every event
regardless of arrival time, so every integral scales by exactly `f` and the
frame is unchanged. That exact self-consistency is its principal advantage,
together with being a *controlled* experiment: a contiguous window is a single
moment, and if the patient moved or the tracer redistributed the result is a
different scan rather than a lower-count version of the same one.

`--window time` updates `frame_duration_ms` to the true `f·T`, which is all that
`osem.stitch.decay_factor` requires, since it divides by the mean activity over
that duration and the window starts where the frame does, leaving
`bed_start_time` untouched. `delays` follows the randoms factor, so
`randoms/delays` remains approximately 0.99.

> One residual is small and deliberately not pursued.
> `quant.scan_start_factor` notes that the frame-duration term cancels against
> GE's, because GE applies the same mean-activity-over-frame correction. With a
> 9 s frame against GE's 90 s it no longer cancels exactly, leaving a 0.42 %
> offset, which `K` absorbs since `K` is fitted. It is well below the 3.5–5 %
> residual the dose ladder already shows.

`--window time` is restricted to `--mode low-count --sinogram derived` and
refuses `--replicates`: combining it with `low-dose` would reduce activity and
duration at once, giving a third set of factors that has not been measured here;
it requires timestamps; and time-separated replicates are not exchangeable.

### Mode 2, `low-dose`

Thinning bin `b` with `q_b` gives exactly `Poisson(f(T+S) + f²R)`: the correct
mean and the correct Poisson variance. `q_b ≤ f ≤ 1` always holds, so it is a
valid probability for every `f`.

#### `ρ` must be estimated per (plane, tangential bin): `--rho`

That guarantee holds only if `ρ_b` is the bin's own randoms fraction, and `ρ`
cannot be estimated per bin: raw bins hold about 0.06 counts, so `ρ > 1` occurs
constantly from Poisson noise alone. It must therefore be aggregated, and the
axis it is aggregated over decides whether Mode 2 is correct:

| `--rho` | cell | counts per cell, fdg26081008 bed 1 | tail error at `f = 0.1` |
|---|---|---|---|
| `tangential` (default) | (plane, u), summed over 288 views | ~70 | none, by construction |
| `plane` | the whole plane | ~26,800 | 6.1× too many counts |

`ρ` varies far more across `u` than along the axis: a tail LOR misses the
patient and is almost pure randoms (`ρ → 1`), while a central one is mostly
trues (`ρ ≈ 0.2`). Giving every bin in a plane the plane mean (`ρ̄ = 0.435` on
bed 1) makes the tails keep `q̄ = 0.061` where they should keep `f² = 0.010`,
while the randoms sinogram there was scaled by exactly `f²`. Data and model then
disagree by a factor of six in the tails.

This is not a theoretical concern; it broke the pipeline. The scatter TOF
estimator selects its tail ring where `T/S` is small, with `T = ΣP − R − S`. The
unsubtracted randoms excess inflated `T` so that `min T/S` moved from 0.000 to
0.712, the ring collapsed from 165 tangential bins to 4, and
`scatter_tof_weights` refused. Beyond that it would also bias the
reconstruction, whose background term under-subtracts in the tails.

Note what did not catch it: `Σ q̄·y = f(T+S) + f²R` still holds *in aggregate*,
so the per-plane binomial check passed (10 of 3871 planes outside 3 sd) and the
invariants reported zero. The error lies purely in the redistribution *within*
each plane. `--rho plane` is retained for a bed too sparse for a `(plane, u)`
cell to be subdivided, and is wrong in the tails whenever it is used.

#### `ρ` also depends on TOF, and that is modelled: `--tof-rho`

The same argument applies one axis further in. Randoms are flat in TOF and trues
are not, so within a single LOR the randoms fraction climbs steeply away from
the TOF peak, and giving every TOF bin of that LOR one `q` repeats the error.
Measured on fdg26081008 bed 1, central LORs, `f = 0.1`:

| TOF bin | offset | % of counts | `ρ(t)` | correct `q` | non-TOF `q` | error |
|---|---|---|---|---|---|---|
| 27 | 0 (peak) | 5.10 % | 0.081 | 0.0927 | 0.0795 | 0.86× |
| 17 | 10 | 1.99 % | 0.208 | 0.0813 | 0.0795 | 0.98× |
| 12 | 15 | 0.63 % | 0.658 | 0.0408 | 0.0795 | 1.95× |
| 7 | 20 | 0.43 % | 0.962 | 0.0134 | 0.0795 | 5.92× |
| 2 | 25 | 0.42 % | 0.996 | 0.0103 | 0.0795 | 7.71× |

Across the whole bed that is 10.6 % excess counts (95,874 of 906,694), sitting
in the deep TOF tails as unthinned randoms, while the *total* comes out only
0.7 % low because the 12 % deficit at the peak nearly cancels it. It is
therefore invisible in any count total, and it reaches only the list-mode TOF
path: the sinogram is non-TOF and receives
`E[y'_b] = f(T_b+S_b) + f²R_b` exactly either way.

`ρ` cannot be *estimated* per `(bin, TOF)`, which would be about 0.001 counts
per cell. It is modelled instead, using the one fact that makes this possible —
randoms are flat in TOF, measured at a coefficient of variation of 0.0404
against a Poisson floor of 0.0393:

```
ρ(b,t) = (R_b / n_tof) / y(b,t)  and  y(b,t) = y_b · φ(u,t)
      ⟹  ρ(b,t) = ρ_nonTOF(b) · (1/n_tof) / φ(u,t)
```

The non-TOF `ρ` already carries the axial and radial structure, and the factor
depends only on `(u, t)`. `φ(u,t)` is the prompts' TOF profile at radial offset
`u`, measured on the *source* at `n_tang × n_tof` cells of about 707 counts
each, so it adds no meaningful noise. The single approximation is that the TOF
shape at a given `u` does not depend on plane or view.

This was checked against a direct measurement of `ρ(t)` on that bed — TOF bin 27
gives 0.081 both ways and bin 7 gives 0.962 both ways — and
`tests/test_lowdose.py` asserts the identity algebraically. `--tof-rho off`
reproduces the earlier behaviour.

Two consequences are worth noting:

* A flat `φ` is a no-op. A bin that really is 100 % randoms has a flat TOF
  profile — the two are the same statement — so the factor is exactly 1 there
  and `q` remains `f²`.
* The TOF peak does not reach `q = f`. A flat randoms floor lies beneath the
  peak, so even there `ρ ≈ 0.14` rather than 0. The measured value is
  `q = 0.877 f`.

Because `q` now varies *within* a `(plane, u)` cell, the binomial check can no
longer form its expectation from one probability per cell. It must not model
that expectation either. The first attempt did, weighting `q` by the global
`φ(u,·)` within each `(plane, u)`; oblique planes have broader TOF distributions
than the global average, so their events drew lower `q` than the prediction
assumed, giving 98 of 3871 planes outside the 3 sd band with every bed biased
the same way. The data were correct; only the prediction was not.

The expectation now contains no model. A thinned plane total is a sum of
independent Bernoulli draws, one per event, so

```
mu  = Σ q_i        var = Σ q_i (1 − q_i)        over the events in that plane
```

`thin.event_q` returns the `q_i` the draw is made against, `thin.expectation`
sums them, and `verify.binomial` takes the pair as absolute counts. Re-measured
on fdg26081008: 11 of 3871 planes, 0.28 %, with residuals scattering both ways.

> **A caveat that belongs in any write-up.** The per-ring randoms distribution
> Mode 2 inherits is known to be 20–40 % out, so its *total* randoms scaling is
> correct while its *spatial* redistribution carries that error — and at small
> `f` the tails go to `f²`, which is where the error lies. Report both modes and
> lead with Mode 1.

### Mode 4, `--replicates`

Independent multinomial assignment of a Poisson total gives exactly independent
Poisson subsets, unlike repeated thinning whose realisations overlap. `k = 2` is
a Noise2Noise pair. The labels are drawn from an RNG seeded on the event count
alone, so replicates `k` and `k'` of the same bed are guaranteed disjoint. It
partitions the stream, which is uniform by construction, and is therefore a
`low-count` operation only.

> **Correlation.** A thinned dataset is a *subset* of the full one, so comparing
> a low-dose reconstruction with a full-dose one carries a positive correlation
> bias. For clean noise measurements, compare against the complement replicate
> rather than the full scan.

The earlier names `uniform` and `randoms` remain accepted wherever a mode is
accepted, including in the `mode` field of a `lowdose.json` already on disk.

## Both forms of the data are thinned

The exam exists as an event table *and* as a histogram, and both reach a
reconstructor: `d710 lm` reads `decoded/bed<n>.lm.npy` and `d710 osem` reads
`decoded/bed<n>.s`. Both are thinned, by default from a single draw:

* `--sinogram derived` (the default) thins the events and then histograms the
  survivors. The two paths then see literally the same thinned data, and
  thinning a Poisson variable gives a Poisson variable, so this is
  distributionally identical to thinning the histogram. One decimator serves
  both paths and the equivalence is guaranteed by construction rather than by
  argument.
* `--sinogram binomial` draws `y'_b ~ Binomial(y_b, q_b)` directly from
  `decoded/bed<n>.s`. The distribution is the same, and it is what allows the
  simulator to run on a bed with no event table at all, which is otherwise a
  hard error. The result has no `bed<n>.lm.npy`, so `d710 lm` cannot run on it.
  This is a fallback rather than a second opinion: two independent draws would
  give the two reconstructors different data.

`derived` is correct only if the event table reproduces the decoded sinogram in
the first place, so `verify.lm_matches_sinogram` checks that per bed before any
work is done. `decode_in.sh` enforces the same property at decode time from the
`.lm.json` sidecar; a thinned case has no sidecar, so it is measured here.
Measured on fdg26081008, the check is exact on all seven beds with no event
falling outside the sinogram.

The method itself is the standard one. It was validated against real
dual-injection low-dose and standard-dose human scans (Schaefferkoetter 2019)
and is how the UDPET dataset was built.

## `thin.py` — the decimators

| | |
|---|---|
| `canonical(mode)`, `randoms_power(mode)` | either spelling to the mode; `1` or `2` |
| `keep(e, f, "low-count", rng)` | keep each event with probability `f` |
| `keep(e, f, "low-dose", rng, bins, rho)` | keep with `q_b = f(1−ρ_b) + f²ρ_b` |
| `binomial_sinogram(y, q, rng)` | thin a histogram directly, one plane at a time |
| `rho_bins(P, R, binmap, axis)` | `ρ = randoms/prompts` at `tangential` or `plane` |
| `time_window(t_ms, f, frame_ms)` | keep the first `f` of the frame, without an RNG |
| `decay_scales(T½, frame_ms, f)` | the `(linear, quadratic)` factors for that window |
| `tof_rho_factor(φ, n_tof)` | `(1/n_tof)/φ(u,t)`, making `ρ` TOF-resolved |
| `event_q(f, bins, ρ, tof)` | each event's keep probability, the quantity actually drawn against |
| `expectation(q, plane, n_plane)` | `Σq` and `Σq(1−q)` per plane, the draw's exact mean and variance |
| `split(n, k, rng)` | `k` disjoint, mutually independent replicates |

## `write.py` — the case it writes

Everything the simulator generates goes into `<case>/raw_simulation/` and is
symlinked into `decoded/` and `work/bed<n>/` under its usual name:

```
d710_out/<case>/
  lowdose.json                       dose fraction, mode, seed, per-bed counts, k_scale
  raw_simulation/
    README.txt                       provenance, in plain text
    bed<n>.hs  bed<n>.s              thinned prompt sinogram
    bed<n>.lm.npy                    thinned event table, from the same draw
    bed<n>/randoms.{hs,s}            × f, or × f² in low-dose mode
           scatter.{hs,s}            × f
           background.{hs,s}         randoms + scatter, rebuilt rather than scaled
  decoded/bed<n>.{hs,s,lm.npy}  ->  ../raw_simulation/...
  work/bed<n>/{randoms,scatter,background}.{hs,s}  ->  ../../raw_simulation/bed<n>/...
  work/bed<n>/{normdt,norm_only,attn}.{hs,s}, scatter_tof.npy   copies of the source
```

One directory therefore holds exactly what thinning changed and nothing else,
which is also what distinguishes a simulated case from a measured one at a
glance. The links are relative, so they resolve identically on the host and
inside `d710:full`, which bind-mounts the output root at its
own path; and they stay inside the case, so `rm -rf <case>` still removes
everything. Links rather than second copies, because the thinned prompts are
116 MiB per bed and each scaled term 232 MiB.

What changes, and only what physics requires:

| | |
|---|---|
| prompts | thinned event by event, then histogrammed |
| `scatter` | `× f`, being linear in activity |
| `randoms` | `× f`, or `× f²` in low-dose mode |
| `background` | rebuilt as `randoms + scatter`, the halves no longer sharing a factor |
| `delays` (sidecar) | `× f^power`, so that `randoms/delays` remains ≈ 0.99 |
| `normdt`, `norm_only`, `attn` | copied, since sensitivity does not depend on dose |
| `scatter_tof.npy` | copied, since it is a shape rather than an amplitude |
| `scatter_tof_profile.npy` | measured on the full-count source, as described below |
| `to_stir.json` | rewritten rather than copied, as described below |
| `K` | `× 1/f`, recorded in `lowdose.json` |

### The scatter TOF profile is measured on the source

When the source has no `scatter_tof.npy` — that is, when GE's scatter was
estimated non-TOF — the list-mode path measures the scatter's TOF profile from
the bed's own tail ring. A thinned bed is the wrong place to do that: the
profile is a dose-independent shape, but the tails it is measured from are
exactly what thinning destroys. Measured at DRF 10 on fdg26081008:

| bed | `lowcount` profile against the source's | centroid shift |
|---|---|---|
| 1 | 8.8 % of peak | −0.33 TOF bins |
| 4 | 9.2 % of peak | −0.08 TOF bins |

In low-dose mode the tail ring disappeared entirely.
`write.carry_tof_profile` therefore measures it once on the *source's* events at
the full 55 bins and writes `work/bed<n>/scatter_tof_profile.npy`.
`utils.terms.measured_tof_weights` picks it up and rebins it to whatever
`--tof-bins` requests, and both `utils/terms.py` and `lm/terms.py` consult it
between GE's own weights and the in-place fallback. No flag has to be
remembered, and the thinned case remains self-sufficient.

This is an accuracy improvement rather than a prerequisite: if the source has no
measurable tail ring either, the failure is reported in the build output and the
reconstruction falls back to its own measurement.

`work/bed<n>/lm.npz`, the keyed cache `d710 lm recon` leaves behind, is
deliberately not copied. It holds prepared full-dose data, and a thinned case
that inherited it would reconstruct the source exam while every count on disk
said otherwise.

`to_stir.json` has to travel, because `utils.terms.ct_dir` and
`d710_isolate_stir.sh` read `estimate.ct` from it to locate the CT, but most of
it measures the *source*. Copied verbatim it made the derived case claim a
bit-exactness proof for prompts it does not have, and carry `stats` for terms
that had since been scaled; `tests/test_pipeline_data.py` detects exactly that.
`verified.prompts` is therefore rewritten, `stats` is dropped, and a `lowdose`
stanza records what was scaled.

Headers are cloned from the source and never regenerated: a fresh header
desynchronises ExamInfo, and STIR complains only much later, inside
`make_Poisson_loglikelihood`. Term scaling is element-wise, so the segment
layout never has to be known.

`lowdose.json` carries the dose fraction, the mode, the seed, the per-bed counts
and `k_scale = 1/f`. `utils.quant.lowdose_k_scale` reads it and
`utils/export.py` applies it, so SUV remains flat along the dose ladder without
anyone having to remember a flag.

## `verify.py` — what runs after every write

| | |
|---|---|
| `lm_matches_sinogram(src, beds)` | before thinning: `Σ bed<n>.s` equals the event count |
| `binomial(src, dst, ...)` | per plane, `Σy'` within 3 sd of `Binomial(Σy, q)` |
| `invariants(dst, ...)` | `Σp ≥ Σr` and `Σs ≤ Σ(p−r)` per plane, which must be zero |
| `plane_tang_sums(...)` | the `(plane, u)` aggregate the last two require |

All three report per plane. The raw sinogram runs at about 0.06 counts per bin,
so `p < r` is true at roughly 82 % of bins from Poisson noise alone and a
per-bin assertion says nothing. `--no-check` skips them.

The binomial check must build its expectation at whatever resolution `q` has:
`Σ q_b y_b`, not `f · Σy`, as soon as `q` varies within the plane. It therefore
reads the shape of `q` — scalar, `(n_plane,)` or `(n_plane, n_tang)` — and uses
`plane_tang_sums` for the last. Passing a scalar `f` for a low-dose run fails
553 of 553 planes on perfectly good data.

These checks have a known blind spot. All three aggregate, and the `--rho plane`
defect described above passed every one of them: the totals were correct and
only the distribution within each plane was wrong. An aggregate check is not a
proof of spatial correctness.

## Measurements

Paediatric bed 1, 2026-09-03:

| mode | kept | expected | planes outside 3 sd | invariants |
|---|---|---|---|---|
| low-count DRF 10 | 1,876,897 (0.1001) | 1,875,929 | 2/553 | 0 |
| low-dose DRF 10 | 1,158,231 (0.0617) | 1,158,247 | 4/553 | 0 |
| split 2 | 0.4999 / 0.5001 | — | — | — |

fdg26081008, all seven beds, DRF 10 — three cases carried end to end through
OSEM, list-mode OSEM, export and a `K` fit against GE's own image of the same
exam:

| | low-count uniform | low-count `time` | low-dose |
|---|---|---|---|
| kept | 0.0999–0.1001 | 0.1004–0.1008 | 0.0528–0.0788 |
| randoms / scatter | 0.1000 / 0.1000 | 0.100855 / 0.100427 | 0.0100 / 0.1000 |
| `frame_duration_ms` | 90,000 | 9,000 | 90,000 |
| planes outside 3 sd | 11/3871 | 9/3871 | 11/3871 |
| invariants | 0 | 0 | 0 |
| `k_ls` sinogram (10× = 629,713) | 653,212 (1.0373) | 651,091 (1.0339) | 651,669 (1.0349) |
| `k_ls` list mode (10× = 1,241,780) | 1,289,312 (1.0383) | 1,290,437 (1.0392) | 1,304,047 (1.0501) |

`k_scale = 1/f` leaves a residual of 3.4–5 %, while the choice of method moves
`K` by only about 0.3 %. That residual is therefore the nonlinearity of
two-iteration OSEM in counts rather than the thinning scheme: uniform and `time`
agree to 0.3 % on the sinogram path and 0.1 % on the list-mode path, exactly as
the decay analysis predicts for F-18. The correlation `r` falls from 0.974 to
0.949 and from 0.982 to 0.966, as a tenfold increase in noise predicts. Each
case occupies roughly 12 GB.

## Mode 3, dead time, is deliberately not implemented

Lower activity means less dead time and therefore higher live sensitivity. The
measured livetime across paediatric beds 1–6 runs from 0.9569 to 0.9285 over
208 to 969 kcps, so the correction is bounded by about 7 % and would be applied
as `normdt'(f) = normdt · livetime(f·kcps)/livetime(kcps)`, a fit to that
six-point curve requiring no reverse engineering. It is a roughly 5 %
quantitative refinement on top of a simulator whose leading error lies
elsewhere, so it is recorded here rather than implemented. Note that it is a
*low-dose* effect: a shorter scan of the same patient has the same count rate
and the same dead time.

## The realistic-background ablation

Everything above gives the low-dose reconstruction an oracle background,
estimated from full-count data. A real low-dose scan estimates randoms and
scatter from its own noisy data. That ablation consists of re-running
randoms-from-singles with singles scaled by `f` and the scatter estimate on the
thinned sinogram. It is expected to degrade the result, and the scatter fit's
tail numerator is precisely where the randoms model is weakest.
