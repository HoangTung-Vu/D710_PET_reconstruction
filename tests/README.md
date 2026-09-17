# tests

```bash
conda activate petct_recon                 # the host runtime (environment.yml)
export D710_OUT=~/UET/d710_out             # so the data-backed tests find the beds
cd D710

python -m pytest -q                        # everything
tests/run_tests.sh                         # with an environment and bed inventory
tests/run_tests.sh --no-data               # the synthetic tests only
tests/run_tests.sh --case nema             # the beds of one case only
```

Without the environment activated, `stir` and `sirf.STIR` cannot be imported and
the tests that require them skip with a stated reason. No test fails for want of
an environment.

There are two environments and they give two different skip counts; both report
zero failures:

| environment | description | the SIRF/STIR tests |
|---|---|---|
| `petct_recon` | the host runtime, built from `environment.yml` | skipped; this environment deliberately has no SIRF |
| `petct_reconstruction` | the environment SIRF was built from source in, holding `dlevel/` | executed |

The names are similar enough to be confused; see the note in the top-level
`README.md`.

## The miniature scanner

A real bed is `553 × 288 × 381`: the prompts occupy 121 MB and each correction
term 231 MB. No test should carry that. Every geometric rule involved is a rule
of **span 2** rather than of 24 rings, so `synth_hs.py` shrinks the scanner while
preserving the rules:

| | rings | detectors per ring | views | tangential | planes |
|---|---|---|---|---|---|
| the real D710 | 24 | 576 | 288 | 381 | 553 |
| `mini_hs` | 6 | 16 | 8 | 9 | 31 |
| `bed24` | 24 | 48 | 24 | 9 | 553 |

`bed24` retains all 24 rings because `utils.attenuation.mu_image` requires a
bed's exact 47-plane image grid; it shrinks only the two axes that grid does not
depend on.

`synth_ct.py` builds a synthetic CT series: a water cylinder in air with a dense
insert displaced towards +y. That displacement is a necessary condition — a
symmetric phantom cannot distinguish a y flip from the identity, and exactly one
y flip constitutes the whole orientation convention of the pipeline.

## Tests that require real data

`test_pipeline_data.py` runs on decoded beds in `$D710_OUT/<case>/decoded/` and
terms in `$D710_OUT/<case>/work/bed<n>/`. Both are derived from patient data and
lie entirely outside the source tree, so the tests skip when they have not been
built, and likewise when `$D710_OUT` is unset. Build them with:

```bash
d710 exam --raw <petRDFS/.../DIR> --ct <CT series> --case ped
```

Each bed found becomes a separate parameter (`ped-bed4`, `nema-bed2`, and so
on). Sinograms are read by memory map rather than through
`sirf.AcquisitionData`, so all seven beds currently available run in about
12 seconds instead of consuming 8 GB of RAM.

`D710_CASE=ped` narrows the run to one case. `D710_CT=<CT directory>` enables an
additional check of the CT's `FrameOfReferenceUID` against the bed's
`sop_instance_uid`.

## The import path

The import path is declared in `pytest.ini` (`pythonpath = . vendor`) rather
than by `sys.path` manipulation in `conftest.py`. Both entries are relative to
that file.

* `.` — `utils` and `osem` are genuine packages: `from utils import
  attenuation`.
* `vendor` — `estimate.py`, `to_stir.py` and their neighbours are *scripts*, run
  as `python3 vendor/x.py`, so their own directory is `sys.path[0]` and they
  import one another by bare name. The tests place them on the path the same
  way rather than inventing a package layout the shipped code does not have.

`tests/` itself is added by pytest under prepend import mode, which is what
makes `import interfile`, `import synth_hs` and `from cases import ...` work.

## Notebooks must not contain code

`test_forward_model.py` fails if a code cell of any notebook in the tree defines
a `def` or a `class`, or exceeds 15 statements. There is currently no notebook,
so those two tests skip with a stated reason; they remain because the rule
applies to notebooks added in future rather than to the one that was removed.
This is a machine-enforced constraint on something that actually happened:
`utils/` was once copied into a notebook, the two copies diverged, and both
continued to run while no longer computing the same thing.

## What the tests target

Each is a place where an error produces a plausible-looking image rather than an
exception:

* **GE-to-STIR bin order** — `stir[0, plane, 287 − ge_view, u]`. Reversing the
  view axis incorrectly mirrors the image transversally, which is not visually
  obvious.
* **Span 2** — segment 0 merges two ring pairs into each odd plane. `normdt`
  already carries that multiplicity
  (`test_span_2_doubles_the_odd_planes_of_every_term`), so multiplying by
  `ring_pair_multiplicity()` again squares it.
* **The direction of `normdt`** — it is a *sensitivity*, so dividing is the
  correction. `test_the_sensitivity_multiplies_rather_than_divides` fixes the
  direction using SIRF itself, and `test_dead_time_is_a_livetime_fraction` fixes
  it against data.
* **`b` bypasses `S`** — `test_the_forward_model_is_s_times_gx_plus_b`
  assembles `y = S·(Gx) + b` and compares it against known `S` and `b`.
* **Plane order across a disk round trip** — SIRF writes headers in a different
  layout from the decoded files (segments ascending, and the view axis before
  the axial axis), yet `as_array()` returns the same order. Since the pipeline
  multiplies `normdt` by `attn` as numpy arrays, this is required, and it is not
  obvious.
* **Exactly one y flip** — CT enters through `mu_image` (flipped) and leaves
  through `write_dicom` (flipped back);
  `test_a_ct_feature_comes_back_at_the_same_patient_coordinate` traverses the
  whole round trip and compares in millimetres.
* **The unit of mu** — STIR uses 1/cm and PIFA uses 1/mm. Confusing them is a
  factor of ten with nothing to signal it.

## Conventions

One behaviour per test, and each test short. No xfail. Any figure that was
*measured* is recorded with its date of measurement in the docstring, so that a
later reader can tell a result from a value chosen to fit.
