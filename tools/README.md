# tools

Manually invoked utilities. None lies on the `d710 exam` → `osem` → `export`
path, and nothing here is imported by the pipeline; removing a file in this
directory does not affect reconstruction.

## In use

| file | purpose |
|---|---|
| `compare_vendor.py` | compares a reconstruction with GE's BQML series for the same case; `--json` writes `calib_<path>.json` |
| `calib_k.py` | combines the `calib_*.json` files into the two `K` constants recorded in `utils/scanner.py` |
| `dicom_suv.py` | any PET DICOM directory to a SUV NIfTI |
| `compare_suv.py` | compares two SUV volumes |
| `ct_nifti.py` | CT DICOM to a Hounsfield-unit NIfTI, on the affine `export` uses |
| `migrate_out.sh` | migrates an earlier output tree into `$D710_OUT`; performs a dry run by default |

`../run_all_ok.sh`, `../export_all_ok.sh` and `../rerun_lm_ok.sh` invoke the
first four.

## Diagnostics, retained because they are referenced

| file | reason for retention |
|---|---|
| `tof_direction.py` | measures the direction of the TOF axis; `utils/terms.py` names it in an error message |
| `tof_profile.py` | measures the scatter's TOF profile; `utils/terms.py` prints this exact command when needed |
| `lm_frame.py` | measures the list-mode reference frame; `tests/test_lm_geom.py` cites it as the source of the established frame. It is blind to a shared angular error: a 10.04° offset once passed through it |

## One-off measurements, retained as evidence

Neither is called, and both are kept deliberately: they are the measurements
behind two architectural decisions rather than dead code awaiting removal.

| file | what it established |
|---|---|
| `projector_bench.py` | the cost of the projector, and hence why subsets buy nothing with parallelproj |
| `pytomo_lm_probe.py` | that PyTomography accepts D710 list-mode data, which is the basis for choosing it |

`pytomo_lm_probe.py` builds its own geometry and has diverged from `lm/geom.py`:
it uses `(cos, sin)` rather than `(sin, −cos)`, bare `R_MM` rather than
`R_EFF_MM`, and does not reverse the TOF axis. Its docstring states that the
constants come from `utils/scanner.py`, which is true of the *constants* but not
of the geometry. It must not be read as a reference implementation.

## Stubs

`stubs/kornia_rs.py` is an inert replacement for the `kornia_rs` extension, for
CPUs without AVX2. It is not on the default `PYTHONPATH` and is enabled per
machine; see the top-level `README.md`.
