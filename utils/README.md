# utils

Everything that does not belong to a specific reconstruction algorithm. The
algorithm lives in `osem/`; a later algorithm (FBP, MLEM, a deep prior) belongs
in its own package at the same level and reuses these modules.

The governing rule is that a function meaningful only to OSEM belongs in
`osem/`.

| module | responsibility |
|---|---|
| `paths.py` | `$D710_OUT/<case>/...`; the only module that knows the output tree |
| `container.py` | the only module that knows how to invoke `docker` from Python |
| `attenuation.py` | CT DICOM to mu-map (`load`, `hu_to_mu`, `mu_image`, `factors`) |
| `geometry.py` | the D710-to-STIR bin index conventions and the crystal positions (`crystal_to_det`, `det_pair_map`, `crystal_positions`); plain numpy, no STIR |
| `interfile.py` | the projdata header reader, which does not need STIR |
| `binmap.py` | crystal pairs to sinogram bins, and the one statement of the detector-to-ring pairing |
| `attn_proj.py` | attenuation factors by parallelproj line integral, in PyTomography's world frame |
| `terms.py` | loading one bed's terms, plus the summary and invariant tables |
| `attn.py` | per-bed `af`, cached in `work/bed<n>/attn.hs`; builds it with `attn_proj`, reads it with `np.fromfile` |
| `sirf_env.py` | changing into the scratch directory and keeping `MessageRedirector` alive — used by `osem/` alone, the one command that still needs SIRF |
| `quant.py` | counts per voxel to Bq/mL to SUV, and the constant `K` |
| `export.py` | writing NIfTI and DICOM (`python3 -m utils.export` is `d710 export`) |
| `plots.py` | figures for inspecting sinograms and images; currently uncalled |
| `scanner.py` | every machine constant and the image grid, in one place |

## Scanner constants

`scanner.py` is the single source for every number describing the machine and
the grid it is reconstructed on. Nothing in it is a tuning parameter: each value
is measured, read from a vendor header, or derived from one that is.

| constant | value | provenance |
|---|---|---|
| `NRINGS`, `NDET` | 24, 576 | detector geometry |
| `R_MM` | 405.10 | header `Inner ring diameter (cm) := 81.02`, halved |
| `XTAL0_OFFSET_DEG` | −5.0210 | azimuth of GE crystal 0 in the gantry frame; `cmcfg.XR.xml` and every RDF |
| `XTAL_PITCH_DEG` | 360/576 | GE's own `deltaAngle` |
| `VIEW_OFFSET_DEG` | +4.3960 | STIR's `psi_offset`, a different quantity in a different frame from `XTAL0_OFFSET_DEG` |
| `DOI_MM` | 8.4 | depth of interaction |
| `PLANE_MM` | 3.2699997 | axial plane pitch |
| `NSEG0` | 47 | axial positions in segment 0 |
| `N_TOF_RAW`, `TOF_LSB_PS` | 55, 89.2459 | header `coincTimingPrecision` |
| `BIN_MM`, `DR_MM` | 2.1306 | tangential bin size and transverse voxel pitch |
| `XY` | 337 | the only matrix size giving 2.130600 mm in both SIRF builds, which is what lets `osem` and `lm` be compared |
| `PSF_XY_MM`, `PSF_Z_MM` | 4.87, 4.45 | GE's resolution model as Gaussian FWHMs: `psfLUT.XR` radial kernels fitted over \|s\| < 195 mm, and `sharcAp.cfg.XR` `PSF_AXIAL_KERNEL` [1/6, 2/3, 1/6] |
| `PSF_FWHM_MM` | (4.87, 4.87, 4.45) | the two above in PyTomography's object order (x, y, z), the default of `lm` |
| `POST_FILTER_FWHM_MM` | 6.4 | `(0009,10BB) post_filt_parm`: GE's post-filter, not a PSF |

`VIEW_OFFSET_DEG` is provisional and is not the same quantity as
`XTAL0_OFFSET_DEG`; copying one into the other costs 10.04° of image rotation on
both reconstruction paths, since they share the header. Only a NEMA scan
separates +4.396 from +5.021. The derivation and the open conflict are recorded
in `GEOMETRY_AUDIT.md`.

## Code that is retained but currently uncalled

The tree once contained `osem_pipeline.ipynb`. The notebook has been removed and
the parts of `utils/` that served it have been kept. They are listed here so
that the state does not have to be rediscovered by grep, and so that they are
not mistaken for an oversight.

| symbol | original purpose |
|---|---|
| `plots.py`, the whole module | notebook figures; `slices` and `busiest_plane` are called only by `terms.collect` |
| `terms.collect`, `bed_table`, `invariant_table`, `invariants`, `summarise` | the notebook's table and invariant cells, now covered by `tests/test_pipeline_data.py` |
| `quant.suv_table`, `suv_bsa`, `bsa_m2`, `body_mask`, `voxel_ml` | the body-surface-area SUV branch beneath `suv_table` |
| `geometry.open_projdata` | used only by `tests/test_geometry.py` |
| `geometry.ring_pair_multiplicity` | not dead code: it is the oracle of `tests/test_lm_geom.py`, and its correctness consists in nothing else calling it |
| `osem.stitch.plane_index` | plane index lookup, with no remaining caller |

Verify this list by counting references through the AST across the whole tree
rather than by grepping function names: several of these names (`collect`,
`slices`) are common words.

The four correction terms are not built here. They are taken directly from GE's
kernel:

```bash
d710 estimate --raw <petRDFS directory> --ct <CT DICOM directory> --case <case>
```

## Two conditions recorded here

`geometry.ring_pair_multiplicity()` must not be applied on the vendor path.
GE's `normdt` already carries the span-2 multiplicity, and multiplying by it
again squares it. See the function's docstring and
`tests/test_pipeline_data.py`.

`attenuation.to_radiological()` is its own inverse. `mu_image` calls it, because
STIR flips y relative to DICOM, and `export.to_dicom_order` calls it again to
undo the flip. The flip must not be reimplemented in a third place.

## Why `container.py` exists

`D710/` once reached up into `../../custom_tool/` for the decoder and the
calibration tree. Both are now present inside the image:

| formerly on the host | inside the image |
|---|---|
| `custom_tool/ge_rdf_tool.py` | `/opt/custom_tool/ge_rdf_tool.py` |
| `custom_tool/petsw/.../cal/*.3dnorm` | `/usr/PET/systemConfig/cal/*.3dnorm` |
| `.../cal/*.3dwcc` | the same directory |

`D710/` therefore holds no reference to `custom_tool/`, and the decode, estimate
and tostir steps require only bash, docker and the Python standard library. A
single point of entry also means the tests have only one place to stub.
