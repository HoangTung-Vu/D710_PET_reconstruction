"""Shared infrastructure — everything NOT belonging to a specific reconstruction algorithm.

| module | contents |
|---|---|
| `paths` | `$D710_OUT/<case>/...`; the only place that knows the output tree |
| `attenuation` | DICOM CT -> mu-map -> attenuation factors |
| `geometry` | D710 -> STIR bin index conventions |
| `terms` | load one bed's terms from Interfile |
| `attn` | per-bed `af`, cached on disk (`work/bed<n>/attn.hs`) |
| `sirf_env` | chdir into scratch + keep the `MessageRedirector` alive |
| `quant` | count/voxel -> Bq/mL -> SUV; the constant `K` |
| `export` | write NIfTI / DICOM |
| `plots` | figures for the notebook |

The algorithms live elsewhere: `osem/` is one, and a later algorithm (FBP, MLEM,
deep prior) gets its own sibling package and reuses this same `utils/`. So
**nothing OSEM-specific belongs here** — if a function only makes sense for OSEM,
its place is `osem/`.
"""
