# cal

The scanner's own calibration, kept alongside the tool.

`estimate.py` normally locates the norm scan itself, by following the chain the
exam declares:

```
emission SINO*  header 0xEEC = norm_cal_uid
  -> <uid>.3dnorm            (0017,1005) "PET 3D Normalization"
                             (0017,1007) /petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001
  -> that file inside the exam's own drop
```

The copies held here allow the tool to run when a drop does not carry its
calibration, and preserve the provenance independently of any user directory.

| file | description |
|---|---|
| `norm_DXRM3_20231020.rdf` | the 3D normalisation scan itself, byte-identical to `12082026/petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001` (md5 `01fd1024…`), 125,646 B, 20 min, 188.4 M prompts, acquired 2023-10-20 |
| `1.2.840…195435.3dnorm` | GE's bookkeeping record naming the scan above |
| `1.2.840…151600.3dwcc` | the matching well-counter record, Activity Factor 4.062769 |

`.3dnorm` and `.3dwcc` are DICOM files, read with
`pydicom.dcmread(..., force=True)`; only the `SINO*` is non-DICOM RDF.

## The two drops differ by one byte, which is not a coefficient

`11082026` carries its own copy of the same scan under a different md5
(`1d2a0151…`). The two files differ in exactly one byte, at offset 83: bit 31 of
the u32 at offset 80 (`11641` in the NEMA drop, `0x80002D79` in the patient
drop), a flag inside the header.

```
crystal efficiencies  [8192:118784]   0 bytes differ   -> identical
3D geometry factors   [118784:end]    0 bytes differ   -> identical
```

The calibration data is therefore bit-identical and the bundled copy is safe for
either exam. `estimate.py` still prefers the exam's own drop, so this fallback
matters only for a drop shipped without its calibration.

## Contents of the norm scan

`ge_rdf_tool.py info` reports the file as header-only (`payload_offset: 0`),
which is incorrect. The 125,646 bytes hold:

```
byte   8192 .. 118784   27,648 float32 = 2 × 24 × 576 crystal efficiencies
                                          (min 0.855  max 1.247  mean 1.005)
byte 118784 .. end       6,862 B       = 3D geometry factors, 381 × 9 uint16
```

## Which scanner, and when

`(0019,1002)` and `(0017,1002)` hold the station identifier, and the console's
calibration directory contains records for two scanners. This one is DXRM3, with
UID root `1.2.840.113619.2.290.663120.*`:

| station | scanner | WCC dates | Activity Factor |
|---|---|---|---|
| DXRM3 | Discovery TOF — this D710 | 2023-10-20, 2024-06-07, 2025-02-24 (×3) | 4.0628 → 3.9854 → 4.0089 |
| HRBME | Discovery ES — a different scanner | 2022-06-08, 2025-10-23 | 0.2127, 0.1295 |

The newest WCC file on the console (2025-10-23) is therefore not this scanner's.
For DXRM3 the newest calibration is 2025-02-24.

The 2026 exams nonetheless reference the 2023-10-20 pair. That is what their
headers state, and it is why this is the copy kept here. See `../README.md`,
section 0b.
