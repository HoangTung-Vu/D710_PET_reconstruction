# Tham số đọc từ `pet_recon` **lúc chạy**

Nguồn: `apcfg_live.txt`, dump bằng `probes/dump_cfg.gdb` trong container `d710`,
sau khi `sharcCmpOpenDataFiles(&IgJobReq)` trả 0 trên job XR thật
(`selftest_kh_3dir.job` = D710).

Đây **không** phải đọc file `.cfg`. Đây là hai global `apCfg` và `sysGeometry`
sau khi chính reader của GE nạp chúng từ header RDF + `sharcAp.cfg.XR`. Tên
trường lấy từ DWARF, nên **không còn trường nào phải đoán offset**.

### ⚠ Đây là tham số của scanner trong file selftest, KHÔNG phải của máy mình

Job dùng để dump là `selftest_kh_3dir.job`, và header RDF của nó ghi
`patientID PETWCC3D`, `hospitalName bay87ct`, `scannerDesc Discovery XR`. Máy
mình là **bay85ct**. Cùng model (XR = D710) nên hằng số vật lý và bố cục dùng
chung được, nhưng những thứ **hiệu chuẩn theo từng máy** thì **KHÔNG**:

* `sysGeometry.vqc_*` (căn chỉnh gantry), `transaxial_crystal_0_offset`
* `sysGeometry.dt_*PileUp_factors`, `avgBlockDeadtime`, `avgCrystalSingles`
* `apCfg.wccActivityScaleFactor*`
* hai bảng `scatterHr/HsParameters`

Muốn bộ của **máy mình** thì làm đúng quy trình dưới, nhưng thay
`inputEmissionFileName[0]` / `normalizationSinogramFile` trong `.job` bằng
`SINO*` của mình. `sharcCmpOpenDataFiles` sẽ nạp lại `apCfg`/`sysGeometry`
từ header RDF đó. **Làm việc này trước khi dùng bất kỳ con số nào dưới đây cho
máy mình.**

### Dựng lại bảng này

```bash
cd D710/vendor
./run.sh --out $D710_OUT/_probes probes/dump_cfg.gdb   # -> $D710_OUT/_probes/apcfg.txt
```

---

## 1. Dead time — bộ tham số giờ đã ĐỦ TÊN

Đọc tĩnh dựng được công thức nhưng không có tên cho các trường của struct
deadtime (offset 0x78/0x7c/0x80 nuôi k₁, k₂). Bản sống có đủ:

```
apCfg.normDeadtimeSlopeInnerRing   4.74      normDeadtimeSigmaInner  0.167
apCfg.normDeadtimeSlopeOuterRing   4.57      normDeadtimeSigmaOuter  0.170
apCfg.normDeadtimeFitFactor        1.398
apCfg.norm3dCorrXtalEffClipValue   10.0      <- chính là `clip`
apCfg.blockBusyScaleFactor         1.7
apCfg.normDeadtimeCrystalPileupFactors[27] =
    2.36 2.18 2.09 2.19 1.78 1.44 2.10 1.45 1.30 1 1 1 1 1 1  (rồi 12 số 0)
    -> đúng 15 giá trị khác 0, khớp `(6+1)/2 × (9+1)/2 = 15`
```

### Scalar toàn cục `(a + b + c) / a` — đã có tên và giá trị

Đọc tĩnh chỉ ra `scalar = (a+b+c)/a` với a,b,c là `deadtimeStruct[0x10],
[0x00], [0x08]` mà chưa biết tên. Chúng nằm trong `sysGeometry`, và bộ **3d**
mới là bộ đường 3D dùng:

```
sysGeometry.dt_3dintCorrectionConstant     1.85
sysGeometry.dt_3dmuxCorrectionConstant     0.0
sysGeometry.dt_3dtimingCorrectionConstant  0.20
    -> (1.85 + 0.0 + 0.20) / 1.85 = 1.1081

(bộ 2D để đối chiếu: dt_intCorrectionConstant 1, dt_mux 0, dt_timing 0.11)
```

### `A[i]` / `B[j]` trong `(1−S·A[i])(1−S·B[j])` — chính là mảng này

```
sysGeometry.dt_3dcrystalpileUp_factors[54 giá trị khác 0]
  1.049 1.036 0.975 0.940 0.936 0.940 0.975 1.036 1.049
  1.017 1.005 0.882 0.832 0.827 0.832 0.882 1.005 1.017
  0.991 0.948 0.832 0.767 0.747 0.767 0.832 0.948 0.991
  0.991 0.948 0.832 0.767 0.747 0.767 0.832 0.948 0.991
  1.017 1.005 0.882 0.832 0.827 0.832 0.882 1.005 1.017
  1.049 1.036 0.975 0.940 0.936 0.940 0.975 1.036 1.049
```

**54 = 9 × 6 = radialCrystalsPerBlock × axialCrystalsPerBlock.** Bảng đối xứng
cả hai chiều, trũng ở giữa block — đúng dạng vật lý của pile-up.

**Bảng này KHÔNG giải thích được sai số ring của randoms-from-singles.** Phép
khớp `eps` theo vị trí crystal trong block 6 (trục) cho
`[0.90, 1.03, 1.08, 1.04, 1.02, 0.92]`. Cột trục của bảng trên (bước 9) là
`1.049, 1.017, 0.991, 0.991, 1.017, 1.049` — cũng đối xứng, nhưng **lồi** ở mép
chứ không lõm. Hai thứ **không** khớp nhau; nên bảng này *không* phải lời giải
cho sai số ring của randoms, và đó là một câu trả lời phủ định có giá trị.

Kèm theo:
```
sysGeometry.dt_crossRingFactors   1 1 1 1 0 0
sysGeometry.avgBlockDeadtime      0.00483
sysGeometry.avgCrystalSingles     63.4166     <- thang của S, xem dưới
sysGeometry.numCoincAsics         7
dt_hsPileUp_factors[47]           1.28 1 0.87 0.80 … (đối xứng)
dt_hrPileUp_factors[47]           toàn 1
```

**`avgCrystalSingles = 63.4166`** gần như chắc là chìa cho chỗ lệch năm bậc độ
lớn ở đơn vị của `S`: dạng `(1 − S·k)` với `k ≈ 4.74` buộc `S` phải là **tốc độ
đã chuẩn hoá**. Chia singles từng crystal cho một đại lượng cỡ này là
ứng viên số một. Cần kiểm: đáp số đúng phải ra livetime **1.0–1.3**, không phải
dính trần 10.

## 2. Scatter

```
apCfg.scatScaleFactorLimit        1.5        scatTailfitAngleWindow   31
apCfg.scatTailfitArrayFlag        0          scatWeightingFactor      1
apCfg.scatPostScale               1          scatRtTableRodMinRange   22
apCfg.exCtacFrameTableLocTolerance 1.975     scatRtTableRodMaxRange   50
apCfg.scatDetEffFilePath          /usr/PET/systemConfig/local/detEff.lyso
apCfg.scatCountRateFuncCoeff[11]  0.4509 0.0709 0.0193 −0.2785 −0.5523
                                  0.1178 0.4944 −0.0057 −0.1703 −0.0031 0.0210
```

### HR và HS là hai bộ tham số scatter riêng — đây là chúng

`sysGeometry` có **hai bộ tham số scatter 10 phần tử**, khác nhau đúng ở
phần tử 0, 4 và 8:

```
scatterHrParameters  −0.0018048  −0.00546  1.204  −0.3919  −6.15e−06
                     −0.0427     −9.62     −0.366  0.0250   0.0621
scatterHsParameters  −0.0023808  −0.00535  1.204  −0.3919  −8.78e−06
                     −0.0427     −9.62     −0.366  0.0321   0.0621
```

## 3. Normalisation

```
apCfg.norm3dGeometryFactors    0     <- 0 = chỉ theta 0 (khớp tài liệu)
apCfg.norm3dGeometryPeriod     9     <- nên khối có 3429 = 381 × 9
apCfg.norm2dCorrClipValue      100
apCfg.norm3dCorrXtalEffClipValue 10
```

## 4. Hình học — dùng chung cho cả hai module

```
radialModulesPerSystem 32   radialBlocksPerModule 2   radialCrystalsPerBlock 9
axialModulesPerSystem  1    axialBlocksPerModule  4   axialCrystalsPerBlock  6
   -> 32 × 2 × 9 = 576 detector/ring ✓      1 × 4 × 6 = 24 ring ✓
effectiveRingDiameter    827.0        detectorRadialSize 405.1
sourceRadius             369.4        detectorAxialSize  156.7
axial/radialBlockGap     1.5          transaxial_crystal_0_offset −5.021
interCrystalPitch        0.0103       interBlockPitch    0.00547
delaysCorrectionFactor   1.0          blockRepeatFactor  9
VQC: X −4.1552  Y 0.4685  Z 1.3  tilt −0.1739  swivel −0.0385  roll 0.2000
```

⚠ **Dòng VQC trên là của file selftest, KHÔNG phải của bay85ct.** Giá trị thật
của máy này, và mọi đối chiếu hình học khác với `petsw/`: xem `GEOMETRY_AUDIT.md`.

`576` và `24` khớp `GE_D710_raw_data_report.md`. `delaysCorrectionFactor = 1`
nghĩa là GE **không** nhân thêm hệ số nào vào randoms.

## 5. Hai thứ ngoài lề nhưng đáng ghi

* **`apCfg.dataType = S_CMPI_TOF9x6_DATA`.** Job XR chạy ở chế độ **ToF**, và
  `apCfg.timingResolutionInPico = 675` — GE gán kiểu dữ liệu ToF cho chính
  model này.
* **WCC**, ghi lại để đối chiếu: `wccActivityScaleFactorForTOF3DIR =
  0.97946`, khớp đúng `HANDOFF.md:359`. Kèm `ForRP 2.99610`,
  `ForFORE 0.0551430`, còn `FBP2D / 3DIR / 2DIR` đều `1`.

## 6. Header RDF `pet_recon` in ra — xác nhận độc lập

Bật `sharcCmpDebugFlag = 0x1FF` thì `sharcCmpOpenDataFiles` in **toàn bộ** header
đã parse. Những dòng đáng giá (từ file selftest):

```
acqParams.RDFEdcatParameters.coincTimingPrecision   0.089246 ns
   -> đúng 89.2459 ps trong  2τ = 55 × 89.2459 ps = 4.909 ns
acqParams.RDFAcqRxScanParams.delayedEvents          0     <- không ghi delayed
acqParams.RDFAcqRxScanParams.singleCollect          1     <- có giữ singles
acqParams.RDFAcqRxScanParams.deadtimeCollect        1
acqParams.RDFAcqRxScanParams.tofCompressionFactor   1
acqParams.RDFEdcatParameters.delayWindowOffset      32
acqParams.RDFEdcatParameters.pos/negCoincidenceWindow  37 / 37
acqParams.RDFEdcatParameters.pos/negAxialAcceptanceAngle  23 / 23
acqParams.RDFBackEnadAcqFilters.maxRingDiff         23
acqParams.RDFEdcatParameters.upper/lower_energy_limit  650 / 425 keV
   -> khớp detEff.lyso.425 đang dùng
acqParams.RDFEdcatParameters.transAxialFOV          70 cm
sorterData.acqDataSegmentParams[4].segmentType      9
sorterData.acqDataSegmentParams[4].dataSegmentOffset    118784
sorterData.acqDataSegmentParams[4].scaleFactorsOffset   125642
   -> đúng khối hệ số hình học tìm được bằng đọc byte file norm
rdfConfig.fileVersion    8.0        singlesVersion 1     deadTimeVersion 0
```

`delayedEvents = 0` + `singleCollect = 1`: máy **không** ghi sinogram delayed,
nên randoms buộc phải suy từ singles.

## 7. Cách lấy lại từng nhóm số

| cần | lệnh gdb |
|---|---|
| một trường | `print apCfg.<tên>` / `print sysGeometry.<tên>` |
| toàn bộ tên trường + offset | `ptype /o apCfg` |
| header RDF | `set var sharcCmpDebugFlag = 0x1FF` trước `sharcCmpOpenDataFiles` |
| bit mask debug | `0x1` chung, `0x4` LOAD, `0x8` RDF, `0x80` AP_CONFIG, **`0x100` AP_RANDOMS**, `0x200` AP_FORE |
