# List-mode TOF recon — kế hoạch

Viết 2026-08-27, ngay sau khi scatter TOF của hãng chạy được
(`TOF_SCATTER_REVERSE.md`). Mọi khẳng định dưới đây đều có nguồn: hoặc mã nguồn
STIR trong `$CONDA_PREFIX/dlevel/build/sources/STIR`, hoặc `custom_tool/gerdf`,
hoặc đo trực tiếp trên máy này hôm nay.

> **CẬP NHẬT 2026-08-30 — đọc §0 và §5 trước.** Đã đo được *vì sao* một bed TOF
> mất 2,2 h, và xuất hiện **đường C** rẻ hơn hẳn cả A lẫn B: đưa thẳng event vào
> PyTomography, **~25 s/bed**, không cần build lại STIR, không cần RDF9 writer.
> Ba khẳng định trong bản 08-27 đã sai và được sửa tại chỗ, mỗi chỗ có ghi chú:
> "bẫy lớn nhất của đường B" (§4), "không có reader PETSIRD" (§4, §7), và thứ tự
> làm việc (§6). Hai script kiểm chứng: `tools/projector_bench.py`,
> `tools/pytomo_lm_probe.py`.

---

## 0. Kết luận trước

**Phát hiện quan trọng nhất: record 6 byte của D710 khớp từng trường với record
coincidence RDF9 mà STIR đã biết đọc sẵn.** GE giữ nguyên cách mã hoá sự kiện,
chỉ đổi vỏ chứa (blob nén GLEPL → HDF5). Nên đường LM-OSEM thật **không cần
viết codec nào** — chỉ cần đóng gói lại.

Thứ tự đề nghị:

| | việc | được gì | rủi ro |
|---|---|---|---|
| **1** | `listmode → sinogram TOF` (histogram) | dùng lại **nguyên** đường ống hôm nay; mở khoá frame động + gating | thấp, mapping đã chứng minh |
| **2** | gói lại thành RDF9-HDF5 → LM-OSEM thật của STIR | không mất thông tin do binning; `set_time_interval` | trung bình, xem §4 |

Làm **1 trước**. Nó rẻ, và nó là thứ duy nhất cần thiết nếu mục đích là *frame
động / gating* chứ không phải *nghiên cứu LM-OSEM*.

### Bổ sung 2026-08-30: đường C, và vì sao B tụt xuống cuối

| | việc | được gì | rủi ro |
|---|---|---|---|
| **C** | event → **PyTomography** LM-OSEM (bỏ qua STIR) | **~25 s/bed** thay vì ~2,2 h; TOF sẵn; không build lại gì | thấp về API (đã chạy), việc thật nằm ở `weights`/`additive_term` |

**Cái đắt không phải list-mode hay histogram, mà là sinogram TOF có 303 M bin để
chứa 14,8 M event.** Đếm trên `ped2tof` bed 1:

| | số bin | số event | bin ≠ 0 | count/bin≠0 |
|---|---|---|---|---|
| non-TOF | 60.679.584 | 14.809.731 | 20,3 % | 1,20 |
| TOF 5 bin | **303.397.920** | **14.809.731** | **4,5 %** | 1,08 |

Ba hệ quả cho thứ tự làm việc (§6):

* **A vẫn đứng đầu, nhưng vì lý do khác.** Nó *không* giải quyết 2,2 h —
  histogram xong vẫn rơi vào đúng đường sinogram chậm. Nó đứng đầu vì bảng tra
  `crystal_to_det`/`det_pair_map` mà nó cần **cũng chính là** thứ C cần để dựng
  `weights`/`additive_term`. Làm một lần, dùng cho cả hai.
* **C là đích**, và sau A thì nó gần như free.
* **B tụt xuống cuối.** Sau khi có C, B chỉ còn hơn ở đúng `set_time_interval`.

---

## 1. Điểm xuất phát — cái gì đã có sẵn

`custom_tool/gerdf/listmode.py` giải mã xong `LIST*.BLF`:

* GLEPL giải nén bằng `unglepl` của chính console (`gerdf/vendor.py`), ~200 MB/s;
* stream 6 byte/record đã đảo hết, kiểm chéo với `rdfListDecode` của hãng trên
  **toàn bộ 26.204.945 record** của một bed thật;
* mỗi sự kiện có: **bin TOF −27..+27** (55 giá trị), ring 0..23 và transaxial
  0..575 của cả hai tinh thể, cộng `t_ms` (nội suy từ time marker mỗi ms).

Đầu ra hiện tại: `.npy` (bảng sự kiện) hoặc `.petsird`.

**Chưa có: bộ histogram từ event → sinogram.** `estimate.py` nói "phải histogram
thành SINO* trước" nhưng công cụ đó chưa tồn tại — `d710 decode --listmode` chỉ
ghi `.prd` (PETSIRD). Đây là lỗ hổng thật, và là việc số 1.

Hai thứ đã có và dùng lại được nguyên vẹn:

* **Mapping bin** — `utils/geometry.py`: `crystal_to_det` (GE→STIR, đảo chiều +
  offset 288) và `det_pair_map`, hiệu chuẩn bằng cách histogram list-mode NEMA
  bed 2 rồi chấm với sinogram hãng giải mã: **corr 1.000000**.
* **Số hạng hiệu chỉnh TOF** — randoms + scatter TOF của hãng, đã xong hôm nay.
  Cả hai đường dưới đây đều ăn cùng bộ này.

---

## 2. Phát hiện: record của ta CHÍNH LÀ record RDF9

STIR đọc list-mode GE qua `CListRecordGEHDF5.h`:

```c
uint16 eventLength      : 2
uint16 eventType        : 1
uint16 hiXtalShortInteg : 1
uint16 loXtalShortInteg : 1
uint16 hiXtalScatterRec : 1
uint16 loXtalScatterRec : 1
int16  deltaTime        : 9     /* TOF, CÓ DẤU */
uint16 hiXtalAxialID    : 6
uint16 hiXtalTransAxID  : 10
uint16 loXtalAxialID    : 6
uint16 loXtalTransAxID  : 10
```

Đặt cạnh cái `gerdf/listmode.py` đã đảo ra từ D710:

```
w0 >> 6  (signed)   bin TOF, −27..+27
w1 & 0x3f           ring   tinh thể "high"     w1 >> 6   transaxial "high"
w2 & 0x3f           ring   tinh thể "low"      w2 >> 6   transaxial "low"
```

Ba từ 16-bit, mỗi từ chia 10/6 cho (transaxial, axial), TOF có dấu lấy từ 10 bit
cao của từ đầu — **trùng từng trường**.

⚠ Phải kiểm bit-exact bằng round-trip trước khi tin, đừng tin bảng so sánh này:
đóng gói N sự kiện, cho STIR đọc lại, so `get_tof_bin()` và cặp tinh thể với
bảng `.npy` gốc. Bitfield C++ phụ thuộc thứ tự nhồi của trình biên dịch.

### PHẠM VI: ba phát hiện, ba tầm ảnh hưởng KHÁC NHAU

Đừng đọc cả mục này như chuyện riêng của list-mode.

| phát hiện | ảnh hưởng tới |
|---|---|
| cách nhồi bitfield ở trên | **chỉ đường B**. Đường A histogram từ bảng `.npy` của bộ giải mã ta, không nhờ STIR đọc bit |
| `is_prompt() { return true; }` (kèm `// TODO`) — STIR coi mọi sự kiện là prompt, vừa khít với stream chỉ-prompt của D710 | **chỉ đường B** |
| **`get_tof_bin() { return -deltaTime; }`** | **MỌI thứ có trục TOF** — đường A, đường B, *và cả đường sinogram đang chạy hôm nay* |

### Vì sao dấu TOF lan ra ngoài list-mode

STIR đánh số timing position **có dấu, đối xứng quanh 0**, và index trong file
chạy **tăng dần từ âm nhất**. Đo trên chính `pedtof5/decoded/bed1.hs`:

```
5 bin -> timing_pos −2..+2,  index file 0 = timing_pos −2
  timing_pos −2  ->  k = −294.31 mm   (ProjDataInfo::get_k)
  timing_pos +2  ->  k = +294.31 mm
```

`k` là **độ dời CÓ DẤU dọc LOR**: nó quyết định photon phát ra ở *phía nào*.
Mà reader GE của STIR khai `timing_pos = −deltaTime`. Nên nếu RDF sắp bin theo
`deltaTime` tăng dần, index 0 của file ta ứng với `deltaTime −27`, tức timing_pos
**+27** — **trục bị soi gương**.

Nếu sai: prompts và scatter vẫn nhất quán với nhau (cùng một trục), tổng đếm vẫn
đúng, mọi invariant trong `test_pipeline_data.py` vẫn xanh — nhưng OSEM đặt hoạt
độ sang **nửa sai của LOR** và ảnh TOF sẽ tệ hơn cả non-TOF. Không có cửa kiểm
nào hiện tại bắt được lỗi này.

### ĐÃ KIỂM 2026-08-27 (trục bị soi gương) — ĐÃ SỬA 2026-08-29

`tools/tof_direction.py`: dựng bed 1 **không TOF** (ảnh tham chiếu không thể bị
chính trục đang nghi ngờ làm thiên vị), chiếu thuận qua mô hình TOF của STIR, so
với dữ liệu theo hai chiều. Trên 10 % LOR nhiều đếm nhất — nơi trues áp đảo
scatter, nên scatter không thể là nguyên nhân:

| | corr sinogram | corr centroid | mean \|Δcentroid\| |
|---|---|---|---|
| như đang ghi | +0.734 | **−0.977** | 0.541 bin |
| **lật trục** | **+0.983** | +0.977 | **0.055 bin** |

Và nó khớp đúng cái `timing_pos = −deltaTime` suy ra từ mã STIR ở trên — hai
đường độc lập, cùng kết luận.

Trục đã được đảo ở chỗ ghi file (`gerdf.cli._tof_to_stir` cho prompts,
`vendor/to_stir.py` cho trọng số scatter) và chạy lại `tof_direction.py` cho
đúng hai cột đổi chỗ: "AS WRITTEN" thắng, 0.055 bin. Chi tiết + cửa kiểm mới ở
`TOF_SCATTER_REVERSE.md` §8.

**Hệ quả cho đường list-mode:** đường A phải đảo bin TOF **ở bước histogram**
(bước 4 của §3, cùng chỗ `mash_tof`), vì `listmode.py` giữ nguyên quy ước GE —
đúng như PETSIRD muốn, và không nên đổi ở đó. Đường B thì ngược lại: STIR tự đọc
`deltaTime` qua `get_tof_bin()` của chính nó, nên payload phải giữ **nguyên** bit
của GE, không đảo gì cả.

---

## 3. Đường A — histogram thành sinogram TOF (làm trước)

### Việc

Thêm `gerdf listmode-histogram` (hoặc mở rộng `listmode-decode --format hs`):

```
LIST*.BLF  →  bảng sự kiện  →  (view, plane, tang, tof)  →  cùng Interfile 5-D
                                                            mà đường SINO sinh ra
```

Cụ thể, mỗi sự kiện:

1. `crystal_to_det` cho cả hai tinh thể → cặp detector STIR;
2. `det_pair_map` ngược → `(view, tangential)`;
3. cặp ring → plane, theo đúng bảng span-2 trong `geometry.plane_ring_pairs`;
4. bin TOF `−27..+27` → chỉ số `0..54`, rồi `--tof-mash` **y hệt** đường SINO
   (`gerdf.cli.mash_tof`);
5. `np.add.at` / `bincount` trên chỉ số phẳng.

### Cửa kiểm — bắt buộc, và đã có sẵn mẫu

* tổng đếm giải mã **phải bằng** `prompts` trong header (đúng cái check `MATCH`
  mà `convert` đang làm);
* histogram của **cả frame** phải khớp `SINO*` của cùng bed — mỗi bed list-mode
  trên đĩa đều có một SINO* cùng acquisition, nên đây là đối chứng thật, không
  phải tự kiểm. Đặt ngưỡng: corr > 0.999 và tổng lệch 0.

### Sau đó không phải sửa gì nữa

`estimate` / `tostir` / `osem` chạy nguyên. Cái mở khoá được:

* **frame động**: cắt theo `t_ms` rồi histogram từng frame;
* **gating** hô hấp / tim, nếu có tín hiệu ngoài.

### Bẫy: hiệu chỉnh cho SUB-FRAME

`randoms` và `scatter` lấy từ estimate của **cả** frame. Cắt ra một sub-frame
thì:

* **scatter** ~ tuyến tính theo số đếm → nhân theo tỉ lệ prompts là chấp nhận
  được;
* **randoms KHÔNG tuyến tính** — randoms ∝ (tốc độ singles)², nên chia theo thời
  gian là **sai**. Phải dựng lại từ singles của chính sub-frame đó, hoặc ít nhất
  scale theo `(S_sub/S_full)²`. Xem `d710-randoms-from-singles`: dạng công thức
  đã đúng, phân bố theo ring còn lệch 20–40 %, nên đây là chỗ cần cẩn thận nhất
  của toàn bộ đường động.

---

## 4. Đường B — LM-OSEM thật

### STIR/SIRF trên máy này có gì

```
sirf.STIR.ListmodeData(filename)
sirf.STIR.PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin
```

Reader list-mode STIR build này có: `ECAT`, `ECAT8_32bit`, `GEHDF5`, `PENN`,
`ROOT`, `SAFIR`. **Không có PETSIRD.**

Có TOF: `GEHDF5`, `PENN`, `ROOT`. **`SAFIR` không có TOF** — nên đường "format
tự chế đơn giản nhất" tự loại, vì nó vứt đúng cái trục ta cần.

→ còn đúng một cửa: **GEHDF5 (RDF9)**.

> **Sửa 2026-08-30 — "không có PETSIRD" đúng với *build này*, nhưng reader thì
> CÓ tồn tại.** Env dựng từ tag `rel_6.4.0`; PETSIRD nằm ở nhánh
> `origin/PETSIRD` **đã fetch sẵn trong clone local**, và nó **29 commit trên
> `rel_6.4.0`, 0 commit sau** — tức rebase thẳng lên đúng cái release này. Nhánh
> đó có đủ bộ: `CListModeDataPETSIRD`, `CListRecordPETSIRD`,
> `PETSIRDCListmodeInputFileFormat`, `PETSIRDInfo`, và cả
> `BinNormalisationFromPETSIRD`; đăng ký ở `IO_registries.cxx:165`.
>
> Trục TOF hai đầu đã khớp sẵn: `CListRecordPETSIRD.cxx:55` làm
> `tof_idx + get_min_tof_pos_num()`, đúng phép nghịch đảo của
> `offset = num_tof_bins // 2` trong `gerdf/petsird_out.py`.
>
> SIRF **không phải sửa dòng nào** — `STIRListmodeData(filename)`
> (`cstir.cpp:407`) đi qua `InputFileFormatRegistry` chung.
>
> Giá phải trả: `find_package(PETSIRD CONFIG)` cần **thư viện PETSIRD C++**
> (hiện chỉ có bản Python; `yardl` chưa cài), rồi build lại STIR **và** SIRF.
> Đó là lý do đường C (§5) được ưu tiên: nó không cần bất kỳ bước nào ở đây.

### Việc: gói lại, không mã hoá lại

`GEHDF5Wrapper.cxx` đọc những dataset này, và **mọi trường đều đã có trong header
RDF legacy** — `PARAMS.md` ghi đúng từng cái:

```
/HeaderData/RDFConfiguration/fileVersion/majorVersion   <- phải là 9
/HeaderData/RDFConfiguration/isListFile
/HeaderData/ListHeader/isListCompressed                 <- phải là 0
/HeaderData/ExamData/scannerDesc                        <- "Discovery 710"
/HeaderData/SystemGeometry/{radial,axial}*              <- 32/2/9, 1/4/6
/HeaderData/AcqParameters/EDCATParameters/coincTimingPrecision
                                        /pos,negCoincidenceWindow
                                        /upper,lower_energy_limit
/HeaderData/Sorter/numTOF_bins
/HeaderData/AcqStats/{scanStartTime,frameStartTime,frameDuration}
/HeaderData/ExamData/{halfLife,positronFraction,radionuclideName}
```

`scannerDesc` đi qua `Scanner::get_scanner_from_name()`, và STIR đã biết
"Discovery 710" (header Interfile của ta đang dùng đúng tên đó).

Payload: ghi thẳng stream 6 byte đã giải nén — **không đụng tới từng sự kiện**,
nếu §2 kiểm ra đúng.

### Ràng buộc cứng, đọc từ mã STIR

* `error("Only RDF version 9 supported. Aborting")` → phải khai major = 9;
* `error("The RDF9 Listmode file is compressed ... Aborting")` → phải ghi bản đã
  giải nén (dung lượng bằng `.BLF` bung ra, ~150–300 MB/bed);
* STIR sẽ **cảnh báo và tự chỉnh** nếu `effectiveRingDiameter` /
  `timingResolutionInPico` lệch với định nghĩa scanner của nó — không phải lỗi,
  nhưng phải đọc log để biết nó đã chỉnh gì.

### ~~Bẫy lớn nhất của đường B: ma trận chiếu~~ — SAI, sửa 2026-08-30

`PoissonLogLikelihood...ListModeDataWithProjMatrixByBin` — tên nói hết:
**bắt buộc `ProjMatrixByBin`, không dùng được `parallelproj`.** Mà đó đúng là
cái vừa làm nổ RAM hôm nay: sinogram TOF 5 bin, một bed, cache ma trận
ray-tracing leo tới **30 GB** rồi phải kill (`osem/recon.py` giờ mặc định
`parallelproj` khi có TOF vì lý do này).

> **Mất `parallelproj` KHÔNG tốn gì cả.** Đo head-to-head trên ped2 bed 1
> (60,7 M bin, 16 thread, đã warm — `tools/projector_bench.py`):
>
> | projector | fwd TOÀN BỘ | fwd 1/24 | back TOÀN BỘ | back 1/24 |
> |---|---|---|---|---|
> | parallelproj | 10,7 s | **10,7 s** | 5,7 s | 0,3 s |
> | ray lors=5 cache=off | 10,1 s | **0,7 s** | 7,3 s | 0,3 s |
> | ray lors=5 cache=on | 8,1 s | 0,5 s | 4,2 s | 0,2 s |
>
> Chiếu toàn bộ thì hai bên **ngang nhau**; backprojection parallelproj còn
> nhanh hơn. Toàn bộ khoảng cách 16x trong một subiteration OSEM (0,7 vs 11,0 s)
> chỉ đến từ **xử lý subset ở forward**:
> `ForwardProjectorByBinParallelproj::set_input()` chiếu *mọi* LOR, còn
> `actual_forward_project()` chỉ memcpy viewgram ra — nó thậm chí `error()` nếu
> bị hỏi một dải không đầy đủ ("current only handles projecting all data"). Mà
> `distributable.cxx:409` gọi `set_input` **một lần mỗi subset**.
>
> **Đây mới là lời giải cho 2,2 h/bed: chi phí ∝ n_subsets × n_iters, subset
> không mua được gì.** 48 subiteration × ~166 s (TOF) ≈ 2,2 h. Hệ quả kèm theo:
> `--lors` vô hiệu trên đường parallelproj, và các docstring trong
> `osem/recon.py` nói cache đáng "5x" thì thực đo chỉ 1,25x (`--lors` 5→1 là
> 1,9x, không phải 5x).
>
> Nên bẫy thật của đường B chỉ còn là **RAM của cache**, không phải tốc độ — và
> mục dưới đã có sẵn núm chỉnh cho nó.

Tin tốt: đường list-mode có núm chỉnh mà đường sinogram không có —

```
set_cache_max_size(...)    set_cache_path(...)    set_recompute_cache(...)
set_subsensitivity_filenames(...)                 set_recompute_sensitivity(...)
```

Cache có **trần** và **đổ ra đĩa** được, còn sensitivity tính một lần rồi cache
ra file. Nên B khả thi về RAM, nhưng phải đặt mấy cái này ngay từ lần chạy đầu,
đừng để mặc định.

### Cái đáng giá của B

`set_time_interval(...)` — chọn cửa sổ thời gian mà **không** phải histogram
lại. Đó mới là lý do thật để làm LM-OSEM thay vì đường A.

---

## 5. Đường C — PyTomography trực tiếp (MỚI 2026-08-30, làm trước)

Bỏ qua STIR hoàn toàn cho bước recon. Không PETSIRD, không RDF9 writer, không
build lại gì. Kiểm chứng: `tools/pytomo_lm_probe.py`.

### Vì sao vào được — ba sự thật, mỗi cái đọc từ mã đã cài

1. **`PETLMProjMeta` nhận thẳng `scanner_LUT`**, và `info` **không được đọc ở
   bất kỳ đâu** trong `PETLMSystemMatrix` (mọi kết quả grep đều là chữ
   "information" trong docstring). Nên D710 **không** phải nhét vừa mô hình phân
   cấp kiểu GATE (crystal/submodule/module/rsector) của `pet_scanner_info.txt`.
2. **Quy ước detector id đã trùng sẵn.** `EVENT_DTYPE.xtal_a` của ta là
   `ring * detectors_per_ring + trans`; `clinical.get_detector_ids_hdf5` dựng
   `NrCrystalsPerRing*ring + crystal`. Cùng một thứ. Không phải map lại.
3. **`norm_BP` (sensitivity trên toàn bộ LOR hợp lệ) dựng ngay trong
   `PETLMSystemMatrix.__init__`** (dòng 54, `self.norm_BP = self._backward_full()`)
   — nên thời gian constructor CHÍNH LÀ chi phí sensitivity, đừng tìm chỗ khác.

### Cần đưa vào những gì

| | giá trị | nguồn |
|---|---|---|
| `detector_ids` | `(N,3)` int32 `[xtal_a, xtal_b, tof_idx]` | bảng `.npy` của `listmode.py`, **cộng `num_tof_bins // 2`** để đổi bin có dấu → 0-based (đúng phép dịch `petsird_out.py` đang làm) |
| `scanner_LUT` | `(13824, 3)` | 5 dòng từ `Geometry`: 24 ring × 576, R = 405,10 mm, pitch = 156,70/24 = 6,529 mm |
| `tof_meta` | `PETTOFMeta(55, 733.7, 82.4)` | 55 bin × 89 ps → `tof_range = n·c·lsb/2`; fwhm từ timing resolution 550 ps |

Kiểm chéo cho `tof_range`: dòng `discovery_MI` trong `pet_scanner_info.txt` ghi
735,77 mm, ta ra 733,7 mm — cùng tầm vật lý, đúng như mong đợi.

⚠ **Trục TOF.** `listmode.py` giữ nguyên quy ước GE, và §2 đã chứng minh trục GE
bị soi gương so với STIR. Đường C chưa được kiểm chiều — phải chạy lại phép kiểm
kiểu `tools/tof_direction.py` trước khi tin bất kỳ ảnh nào, vì đây đúng là loại
lỗi không invariant nào bắt được.

### Đo được (2026-08-30, CPU này, 16 thread, 14.809.731 event, PSF 6,4 mm)

```
PETLMProjMeta built in 0.0s   info=None accepted
sensitivity over 4,000,000 LORs: 0.9s -> 14s cho tập thật 60.679.584
OSEM 2 iters x 24 subsets on 14,809,731 events: 11.8s
TOTAL for one bed: ~25s
```

Sensitivity tuyến tính sạch: 0,22 s/triệu cặp (đo tại 1 M / 4 M / 12 M). Tổng
**~25 s/bed** so với ~2,2 h của đường sinogram — cùng CPU, cùng 48 subiteration,
cùng PSF.

### Việc thật còn lại: `weights` và `additive_term`

Đây là phần chưa làm, và là toàn bộ độ khó còn lại của đường C. Cần tra **theo
từng event**:

* `weights` = norm × dead-time × suy giảm → từ `work/bed<n>/normdt.hs` × af;
* `additive_term` = randoms + scatter, **chia cho `weights`** (đúng như notebook
  `PyTomo/t_GE_HDF5.ipynb` làm: `additive_term=additive_term/weights`);
* `weights_sensitivity` + `detector_ids_sensitivity` = mọi cặp hợp lệ và trọng số
  của chúng.

`pytomography.io.PET.shared.sinogram_to_listmode()` làm đúng việc này nhưng đòi
`info` kiểu GATE, nên phải tự viết theo quy ước bin của D710 — thứ đã hiệu chuẩn
bit-exact rồi (`utils/geometry.py`, `crystal_to_det` + `det_pair_map`, corr
1.000000). Nói cách khác: **cùng một bảng tra mà đường A cần ở §3 bước 1-3**, chỉ
khác là dùng để *đọc ra* thay vì *cộng vào*. Làm A trước thì C gần như free.

### Cái đường C KHÔNG cho

* **Không** có `set_time_interval` — muốn frame động vẫn phải tự cắt theo `t_ms`,
  nhưng cắt xong thì đưa thẳng vào C chứ không phải histogram (rẻ hơn A).
* **Không** dùng lại `estimate`/`tostir`/`osem` — ra khỏi hệ SIRF thì phần export
  DICOM/SUV phải nối lại từ mảng numpy.
* **Chưa kiểm đúng sai bao giờ.** Số ở trên là API + tốc độ, không phải kết quả.

### Ngõ cụt đã loại: PETSIRD → PyTomography

`pytomography/io/PET/petsird.py` + thư mục `prd/` đi kèm là ảnh chụp
`ETSInitiative/PRDdefinition` **đời cũ**, không đọc được file decoder ta ghi:

| | `prd` của pytomography | `petsird` đã cài (decoder ghi) |
|---|---|---|
| protocol | `PrdExperiment` | `PETSIRD` |
| event | `detector_1_id`, `detector_2_id`, `tof_idx`, `energy_*_idx` | `detection_bins: list[uint32]`, `tof_idx` |
| tof edges | `scanner.tof_bin_edges` phẳng | `tof_bin_edges=[[...]]` lồng theo cặp module |

yardl nhúng protocol + schema vào header file và validate lúc đọc, nên
`BinaryPrdExperimentReader` từ chối thẳng. Docstring của chính pytomography thừa
nhận đây là bản tạm chờ package chính thức. **Đừng mất thời gian ở đây** — đường
C không đi qua file, nó nhận tensor.

---

## 6. Thứ tự

0. ~~**Chốt CHIỀU trục TOF**~~ — **XONG 2026-08-29**: đảo ở `gerdf/cli.py` +
   `to_stir.py`, dựng lại `d710:full`, giải mã lại `pedtof`/`pedtof5`, và
   `tools/tof_direction.py` xác nhận hai cột đã đổi chỗ. Header prompts giờ đóng
   dấu `; TOF axis` và `utils.terms` từ chối dữ liệu giải mã trước ngày đó.
1. **`listmode-histogram` + cửa kiểm vs SINO*** (đường A, §3). Lên đầu vì bảng
   tra `crystal_to_det` / `det_pair_map` mà nó cần **cũng chính là** thứ đường C
   cần để dựng `weights`/`additive_term`. Làm một lần, dùng cho cả hai.
2. **Đường C chạy được đầu-cuối trên một bed thật** (§5): thay event tổng hợp
   bằng `.npy` thật, nối `weights` + `additive_term`, rồi so ảnh với
   `work/bed<n>/osem.npz` của đường sinogram. **Đây là phép kiểm quan trọng
   nhất** — hai đường độc lập hoàn toàn, cùng dữ liệu, phải ra cùng một ảnh.
3. **Chiều trục TOF cho đường C** — chạy lại phép kiểm kiểu
   `tools/tof_direction.py`. Đừng bỏ qua vì §2 đã sửa cho đường sinogram: đường C
   đọc thẳng bảng `.npy` giữ quy ước GE, nên nó là một trục **khác**.
4. **Frame động qua A hoặc C**, kèm randoms dựng lại từ singles của sub-frame —
   đây là phần vật lý khó, không phải phần code.
5. *(chỉ khi thật cần `set_time_interval`)* Round-trip bit-exact §2, rồi RDF9
   writer + LM-OSEM (đường B), với cache có trần ngay từ đầu. Sau khi có C thì
   động lực cho B tụt hẳn — B chỉ còn hơn C ở đúng `set_time_interval`.

---

## 7. Không nằm trong plan này

* **PETSIRD → STIR**: ~~build này không có reader~~ — reader CÓ, ở nhánh
  `origin/PETSIRD` (29 commit trên `rel_6.4.0`, đã fetch sẵn), chi tiết ở §4.
  Vẫn để ngoài plan vì phải build thư viện PETSIRD C++ rồi build lại cả STIR lẫn
  SIRF, trong khi đường C không cần bước nào trong số đó.
* **PSF / 2×24 subset / post-filter**: xem `TOF_PLAN.md` §5.
* **Randoms từ singles cho cả frame**: đã có, còn lệch ring 20–40 %; sub-frame
  chỉ làm nó lộ rõ hơn chứ không phải nguyên nhân mới.
