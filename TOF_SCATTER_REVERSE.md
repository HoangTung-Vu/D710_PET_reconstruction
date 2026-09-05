# Scatter TOF: XONG. Một trường job, không phải một hàm phải reverse

Viết lại 2026-08-27. Bản trước của file này liệt kê những gì còn thiếu để đảo
`OsemTof::COsemTofMain::GetScatterViewDataTof`. **Không cần đảo hàm đó nữa** —
và `TOF_PLAN.md` §0.2 đoán sai chỗ: nó viết `CalcScatterEstimateTof` "không phải
thứ cần", trong khi đó chính là thứ cần.

```
IgJobReq.reconMethod = 3
```

Hết. Mọi thứ dưới đây chỉ là chứng minh.

---

## 1. Chuỗi bằng chứng

`CIgManager::Do3dEmissionImage`, tại `0x42a6fc`:

```asm
cmpl  $0x3, IgJobReq+0x59c        ; +0x59c = reconMethod
sete  %cl
call  CScatterFully3dModel::CScatterFully3dModel(this, ig, nThreads, cl)
```

Constructor, tại `0x48e184`:

```asm
movzbl 0x17(%rsp),%eax            ; đúng cái bool vừa truyền
mov    %al, 0x2d8(%rbp)           ; -> m_bTOFDim
```

`ptype /o CScatterFully3dModel` xác nhận offset:

```
/*    728      |       1 */    bool m_bTOFDim;
```

`CreateTaskList` test đúng byte đó năm chỗ (`cmpb $0x0,0x2d8(%r13)` tại
`48cbc2, 48cecc, 48d030, 48d47f, 48d4e4`), và trong nhánh `true` nó xếp thêm
năm task — task type là `movl $0x14..$0x18, 0x70(%rsp)`:

| # | task | hàm |
|---|---|---|
| 20 | `MSCAT_CREATE_IMAGE_PATHS_TOF` | `CreateImagePathsTof` |
| 21 | `MSCAT_CALC_SCAT_ESTIMATE_TOF` | `CalcScatterEstimateTof` |
| 22 | `MSCAT_COMBINE_DS_SINGLE_SCAT_SINO_TOF` | `CombineDSSingleScatSinoTof` |
| 23 | `MSCAT_PHI_UPSAMPLE_TOF_SCAT` | `PhiUpsampleTofScatter` |
| 24 | `MSCAT_CONVERT_PHIUP_SCAT_TO_3D` | `ConvertPhiUpsampledScatterTo3d` |

(Tên enum lấy từ `ptype MODEL_SCAT_TASK`; số thứ tự khớp bảng nhảy `DoTask`
ở `0xb5a2e8`.)

Nói cách khác: **scatter TOF không phải một mô hình thứ hai và cũng không nằm ở
tầng khác.** Nó là đúng cái `CScatterFully3dModel` mà `vendor/extract.gdb` vẫn
đang lái, chỉ khác là được bảo giữ lại trục thời gian.

## 2. Vì sao trước đây không thấy

`extract.gdb` dựng object đó bằng tay:

```
("$scat", 864, "...0x48e070)($scat, $ig, 1, (char) 0)")
```

`(char) 0` — tham số thứ ba, chính là `m_bTOFDim`. Mọi scatter dự án này từng
sinh ra đều non-TOF vì dòng đó, không phải vì tầng prep "by design là non-TOF".

## 3. Đích đến đã được cấp phát sẵn từ trước

`CCorrDataMem::m_pScatterTOF` (offset 56). Log của **mọi** lần chạy cũ đều có:

```
CorrDataMem.cpp(715) ReallocateBuffs(): m_pScatterTOF: 41.6MiB
CViewBuffer:: Reallocating CorrScatterTOF to 288 views with 151360 size
```

Nó được cấp phát vì header RDF khai `emissionSorterData.dataOrientation == 7`
(dữ liệu có TOF). Thiếu `reconMethod = 3` thì nó chỉ nằm đó rỗng.

## 4. Layout — đọc ra, không đoán

Câu lệnh cuối của `PhiUpsampleTofScatter` (`0x484b92`):

```c
permute_41253(buf, 4, ds_nu, number_phi, 4, numTOF_bins, m_pScatterTOF)
```

Hoán vị 5 chiều, quy ước **column-major** (nhanh nhất trước). Ra:
`number_phi` chậm nhất → mỗi view là một khối liền → trong một view, thứ tự C là

```
[numTOF_bins][ds_nu][4][4]  =  55 x 43 x 16  =  37840 float  =  151360 B  ✓
```

`ds_nu = 43` là trục tiếp tuyến đã hạ mẫu (đọc từ `scatterData.ds_nu` lúc chạy);
hai số 4 là lấy mẫu trục đã hạ (`MSCAT_DOWNSAMPLE_EMIS_IMG In: 0 47 Out: 0 4`).

Kiểm bằng số, trên ped bed 1: profile theo trục 43 tương quan **+1.0000** với
profile `u` của `scatter.f32` đầy đủ. Sai thứ tự trục thì con số đó sập.

## 5. Đã đo được gì

Chạy thật (`vendor/probes/tofscat.gdb`, rồi `estimate.py`), ped bed 1:

| | |
|---|---|
| `m_bTOFDim` đọc lại | `1` |
| task TOF được xếp | đủ 5, mỗi loại 1 lần, sau 5 vòng SSS non-TOF |
| `scatter_tof.f32` | 41.6 MB, **100 % khác 0** |
| `randoms.f32` so bản non-TOF | **giống hệt từng bit** |
| `scatter.f32` so bản non-TOF | lệch **0.04 %** tổng |
| đỉnh TOF | bin 27/55, max/mean 3.96 |
| tâm profile theo `(view, u)` | **21.4 → 31.4** bin, tức trải 10 bin = 890 ps = 134 mm |

Dòng cuối là điểm đáng giá nhất: **một profile toàn cục sai 20.6 %** (khoảng
cách L1, có trọng số theo số đếm) so với phân bố thật. Đó chính là phần
`utils.terms.scatter_tof_profile` không thể biết.

Và cái thước cũ trong bản trước của file này vẫn đúng: profile đo từ vành đuôi
có đỉnh bin 28, còn của GE là bin 27 — lệch 1 bin, đúng dấu đã dự đoán (vành
đuôi còn lẫn trues, mà trues hẹp hơn scatter).

## 5b. Còn randoms TOF thì sao? **Không có, và đó là câu trả lời đúng**

Toàn bộ mặt cắt randoms trong `pet_recon`:

```
nm -C --defined-only pet_recon | grep -i random
```

* **Không một symbol nào có hậu tố `Tof`.**
* Getter chạy **bên trong** vòng TOF OSEM là
  `OsemTof::COsemTofMain::GetRandomsViewData(unsigned long view, float* out)` —
  một tham số view, không có chỉ số TOF. Nó phát lại đúng buffer non-TOF cho
  mọi bin.
* `CCorrDataMem` có `m_pRandoms` và `m_pOverlapRandoms`, không có
  `m_pRandomsTOF`. Buffer `...TOF` duy nhất trong cả class là `m_pScatterTOF`.

Nên "chạy vendor code để lấy randoms có TOF" là việc **không tồn tại**: hãng
không mô hình hoá nó, vì trùng phùng ngẫu nhiên không có tương quan thời gian —
55 bin × 89.2459 ps = 4.909 ns đúng bằng cửa sổ trùng phùng.

Hệ số `1/n_tof` thì **dữ liệu tự chứng minh**, không cần đọc code. Ped bed 1,
mash 5: `randoms / 11 = 724.932` mỗi bin, còn prompts ở bin ngoài cùng (t0) là
**726.894**. Nếu dùng nguyên randoms cho mỗi bin thì t0 có background 7,97 triệu
đối lại 727 nghìn prompts — âm gấp 11 lần. Chia đều thì t0 còn dư 1.937 trues.
Đo trực tiếp cũng xác nhận: CoV **0.0404** so với sàn Poisson **0.0393** trên
LOR trượt hẳn ngoài người.

## 6. Đường đi trong code bây giờ

```
d710 estimate            --> IgJobReq.reconMethod = 3   (vendor/extract.gdb)
                             m_bTOFDim = 1
                         --> vendor/bed<n>/scatter_tof.f32   288 x 55 x 43 x 4 x 4
d710 tostir              --> work/bed<n>/scatter_tof.npy     55 x 288 x 43, view đảo về STIR
d710 osem                --> utils.terms.vendor_tof_weights
                             mash 55 -> n_tof, nội suy 43 -> 381, chuẩn hoá theo TOF
                             b[t] = randoms/n_tof + scatter * w[t]
```

`--no-tof` quay lại `reconMethod = 2` ở cả hai bước.

## 7. Còn lại gì

* **Hai trục 4×4 bị cộng bỏ** trong `to_stir.py`. Đo được: mất trung vị 0.45
  bin tâm (40 ps, 6 mm), so với 10 bin biến thiên theo `(view, u)` được giữ.
  Muốn giữ thì phải biết 4 mẫu trục ứng với 553 plane thế nào — chưa dò.
* ~~**Chiều của trục TOF — SAI, trục bị soi gương.**~~ **ĐÃ SỬA 2026-08-29.**
  Xem §8 ngay dưới.
* `GetScatterViewDataTof` vẫn chưa đọc. Bây giờ nó chỉ còn là **cách GE nội suy
  bảng này lúc chạy OSEM** — `terms.vendor_tof_weights` đang làm việc đó bằng
  nội suy tuyến tính. Muốn khớp bit thì mới cần đọc.

---

## 8. Chiều trục TOF: đã sửa 2026-08-29

Trước ngày này bộ giải mã ghi thẳng thứ tự bin của GE ra file STIR. Sai. Mọi ảnh
TOF dựng trước đó **không dùng được** — không phải kém, mà là đặt hoạt độ sang
**nửa sai của LOR**, nên tệ hơn cả không TOF.

### Vì sao sai

STIR đánh số timing position **có dấu**, index file 0 = âm nhất, và
`ProjDataInfo::get_k` biến nó thành **độ dời có dấu dọc LOR** (đo trên
`pedtof5`: timing_pos −2 → −294.31 mm, +2 → +294.31 mm). Tức trục này quyết định
photon phát ra ở *phía nào* của LOR — không phải một hoán vị thẩm mỹ.

Reader GE của chính STIR khai thẳng quan hệ:
`CListRecordGEHDF5::get_tof_bin() { return -deltaTime; }`. RDF sắp bin theo
`deltaTime` tăng dần, nên index 0 của ta ứng với `deltaTime −27`, tức timing_pos
**+27** — trục soi gương.

### Đo, độc lập với suy luận trên

`tools/tof_direction.py`: dựng bed 1 **không TOF** (ảnh tham chiếu không thể bị
chính trục đang nghi ngờ làm thiên vị), chiếu thuận qua mô hình TOF của STIR, so
với dữ liệu theo hai chiều. Trên 10 % LOR nhiều đếm nhất — trues áp đảo scatter,
nên scatter không thể là nguyên nhân:

| ped bed 1 | corr sinogram | corr centroid | mean \|Δcentroid\| |
|---|---|---|---|
| **trước khi sửa**, như đang ghi | +0.734 | **−0.977** | 0.541 bin |
| **trước khi sửa**, lật trục | +0.983 | +0.977 | 0.055 bin |
| **sau khi sửa**, như đang ghi | **+0.983** | **+0.977** | **0.055 bin** |
| **sau khi sửa**, lật trục | +0.734 | — | 0.541 bin |

Hai cột đổi chỗ đúng như dự đoán. Kết luận giữ nguyên trên mọi ngưỡng đếm (top
50/20/10/1 %). Hai đường độc lập — mã STIR và phép đo — cùng một kết luận.

### Sửa ở đâu — hai nơi, **một** thay đổi

Cả hai đều là phép đổi "GE → STIR", cùng tinh thần với `_view_index` đang đảo
trục view. Làm một nửa thì **tệ hơn không làm gì**: prompts và scatter lệch nhau.

1. `custom_tool/gerdf/cli.py:_tof_to_stir` — đảo trục TOF trên cả hai đường ghi
   (`_convert_native` trục 1 của `(radial, tof, plane)`, `_convert_registry`
   trục 1 của `(plane, tof)`). Đảo trước hay sau `mash_tof` là như nhau, vì
   `tof_plan` bắt mash phải chia hết số bin.
   Nằm trong image, nên phải **dựng lại `d710:full`**:
   ```
   docker tag d710:full d710:pre-tofaxis            # đường lui
   docker build -t d710:full -f D710/Dockerfile .   # ~3 s, cache ăn tới COPY gerdf
   ```
2. `vendor/to_stir.py:convert_scatter_tof` — `a.transpose(1,0,2)[::-1, ::-1, :]`,
   đảo cả view lẫn TOF. `peak_tof_bin` trong sidecar cũng báo theo trục đã ghi.

`--view-order ge` giữ thứ tự GE trên **cả hai** trục, và khi đó header không
đóng dấu (xem dưới).

### Cửa kiểm — thứ trước đây không có

Đây mới là phần đáng giá: dữ liệu cũ trên đĩa **không phân biệt được** với dữ
liệu đúng. Cùng kích thước, cùng tổng đếm, `Σp ≥ Σr` vẫn đúng, không bin TOF nào
âm, mọi test trong `test_pipeline_data.py` vẫn xanh. Chỉ ảnh khác.

Nên bây giờ có dấu, ở cả hai phía:

* header prompts có dòng comment `; TOF axis := STIR timing positions ...`
  (`gerdf.interfile.TOF_AXIS_KEY`; Interfile coi `;` là comment, `KeyParser` bỏ
  qua không cảnh báo). `utils.terms.check_tof_axis` **từ chối** prompts TOF
  không có dòng này.
* `to_stir.json` có `scatter_tof.tof_axis = "stir"`.
  `utils.terms.vendor_tof_weights` từ chối file thiếu nó.

Dữ liệu giải mã trước 2026-08-29 phải chạy lại — rẻ, vài giây một bed:

```
d710 decode --raw <SINO dir> --case <n> --tof --force
d710 tostir --case <n>
```

Và `tools/tof_direction.py` từ nay là **regression check**, không còn là phép
dò: kỳ vọng "AS WRITTEN wins".

```
PYTHONPATH=<D710> python3 tools/tof_direction.py \
    --nontof-case ped --tof-case pedtof5 --bed 1
```

### Chỗ khác cũng đi theo

`tools/tof_profile.py` đọc view thô (thứ tự GE) nên mọi số nó **in ra** vẫn theo
GE; nhưng `--save` ghi profile **đã đảo**, vì người tiêu thụ duy nhất là
`d710 osem --tof-scatter`, chạy trên prompts đã đảo. Peak bin in ra và peak bin
`utils.terms` báo là ảnh gương của nhau, và cả hai đều đúng.

Đường list-mode (`gerdf/listmode.py`, `.prd`) **chưa** đụng tới: nó giữ nguyên
quy ước GE, đúng như PETSIRD muốn. Khi làm `listmode-histogram`
(`LISTMODE_TOF_PLAN.md` §3, bước 4) thì phải đảo ở đó, cùng chỗ mash.
