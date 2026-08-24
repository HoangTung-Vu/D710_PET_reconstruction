# vendor — chạy chính `pet_recon` của GE trên máy này

Mục tiêu: lấy **randoms, scatter, normalisation, dead time** từ binary của hãng
thay vì dựng lại bằng Python.

Cả bốn thứ đó đều lấy được: §3 là bảng trạng thái, §3a chuỗi lệnh đúng, §3b
**norm × dead time**, §3c **randoms + scatter**, §3d **đưa mu-map tự tính vào
cho SSS**.

---

## 0. Đóng gói — cần đúng những gì

**Docker image `d710:full` chứa cây vendor của GE. Ngoài ra `run.sh` chỉ mount
đúng hai thứ: thư mục này (`/vendor`, read-only) và `out/` (`/out`, ghi được).**

```
extract.gdb   quy trình chính
 ├─ boot.gdb  đưa pet_recon tới idle; publish 3 global AP; clear DF; lock
 ├─ lib.gdb   helper python: say/ex/show/setv/deep/dump_views/unlocked/clear_df
 └─ job.gdb   nạp .job vào IgJobReq (524 trường) — sinh bởi job2gdb.py
```

Bốn file đó **phải nằm cạnh nhau** trong `/vendor`. Còn lại chạy trên **host**,
không vào container:

```
estimate.py    >>> ĐIỂM VÀO <<< raw + CT -> cả bốn sinogram, một lệnh
run.sh         chạy một script gdb trong container (estimate.py gọi cái này)
Dockerfile     dựng d710:full — pet_recon + custom_tool, tự chứa (một lần)
job2gdb.py     .job  -> job.gdb
read_out.py    đọc out/*.f32 + sidecar .json
make_pifa.py   dựng / giải mã PIFA (hộp đựng mu-map)
ct_to_pifa.py  CT DICOM -> mu(mm^-1) -> PIFA, cho ca thật
```

`probes/` là giàn giáo — **không cần để chạy**, giữ làm bằng chứng cho các kết
luận trong tài liệu này (xem `probes/README.md`). `out/` là kết quả.

### Chạy — MỘT lệnh

```bash
# một lần: image được bàn giao nguyên con, KHÔNG dựng lại
docker load -i d710_full.tar
export D710_OUT=~/UET/d710_out

# rồi mỗi ca — mọi bed một lượt:
d710 estimate --raw <thư mục petRDFS của exam> \
              --ct  <thư mục series CT DICOM> \
              --case <tên ca>
```

Một bed riêng lẻ: thêm `--bed 4`. Ra ở `$D710_OUT/<ca>/vendor/bed<n>/`.

Không có `--norm`: exam tự khai file norm của nó và tool tự tra (§0b). Cờ đó chỉ
để ép một file khác khi cần.

`estimate.py` tự làm hết: đọc header RDF lấy **table position** của bed → dựng
mu-map từ CT trên lưới PIFA → sinh job → chạy `pet_recon` trong container với
dữ liệu mount read-only, `--out` mount thẳng lên `/out` nên kernel ghi kết quả
tại chỗ.

**Nó chạy bằng `python3` trần.** Không conda, không numpy, không pydicom: mọi
bước cần thư viện đều chạy trong image (`ct_to_pifa.py`, đọc DICOM hiệu chuẩn),
và mọi file của hãng nó đọc cũng đã ở trong image. Nó ở lại host chỉ vì nó là
thứ *gọi* `docker`. `tests/test_estimate.py` chốt điều đó bằng cách quét import
của chính file này.

Ví dụ đã kiểm trên NEMA bed 2 (dữ liệu thật):

```
prompts 13.954.115  ==  header RDF     -> đúng dữ liệu của mình
randoms/prompts            12,3 %
scatter fraction S/(T+S)   33,0 %
gồ ghề bán kính: scatter 0,071  vs  prompts 1,747
```

Nếu tra không ra và cũng không có `--norm`, tool lùi về norm selftest của GE và
**cảnh báo to**: randoms/scatter không ảnh hưởng, nhưng `normdt.f32` /
`norm_only.f32` khi đó là độ nhạy của máy test bên GE, không phải máy mình.

> **List-mode không nhận trực tiếp.** `LIST*.BLF` phải histogram thành `SINO*`
> trước (`d710 decode --listmode`). Mọi `LIST*.BLF` trên đĩa đều có `SINO*`
> cùng acquisition — đưa file đó vào.

Muốn chạy tay từng bước thì vẫn được:
`./run.sh --out <thư mục> extract.gdb` dùng `job.gdb` mặc định. `--out` bắt
buộc và không có mặc định — trước kia nó là `vendor/out/` cố định, tức là mọi
lần chạy đi qua **một** thư mục chung: hai bed chạy song song đè kết quả của
nhau, và 7,5 GB kết quả mọc trong cây mã.

---

## 0b. `--norm`: máy mình CÓ file norm không, và sao biết?

**Có, đúng một file**, và không phải trùng hợp: bản export kèm theo đúng scan
hiệu chuẩn mà exam tham chiếu, vì exam khai nó trong chính header của mình.

Header RDF offset **0xEEC = `norm_cal_uid`** (`ge_rdf_tool.py info` vẫn in nhãn
sai là `study_instance_uid`; `0xF74 = wcc_cal_uid` bị gọi nhầm là
`series_instance_uid`). Chuỗi đầy đủ, kiểm được từng bước — và `estimate.py`
đi đúng chuỗi này tự động, nên **không cần `--norm`**:

```
NEMA SINO0001  header 0xEEC = 1.2.840.113619.2.290.663120.1697775916.195435
   -> cal/1.2.840...195435.3dnorm         (508 B DICOM, thuần bookkeeping —
        (0017,1005) PET 3D Normalization   KHÔNG có hệ số nào trong đó)
        (0017,1007) /petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001
        (0017,1004) 20231020112516.00
   -> ~/Documents/12082026/petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001   ✓ có
```

Trùng **đường dẫn tuyệt đối** nên không có chỗ cho nhầm lẫn. Bản sao nằm sẵn
trong repo (`cal/`), nên tool vẫn chạy khi bản export không kèm hiệu chuẩn — nội
dung file norm và bảng hai máy trên console: xem **`cal/README.md`**.

### Sổ hiệu chuẩn có 7 bản, và WCC không phải 3D norm

| ngày | loại | scan nguồn | có trên đĩa? |
|---|---|---|---|
| 2022-12-29 | **3DWCC Annulus** | `/petRDFS/HBJMGPAB/...` | không |
| **2023-10-20** | **PET 3D Normalization** | `/petRDFS/JFEJGPAB/SYSNDZAQ/PQRPXCCJ/SINO0001` | **có** |
| 2024-06-07 | PET 3D Normalization | `/petRDFS/NQNGGPAB/SYUKRVRX/ZVOAMELQ/SINO0001` | không |
| 2025-02-21 | PET 3D Normalization | `/petRDFS/NQPDZCAC/.../SINO0002` | không |
| 2025-02-21 | PET 3D Normalization | `/petRDFS/NQPDZCAC/.../SINO0003` | không |
| 2025-02-24 | PET 3D Normalization | `/petRDFS/NQNGGPAB/SYEWRVTO/FRLMEOTO/SINO0001` | không |
| 2025-02-24 | PET 3D Normalization | `/petRDFS/NQNGGPAB/MNICTYEA/RNHFDCKE/SINO0001` | không |

**WCC (well-counter) và 3D normalization là hai thứ khác nhau** — GE gán nhãn
riêng ở `(0017,1005)`. Scan 3DWCC Annulus thật nằm ở `HBJMGPAB/...` và **không**
có trong bản export.

### Exam 2026 vẫn trỏ về bản 2023-10-20 — console xác nhận

Máy **có** hiệu chuẩn lại 2024-06-07, 2025-02-21 (×2) và 2025-02-24 (×2) cho
đúng station DXRM3, nhưng cả ba exam 2026 vẫn khai `norm_cal_uid` /
`wcc_cal_uid` của **2023-10-20**. Đây không phải suy đoán từ header: console giữ
sổ đăng ký bản đang active ở `systemConfig/cal/.default`, và nó ghi thẳng ra —

```
3dgeom | 1.2.840.113619.2.131.1459872910.1219799569.748327
3dnorm | 1.2.840.113619.2.290.663120.1697775916.195435     <- 2023-10-20
3dwcc  | 1.2.840.113619.2.290.663120.1697776129.151600     <- 2023-10-20
```

File sửa lần cuối **2026-07-17**, tức 11 ngày trước ca nhi; `.default.old`
(2026-03-11) ghi y hệt. **Cả ba** trường UID của header RDF khớp từng ký tự với
ba dòng này — kể cả trường `ge_rdf_tool` gọi là `device_uid`, thực chất là UID
của cal `3dgeom`. Đó là trường thứ ba bị gán nhãn sai, cùng loại 0xEEC/0xF74.

Xác nhận thêm cho cách đọc 0xEEC: chính scan norm 2023-10-20 khai 0xEEC của nó
là `...1692003848...` = **2023-08-14**, tức bản active lúc *nó* được chụp.

Vậy bản đang dùng cũ **~2,8 năm**, ngoài mọi khuyến nghị (GE: hàng quý;
ACR/IAEA/EANM: tối thiểu hàng năm). Dấu hiệu cho thấy các lần sau có chạy nhưng
không được commit: ngày **2025-02-24 có tới ba file WCC** cách nhau ~10 phút
(12:12:56, 12:18:45, 12:32:38) và **cùng một AF 4,008868** — dạng thử đi thử
lại. Trên console GE, một lần WCC/norm phải được chấp nhận và cài đặt thì mới
thay bản cũ, và `.default` là chỗ ghi việc đó.

**Ảnh hưởng tới việc đang làm thì nhỏ:** AF trôi 4,0628 → 3,9854 → 4,0089, tức
~±1 %, và `K` thì tự đo trên NEMA nên thang tuyệt đối của GE không vào thẳng kết
quả. Đáng lo hơn là **đồng nhất theo detector** sau 2,8 năm, không phải hệ số tổng.

### File norm được nạp, và có tác dụng

`CNorm3d::Initialize()` trả `GRE_IG_SUCCESS`, và kết quả **đổi thật** so với
norm selftest:

```
norm selftest :  normdt  0,134 … 4,478   mean 1,011
norm của máy  :  normdt  0,466 … 3,280   mean 1,029   <- gọn hơn, dạng norm hơn
```

> Điều đó chứng minh file **được nạp và có tác dụng**, **không** chứng minh mọi
> hệ số được diễn giải đúng — `Initialize()` có thể trả SUCCESS trên dữ liệu nó
> hiểu sai. Muốn chắc thì kiểm bằng NEMA: norm đúng phải làm **phẳng** phantom
> đồng nhất (hiện lệch ±15 % theo trục).

### Kết quả (`out/`, mỗi `.f32` kèm `.json` ghi shape + thống kê + xuất xứ)

| file | nội dung | dtype / shape |
|---|---|---|
| `randoms.f32` | randoms | f32, 288 × 553 × 381 |
| `scatter.f32` | scatter SSS | f32, 288 × 553 × 381 |
| `normdt.f32` | **norm × dead time** (độ nhạy) | f32, 288 × 553 × 381 |
| `norm_only.f32` | chỉ norm → `dead time = normdt / norm_only` | f32, 288 × 553 × 381 |
| `prompts.u16` | sinogram phát xạ như đã nạp | u16, 288 × 553 × 381 |
| `singles.i32` | singles từng crystal | i32, 576 × 24 |
| `dt_int.f32`, `dt_mux.f32` | dead time theo block | f32, 256 |
| `estimate.json` | xuất xứ: đường dẫn vào, table position, header RDF | JSON |

**Không có WCC ở bất kỳ đâu** — thang tuyệt đối để tự hiệu chuẩn.

### Đổi sang dữ liệu khác — bằng tay

`estimate.py` sinh `job.gdb` từ job XR thật của hãng và **chỉ thay ba đường
dẫn**; 521 trường còn lại giữ nguyên của GE. Muốn tự làm:

```bash
./job2gdb.py <file.job> /out/ovl > job.gdb    # sinh lại từ .job khác
# hoặc sửa tay ba dòng trong job.gdb:
#   inputEmissionFileName[0]      <- sinogram phát xạ
#   normalizationSinogramFile     <- norm
#   inputTransmissionFileName[0]  <- PIFA (mu-map)
```

`extract.gdb` đọc job từ biến môi trường `D710_JOB` (mặc định
`/vendor/job.gdb`), và `run.sh` mount thêm `D710_DATA` vào `/data` — đó là
cách `estimate.py` đưa dữ liệu ngoài image vào.

**PIFA cho ca thật** (cả đĩa chỉ có 4 file selftest, không có cho ca nào của
mình) do `ct_to_pifa.py` dựng; `estimate.py` gọi sẵn. Gọi riêng thì:

```bash
python ct_to_pifa.py <CT dir> out/mu.pifa --table-location -125.17
python make_pifa.py --inspect out/mu.pifa
```

`ct_to_pifa.py` mặc định lấy `FrameOfReferenceUID` của chính series CT, tức thứ
`ValidateCTAC` đòi (§3d), nên thường không phải nghĩ.

---

## 1. Chốt lại: cái gì đã mở khoá

`pet_recon` cần 10 thư viện `.so` 64-bit. Cả 10 đều có, ở hai chỗ:

```
petsw/usr/g/PET/lib64/linux2/    libpetcfg libconfigMgr librdf libErr libpetlwc
                                 (+ libOpenCL.so.1)   <- /usr/PET/lib64 trỏ vào đây
petsw/usr/lib64/                 libreadcfg libeventmgr libmsghand libcupipc
                                 libstartup.so.1
```

`ldd` trong namespace: **0 thư viện thiếu**. `pet_recon` khởi động, đọc
`coremap.cfg.parc3`, dựng 6 processing thread.

### GPU không cần cho bốn thứ đang tìm

Đã kiểm bằng `nm`: **mọi** lớp `Ocl*` trong binary đều nhận
`OsemTofGpu::COsemTofValues` — `OclFwdProj`, `OclBackProj`, `OclRatioStep`,
`OclImageUpdate`, `OclPsfAxial/Radial`, `OclScatUpsample`. Tức GPU chạy **OSEM**,
không chạy correction. Randoms/scatter/norm/deadtime đều là kernel `sharcAp*`
trên CPU.

Nên bỏ GPU **không mất gì** cho việc này. (Và cũng không có lựa chọn: máy không
có OpenCL platform, và GE **không** ship mã nguồn kernel — `find petsw -name
'*.cl'` → 0 file. Chúng nằm trên chassis accelerator 10.1.1.x.)

---

## 2. Về cái image — vì sao nó như vậy

Cách chạy nằm ở §0; mục này chỉ giải thích image.

`d710:full` **chứa sẵn** `/usr/PET`, `/usr/g`, `/vendorlib` — không mount
`petsw/` nữa. Chỉ mount hai thứ nhỏ: `/vendor` (thư mục này, read-only, để sửa
script gdb không phải build lại) và `/out` (ghi kết quả).

`./run.sh --mount ...` là đường lùi dùng stage nền: bind `petsw/` từ host, và
khi đó phải có bản copy ghi được của `systemConfig` (~1.4 GB) — `run.sh` tự tạo
trong `--cache` (mặc định `$D710_OUT/.cache`) nên nó sống qua các bed.
`d710:full` sinh ra chính là để khỏi phải thế.

### `COPY`, không phải `docker commit`

Trước đây việc này do `bake.sh` làm: bind-mount `petsw/` vào một container tạm,
copy bên trong, rồi `docker commit`. Lý do là build context — `COPY` bắt docker
đẩy cả 18 GB sang daemon.

Image do `commit` sinh ra **hoàn toàn push/pull được**, nó là image bình thường;
mấy dòng `-v` chỉ sống lúc copy chứ không tham chiếu ngược về máy build. Vấn đề
là **không ai dựng lại được nó**: Dockerfile không mô tả nội dung của nó, nên
người nhận có thứ chạy được mà không sửa được.

`.dockerignore` ở thư mục gốc giải quyết vấn đề context đúng chỗ, nên bản copy
giờ là `COPY` thật. Nó cũng chính là chỗ **loại rác** — phần lớn 18 GB đó
`pet_recon` không đụng:

| bỏ | cỡ |
|---|---:|
| `usr/g/service/log/pet_iow_service.log` | **9.2 GB** |
| `usr/g/service/matlab/library{32,64}` | ~450 MB |
| `usr/g/service/tools/.../vct_helical.ffp` | 233 MB |
| `usr/g/service/dicomtools` | |

Bỏ thêm `usr/lib` (842 MB, thư viện 32-bit của console — thứ thật sự được nạp
nằm trong `usr/PET/lib/linux2`) và 1.1 GB `usr/lib64` trừ đúng 5 file. Còn lại
`usr/PET` 1.8 GB + `usr/g` 1.8 GB → context ~3.6 GB, image **7.3 GB** (gồm cả
toolchain i386 + numpy/scipy/pydicom/petsird cho phía decode).
`service/log/` được tạo lại rỗng vì `pet_recon` ghi `IG1.timers.log` vào đó.

> File log 9.2 GB kia là log dịch vụ console, **không phải dữ liệu** — xoá được
> nếu cần chỗ trống, nhưng nó nằm trong cây vendor nên `.dockerignore` chỉ bỏ nó
> khỏi context chứ không đụng bản gốc.

### Dữ liệu thử — có sẵn, đúng model, kèm đáp án

```
/usr/PET/release/petig/selftest/          (đã nằm trong image)
  test_protocols_kh/XR/selftest_kh_3dir.job     job XR = D710, thật
  data/selftest_kh3d_ex.rdf                     emission
  data/selftest_kh3d_norm.rdf                   norm
  data/selftest_kh3d_pifa.dat                   PIFA (mu-map cho scatter)
  gold_standard/job1.0 … job1.N                 ảnh chuẩn để đối chiếu
```

---

## 3. Trạng thái — cái gì đã chứng minh, cái gì chưa

| bước | trạng thái |
|---|---|
| `pet_recon` nạp đủ thư viện, khởi động | **xong** |
| boot tới idle (comm thread + GPU đã vô hiệu hoá) | **xong** |
| nạp `.job` vào `IgJobReq` (524 trường, cả string lẫn 47 packet) | **xong** |
| `sharcCmpOpenDataFiles(&IgJobReq)` | **xong, trả 0** |
| `apCfg` / `sysGeometry` được điền từ header RDF | **xong** |
| image Docker tự chứa `d710:full` (6.25 GB), boot OK | **xong** |
| dump toàn bộ `apCfg` + `sysGeometry` lúc chạy | **xong** → `PARAMS.md` |
| `InitParamStruct` → hình học sống (381 × 553 × 288, 55 TOF) | **xong** |
| `Deallocate` + `ReallocateBuffs` cấp phát thật (1 GB) | **xong** |
| `CIgManager::LoadRawData()` | **xong**, GRE_IG_SUCCESS |
| nạp singles (`sharcCmp3dRemoteSinglesLoad`) | **xong** |
| nạp dead time (`sharcCmp3dRemoteDeadtimeLoad`) | **xong** |
| nạp norm (`setup3dEmissJob(jobReq, false)`) | **xong** |
| **sinogram norm × dead time ghi ra đĩa** | **XONG** → `out/normdt.f32` |
| **sinogram randoms ghi ra đĩa** | **XONG** → `out/randoms.f32` (288 view, 100% ≠ 0) |
| **sinogram scatter (SSS) ghi ra đĩa** | **XONG** → `out/scatter.f32` |
| mu-map tự tính nạp vào `m_pAttn` cho SSS | **XONG** — §3c/§3d |
| sinogram attenuation qua `CCTAC_3D` | **KHÔNG làm** — `DoTask` SIGSEGV, và không cần: suy giảm dựng lại bằng `utils/attenuation.py` |
| `sharcCmpProcessJobOnAp(...)` chạy tới cùng | **KHÔNG làm** — deadlock, §3a thay thế |

---

## 3a. Chuỗi lệnh đúng — chép từ `CIgManager::RunCyclic`

`RunCyclic` là hàm điều phối, và thứ tự nó gọi mới là thứ tự bắt buộc:

```
CleanDataBuffers(false)
InitParamStruct()                 <- điền GRE_IG_PARAMETER_STRUCT từ apCfg/RDF
CRawDataMem::DeallocateBuffs(false, ig)
CCorrDataMem::DeallocateBuffs(false, ig)
CRawDataMem::ReallocateBuffs(false, ig)
CCorrDataMem::ReallocateBuffs(false, ig)
CImageDataMem::ReallocateBuffs(false, ig)
LoadRawData()                     <- CRawDataLoad trên thread pool
DoProcessing()                    <- InitLuts, InitRandomsFromSingles, ...
```

**`Deallocate` trước `Reallocate` không phải thủ tục thừa.** `ReallocateBuffs`
mở đầu bằng `if (GetNumAllocViews() != 0) skip`, mà một `CViewBuffer` vừa dựng
đã báo **1 phần tử 2 byte**. Không `Deallocate` thì mọi `Reallocate` trả
`GRE_IG_SUCCESS` mà **không cấp phát gì**, và `CRawDataLoad` chỉ đặt được
view 0 rồi báo `can't reserve view 1` (RawDataLoad.cpp:154). Đó là nguyên nhân
duy nhất khiến `LoadRawData` hỏng — không phải cờ `bool` của `ReallocateBuffs`,
cờ đó `RunCyclic` luôn truyền `false`.

Sau khi cấp phát đúng:

```
RawPrompts   288 view × 421386 B  = 121 MB   (2 B × 381 × 553)
RawDelays    288 view × 421386 B
RawNorm       47 view × 647168 B
CorrPrompts  288 view × 842772 B  = 242 MB   (4 B × 381 × 553)
CorrRandoms  288 view × 842772 B
CorrScatter  288 view × 842772 B
```

### Ba global mà `CPCCommThread` phải publish

`boot.gdb` giết `CPCCommThread` ở entry vì nó `opendir("/petRDFS/OVLFILES")` rồi
`exit(-1)`. Nhưng **ba câu lệnh đầu tiên** của nó (cpcMain.cpp:234-236) không
phải việc socket — chúng công bố ba global mà mọi kernel `sharcCmp*`/`sharcAp*`
dereference:

```
m_pRawDataMem   = arg[0]   @ 0xf38948
m_pImageDataMem = arg[1]   @ 0xf38950
m_pParamStruct  = arg[2]   @ 0xf38958
```

Bỏ ba dòng đó là nguyên nhân `LoadRawData()` SIGSEGV ở `0x516da5`, trong
`sharcCmp3dRemoteLoadPrompts`: `mov m_pRawDataMem,%rax ; mov (%rax),%r14` trên
global NULL. `CIgManager::m_pRawDataMem` (member) vẫn tốt suốt — **phía AP
không bao giờ nhìn vào member đó.** `boot.gdb` giờ chép lại ba lệnh ghi rồi mới
`return`.

### Nạp singles / dead time / norm — ba lối AP riêng, không handshake

`LoadRawData` **chỉ** nạp prompts. Ba thứ còn lại có entry point riêng, và tất cả
đều là phía AP nên không cần FIFO:

| cần | gọi | điều kiện |
|---|---|---|
| singles | `sharcCmp3dRemoteSinglesLoad(&IgJobReq, 0xf38968)` | `emissionRandomsFlag == 3`, `singlesType == 0` |
| dead time | `sharcCmp3dRemoteDeadtimeLoad(&IgJobReq, 0xf38980)` | `(1 << deadtimeType) & 0xd` |
| norm (+attn) | `setup3dEmissJob(&IgJobReq, false)` | `normalizationFlag`, `attenuationFlag` |

`setup3dEmissJob` với tham số `bool` = **false** thì bỏ vòng nạp emission
segment (`LoadRawData` làm rồi) và chỉ còn nạp norm + attenuation.

Hai struct `loadSinglesStruct` / `loadDeadtimeStruct` là global file-scope, kiểu
**anonymous trong DWARF** nên gdb không gọi tên trường được — ghi thẳng qua địa
chỉ cố định (binary non-PIE). Cả hai đều là
`{s32 status; s32 dragonSocketStatus; n32 type;}`.

---

## 3b. Norm × dead time — ĐÃ LẤY ĐƯỢC

`out/normdt.f32` — **288 × 553 × 381 float32, 231.5 MB, 100% khác 0**,
min 0.1229 max 4.0872 mean 0.9246, độ tản giữa các view chỉ 4.5%.

Công thức **là của GE**, lấy từ `CAccelIntfNormDTData::ApplyNormDeadtime`
(0x4a39e0) — hàm dựng sinogram NormDT cho accelerator — và được
`COsem3dPrep::DoPrep` lặp lại y hệt:

```c
GE_vfill(1.0f, work, number_v_theta * numberSamples);        // một view toàn 1.0
if (IgJobReq.normalizationFlag)
    CNorm3d::ApplyNormalization(work, phi, scratch, false);
if (IgJobReq.emissionDeadTimeFlag)
    CDeadtime3d::ApplyDeadtime(work, phi, work, 0, &f1, &f2, scratch, false);
// rồi PadData / projTranspose / p_rev4  <- chỉ là đóng gói cho accelerator,
//                                          extract.gdb KHÔNG chép phần này
```

**Cho kernel ăn một sinogram toàn 1.0 chính là mẹo** biến kết quả thành *hệ số
hiệu chỉnh* thay vì *dữ liệu đã hiệu chỉnh*.

`scratch` phải là **5 ×** một view (`NewReconRequest` cấp đúng
`5 * numberSamples * number_v_theta` float).

### Không có WCC trong đó

Đúng như yêu cầu. `ApplyNormalization` và `ApplyDeadtime` **không** đụng
`wccActivityFactor`, `wccSensitivityFactor` hay `m_fwccScaleFactor`, và
`extract.gdb` không gọi gì khác chạm vào chúng. Thang tuyệt đối để tự đo.

`extract.gdb` xuất **hai** file để tách được hai thừa số:

```
normdt.f32     = norm × dead time
norm_only.f32  = norm (bỏ qua bước ApplyDeadtime)
dead time      = normdt.f32 / norm_only.f32
```

### Dead time PHỤ THUỘC TỐC ĐỘ ĐẾM — đừng so với một hằng số tĩnh

Kernel dead time là

```
livetime = 1 / max((1−S·k1)(1−S·k2)(1−S·A)(1−S·B), 1/clip)
```

tức là hàm của **tốc độ singles `S`**, nên nó không thể bằng một hằng số ở mọi
ca. `PARAMS.md` §1 có suy ra được một scalar dead time 3D **tĩnh** từ
`sysGeometry` — `(dt_3dint + dt_3dmux + dt_3dtiming) / dt_3dint =
(1.85 + 0.0 + 0.20) / 1.85 = 1.10811` — nhưng **đối chiếu `normdt/norm_only`
với số đó là sai về nguyên tắc**, không chỉ sai con số.

Đo trên toàn bộ 60 679 584 bin, hai ca:

| ca | singles | livetime | 1/livetime |
|---|---|---|---|
| NEMA bed 2 | 1,88 Mcps | 0,9874 | 1,0128 |
| ca nhi bed 4 | 11,76 Mcps | 0,9339 | 1,0708 |

Livetime **giảm khi singles tăng**, đúng như dead time phải hành xử — và đó mới
là bằng chứng dùng được: nó cho thấy `normdt` là **độ nhạy**, không phải hệ số
hiệu chỉnh. Chia dữ liệu cho nó mới là hiệu chỉnh.

Thêm một kiểm chéo nữa: `out/dt_mux.f32` **toàn 0**, khớp
`dt_3dmuxCorrectionConstant = 0.0`.

### Mảng dead time là theo BLOCK, không theo crystal

`m_pIntDeadtime` / `m_pMuxDeadtime` dài đúng **`blocksPerSystem` = 256**
(= 64 blocksPerRing × 4), không phải 576 × 24. Đọc quá độ dài đó thì ra những
giá trị kiểu 7.9e34 — dấu hiệu duy nhất báo là đã đi quá. `extract.gdb` đọc
`blocksPerSystem` từ param struct sống chứ không hằng số hoá.

### Dựng `CNorm3d` / `CDeadtime3d` bằng cách ghi trường, không gọi ctor

Hai ctor chỉ là lệnh gán thuần:

```
CNorm3d     (32 B): [0]=CRawDataMem*  [8]=paramStruct*  [0x10]=0 [0x18]=0
CDeadtime3d (56 B): [0]=CRawDataMem*  [8]=paramStruct*  [0x10..0x28]=0
                    [0x30]=1.0f
```

Nên `extract.gdb` **ghi thẳng byte**. Gọi ctor qua con trỏ hàm ép kiểu thì gdb
từ chối, và nó báo lỗi như một lỗi lệnh bình thường — object lặng lẽ vẫn là
bộ nhớ `malloc` chưa khởi tạo, rồi `Initialize()` mới sập ở một chỗ hoàn toàn
khác.

---

## 3c. Randoms VÀ scatter — XONG cả hai

`CIgManager::InitRandomsFromSingles()` **không** sinh sinogram. Nó chỉ gọi
`CalculateRFS()` → `GE_meanvx` + `initRFSParams`, tức tính **tham số**
randoms-from-singles. Đó là lý do nó trả `GRE_IG_SUCCESS` mà `m_pRandoms` vẫn
toàn 0.

Nơi randoms 3D thực sự được ghi là `COsem3dPrep::DoPrep(view, ...)`:

```
GetRawPromptsBuff / GetDelaysBuff / GetCorrPromptsBuff
Preprocess(...)                          -> sharcApPreprocess
SetCorrPromptsBuffValid
GetCorrRandomsBuff(view) -> Preprocess -> SetCorrRandomsBuffValid   <- RANDOMS
GetCorrNormDTBuff(view)  -> GE_vfill  -> ApplyNormalization/ApplyDeadtime
GetCorrScatterPromptsBuff(view) -> Preprocess
```

Dòng `GetCorrNormDTBuff → GE_vfill → ApplyNormalization → ApplyDeadtime` là
**xác nhận độc lập** rằng công thức §3b đúng là của hãng.

Ba context dựng được hết (`probes/prep.gdb`), entry point lấy từ vtable:

```
CCTAC_3D              Init 0x43eef0  TaskList 0x441550  DoTask 0x441030
CScatterFully3dModel  Init 0x489a30  TaskList 0x48bba0  DoTask 0x48b5d0
COsem3dPrep           Init 0x482290  TaskList 0x434de0  DoTask 0x435e70
```

`probes/prep2.gdb` chạy `COsem3dPrep::DoTask` và **model scatter chạy hết các pha**
— `MSCAT_CONVERT_SCAT_TO_3D`, `MSCAT_TAILFIT_SCAT_3D`, `MSCAT_TAILSCALE_SCAT_3D`
— rồi `CorrPrompts/CorrRandoms/CorrScatterPrompts::InsertedNewViews 0..48`.

**Chỗ kẹt là attenuation, không phải randoms hay scatter.** `DoPrep` quay vòng ở

```
CCTAC_3D:: AreViewsAvailable not ready Start: 0 Num: 47
```

vì `CCTAC_3D` là context **riêng**; trên console thread pool chạy nó song song,
còn ở đây một luồng thì phải chạy xong trước. Chạy nó trước thì
`CCTAC_3D::DoTask` **SIGSEGV ở CTAC3D.cpp:320** — bộ ảnh CTAC chưa bao giờ được
nạp (đường khác với PIFA mà `sharcCmpOpenDataFiles` đọc).

Attenuation không nằm trong bốn thứ cần, nên `probes/prep4.gdb` đặt
`IgJobReq.attenuationFlag = 0` và `fCTACFlag = 0` để `DoPrep` bỏ nhánh CTAC.
Làm vậy thì `DoPrep` chạy thật, view sau view — `Prep3d uiPhiIndex = N
Processing Done` — nhưng **dừng ở view 28/288**: đo `/proc` thấy utime đứng yên,
cả 8 thread ngủ.

### Randoms/scatter là PRE-ESTIMATE, không nằm trong vòng lặp OSEM

Prep **không** chờ OSEM tiêu thụ buffer. Dựng lại `Do3dEmissionImage` theo
**thứ tự dòng nguồn** (không phải thứ tự địa chỉ) thì thấy rõ:

```
IgManager.cpp:231   dựng COsem3dPrep / CPrep3d
             247    CCorrDataMem::InvalidateBuffs
             248    CThreadPool::StartContext      <- thả prep chạy ASYNC
            1838    CCTAC_3D(ig)
            1911    CThreadPool::WaitContext       <- rào chắn
            1915    CThreadPool::WaitContext       <- rào chắn
            1944    CScatterFully3dModel(ig, n, bool)
            1948    WriteCorrDataByBrp
            1962    CThreadPool::WaitContext       <- rào chắn
            1994    CreateReconContext()           <- dựng OSEM; bên trong gọi
                                                      CombineAdditiveCorrBuffers
            2005    CThreadPool::RunContext        <- OSEM CHẠY Ở ĐÂY (blocking)
            2039    Do3dOverlapProcessing
```

**Cả ba `WaitContext` đều xong TRƯỚC `CreateReconContext` (1994) và trước
`RunContext` của OSEM (2005).** Mà `CreateReconContext` chính là chỗ gọi
`CombineAdditiveCorrBuffers()` — gộp randoms + scatter thành số hạng cộng.

Tức là randoms và scatter được **ước lượng một lần, xong hẳn, rồi OSEM mới bắt
đầu**. Trong vòng lặp OSEM chỉ có đọc lại:
`OsemTof::COsemTofMain::GetRandomsViewData(view, float*)` chỉ **copy** từ buffer
đã có, không tính gì.

(`IgJobReq.modelScatterIterations` là vòng lặp **nội bộ của SSS** — tail-fit
tinh dần — không phải vòng lặp OSEM.)

Vậy chỗ kẹt ở view 28 **không thể** là "chờ OSEM": lúc đó OSEM còn chưa được
dựng. Nguyên nhân là `CRendezvous(int)` + `CRendezvous::Join()` — **rào chắn đếm
số luồng**, và `CScatterFully3dModel::SyncBeforeNextIter` dùng một cái. Dựng
context với `nThreads = 6` (số `GetNumOfProcessingThreads` báo) rồi chỉ chạy
`DoTask` trên **một** luồng thì rào chắn chờ đủ 6 lượt không bao giờ mở — đúng
hình dạng của chỗ kẹt: 0% CPU, mọi thread ngủ.

### Đổi thành `nThreads = 1` là xong: randoms ra đủ 288 view

```
COsem3dPrep(ig, 1, ctac, scat)          CScatterFully3dModel(ig, 1, false)
-> 28 view  =>  281 view in ra, m_pRandoms->m_uiCounterElement = 288
```

`out/randoms.f32` — **288 × 553 × 381, 100% khác 0**, 0.0352 … 0.2473.

Kiểm chứng vật lý:

```
tổng prompts   2.143e7 counts
tổng randoms   4.698e6 counts      -> randoms/prompts = 21.9 %
tản theo view: randoms 2.0 %   vs   prompts 5.7 %
```

Randoms mượt hơn hẳn theo view — đúng dạng của ước lượng from-singles, không
phải rác.

### Vì sao "hai cái phải chạy cùng lúc" — thật ra KHÔNG phải hai context

Giải mã offset vtable trong `CPrep3d::DoTask` (address point =
`_ZTV11COsem3dPrep+16`):

```
*0x20 DoTask            *0x30 Finalize       *0x80 DoPrep / GetScatterCounts
*0x88 AddViewsToCorr    *0x90 AreViewsAvailable
*0xa0 DoPostCTACPrep    *0xa8 DoPostScatterPrep   *0xb0 DoPostRandomsPrep
```

Ghép với số dòng nguồn:

```
Prep3d.cpp:406  GetNextTask + AddViewsToCorr
           422  AreViewsAvailable
           462  *0x20  -> gọi DoTask() của MODEL SCATTER, ngay trong hàm này
           473  CRendezvous::Join()      <- rào chắn 1
           475  DoPostCTACPrep
           484  CRendezvous::Join()      <- rào chắn 2
           500  *0x20  -> scatter DoTask lần nữa
           509  CRendezvous::Join()      <- rào chắn 3
           515  Finalize
           519  DoPostScatterPrep
           527  CRendezvous::Join()      <- rào chắn 4
           540  DoPostRandomsPrep
```

**Scatter không phải một context chạy "song song bên cạnh" prep.** Nó là một
**pha nằm trong chính `CPrep3d::DoTask`**, được gọi trực tiếp qua con trỏ
`CScatter*` mà ctor lưu ở `this+0x190` (lưu **vô điều kiện**, ctor+264).

Cái chạy song song là **N luồng của thread pool cùng chạy một hàm `DoTask`
đó**, gặp nhau ở bốn `CRendezvous::Join()`. `CRendezvous` được dựng trong ctor
của `CPrep3d` với **đúng `nThreads`** (ctor+289). Nên:

* `nThreads = 6` + chỉ 1 luồng gọi `DoTask` → rào chắn chờ 6 lượt, **không bao
  giờ mở** → kẹt ở view 28, 0% CPU.
* `nThreads = 1` → rào chắn mở ngay với 1 lượt → chạy hết 288 view → **randoms
  ra được**.

### Cái tắt scatter là `attenuationFlag`, KHÔNG phải `nThreads`

So `prep2` với `prep6`, biến thật sự đổi giữa hai lần chạy là `attenuationFlag`,
không phải `nThreads`:

| lần chạy | attenuationFlag | nThreads | số dòng `MSCAT_` |
|---|---|---|---|
| `prep2` | **2 (bật)** | 6 | **97** |
| `prep4` | 0 (tắt) | 6 | 0 |
| `prep6` | 0 (tắt) | 1 | 0 |

Tắt attenuation để né cú sập của `CCTAC_3D` thì **tắt luôn SSS scatter** — hợp
lý về vật lý, vì SSS cần thông tin mu.

Nên bài toán scatter thật ra là: **scatter cần `attenuationFlag` bật; mà bật thì
`CCTAC_3D` phải sinh được 47 view attenuation; mà `CCTAC_3D::Initialize` fail
với `ValidateCTAC radialModulesPerSystem EX: 32 CTAC: 0`** — struct sysGeometry
phía CTAC toàn 0 vì chưa bao giờ có file CTAC nào được nạp.

Chép nguyên 2048 byte `sysGeometry` sang `TransSysGeometry` là qua được
`ValidateCTAC` (chi tiết ở §3d). Nhưng qua được vẫn chưa đủ: `CCTAC_3D` đói **đầu vào**. Đo được:

```
$rd->m_pAttn   counter=0  alloc=47  elementSize=65536   <- 128*128*4, một lát mu
fileStatus[1] = 0        <- PIFA chỉ được parse HEADER, voxel chưa ai nạp
```

Nạp thẳng 47 lát mu vào `m_pAttn` rồi `InsertedNewViews` + `SetViewsAsValid`:
số lần poll `AreViewsAvailable not ready` **387 → 0**.

### Công thức chạy được — bốn điều kiện phải đúng CÙNG LÚC

```
1. attenuationFlag  BẬT  (giữ nguyên 2 như job)      <- tắt là mất scatter
2. TransSysGeometry := sysGeometry (2048 byte)        <- qua ValidateCTAC
3. mu-map nạp vào CRawDataMem::m_pAttn (47 view x 128x128 f32, mm^-1)
                   + InsertedNewViews(0,47) + SetViewsAsValid(0,47)
4. nThreads = 1 khi dựng CScatterFully3dModel và COsem3dPrep
```

Kết quả: **288/288 view, 0 poll, `m_pScatter` counter = 288.**

```
gồ ghề theo bán kính:  scatter 0.071  vs  prompts 1.317  -> mượt hơn ~19 lần
```

Sinogram scatter **bắt buộc** mượt và tần số thấp — tỉ số gồ ghề đó là chữ ký
của nó, và là bằng chứng đây là ước lượng thật chứ không phải nhiễu hay bản sao
dữ liệu.

`DoTask` vẫn kết thúc bằng SIGSEGV trong
`CScatterFully3dModel::GetScatterCounts` (ScatterFully3dModel.cpp:5025) **sau
khi** cả 288 view đã ghi xong, nên không ảnh hưởng kết quả.

**Điểm nạp mu-map ở (3) chính là chỗ đưa mu tự tính vào** — xem §3d và
`ct_to_pifa.py`.

## 3d. Mu-map tự tính đưa vào pipeline của hãng — ĐƯỢC, và chỉ sửa 1 dòng

Trả lời cho hướng **SSS**: có, và không cần console.

File PIFA **chính là** hộp đựng mu-map mà model scatter đọc.
`sharcCmpOpenDataFiles` parse nó vào global `transmissionCTACHeader`, rồi
`CScatterFully3dModel` dựng mu image từ đó. Job trỏ tới nó bằng **đúng một
trường**:

```
IgJobReq.inputTransmissionFileName[0]      <- job.gdb dòng 12
```

### Định dạng — toàn bộ trường, tên và offset lấy từ DWARF

`ptype /o transmissionCTACHeader` (global là `[3]`, mỗi acquisition một cái):

```
  0  f32 versionID          (1.0)      32  s8  frame_of_reference[64]
  4  n32 cmpJobID                      96  s8  spareFields[64]
  8  n16 xMatrix            (128)     160  n32 offsetToStartOfImage = 164
 10  n16 yMatrix            (128)     --- hết 164 byte, tiếp theo là khối ảnh:
 12  n16 zMatrix            (47)          xMatrix*yMatrix*zMatrix float32
 16  f32 ctacDfov           (700 mm)      mu theo mm^-1, x chạy nhanh nhất
 20  n32 patientEntry
 24  n32 patientPosition
 28  f32 tableLocation      (−47.92)
```

Kiểm trên chính file selftest: 164 + 128×128×47×4 = 3 080 356 byte = đúng cỡ
file. **Đơn vị là mm⁻¹, không phải cm⁻¹**: max đo được 0.009336 mm⁻¹ = 0.0934
cm⁻¹, khớp nước ở 511 keV (~0.096 cm⁻¹). STIR dùng cm⁻¹ → chia 10.

### Thứ tự chiều của khối mu — CHỐT được, không phải đoán

Câu hỏi "PIFA đúng tổng dữ liệu nhưng chiều thì sao?" — trả lời bằng chính cách
binary **tiêu thụ** khối đó, chứ không phải bằng ước đoán.

`CRawDataMem::m_pAttn` (buffer mà `CCTAC_3D` đọc vào) được cấp phát:

```
$rd->m_pAttn   alloc = 47 view      = zMatrix
               elementSize = 65536  = 128 × 128 × 4 = xMatrix × yMatrix × f32
```

Một view = **một lát z**. Nên bố cục bắt buộc là

```
mu[z][y][x]   z chậm nhất  ->  y  ->  x nhanh nhất
```

Đây là **ràng buộc cấu trúc**, không phải quy ước tự chọn: nếu viết khác thì
kích thước view không khớp và `CCTAC_3D` đọc lệch ngay lát đầu.
`ct_to_pifa.py` ghi đúng thứ tự này.

### `reorderDims*` — GE tự hoán vị, và nó KHÔNG phải chuyện hướng bệnh nhân

Binary có nguyên một họ hàm hoán vị 3 chiều:

```
reorderDims132  reorderDims213  reorderDims231  reorderDims312  reorderDims321
        (int n1, int n2, int n3, float* in, float* out)
```

Giải mã `reorderDims312` từ assembly:

```
vào :  idx = i1 + n1·i2 + n1·n2·i3      (i1 nhanh nhất)
ra  :  idx = i3 + n3·i1 + n3·n1·i2      (i3 nhanh nhất, rồi i1, rồi i2)
```

→ tên `reorderDimsABC` = **thứ tự trục của đầu RA, nhanh-trước, là A,B,C**.

`CCTAC_3D::DoTask` gọi `ReorderDims312` **hoặc** `ReorderDims132`, chọn bằng
một bảng nhảy 7 nhánh (`jmp *0xb4f668(,%rax,8)`, chặn `cmp $0x6`). Chỉ số đó
**không phải** cờ hướng bệnh nhân; nó là
`CTAC_TASK_TYPE::taskType`, tức **giai đoạn pipeline**:

```
struct CTAC_TASK_TYPE { CTAC_TASK taskType; uint uiInputStartView; ... }
```

và log của chính GE in `CCTAC_3D:: CTAC task 6 Input: 0 47 Output: 0 47` —
task 6 → nhánh 6 → `ReorderDims312`. Đây là chuyển vị **nội bộ giữa các pha**
(khối CT → thứ tự mà forward-projector muốn), xảy ra **bất kể** mình ghi PIFA
thế nào. Nó không đọc `patientEntry`/`patientPosition`.

### Chiều dương — ĐO được, bằng chính code masking bàn của GE

`reorderDims*` là **chuyển vị thuần, không lật dấu**, nên chúng không nói gì về
việc hàng 0 là trước hay sau. Nhưng **module scatter thì nói**.

`CScatterFully3dModel::MaskTableMuImage` gọi `sharcApTableMaskModified`, và nó
mô hình **bàn bệnh nhân là một hình tròn cố định**:

```
hằng số trong binary:  r = 464,05 mm      offset = 45,974
vị trí lấy từ:  paramStruct+0x8c18 / +0x8c1c
                = ctacXaxisTranslation / ctacYaxisTranslation
                (XOR với −0.0 → đảo dấu, tức dùng giá trị ÂM)
rồi:  rotate(mu, out, 128, 128, sysGeometry.vqc_ZaxisRoll)
```

Tức là GE **biết bàn nằm ở đâu** trong khối mu và cắt nó đi. Đó chính là cái
phân biệt được chiều: **nếu lật y thì mask cắt nhầm sang mô, bàn ở lại trong
mu-map như vật chất đặc, và SSS thổi phồng scatter.**

Đo thật, cùng một ca NEMA bed 2, chỉ khác chiều của khối mu đưa vào:

| mu-map | scatter sum | S/(T+S) | max |
|---|---|---|---|
| **LPS nguyên bản** | 4,037e6 | **32,98 %** | 0,673 |
| lật y | 6,031e6 | **49,28 %** | 0,924 |
| lật x | 4,133e6 | 33,77 % | 0,683 |

**Trục y: chốt.** 32,98 % nằm đúng dải vật lý của 3D PET (30–45 %); 49,28 % thì
không. LPS nguyên bản là đúng — `ct_to_pifa.py` ghi theo toạ độ bệnh nhân DICOM,
lấy thẳng `x0`,`y0` từ `ImagePositionPatient`, và đó là cái GE muốn.

Trên NEMA, **trục x không chốt được** (33,77 so 32,98 — lệch 2,4 % tương đối,
cả hai đều hợp lý): phantom gần đối xứng qua x nên lật x hầu như vô hình. Bàn
lệch theo y nên chỉ y bị ràng buộc mạnh.

### Trục x — chốt bằng ca bệnh nhân thật

Lặp lại đúng phép đo trên **một bed ngực–bụng của bệnh nhân** (`11082026`,
`SINO0004`, bàn −270,68 mm, 43,8 M prompts). Chọn bed này vì mu-map của nó
**bất đối xứng trái–phải 22,1 %** — đúng thứ NEMA không có:

| mu-map | S/(T+S) | tail residual lệch L–R |
|---|---|---|
| **LPS nguyên bản** | **37,28 %** | 83,8 % |
| lật x | **76,48 %** | 105,3 % |

**76 % là vô lý** — scatter không thể chiếm 76 % số trues. Nên **LPS nguyên bản
cũng đúng ở trục x**. Ca bệnh nhân phân biệt mạnh hơn NEMA rất nhiều (37 vs 76,
so với 33,0 vs 33,8), đúng như dự đoán từ tính bất đối xứng.

> Một số dễ gây hiểu nhầm trong phép kiểm này: `prompts − randoms − scatter`
> âm ở **54,7 %** số bin. Không phải mô hình sai — **54 % số bin có 0 count**
> (trung vị 0, 94 % số bin ≤ 2 count), nên `0 − randoms − scatter` âm là hiển
> nhiên. Lọc theo số đếm:
>
> ```
> bin có prompts > 0  (46,0 % số bin):  1,5 % âm
> bin có prompts > 2  ( 5,9 % số bin):  0,0 % âm
> ```
>
> Và tổng: `randoms+scatter = 71,9 %` của prompts → còn 28 % là trues. Hợp lý.

**Cả hai trục đều đã đo, và toạ độ DICOM LPS nguyên bản là đúng.** Vì vậy
`ct_to_pifa.py` **không có cờ lật nào** — cả hai phép lật đều đổi kết quả rất
mạnh, nên chúng là thứ phải đo lại chứ không phải thứ để bật thử. Máy hay bản
export nào khác quy ước thì lặp lại đúng phép đo trên.

Nhất quán với kiểm hình học trên chính mu-map:

```
lát giữa: trọng tâm y=67,5 (tâm mảng 64)
tổng mu nửa y<64 (trước) 7,8   nửa y>=64 (sau) 12,5
-> nửa đặc hơn ở y CAO = giường nằm DƯỚI phantom, đúng LPS
```

> Vì sao không so trực tiếp với PIFA thật của hãng: `/usr/g/ctuser/PIFA/<exam>/`
> có trong log console nhưng **không** nằm trong bản export, nên không có file
> đối chứng. Phép đo qua table-mask thay thế được, và mạnh hơn kiểu "nhìn ảnh
> thấy hợp lý".

### Công cụ

```bash
python3 make_pifa.py --inspect <file>.dat            # giải mã, kiểm đơn vị
python3 make_pifa.py mu.npy my.pifa --dfov 700 --units cm-1 \
        --copy-spare-from <pifa_mau>.dat
# rồi sửa job.gdb dòng 12 trỏ vào my.pifa
```

`make_pifa.py` đã **kiểm tra khứ hồi**: ghi lại từ payload của file selftest
cho ra file **giống từng byte**.

Một chỗ chưa chắc chắn: `spareFields[64]` (offset 96..159) trong file của GE
**không rỗng** (có `2.5`, `6`, `3`, `64.0`, vài float nhỏ). DWARF đặt tên nó là
"spare" nên parser của `pet_recon` không có tên để đọc — nhưng đó là suy luận,
không phải phép đo. `--copy-spare-from` chép nguyên 64 byte đó từ một PIFA đã
biết chạy được, để khỏi phải cược.

### `ValidateCTAC` kiểm ~20 trường, không phải 4

Bản `pet_recon` thật sự gọi là overload **0x43e550**
`ValidateCTAC(jobReq, apConfig, petExam, acqParam×2, sorter×2, sysGeo×2)`, và
nó so từng trường một, mỗi trường một câu lỗi riêng:

```
radial/axialModulesPerSystem   radial/axialBlocksPerModule
radial/axialCrystalsPerBlock   detectorRadialSize   detectorAxialSize
effectiveRingDiameter          transaxial_crystal_0_offset
vqc_XaxisTranslation  vqc_YaxisTranslation  vqc_ZaxisRoll
scanner_first_slice   interCrystalPitch   interBlockPitch     <- SYS_GEO
patientEntry   patientPosition   tableLocation                <- header PIFA
```

(Có một overload nhỏ 0x43e400 chỉ so 4 trường — recon thật không gọi bản đó.)
Nhưng ~20 trường lại **dễ hơn**, không khó hơn: mọi trường hình học đều so
`sysGeometry` (EX) với `TransSysGeometry` (CTAC), nên chép 2048 byte cái này
sang cái kia là mọi phép so thành đồng nhất (§3c). Còn lại ba trường trong
header PIFA, `make_pifa.py` / `ct_to_pifa.py` ghi.

**`frame_of_reference` phải khớp exam** — `strcmp`, sai là loại thẳng và
**scatter ra toàn 0**:

```
ValidateCTAC frame_of_reference EX: PIFA CTAC: 1.2.840.113619.2.290.3.663120...
```

Một literal như chữ "PIFA" là loại thẳng. `ct_to_pifa.py` lấy mặc định là
`FrameOfReferenceUID` của chính series CT — đo được là **trùng khít** giá trị
vendor đòi.

### Phân biệt PIFA với CTAC — hai đường hoàn toàn khác nhau

| | đọc từ | dùng cho | trạng thái |
|---|---|---|---|
| **PIFA** | `inputTransmissionFileName[0]` | **model scatter (SSS)** | **chạy được** |
| **CTAC** | bộ ảnh CTAC, đường khác hẳn | sinogram attenuation (`CCTAC_3D`) | SIGSEGV, chưa nạp |

Model scatter chạy hết mọi pha (`MSCAT_CONVERT_SCAT_TO_3D`,
`MSCAT_TAILFIT_SCAT_3D`, `MSCAT_TAILSCALE_SCAT_3D`) **trong khi `CCTAC_3D` đang
hỏng** — bằng chứng trực tiếp rằng SSS chỉ cần PIFA. Muốn scatter thì
`attenuationFlag = 0` là đúng, không phải thiếu sót.

### `PARAMS.md` — tham số sống

Không phải sinogram, nhưng là thứ cần để **cài công thức**: toàn bộ `apCfg` và `sysGeometry` đọc từ tiến trình đang chạy, đủ tên trường từ
DWARF. Nó chốt được ba chỗ trước đây phải đoán:

* scalar dead time `(a+b+c)/a` = `dt_3dint 1.85`, `dt_3dmux 0`,
  `dt_3dtiming 0.20` → **1.1081**;
* `A[i]`/`B[j]` trong livetime = `dt_3dcrystalpileUp_factors`, **54 = 9 × 6**;
* `delaysCorrectionFactor = 1.0` → GE không nhân thêm gì vào randoms.

Chi tiết + hai câu trả lời phủ định có giá trị: `PARAMS.md`.

### Cái đã đo được, và nó tự kiểm chứng

`sharcCmpOpenDataFiles` trả **0** và điền `apCfg` bằng đúng những con số mà việc
đọc tĩnh file cfg + binary đã suy ra trước đó:

| trường `apCfg` | giá trị sống | đọc tĩnh |
|---|---|---|
| `normDeadtimeSlopeInnerRing` | 4.74 | 4.74 ✓ |
| `normDeadtimeSlopeOuterRing` | 4.57 | 4.57 ✓ |
| `scatScaleFactorLimit` | 1.5 | `SCAT_SCALE_FACTORS_LIMIT` ✓ |
| `norm3dGeometryPeriod` | 9 | 9 (nên khối có 3429 = 381×9) ✓ |
| `norm3dCorrXtalEffClipValue` | 10.0 | `clip` của dead time ✓ |
| `scatCountRateFuncCoeff[11]` | 0.4509, 0.0709, 0.0193, −0.2785, … | 11 hệ số count-rate ✓ |

Header RDF in ra cũng khớp: `coincTimingPrecision 0.089246` (ns) — đúng
89.2459 ps dùng trong `2τ = 55 × 89.2459 ps`; `delayedEvents 0`,
`singleCollect 1`; segment type 9 tại offset **118784** — đúng khối hệ số hình
học của file norm (xem "Nội dung file" ở §0b).

**Những con số đó được xác nhận bằng chính binary lúc chạy, chứ không phải
bằng đọc file cfg.**

### Đường job qua `sharcCmpProcessJobOnAp` — không dùng được

`sharcCmpProcessJobOnAp` trả −1 ngay ở `cpcApLib.cpp:412`, chỗ
`write(fdCpcIgFIFO, {2,0}, 8)` — FIFO đó do `CPCCommThread` mở, mà thread đó bị
giết lúc boot. Thay bằng một `pipe()` thì qua được `write()` rồi **deadlock**:
đo trên `/proc` thấy main thread `poll_schedule_timeout`, 6 worker
`futex_do_wait`, **0 giây CPU**, RSS 15 MB. Lý do là lỗi kiến trúc của cách gọi
— `ProcessJobOnAp` là phía host/CPC, nó *giao việc* cho phía IG rồi chờ, nhưng
gdb gọi nó từ chính main thread đang đứng trong `CIgManager::RunCyclic`, tức
phía IG. Thread duy nhất có thể trả lời chính là thread đang chờ.

**§3a thay thế toàn bộ đường này**: gọi thẳng các hàm mà `RunCyclic` gọi, không
qua tầng job, không FIFO, không handshake. `run_job.gdb` / `run_job_fifo.gdb`
giữ lại làm bằng chứng cho kết luận trên — đừng chạy.

### Nếu cần đi sâu hơn nữa: gọi kernel trần

Bốn correction là bốn wrapper mỏng:

```
CNorm3d::ApplyNormalization(float* inBuff, int phi, float* pWorkBuffer, bool)
                                                      -> sharcAp3dNormalization
CDeadtime3d::ApplyDeadtime(float* inBuff, int phi, float* outBuff, int countsFlag,
                           float* totalCounts, float* averageDeadtimeFactor,
                           float* pDeadtimeWorkBuff, bool)
                                                      -> sharcApDeadtime3d_puc
sharcApPreprocess                                     <- randoms, qua CPrep3d
CScatterFully3dModel::*                               -> calcScatter3d
```

Tên tham số ở trên là **tên thật trong DWARF**, gdb in ra ở frame lỗi. Hai
wrapper đầu chỉ rút vô hướng và con trỏ ra khỏi `m_pRawDataMem`, `m_pIgParams`
và hai global `apCfg` + `sysGeometry` — §3a đã điền đúng cả bốn. `extract.gdb`
gọi thẳng hai hàm này.

---

## 4. Bẫy — đừng dẫm lại

### Bẫy gdb, cái đầu là khó thấy nhất

* **CỜ HƯỚNG (DF) rò vào lời gọi inferior.** gdb dựng dummy frame từ trạng
  thái thanh ghi ở chỗ dừng, **kể cả EFLAGS**. DF của `pet_recon` lúc dừng là
  bao nhiêu thì hàm được gọi bắt đầu với đúng bấy nhiêu. Với **DF=1**, `rep
  stos` của `memset` trong glibc chạy **ngược**: `CCyclicMemBuffer::AllocateMem`
  (CyclicMemBuffer.cpp:160) đang xoá buffer 121 MB vừa cấp phát thì chạy tụt
  khỏi **đầu** buffer và SIGSEGV sau 13 byte — `rdi = m_pStartBuffer − 17`,
  `si_addr` là trang ngay dưới. ABI System V bảo đảm DF=0 ở mọi ranh giới lời
  gọi nên trình biên dịch không bao giờ phát `cld`; không có gì trong inferior
  sửa lại.
  **Triệu chứng đánh lừa:** nó *có vẻ* ngẫu nhiên, vì DF phụ thuộc chỗ tiến
  trình tình cờ dừng — dễ bị coi là "flake" trong khi hoàn toàn tất định.
  Cách chữa: `set $eflags = $eflags & ~0x400` trước mỗi lời gọi
  (`clear_df()` trong `lib.gdb`, và một lần trong `boot.gdb`).

* **`print` một hàm trả `void` thì gdb từ chối — và báo như lỗi lệnh thường.**
  Ép kiểu `((void (*)(...)) addr)(...)` rồi `print` thì ctor **không hề chạy**,
  object vẫn là bộ nhớ `malloc` rác, và `Initialize()` sập ở chỗ khác hẳn.
  Ép về `(int (*)(...))` (giá trị trả về vứt đi), hoặc — tốt hơn cho ctor thuần
  gán — **ghi thẳng trường**.

* **Đừng để `show()`/wrapper nuốt thông báo lỗi của gdb.** Bản đầu in
  `<unavailable>` cho mọi thất bại thì một lời từ chối cú pháp trông y hệt một
  cú sập trong callee. `lib.gdb` in nguyên văn lỗi.

* **`set scheduler-locking on` phải đặt SAU `run`.** Trước đó chưa có thread
  nào, gdb báo `Target 'exec' cannot support this command` và **huỷ cả file
  script**. Đặt ở cuối `boot.gdb`.
  Mặc định khoá là đúng: mọi lời gọi inferior resume **tất cả** thread, để 6
  processing thread chạy đua với allocator. Chỉ mở khoá (`unlocked()` trong
  `lib.gdb`) cho những lời gọi thật sự cần thread pool — `LoadRawData` và các
  hàm nạp.

* **`sharcCmpDebugFlag = 0x1FF` in ~50 000 dòng header RDF** qua pipe, chiếm
  phần lớn thời gian một lần chạy. Dùng `0x4` (LOAD) cho việc thường; `0x1FF`
  chỉ khi cần dump header (nguồn của `PARAMS.md`).

* **gdb buffer stdout của chính nó, còn `printf` của inferior thì không.** Log
  đan xen sai thứ tự, và một lần chạy bị treo không để lại dấu vết nó đi tới
  đâu. `lib.gdb` flush sau mỗi dòng (`say()`).

* **Đừng đặt `petsw/usr/lib64` vào `LD_LIBRARY_PATH` của gdb.** gdb sẽ nhặt
  `libstdc++.so.6` cổ của GE và không khởi động nổi. Chỉ symlink 5 thư viện cần
  thiết vào một thư mục riêng, và đặt biến cho **inferior** bằng
  `set environment` trong gdb.
* **Đừng gọi `strcpy` trong inferior để nạp string.** Nó vướng ifunc của glibc,
  và mọi lời gọi hàm trong inferior đều **resume tất cả thread** — đủ để thread
  comm chạy tới `exit()` và giết tiến trình. Ghi thẳng bộ nhớ bằng
  `gdb.selected_inferior().write_memory` (hàm `_s` trong `boot.gdb`).
* **`/petRDFS/OVLFILES` không tạo được trong namespace** (`/` thuộc real root).
  Không cần: đường dẫn overlap nằm trong chính `.job`, `job2gdb.py` nhận tham
  số thư mục để ghi đè.
* **`ptype S_HOST_CMP_JOB_REQ` không có** — struct là anonymous trong DWARF.
  Dùng global `IgJobReq` (`cpcMain.cpp:108`) và ép kiểu con trỏ hàm khi gọi.
* **`pet_recon` chỉ cho chạy MỘT bản một lúc, và bản chết treo sẽ chặn hết.**
  `cupInit.c:186` in `pet_recon already running @ pid=NNN, exiting...` rồi thoát
  code 1 — lúc đó gdb chưa kịp làm gì nên lỗi hiện ra dưới dạng khó hiểu
  (`Cannot access memory at address 0xf389a4`, tức đọc `sharcCmpDebugFlag` của
  một tiến trình đã chết). Một lần chạy bị ngắt giữa chừng để lại tiến trình mồ
  côi **ngồi im, RSS 15 MB, 0% CPU** — nhìn `ps` dễ tưởng không còn gì. Kiểm và
  dọn trước mỗi lần chạy:

  ```bash
  pkill -9 -f pet_recon
  ```

  Không attach gdb vào nó được (`ptrace: Inappropriate ioctl`) vì nó nằm trong
  namespace riêng — chỉ có nước giết. Với Docker: `docker ps` rồi `docker rm -f`.
* **`ps` trong sandbox của agent KHÔNG thấy tiến trình trong namespace.**
  `ps aux | grep pet_recon` ra 0 trong khi `/proc/<pid>/comm` vẫn ghi
  `pet_recon`. Đừng tin `ps`; dùng `/proc` mà kiểm:

  ```bash
  for p in /proc/[0-9]*; do [ "$(cat $p/comm 2>/dev/null)" = pet_recon ] && echo $p; done
  ```
* **Phân biệt "đang tính" với "deadlock" bằng `utime`/`stime` trong
  `/proc/<pid>/stat`**, đừng đoán theo thời gian trôi. `utime=0 stime=0` + RSS
  15 MB = chưa làm gì. `wchan` cho biết nó chờ ở đâu.
* **Docker: `/etc/hosts` KHÔNG bake được.** `RUN ... >> /etc/hosts` trong
  Dockerfile fail exit 2 vì daemon bind-mount file đó lúc chạy. Phải dùng
  `--add-host` (đã có trong `run.sh`). Thiếu → chết ở `gethostbyname` trước mọi
  thứ khác.
* **Docker: `/usr/PET/systemConfig` PHẢI ghi được.** Config manager mở
  `local/cmcfg.xml` read-write; mount `:ro` thì chết ngay lúc khởi động với
  `ConfigManager():Error on trying to open XML file` → `InitializeCM():Error in
  Object creation` → exit 255. `d710:full` để nó ghi được trong image layer.
* **Hàm inline thì gdb không gọi được**: `Cannot evaluate function -- may be
  inlined` (gặp với `CRawDataMem::GetPromptsBuff`). Đọc field trực tiếp bằng
  `ptype /o` để lấy offset.

---

## 5. Vì sao `.job` map thẳng được vào `IgJobReq`

`IgJobReq` là biến static ở `cpcMain.cpp:108`, và **tên trường của nó trùng
từng chữ với nhãn `#...` trong file `.job`** — `emissionScatterFlag`,
`emissionRandomsFlag`, `normalizationFlag`, `emissionDeadTimeFlag`,
`hrActivityFactor`, `scatter3dPatientCutoff`, `modelScatterMultiplesNorm`…
`job2gdb.py` khớp theo nhãn nên sai lệch thứ tự sẽ báo lỗi ngay chứ không âm
thầm ghi nhầm trường.

Nhóm cuối (`cmpProcessingPacketID`, `sliceNumber`, `fileRead/Write3dOverlap`,
`cmpPacketDataType`, `wellCounterValue`) thuộc sub-struct `cmpPackets[100]`;
file `.job` liệt kê phẳng, lặp một cụm cho mỗi packet, nên `job2gdb.py` đánh
lại chỉ số mỗi lần gặp `cmpProcessingPacketID`. Job XR mẫu có 47 packet, khớp
`numberOfProcessingPackets = 47`.
