# D710

Tái tạo PET cho GE Discovery 710 từ raw RDF của chính máy.

Mô hình là `y = S·(G x) + b`. Cả bốn số hạng hiệu chỉnh — randoms, scatter,
normalisation, dead time — **lấy bằng chính kernel của GE** (`pet_recon` chạy
dưới gdb trong container), không tự dựng lại bằng Python. Suy giảm là số hạng
duy nhất dựng ở đây, từ CT.

## Chạy

Chuẩn bị một lần:

```bash
docker load -i d710_full.tar                      # image của hãng, nếu chưa có
cd D710 && conda env create -f environment.yml    # env host, tên `petct_recon`
```

**Môi trường host phải là conda, và thủ phạm là `parallelproj`** — gói mà
`d710 lm` chạm tới qua PyTomography. Nó **không có trên PyPI** (404), và
`pytomography` **cũng không khai nó** trong `Requires-Dist`, kể cả extra. Nên
không resolver nào biết nó tồn tại: `pip`/`uv` cho ra một PyTomography import
được nhưng tái tạo thì không — import là lười, nằm trong
`pytomography.projectors.PET`, chỉ `d710 lm recon` mới chạm tới.

Vì không ai khai nên **không ai giữ trần version hộ mình**, và đó là chỗ chết
người: `parallelproj` 2.x là một gói khác đội cùng cái tên. Phần biên dịch dời
vào `parallelproj_core` — extension CPython riêng theo nền tảng, không còn nạp
bằng ctypes qua `$PARALLELPROJ_C_LIB` — và sáu hàm PyTomography gọi ở top level
(`joseph3d_fwd`, `joseph3d_back`, bốn `joseph3d_{fwd,back}_tof_{sino,lm}`) biến
mất khỏi namespace `parallelproj`: 2.0.2 đổi tên bốn cái TOF và không export cái
nào, `__all__` còn đúng sáu tên metadata. Không ghim thì lần dựng env kế tiếp ra
2.0.2 và **cả sáu ném `AttributeError`**. `environment.yml` ghim **1.10.2**, bản
1.x cuối cùng. Thượng nguồn không có bản vá: 3.4.0 là PyTomography mới nhất.

Dựng trọn một ca, **cả hai đường tái tạo**. Chạy từ trong `D710/`:

```bash
export D710_OUT=~/UET/d710_out         # ĐẦU RA ĐI ĐÂU — không có mặc định
conda activate petct_recon             # hoặc: export D710_PYTHON=<python của env>
CASE=fdg26081901
SRC=~/UET/Handson_PET_CT_Reconstruction/data/cases/20260819_FDG26081901_ok

# 1. RDF -> sinogram + bảng sự kiện + bốn số hạng của GE  (decode+estimate+tostir)
./d710 exam --case $CASE \
    --raw  $SRC/raw/petRDFS/NQLHXWDK/PZAMCDES/USIRBPEU \
    --ct   $SRC/dicom/CT_s002_CT_WB_AC_5mm \
    --listmode --lists $SRC/raw/petLists/NQLHXWDK/PZAMCDES/USIRBPEU

# 2. suy giảm từ CT  (SIRF; --ct tự đọc từ sidecar của bed)
./d710_isolate_stir.sh attn --case $CASE

# 3a. đường sinogram, non-TOF        -> recon.npz
./d710_isolate_stir.sh osem --case $CASE --resume

# 3b. đường list-mode, TOF đủ 55 bin -> recon_lm.npz  (TOF là mặc định ở đây)
#     cần conda: parallelproj không có trên PyPI
env D710_PYTHON=$HOME/miniconda3/envs/petct_recon/bin/python \
    ./d710 lm recon --case $CASE --resume

# 4. Bq/mL + SUV -> NIfTI + DICOM
./d710_isolate_stir.sh export --case $CASE --format both
./d710_isolate_stir.sh export --case $CASE --format both --lm
```

Chỉ bước 3b cần conda; SIRF ở bước 2/3a/4 đến từ image `sirf-local:0.1`, còn
bước 1 chạy bằng python3 nào cũng được. Bỏ `--listmode --lists` ở bước 1 nếu chỉ
cần đường sinogram, khi đó bỏ luôn 3b và lần `export --lm` — và **cả pipeline
sinogram không cần conda ở đâu cả**.

`exam` bỏ qua bed đã xong nên chạy lại sau khi hỏng giữa chừng là an toàn;
`--force` để làm lại từ đầu. `osem`/`lm recon` cũng vậy với `--resume`, và chúng
tự nhận ra thiết lập đã đổi thì dựng lại bed đó chứ không dùng lại nhầm.

| lệnh | làm gì | chạy ở đâu |
|---|---|---|
| `d710 decode` | RDF → Interfile + singles (+ bảng sự kiện `bed<n>.lm.npy`) | `d710:full` |
| `d710 estimate` | kernel GE → `randoms/scatter/normdt/norm_only.f32` | `d710:full` |
| `d710 tostir` | `.f32` → Interfile STIR, tự kiểm bit-exact | `d710:full` |
| `d710 exam` | cả ba, mọi bed | ↑ |
| `d710 attn` | CT → `work/bed<n>/attn.hs` | `sirf-local:0.1` |
| `d710 osem` | OSEM từng bed + ghép trục → `recon.npz` | `sirf-local:0.1` |
| `d710 export` | Bq/mL + SUV → NIfTI/DICOM (`--lm` cho `recon_lm.npz`) | `sirf-local:0.1` |
| `d710 lm` | LM-OSEM list-mode → `recon_lm.npz` | python của host |
| `d710 lowdose` | bản liều thấp của một ca | python của host |
| `d710 read` | đọc một `.f32` của vendor | `d710:full` |
| `d710 shell` | shell tương tác trong image | `d710:full` |

`d710_isolate_stir.sh` chỉ nhận **`attn` / `osem` / `export`**; mọi lệnh khác nó
chuyển thẳng cho `./d710`, nên gõ nhầm wrapper không sai kết quả.

## Hai runtime, tách hẳn nhau

| | SIRF/STIR | PyTomography |
|---|---|---|
| ở đâu | image `sirf-local:0.1`, gọi qua `./d710_isolate_stir.sh` | python của host, env conda `petct_recon` |
| lệnh | `attn`, `osem`, `export` | `lm`, `lowdose` |

Và một chỗ dễ nhầm chết người: chữ **"parallelproj" trong repo này chỉ hai thứ
khác nhau**. Chúng khác soname nên nằm chung một env vẫn không đụng nhau:

| file | ai nạp | đến từ đâu |
|---|---|---|
| `lib/libparallelproj_c.so.1.10.2` | gói Python `parallelproj` → PyTomography (`d710 lm`) | conda-forge, nạp bằng ctypes |
| `dlevel/INSTALL/lib/libparallelproj.so.2.0.7` | STIR → `osem --projector parallelproj` | SIRF-SuperBuild tự dựng, dùng API C++ |

Nên "parallelproj 2.x hỏng" ở mục trên **chỉ nói về cột thứ nhất**. Bản 2.0.7 mà
STIR link vào là thư viện C++, không đi qua namespace Python, không liên quan.

`lm/` và `lowdose/` **không import `sirf` hay `stir`** ở bất kỳ đâu: layout
segment đọc thẳng từ header (`lm/interfile.py`), mọi số hạng đọc bằng
`np.fromfile`. Đổi lại, `utils/attn.py` giờ ghi `attn.hs` bằng header clone từ
prompts — cùng một layout với mọi file khác trong `work/bed<n>/` — thay vì
layout riêng của SIRF. **File `attn.hs` cũ phải dựng lại:**
`./d710_isolate_stir.sh attn --case <ca> --force`.

### Một lưới ảnh duy nhất, cho cả hai runtime

Hai bản SIRF hiểu `--xy` **khác nhau**: bản trong `sirf-local:0.1` khoá FOV ở
718,01 mm rồi cho voxel chạy theo `xy`, bản trên host khoá voxel ở 2,1306 mm rồi
cho FOV chạy theo. Cùng `--xy 256` ra 2,8047 mm trong container và 2,1306 mm trên
host — hai thang khác nhau, và `K` tỉ lệ nghịch với thể tích voxel.

Mọi hằng số hình học và cấu hình máy giờ nằm ở **`utils/scanner.py`**, một chỗ
duy nhất. Mặc định `XY = 337` được **đo** chứ không chọn: đó là kích thước ma
trận duy nhất cho ra 2,130600 mm ở **cả hai** bản (FOV 718,01 mm). `scanner.sirf_grid`
kiểm lại lúc chạy và tự chỉnh `xy` nếu bản SIRF hiện tại sẽ cho voxel khác.

**Mọi `recon.npz` / `lm.npz` / `recon_lm.npz` dựng ở lưới cũ phải chạy lại.**

Chi tiết: `lm/README.md`, `lowdose/README.md`.

Bước 1–3 chỉ cần **bash + docker + một python3 bất kỳ** trên host. Không conda,
không numpy, không pydicom, không i386 multiarch, không checkout `custom_tool/`.
Chỉ `osem`, `attn`, `export`, `lm`, `lowdose` cần môi trường project, vì SIRF và
PyTomography không có trong image.

**`d710` không giả định `python3` là trình thông dịch đúng.** Trên máy có
`/usr/bin` đứng trước conda trong `PATH` — trường hợp phổ biến — thì `python3`
là python hệ thống *ngay cả khi đã activate* `petct_recon`, còn `python`
mới là của conda. Nên `d710` thử lần lượt: `$D710_PYTHON` (nếu đặt thì dùng
đúng cái đó, sai thì báo lỗi chứ không lặng lẽ đổi), rồi `python3`, `python`,
`$VIRTUAL_ENV/bin/python`, `$CONDA_PREFIX/bin/python`, `/usr/bin/python3`,
`/usr/local/bin/python3`. Lệnh nào cần thêm gói mà không tìm được thì in ra
**toàn bộ danh sách đã thử**. Chạy được với env conda, với venv trần, hay không
có conda. `tests/test_python_resolution.py` chốt điều đó.

## Cài phụ thuộc Python — bằng conda, và bắt buộc phải thế

`environment.yml` là **bản export nguyên si** của env host (`conda env export`),
không phải danh sách viết tay. Nó ghim cả version lẫn build string, nên tái lập
đúng bản đã đo — và cũng vì thế chỉ dựng lại được trên **linux-64**.

```bash
conda env create -f environment.yml     # dựng env, tên `petct_recon`
conda activate petct_recon
pytest -q                               # chạy test trong đó
python -m utils.export --case ped --format nifti
```

`d710` tự tìm ra env này (`$CONDA_PREFIX/bin/python` nằm trong danh sách nó
thử), hoặc chỉ đích danh:

```bash
export D710_PYTHON=$HOME/miniconda3/envs/petct_recon/bin/python
```

**Vì sao không phải `uv`/`pip`.** Xem mục "Chạy": `parallelproj` không có trên
PyPI và không gói nào khai nó, nên đây là ràng buộc chứ không phải sở thích. Dự
án từng dùng `uv` với `pyproject.toml` + `uv.lock`; `uv sync` dựng được mọi thứ
*trừ* đúng cái gói làm nên `d710 lm`, nên cả hai file đã bị bỏ.

**`libparallelproj` bị ghim bản `cpu_*`.** Máy đo không có GPU, mà `parallelproj`
chỉ chọn CUDA khi thấy `nvidia-smi` trong `PATH`, nên bản CUDA (~1,2 GB so với
39 KB) không mua được gì. Chạy trên máy có GPU thì đổi sang `cuda129_*` hoặc
`cuda130_*`. Ngược lại `torch` trong file là bản PyPI mặc định, tức **có** kèm
wheel CUDA — bất đối xứng có chủ ý, vì máy tái lập có thể có GPU.

**`sirf` và `stir` KHÔNG có trong `environment.yml`, và cố ý như vậy.** Chúng
được build từ nguồn vào `$CONDA_PREFIX/dlevel/` của một env **khác**
(`petct_reconstruction`), là bản dựng C++ gắn với đúng thư viện của env đó và
cần `LD_LIBRARY_PATH` của env khi nạp — không file phụ thuộc nào tái tạo được ở
máy khác. Trong dùng thường ngày thì không cần: `attn` / `osem` / `export` chạy
bằng image `sirf-local:0.1` qua `./d710_isolate_stir.sh`. Thiếu chúng trên host
thì ba lệnh đó báo rõ đã thử những trình thông dịch nào rồi dừng; `decode`,
`estimate`, `tostir`, `exam`, `lm`, `lowdose` không ảnh hưởng.

> ⚠️ **`petct_recon` không phải `petct_reconstruction`.** Tên gần giống nhau
> nhưng là hai env tách hẳn: `petct_recon` là runtime host của `environment.yml`
> (PyTomography + parallelproj), còn `petct_reconstruction` là env dựng SIRF từ
> nguồn — `conda env export --from-history` cho thấy nó sinh ra để làm đúng việc
> đó (cmake, swig 4.2.1, gcc, boost, eigen, fftw, hdf5). **Đừng bao giờ**
> `conda env update --prune` lên `petct_reconstruction`: prune gỡ toolchain và
> phá bản SIRF trong `dlevel/`, dựng lại mất 30–60 phút.

## Đầu ra: `$D710_OUT`, không bao giờ nằm trong cây mã

`--out` > `$D710_OUT` > **lỗi**. Cố ý không có mặc định.

```
$D710_OUT/<ca>/
    decoded/        bed<n>.{hs,s,json,singles.npy,convert.log,prd}
    vendor/bed<n>/  {randoms,scatter,normdt,norm_only}.f32(+.json),
                    prompts.u16, singles.i32, dt_int.f32, dt_mux.f32,
                    job.gdb, extract.log, estimate.json, data/, ovl/
    work/bed<n>/    {randoms,scatter,background,normdt,norm_only,attn}.{hs,s},
                    to_stir.json
    recon.npz       khối đã ghép, count/voxel — cầu nối osem -> export
    recon_lm.npz    như trên, từ đường list-mode (`d710 lm recon`)
    export/         <ca>_bqml.nii.gz, <ca>_suvbw.nii.gz, dicom/
    scratch/        tmp_*.hs/.s của SIRF — xoá lúc nào cũng được
    logs/
```

Xoá một ca là `rm -rf $D710_OUT/<ca>`. Hai ca chạy song song được — mỗi bed
mount thư mục của riêng nó lên `/out`, không còn thư mục staging chung.

Layout cũ (`raw_prompt/`, `work/<ca>_bed<n>/`, `vendor/out/`) chuyển sang bằng
`tools/migrate_out.sh --from <cây cũ> --to $D710_OUT` (dry-run mặc định, chỉ
`mv`, không copy).

## Cây mã

```
d710              CLI, điểm vào DUY NHẤT
Dockerfile        ghi lại image chứa gì (image được bàn giao, không dựng lại)
environment.yml   env host `petct_recon`, bản `conda env export` nguyên si
decode/           vòng lặp per-bed chạy trong container
vendor/           trình điều khiển kernel của GE + tài liệu tham chiếu chính
osem/             THUẬT TOÁN OSEM trên sinogram, không gì khác
lm/               THUẬT TOÁN list-mode (PyTomography); xem lm/README.md
lowdose/          mô phỏng liều thấp bằng cách tỉa event; xem lowdose/README.md
utils/            mọi thứ không thuộc thuật toán, dùng chung
utils/scanner.py    MỌI hằng số hình học + cấu hình máy + lưới ảnh, một chỗ
tests/            kiểm các quy ước trên; xem tests/README.md
tools/            migrate_out.sh, lm_frame.py, tof_direction.py, ...
```

**Thuật toán sau này** — FBP, MLEM, deep prior — tạo package riêng **cùng cấp
với `osem/`** và dùng lại `utils/`. Đó là lý do `utils/` không được chứa gì
mang tính OSEM: nếu một hàm chỉ có nghĩa với OSEM thì chỗ của nó là `osem/`.

## Ba đường vào OSEM, không hoán đổi được

| | file | gắn thế nào |
|---|---|---|
| `y` prompt thô | `<ca>/decoded/bed<n>.hs` | `recon.set_input` |
| `S` | `<ca>/work/bed<n>/normdt.hs` × af | `set_acquisition_sensitivity` **trước** `set_up` |
| `b` | `<ca>/work/bed<n>/background.hs` | `set_background_term` |

`S` phải gắn **trước** `set_up` để STIR gộp vào sensitivity image — đó là cái
làm phép hiệu chỉnh mang tính định lượng chứ không chỉ đánh trọng số lại. `b` đi
**vòng qua** `S` vì randoms và scatter đã nằm sẵn trong miền count đo được.
`tests/test_forward_model.py` dựng lại đúng `y = S·(Gx) + b` trên máy quét
thu nhỏ và so với `S`, `b` đã biết. Hạn chót của `S` là `set_up` của **bộ tái
tạo**, không phải của acquisition model — đo trên SIRF 3.10.1, gắn trước hay sau
`am.set_up` cho ra sensitivity image giống hệt nhau.

## Kiểm

```bash
conda activate petct_recon
export D710_OUT=~/UET/d710_out
python -m pytest -q            # hoặc tests/run_tests.sh
```

Phần tổng hợp chạy trên một máy quét thu nhỏ nên không cần dữ liệu. Phần dữ
liệu thật đọc `$D710_OUT/<ca>/` và tự skip khi chưa dựng — kể cả khi
`$D710_OUT` chưa đặt. Chi tiết: `tests/README.md`.


## Bốn cái bẫy đã xử sẵn

Cả bốn đều bật ra khi chạy thật, không phải phòng xa:

1. **ExamInfo phải trùng nhau.** `to_stir.py` **clone header từ chính
   `bed<n>.hs`** (chỉ đổi tên data file, number format, bytes per pixel) nên mọi
   số hạng cùng ExamInfo theo cấu tạo. Header sinh mới thì lệch energy window,
   và STIR chỉ ném `BinNormalisation set-up with different ExamInfo` mãi về sau,
   trong `make_Poisson_loglikelihood`.
2. **`tmp_*.hs/.s` 231 MB mỗi cái.** SIRF ghi chúng vào **thư mục hiện hành**,
   mỗi `get_uniform_copy` một cặp, giữ tới khi object bị thu gom.
   `utils.sirf_env.setup()` chdir vào `<ca>/scratch`.
3. **Bất biến phải gộp theo plane.** Sinogram thô ~0,06 count/bin, nên `p < r`
   đúng ở ~82 % số bin chỉ vì nhiễu Poisson. So từng bin là vô nghĩa.
4. **`normdt` đã mang sẵn bội số span-2.** Nhân `ring_pair_multiplicity()` thêm
   lần nữa là bình phương nó (4× ở bin lẻ).

Thêm một chỗ không hiển nhiên: file do SIRF ghi (`attn.hs`) dùng **layout khác
hẳn** file giải mã — segment tăng dần và trục view đứng trước trục axial —
nhưng `as_array()` vẫn trả về cùng thứ tự plane. Pipeline nhân `normdt` với
`attn` dạng mảng numpy nên điều đó là bắt buộc.

## Bốn bất biến trên cả 6 bed của ca nhi

| bed | table mm | kcps | prompts | randoms | scatter | S/(T+S) | livetime | ΣR/delays |
|---|---|---|---|---|---|---|---|---|
| 1 | −767,7 | 208 | 18 759 294 | 7 974 248 | 3 466 187 | 0,321 | 0,9569 | 0,994 |
| 2 | −643,5 | 264 | 23 736 423 | 11 124 030 | 4 194 211 | 0,333 | 0,9493 | 0,994 |
| 3 | −519,2 | 338 | 30 371 564 | 15 019 874 | 4 894 771 | 0,319 | 0,9418 | 0,994 |
| 4 | −394,9 | 411 | 37 010 299 | 20 052 033 | 5 639 968 | 0,333 | 0,9339 | 0,994 |
| 5 | −270,7 | 486 | 43 774 817 | 24 193 882 | 7 299 855 | 0,373 | 0,9285 | 0,994 |
| 6 | −146,4 | 969 | 87 220 937 | 22 169 301 | 14 736 322 | 0,227 | 0,9323 | 0,994 |

`Σp ≥ Σr` và `Σs ≤ Σ(p−r)` **vi phạm 0,00 % số plane trên cả 6 bed**, và ánh xạ
bin bit-exact trên cả 6.

Trên NEMA bed 2 thì `Σs ≤ Σ(p−r)` vượt ở **11/553 plane**, và cả 11 đều nằm
trong bốn plane ngoài cùng của một segment — chỗ gộp ít cặp ring nhất, nên tail
fit của SSS có ít dữ liệu nhất. Phần vượt tổng cộng là 0,036 % lượng scatter của
bed. `tests/test_pipeline_data.py` chốt đúng hai điều đó.

**Livetime bám theo randoms, không bám theo prompts.** Bed 6 có tốc độ prompt
cao nhất (969 kcps) nhưng livetime *không* thấp nhất, vì randoms của nó (22,2 M)
thấp hơn bed 5 (24,2 M) — randoms tỉ lệ với singles², mà dead time ăn theo
singles. Đây là kiểm chéo tự nhiên cho chiều của `normdt`.

## Nguồn của pipeline OSEM

Dựa trên ví dụ chính thức của SIRF, không tự bịa API:

| ví dụ | lấy gì |
|---|---|
| `SIRF/examples/Python/PET/osem_reconstruction.py` | `make_Poisson_loglikelihood` + `OSMAPOSLReconstructor` |
| `.../get_multiplicative_sinogram.py` | `AcquisitionSensitivityModel`, cách ghép norm với suy giảm |
| `.../listmode_reconstruction.py` | `set_acquisition_sensitivity` + `set_background_term` cùng nhau |

Đường dẫn đầy đủ: `$CONDA_PREFIX/dlevel/build/sources/SIRF/examples/Python/PET/`.
Không có ví dụ nào của SIRF ghép **cả bốn** (prompt + randoms + scatter + norm +
CTAC) trên sinogram thật; `osem/` là chỗ ghép.

## Trạng thái

| khâu | trạng thái |
|---|---|
| sinogram + hình học + `.f32` → Interfile | **xong, bit-exact mỗi lần chạy** |
| randoms | **xong** — kernel GE, 12,3 % prompts (NEMA bed 2) |
| scatter (SSS) | **xong** — kernel GE, S/(T+S) = 32,9 % |
| normalisation | **xong** — norm 3D của chính máy, tự tra từ header exam |
| dead time | **xong** — `normdt/norm_only`; **phụ thuộc tốc độ đếm** |
| suy giảm CT | **xong** — hướng mu-map đã đo cả hai trục |
| chạy đủ mọi bed | **xong** — `d710 exam` |
| hiệu chỉnh phân rã + ghép trục | **xong** — quy về thời điểm tiêm, trọng số = sensitivity image |
| xuất DICOM + NIfTI | **xong** — `utils/export.py`, `Units = BQML` |
| **list-mode (PyTomography)** | **xong** — `lm/`, ánh xạ bin bit-exact cả 6 bed; **TOF đủ 55 bin chạy 2m01s/bed, NHANH HƠN non-TOF** |
| **FOV ngang** | **xong** — đĩa bán kính 356,7 mm áp vào ước lượng khởi tạo; trước đó 34% số đếm rơi ra góc lưới vuông |
| **mô phỏng liều thấp** | **xong** — `lowdose/`, kiểm nhị thức + bất biến theo plane |
| **lưới ảnh** | **xong** — `utils/scanner.py`, 337 × 2,1306 mm, giống hệt nhau ở cả hai runtime |
| **hằng số `K`** | **đã đo** — `K_EXPORT` (sinogram) và `K_EXPORT_LM` (list-mode), **HAI hằng số** chứ không phải một; **phải đo lại** sau khi sửa `MU_*_511` (2026-09-06) |

**`K` đã đo, nhưng đang phải đo lại.** Đo bằng cách so với chính bản tái tạo
BQML của GE trên 5 ca FDG (`tools/compare_vendor.py` → `tools/calib_k.py`), chứ
không phải trên NEMA. **Hai hằng số**, vì hai đường tái tạo không đặt cùng số
đếm vào một voxel: dùng chung một `K` lệch khoảng 2,1× (`utils/scanner.py`).

`K` **chỉ đúng cho đúng chuỗi hiệu chỉnh đã đo nó và đúng bước voxel 2,1306 mm**
— projector tích luỹ theo bước voxel chứ không theo thể tích, nên đo ở
2,1306 mm rồi dùng ở 1,3672 mm sẽ đọc cao 1,56×. Ngày 2026-09-06 `MU_WATER_511`
và `MU_BONE_511` được sửa về đúng giá trị máy khai báo (`cmcfg.XR.xml:264/267`),
tức suy giảm đã đổi, nên cả hai hằng số hiện tại **không còn hiệu lực**:
chạy lại `run_all_ok.sh` → `export_all_ok.sh` → `tools/calib_k.py`.

Chi tiết: **`vendor/README.md`** (tài liệu chính), `vendor/PARAMS.md` (tham số
sống đọc từ tiến trình), `vendor/cal/README.md` (hiệu chuẩn),
`decode/README.md` (giải mã), `tests/README.md` (kiểm).
