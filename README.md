# D710

Tái tạo PET cho GE Discovery 710 từ raw RDF của chính máy.

Mô hình là `y = S·(G x) + b`. Cả bốn số hạng hiệu chỉnh — randoms, scatter,
normalisation, dead time — **lấy bằng chính kernel của GE** (`pet_recon` chạy
dưới gdb trong container), không tự dựng lại bằng Python. Suy giảm là số hạng
duy nhất dựng ở đây, từ CT.

## Chạy

```bash
docker load -i d710_full.tar               # image bàn giao nguyên con, một lần
export D710_OUT=~/UET/d710_out             # ĐẦU RA ĐI ĐÂU — không có mặc định

D710/d710 exam --raw ~/Documents/11082026/petRDFS/NQLHQWPU/SYQGRVWD/QONDOBON \
               --ct  ~/Documents/11082026/PESI/p1/e1/s2 \
               --case ped

conda activate petct_reconstruction        # từ đây trở đi mới cần SIRF
D710/d710 osem   --case ped
D710/d710 export --case ped --format both
```

`d710 exam` = `decode` + `estimate` + `tostir` cho **mọi bed**, bỏ qua bed đã
xong, nên chạy lại sau khi hỏng giữa chừng là an toàn (`--force` để làm lại).

| lệnh | làm gì | chạy ở đâu |
|---|---|---|
| `d710 decode` | RDF → Interfile + singles (+ bảng sự kiện `bed<n>.lm.npy`) | container |
| `d710 estimate` | kernel GE → `randoms/scatter/normdt/norm_only.f32` | host điều phối, mọi bước con trong container |
| `d710 tostir` | `.f32` → Interfile STIR, tự kiểm bit-exact | container |
| `d710 exam` | cả ba, mọi bed | ↑ |
| `d710 attn` | CT → `work/bed<n>/attn.hs` | **SIRF** |
| `d710 osem` | OSEM từng bed + ghép trục → `recon.npz` | **SIRF** |
| `d710 export` | Bq/mL + SUV → NIfTI/DICOM (`--lm` cho `recon_lm.npz`) | **SIRF** env |
| `d710 lm` | LM-OSEM list-mode → `recon_lm.npz` | **PyTomography**, KHÔNG cần SIRF |
| `d710 lowdose` | bản liều thấp của một ca | numpy thuần |
| `d710 read` | đọc một `.f32` của vendor | container |
| `d710 shell` | shell tương tác trong image | container |

## Hai runtime, tách hẳn nhau

| | SIRF/STIR | PyTomography |
|---|---|---|
| ở đâu | image `sirf-local:0.1`, gọi qua `./d710_isolate_stir.sh` | conda env `petct_reconstruction` |
| lệnh | `attn`, `osem`, `export` | `lm`, `lowdose` |

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

Bước 1–3 chỉ cần **bash + docker + python3 trần** trên host. Không conda, không
numpy, không pydicom, không i386 multiarch, không checkout `custom_tool/`. Chỉ
`osem` và `export` cần môi trường project, vì SIRF không có trong image.

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
`tests/test_notebook_contract.py` dựng lại đúng `y = S·(Gx) + b` trên máy quét
thu nhỏ và so với `S`, `b` đã biết.

## Kiểm

```bash
conda activate petct_reconstruction
export D710_OUT=~/UET/d710_out
python -m pytest -q            # hoặc tests/run_tests.sh
```

Phần tổng hợp chạy trên một máy quét thu nhỏ nên không cần dữ liệu. Phần dữ
liệu thật đọc `$D710_OUT/<ca>/` và tự skip khi chưa dựng — kể cả khi
`$D710_OUT` chưa đặt. Chi tiết: `tests/README.md`.

**Notebook không được chứa mã.** `test_notebook_contract.py` fail nếu một code
cell định nghĩa `def`/`class` hay dài quá 15 câu lệnh. Ràng buộc bằng máy cho
một chuyện đã xảy ra thật: `utils/` từng bị chép vào notebook rồi hai bản lệch
nhau, và cả hai vẫn chạy.

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
| **hằng số `K`** | **CHƯA** — nhưng **một `K` cho cả hai đường**: thang lệch 0,60 trước đây là do voxel lệch, không phải vật lý |

**`K` là việc còn lại duy nhất.** Ảnh ra là count/voxel, chưa phải Bq/mL. Không
có WCC nào được áp trong `vendor/`, nên thang tuyệt đối phải tự đo trên NEMA:
chạy `d710 exam --case nema`, đo trên vùng nền có nồng độ biết trước. `K` đo
xong **chỉ đúng cho đúng chuỗi hiệu chỉnh này và đúng bước voxel 2,1306 mm** —
projector tích luỹ theo bước voxel chứ không theo thể tích, nên đo ở 2,1306 mm
rồi dùng ở 1,3672 mm sẽ đọc cao 1,56×.

Chi tiết: **`vendor/README.md`** (tài liệu chính), `vendor/PARAMS.md` (tham số
sống đọc từ tiến trình), `vendor/cal/README.md` (hiệu chuẩn),
`decode/README.md` (giải mã), `tests/README.md` (kiểm).
