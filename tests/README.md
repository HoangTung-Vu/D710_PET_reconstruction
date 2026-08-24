# tests

```bash
conda activate petct_reconstruction        # SIRF/STIR chỉ nạp khi env đã activate
export D710_OUT=~/UET/d710_out             # để phần test dữ liệu thật tìm ra bed
cd D710

python -m pytest -q                        # tất cả
tests/run_tests.sh                         # + bảng môi trường và bed đang có
tests/run_tests.sh --no-data               # chỉ phần tổng hợp
tests/run_tests.sh --case nema             # chỉ bed của một ca
```

Không activate env thì `stir` / `sirf.STIR` không nạp được và các test cần
chúng **skip** kèm lý do — không có test nào hỏng vì thiếu môi trường.

## Máy quét thu nhỏ

Một bed thật là `553 × 288 × 381`: prompt 121 MB, mỗi số hạng hiệu chỉnh
231 MB. Không test nào nên vác chừng đó. Mọi quy tắc hình học ở đây là quy tắc
của **span 2**, không phải của 24 ring, nên `synth_hs.py` thu nhỏ máy mà giữ
nguyên quy tắc:

| | ring | đầu dò/ring | view | tang | plane |
|---|---|---|---|---|---|
| D710 thật | 24 | 576 | 288 | 381 | 553 |
| `mini_hs` | 6 | 16 | 8 | 9 | 31 |
| `bed24` | 24 | 48 | 24 | 9 | 553 |

`bed24` giữ đủ 24 ring vì `utils.attenuation.mu_image` đòi đúng lưới ảnh 47 plane
của một bed; nó chỉ thu nhỏ hai trục mà lưới đó không quan tâm.

`synth_ct.py` dựng series CT giả: hình trụ nước trong không khí, kèm một khối
đặc **lệch về +y**. Chỗ lệch đó là điều kiện cần — phantom đối xứng thì không
phân biệt được phép lật y với phép đồng nhất, mà đúng một phép lật y là toàn bộ
quy ước hướng của pipeline.

## Test cần dữ liệu thật

`test_pipeline_data.py` chạy trên bed đã giải mã trong
`$D710_OUT/<ca>/decoded/` và số hạng trong `$D710_OUT/<ca>/work/bed<n>/`. Cả
hai đều sinh từ dữ liệu bệnh nhân và nằm **ngoài cây mã hoàn toàn**, nên khi
chưa dựng — hoặc khi `$D710_OUT` chưa đặt — thì test **skip**. Dựng bằng:

```bash
d710 exam --raw <petRDFS/.../DIR> --ct <CT series> --case ped
```

Mỗi bed tìm thấy thành một tham số riêng (`ped-bed4`, `nema-bed2`, …). Đọc
sinogram bằng `memmap` chứ không qua `sirf.AcquisitionData`, nên cả 7 bed hiện
có chạy hết trong ~12 giây thay vì nuốt 8 GB RAM.

`D710_CASE=ped` thu hẹp về một ca. `D710_CT=<thư mục CT>` bật thêm phép kiểm
`FrameOfReferenceUID` của CT bằng `sop_instance_uid` của bed.

## Đường import

Khai trong `pytest.ini` (`pythonpath = . vendor`), không phải bằng `sys.path`
trong `conftest.py`:

* `.` — `utils` và `osem` là **package** thật: `from utils import attenuation`.
* `vendor` — `estimate.py`, `to_stir.py`… là **script**, chạy bằng
  `python3 vendor/x.py`, nên tự import lẫn nhau bằng tên trần. Test đặt chúng
  lên path đúng như vậy thay vì bịa ra một layout package mà mã thật không có.

`tests/` do pytest tự thêm — đó là cái làm `import interfile` chạy được.

## Notebook không được chứa mã

`test_notebook_contract.py` **fail** nếu một code cell của
`osem_pipeline.ipynb` định nghĩa `def`/`class`, hoặc dài quá 15 câu lệnh. Đây
là ràng buộc bằng máy cho một chuyện đã xảy ra thật: `utils/` từng bị chép vào
notebook rồi hai bản lệch nhau, và cả hai vẫn chạy — chỉ là không còn tính cùng
một thứ.

## Những chỗ test nhắm vào

Đều là chỗ mà **sai thì ra ảnh trông hợp lý**, không phải chỗ ném lỗi:

* **thứ tự bin GE→STIR** — `stir[0, plane, 287 − ge_view, u]`. Đảo view sai thì
  ảnh bị soi gương ngang, nhìn không ra.
* **span 2** — segment 0 gộp hai cặp ring vào plane lẻ. `normdt` đã gánh sẵn
  bội số này (`test_span_2_doubles_the_odd_planes_of_every_term`), nên nhân
  `ring_pair_multiplicity()` thêm lần nữa là bình phương nó.
* **chiều của `normdt`** — nó là *độ nhạy*, chia mới đúng.
  `test_the_sensitivity_multiplies_rather_than_divides` chốt chiều bằng chính
  SIRF, `test_dead_time_is_a_livetime_fraction` chốt bằng dữ liệu.
* **`b` đi vòng qua `S`** — `test_the_forward_model_is_s_times_gx_plus_b` dựng
  cả `y = S·(Gx) + b` rồi so với `S` và `b` đã biết.
* **thứ tự plane khi qua đĩa** — SIRF ghi header theo layout *khác hẳn* file
  giải mã (segment tăng dần, trục view đứng trước trục axial) nhưng
  `as_array()` vẫn trả về cùng một thứ tự. Notebook nhân `normdt` với `attn`
  dạng mảng numpy nên điều này là bắt buộc, và nó không hiển nhiên.
* **một phép lật y duy nhất** — CT vào qua `mu_image` (lật), ra qua
  `write_dicom` (lật lại); `test_a_ct_feature_comes_back_at_the_same_patient_coordinate`
  đi trọn vòng và so bằng mm.
* **đơn vị mu** — STIR dùng 1/cm, PIFA dùng 1/mm. Nhầm là sai hệ số 10 mà không
  có gì báo.

## Quy ước

Một test một hành vi, mỗi test ngắn. Không xfail. Con số nào đã *đo được* thì
ghi kèm ngày đo trong docstring, để lần sau ai đọc còn biết nó là kết quả chứ
không phải giá trị ai đó chọn cho vừa.
