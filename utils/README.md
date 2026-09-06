# utils

Mọi thứ **không** thuộc về một thuật toán tái tạo cụ thể. Thuật toán ở
`osem/`; thuật toán sau (FBP, MLEM, deep prior…) tạo package riêng cùng cấp và
dùng lại đúng những module này.

Luật một dòng: **nếu một hàm chỉ có nghĩa với OSEM thì chỗ của nó là `osem/`.**

| module | dùng ở đâu |
|---|---|
| `paths.py` | `$D710_OUT/<ca>/...`; **chỗ duy nhất** biết cây đầu ra |
| `container.py` | **chỗ duy nhất** biết cách gọi `docker` từ Python |
| `attenuation.py` | CT DICOM → mu-map (`load`, `hu_to_mu`, `mu_image`, `factors`) |
| `geometry.py` | quy ước chỉ số bin D710→STIR (`PLANE_MM`, `crystal_to_det`, `plane_ring_pairs`) |
| `terms.py` | nạp số hạng của một bed (+ bảng tóm tắt / bất biến, xem dưới) |
| `attn.py` | `af` theo bed, cache vào `work/bed<n>/attn.hs` |
| `sirf_env.py` | chdir vào scratch + giữ `MessageRedirector` sống |
| `quant.py` | count/voxel → Bq/mL → SUV; hằng số `K` |
| `export.py` | ghi NIfTI / DICOM (`python3 -m utils.export` là `d710 export`) |
| `plots.py` | hình để xem sinogram/ảnh — **không có chỗ nào gọi**, xem dưới |
| `scanner.py` | MỌI hằng số máy + lưới ảnh, một chỗ duy nhất |

## Mã còn đó nhưng hiện không ai gọi

Cây từng có `osem_pipeline.ipynb`; nó đã bị xoá, còn phần `utils/` phục vụ nó
thì **giữ lại nguyên**. Liệt kê ở đây để không ai phải tra lại bằng grep, và để
không nhầm là sót:

| ký hiệu | vốn để làm gì |
|---|---|
| `plots.py` (cả module) | hình cho notebook; `slices` + `busiest_plane` chỉ `terms.collect` gọi |
| `terms.collect`, `bed_table`, `invariant_table`, `invariants`, `summarise` | các ô bảng/bất biến của notebook — nay `tests/test_pipeline_data.py` làm việc đó |
| `quant.suv_table`, `suv_bsa`, `bsa_m2`, `body_mask`, `voxel_ml` | nhánh SUV theo diện tích da, treo dưới `suv_table` |
| `geometry.open_projdata` | chỉ `tests/test_geometry.py` dùng |
| `geometry.ring_pair_multiplicity` | **không phải mã chết**: là oracle của `tests/test_lm_geom.py`, và đúng đắn của nó thể hiện bằng việc *không ai gọi* |
| `osem.stitch.plane_index` | tra chỉ số plane, không còn nơi dùng |

Kiểm lại danh sách này bằng cách đếm tham chiếu qua AST trên toàn cây, không
phải bằng grep tên hàm — nhiều tên ở đây (`collect`, `slices`) là từ thông
dụng.

Bốn số hạng hiệu chỉnh **không** dựng ở đây — lấy thẳng từ kernel của GE:

```bash
d710 estimate --raw <thư mục petRDFS> --ct <thư mục CT DICOM> --case <ca>
```

## Hai cái bẫy ghi lại ở đây

⚠ `geometry.ring_pair_multiplicity()` **không** dùng cho đường vendor:
`normdt` của GE đã mang sẵn bội số span-2, nhân thêm là bình phương nó. Xem
docstring của hàm và `tests/test_pipeline_data.py`.

⚠ `attenuation.to_radiological()` **tự nghịch đảo**. `mu_image` gọi nó (STIR
lật y so với DICOM), `export.to_dicom_order` gọi lại chính nó để hoàn tác. Đừng
viết lại phép lật ở chỗ thứ ba.

## Vì sao `container.py` tồn tại

`D710/` từng đi ngược lên `../../custom_tool/` để lấy bộ giải mã và cây hiệu
chuẩn. Cả hai đã có sẵn trong image:

| trước ở host | trong image |
|---|---|
| `custom_tool/ge_rdf_tool.py` | `/opt/custom_tool/ge_rdf_tool.py` |
| `custom_tool/petsw/.../cal/*.3dnorm` | `/usr/PET/systemConfig/cal/*.3dnorm` |
| `.../cal/*.3dwcc` | cùng thư mục đó |

Nên `D710/` **không còn tham chiếu nào tới `custom_tool/`**, và bước decode /
estimate / tostir chỉ cần bash + docker + python3 stdlib. Một cửa duy nhất
cũng có nghĩa là test chỉ phải stub một chỗ.
