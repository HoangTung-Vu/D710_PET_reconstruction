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
| `terms.py` | nạp số hạng của một bed, bảng tóm tắt, bảng bất biến |
| `attn.py` | `af` theo bed, cache vào `work/bed<n>/attn.hs` |
| `sirf_env.py` | chdir vào scratch + giữ `MessageRedirector` sống |
| `quant.py` | count/voxel → Bq/mL → SUV; hằng số `K` |
| `export.py` | ghi NIfTI / DICOM (`python3 -m utils.export` là `d710 export`) |
| `plots.py` | hình cho notebook |

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
