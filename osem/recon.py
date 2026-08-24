"""OSEM cho một bed: dựng `y = S·(G x) + b` rồi lặp.

Ba đường vào, **không hoán đổi được**:

| | file | gắn thế nào |
|---|---|---|
| `y` prompt thô | `decoded/bed<n>.hs` | `recon.set_input` |
| `S` | `work/bed<n>/normdt.hs` × af | `set_acquisition_sensitivity` **trước** `set_up` |
| `b` | `work/bed<n>/background.hs` | `set_background_term` |

`S` phải gắn **trước** `set_up` để STIR gộp nó vào sensitivity image — đó là cái
làm phép hiệu chỉnh mang tính định lượng chứ không chỉ đánh trọng số lại. `b` đi
**vòng qua** `S` vì randoms và scatter đã nằm sẵn trong miền count đo được;
`tests/test_notebook_contract.py` dựng lại đúng đồng nhất thức đó trên máy quét
thu nhỏ.

⚠ **Đừng nhân `geometry.ring_pair_multiplicity()` vào đây.** `normdt` của GE đã
mang sẵn bội số span-2; nhân thêm là bình phương nó (4× ở bin lẻ). Xem docstring
của hàm đó.

Nguồn API, không tự bịa — ví dụ chính thức của SIRF ở
`$CONDA_PREFIX/dlevel/build/sources/SIRF/examples/Python/PET/`:
`osem_reconstruction.py` (`make_Poisson_loglikelihood` + `OSMAPOSLReconstructor`),
`get_multiplicative_sinogram.py` (`AcquisitionSensitivityModel`),
`listmode_reconstruction.py` (sensitivity + background cùng nhau). Không ví dụ
nào ghép **cả bốn** số hạng trên sinogram thật; chỗ ghép là ở đây.
"""

from __future__ import annotations

import numpy as np

from utils import terms

#: 288 view, nên số subset phải là ước của 288.
N_SUBSETS = 12
N_ITERATIONS = 3

#: Mặc định 1 LOR/bin là quá thô cho hình học này.
TANGENTIAL_LORS = 5

#: Số voxel ngang. STIR ghim bước voxel ở 2,1306 mm bất kể giá trị này, nên nó
#: quyết định **FOV** chứ không quyết định độ phân giải.
XY = 328


def image_grid(case, bed: int, xy: int = XY):
    """Lưới ảnh dùng chung cho mọi bed — cùng máy, cùng hình học.

    Trả `(acq_template, image_template)`. Phải giữ `acq_template` sống: nó là
    nguồn của ExamInfo cho mọi thứ dựng từ nó.
    """
    import sirf.STIR as pet

    y0 = pet.AcquisitionData(str(case.prompt(bed)))
    x0 = y0.create_uniform_image(1.0, xy)
    # `attenuation.mu_image` đòi đúng lưới 47 plane = 2·24 − 1 của một bed.
    if x0.as_array().shape[0] != terms.NSEG0:
        raise SystemExit("error: lưới ảnh có %d plane, phải là %d"
                         % (x0.as_array().shape[0], terms.NSEG0))
    return y0, x0


def acquisition_model(objs, sensitivity, image, tangential_lors=TANGENTIAL_LORS):
    """`y = S·(G x) + b`, dựng đúng thứ tự STIR đòi."""
    import sirf.STIR as pet

    am = pet.AcquisitionModelUsingRayTracingMatrix()
    am.set_num_tangential_LORs(tangential_lors)
    # TRƯỚC set_up: đó là cái làm S đi vào sensitivity image.
    am.set_acquisition_sensitivity(pet.AcquisitionSensitivityModel(sensitivity))
    am.set_background_term(objs["background"])
    am.set_up(objs["prompts"], image)
    return am


def reconstruct(case, bed: int, af, image, n_sub: int = N_SUBSETS,
                n_it: int = N_ITERATIONS, xy: int = XY):
    """OSEM cho một bed. Trả `(ảnh, sensitivity)`, cả hai `(47, xy, xy)`.

    `sensitivity` là **sensitivity image của chính STIR** — mẫu số mà OSEM chia
    vào mỗi vòng lặp, tức backprojection của `S` qua toàn bộ bin. Nó đã gồm sẵn
    norm, dead time, suy giảm VÀ cách projector lấy mẫu LOR thật, nên dùng nó
    làm trọng số ghép bed là *đo* chứ không phải *giả định hình học*. Lấy nó
    không tốn thêm gì: `set_up` vốn đã tính rồi.

    Nạp lại bed từ đĩa rồi trả RAM khi xong, để chạy sáu bed không cần giữ sáu
    bộ sinogram trong bộ nhớ (~2,5 GB thay vì ~15 GB).
    """
    import sirf.STIR as pet

    objs, A = terms.load(case, bed, af=af)

    S = objs["prompts"].get_uniform_copy(0)
    S.fill(A["sensitivity"])
    del A                       # mảng numpy không cần nữa; S đã giữ bản sao

    am = acquisition_model(objs, S, image)

    obj = pet.make_Poisson_loglikelihood(objs["prompts"], acq_model=am)
    obj.set_num_subsets(n_sub)
    obj.set_up(image)
    # Cộng qua mọi subset -> sensitivity đầy đủ của bed này.
    sens = sum(obj.get_subset_sensitivity(s).as_array() for s in range(n_sub))

    rec = pet.OSMAPOSLReconstructor()
    rec.set_objective_function(obj)
    rec.set_num_subsets(n_sub)
    rec.set_num_subiterations(n_sub * n_it)
    rec.set_input(objs["prompts"])
    rec.set_up(image)           # tính sensitivity image — chậm nhất ở đây
    rec.set_current_estimate(image)
    for _ in range(rec.get_num_subiterations()):
        rec.update_current_estimate()

    out = rec.get_current_estimate().as_array().copy()
    del objs
    return out, sens.astype(np.float32)


def reconstruct_all(case, beds, af: dict, image, out=print, **kw):
    """`reconstruct` cho mọi bed, có đồng hồ. Trả `(img, sens)`, hai dict.

    `sens rìa/giữa` in ra mỗi bed là thứ đáng liếc: nó cho thấy độ nhạy tụt bao
    nhiêu ở đầu bed, tức là vùng chồng khi ghép yếu tới đâu.
    """
    import time

    img, sens = {}, {}
    for n in beds:
        t0 = time.time()
        img[n], sens[n] = reconstruct(case, n, af[n], image, **kw)
        sp = sens[n].mean(axis=(1, 2))
        out(f"bed {n}: {time.time() - t0:5.0f} s   max {img[n].max():9.4g}   "
            f"mean {img[n].mean():9.4g}   sens rìa/giữa {sp[0] / sp.max():.4f}")
    return img, sens
