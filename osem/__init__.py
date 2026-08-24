"""OSEM — thuật toán, và không gì khác.

* `recon` — `y = S·(G x) + b` cho **một** bed, cộng sensitivity image của bed đó
* `stitch` — hiệu chỉnh phân rã rồi ghép các bed, trọng số = sensitivity image

Mọi thứ không thuộc thuật toán (đường dẫn, CT → suy giảm, đọc số hạng, định
lượng, xuất ảnh, vẽ hình) nằm ở `utils/` và được dùng chung. Thuật toán sau —
FBP, MLEM, deep prior — tạo package riêng cùng cấp với package này và dùng lại
đúng `utils/` đó.

    from utils.paths import case
    from utils import sirf_env, attn
    from osem import recon, stitch

    C = case("ped"); sirf_env.setup(C)
    beds = C.beds()
    y0, x0 = recon.image_grid(C, beds[0])
    at = attn.Attenuation(C, ct_dir, x0, y0)
    img = {n: recon.reconstruct(C, n, at.af(n), x0) for n in beds}
"""

from . import recon, stitch  # noqa: F401
