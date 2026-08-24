"""Hình học sinogram D710 — đọc từ header STIR, không chép lại thành bảng.

Ba hằng số ở đây đã được kiểm **bit-exact** với sinogram do chính máy giải mã
(histogram list-mode NEMA bed 2 tái tạo đúng từng bin, corr 1.000000). Chúng là
kết quả đã chứng minh, không phải lựa chọn kiến trúc, nên được mang sang nguyên:

* `crystal_to_det` — GE đánh số crystal ngược chiều STIR: `stir = (287 - ge) mod 576`
* `plane_ring_pairs` — dấu của ring difference theo `pos1 - pos2`
* segment 0 của span-2 gộp hai ring pair vào một plane lẻ
"""

from __future__ import annotations

import numpy as np

#: Khoảng cách plane = nửa khoảng cách ring.
PLANE_MM = 3.2699997

#: GE đánh số crystal ngược chiều STIR, lệch nửa vòng.
#: Hiệu chuẩn bằng cách histogram list-mode NEMA bed 2 qua cả 1152 phương án
#: rồi chấm với sinogram vendor giải mã của đúng bed đó (corr 0.990, á quân 0.981).
CRYSTAL_REVERSE = True
CRYSTAL_OFFSET = 288


def open_projdata(hs: str):
    """``(proj_data, info)`` — phải giữ proj_data sống, info là con trỏ mượn."""
    import stir

    pd = stir.ProjData.read_from_file(hs)
    return pd, pd.get_proj_data_info()


def segment_order(info) -> list[int]:
    """Thứ tự STIR lưu segment: 0, +1, -1, +2, -2, ..."""
    out = [0]
    for k in range(1, info.get_max_segment_num() + 1):
        out += [k, -k]
    return out


def plane_ring_pairs(info, num_rings: int) -> list[list[tuple[int, int]]]:
    """Các cặp ring được cộng vào mỗi plane, dạng ``(ring của pos1, ring của pos2)``.

    Plane tại axial position ``a`` của segment phủ ring difference ``[lo, hi]``
    chứa mọi cặp có tổng ring bằng ``z`` và hiệu nằm trong khoảng đó. Segment 0
    của span-2 phủ -1..+1, nên các ``z`` lẻ chứa **hai** cặp — chỗ gộp mà số hạng
    nền phải tự cộng, vì nó không đi qua sensitivity model.

    Cặp phát ra là ``(r + d, r)`` chứ không phải ``(r, r + d)``: segment có dấu
    của STIR theo ``pos1 - pos2``, ngược với cách đọc hiển nhiên.
    """
    out = []
    for s in segment_order(info):
        lo, hi = info.get_min_ring_difference(s), info.get_max_ring_difference(s)
        z0 = min(abs(d) for d in range(lo, hi + 1))
        for a in range(info.get_num_axial_poss(s)):
            z = z0 + a
            out.append([((z - d) // 2 + d, (z - d) // 2) for d in range(lo, hi + 1)
                        if (z - d) % 2 == 0 and 0 <= (z - d) // 2 < num_rings
                        and 0 <= (z - d) // 2 + d < num_rings])
    return out


def check_ring_pairs(info, pairs: list[list[tuple[int, int]]]) -> None:
    """Ném lỗi nếu cặp ring suy ra không khớp số đếm của chính STIR."""
    p = 0
    for s in segment_order(info):
        for a in range(info.get_num_axial_poss(s)):
            want = info.get_num_ring_pairs_for_segment_axial_pos_num(s, a)
            if len(pairs[p]) != want:
                raise ValueError(
                    f"plane {p} (segment {s}, axial {a}): suy ra "
                    f"{len(pairs[p])} cặp ring, STIR nói {want}")
            p += 1


def ring_pair_multiplicity(info) -> np.ndarray:
    """Số cặp ring mỗi plane gộp vào, dọc trục axial nối liền của STIR.

    ⚠ **ĐỪNG áp hàm này lên đường vendor (`vendor/to_stir.py` + `normdt`).**
    Đo 2026-08-22 trên ped bed 4, tỉ số lẻ/chẵn trong segment 0:

        prompts 1.985   randoms 1.981   scatter 2.007   background 1.986
        normdt  1.992   norm_only 1.993

    `normdt` của GE **đã mang sẵn** bội số này — nó là độ nhạy của cả *bin*, đã
    gộp chuyện bin đó nhận mấy cặp ring, chứ không phải hiệu suất thuần của một
    LOR. Nhân thêm `ring_pair_multiplicity` nữa là **bình phương** nó (4x ở bin
    lẻ). Chuỗi `y = S(Gx) + b` tự khớp: y, b, S đều 2x ở bin lẻ, projector bắn
    một LOR, nên `S·(Gx)` ra đúng 2x. Kiểm trên ảnh: công suất tại tần số
    Nyquist của profile trục còn 0.00-0.49 % (sinogram là 68.9 %).

    Hàm này còn ở đây cho pipeline **tự code cũ** (đã xoá), nơi randoms/scatter/
    norm dựng bằng tay và KHÔNG có bội số, nên phải nhét vào `asm` thủ công.

    **Đây là hình học, không phải normalisation đầu dò.** Segment 0 của span-2
    gộp ring difference +1 và -1 vào các axial position lẻ của nó, nên những bin
    đó thu **hai** LOR trong khi projector của STIR bắn **một**.

    Bỏ qua nó *khi số hạng nhân chưa có nó* thì đọng lại thành vằn chu kỳ 2 dọc
    trục — nằm đúng tần số Nyquist theo trục.
    """
    return np.concatenate([
        np.array([info.get_num_ring_pairs_for_segment_axial_pos_num(s, a)
                  for a in range(info.get_num_axial_poss(s))], dtype=np.float32)
        for s in segment_order(info)])


def det_pair_map(num_views: int, num_tang: int, num_det: int):
    """``(view, tangential) -> (det1, det2)``, hai mảng ``(num_views, num_tang)``."""
    v = np.arange(num_views)[:, None]
    t = (np.arange(num_tang) - num_tang // 2)[None, :]
    d1 = (v + np.floor_divide(t, 2)) % num_det
    d2 = (v - (-np.floor_divide(-t, 2)) + num_views) % num_det
    return d1.astype(np.int32), d2.astype(np.int32)


def crystal_to_det(num_det: int, offset: int = CRYSTAL_OFFSET,
                   reverse: bool = CRYSTAL_REVERSE) -> np.ndarray:
    """Bảng tra: chỉ số crystal ngang của GE -> số hiệu detector của STIR."""
    d = np.arange(num_det)
    return np.roll(d[::-1] if reverse else d, offset)


def tangential_s_mm(hs: str) -> np.ndarray:
    """Độ lệch xuyên tâm ``s`` của từng bin ngang, mm, lấy từ header.

    Không arc-correct, nên bin **không** cách đều: trên máy này chạy
    -356.7 .. +356.7 mm qua 381 bin, 2.261 mm ở giữa và khít dần ra rìa. Cái gì
    cần khoảng cách theo mm thì phải hỏi hàm này, đừng nhân với bề rộng danh nghĩa.
    """
    import stir

    pd = stir.ProjData.read_from_file(hs)
    info = pd.get_proj_data_info()
    lo = info.get_min_tangential_pos_num()
    return np.array([info.get_s(stir.Bin(0, 0, 0, lo + t))
                     for t in range(info.get_num_tangential_poss())])
