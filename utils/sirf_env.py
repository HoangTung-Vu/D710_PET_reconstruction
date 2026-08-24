"""Hai cái bẫy của SIRF phải xử **trước** khi chạm vào bất kỳ `AcquisitionData` nào.

Cả hai đều đã bật ra khi chạy thật, không phải phòng xa:

1. **`tmp_*.hs`/`.s`, 231 MB mỗi cặp.** SIRF ghi chúng vào **thư mục hiện hành**
   của tiến trình — không phải cạnh file nguồn — và mỗi `get_uniform_copy` sinh
   một cặp, giữ tới khi object bị thu gom. Lần chạy đầu để lại 926 MB ngay cạnh
   notebook. Nên: `chdir` vào `<ca>/scratch`.
2. **`MessageRedirector` phải được GIỮ tham chiếu.** Thả ra là nó bị thu gom
   ngay và STIR đổ hàng nghìn dòng INFO ra stdout. `setup()` giữ hộ.

Và một cái thứ ba mà `chdir` tự tạo ra: mục `''` trong `sys.path` (Jupyter và
`python script.py` đều chèn) được giải nghĩa **tại thời điểm import**, nên sau
`chdir` nó trỏ vào `scratch/` và `import osem` gãy giữa chừng notebook.
`setup()` neo nó thành đường tuyệt đối trước khi đổi thư mục.
"""

from __future__ import annotations

import os
import sys

from .paths import ROOT, Case

#: Giữ `MessageRedirector` sống suốt đời tiến trình. Đây là toàn bộ lý do nó tồn tại.
_redirector = None


def anchor_sys_path() -> None:
    """Biến mọi mục tương đối trong `sys.path` thành tuyệt đối, và bảo đảm có `D710/`.

    Gọi trước bất kỳ `chdir` nào. Không gọi thì `import utils` / `import osem`
    sau `chdir` sẽ đi tìm trong `scratch/`.
    """
    root = str(ROOT)
    sys.path[:] = [os.path.abspath(p) if p in ("", ".") else p for p in sys.path]
    if root not in sys.path:
        sys.path.insert(0, root)


def setup(case: Case, quiet: bool = True):
    """Vào `<ca>/scratch`, chặn INFO của STIR. Trả về thư mục scratch.

    Idempotent: gọi lại trong cùng tiến trình không tạo thêm redirector nào.
    """
    global _redirector

    anchor_sys_path()
    case.scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(case.scratch)

    if quiet and _redirector is None:
        import sirf.STIR as pet

        # Ba file này rơi vào scratch, cùng chỗ với tmp_*.hs — cùng số phận.
        _redirector = pet.MessageRedirector("info.txt", "warn.txt", "err.txt")
    return case.scratch


def clear_scratch(case: Case) -> int:
    """Xoá `tmp_*` trong scratch. Trả số file đã xoá.

    An toàn giữa hai lần chạy, KHÔNG an toàn khi đang có `AcquisitionData` sống:
    SIRF vẫn giữ file mở, xoá xong là object rỗng.
    """
    n = 0
    if not case.scratch.is_dir():
        return 0
    for p in case.scratch.glob("tmp_*"):
        p.unlink()
        n += 1
    return n
