"""Nơi mọi thứ sinh ra lúc chạy được đặt — và chỗ DUY NHẤT biết điều đó.

Cây mã (`D710/`) chỉ chứa mã. Mọi file sinh ra lúc chạy đi vào một gốc do người
chạy chỉ định:

    $D710_OUT/<ca>/
        decoded/       bed<n>.{hs,s,json,singles.npy,...}   <- d710 decode
        vendor/bed<n>/ {randoms,scatter,normdt,norm_only}.f32, prompts.u16, ...
        work/bed<n>/   {randoms,scatter,background,normdt,norm_only,attn}.{hs,s}
        export/        <ca>_bqml.nii.gz, <ca>_suvbw.nii.gz, dicom/
        scratch/       tmp_*.hs/.s của SIRF — xoá lúc nào cũng được
        logs/
        recon.npz      thể tích đã ghép (count/voxel), cầu nối osem -> export

**Không có gốc mặc định.** Thiếu cả `--out` lẫn `$D710_OUT` là lỗi, chứ không
rơi về thư mục mã — đó chính là cái refactor này dọn đi.

Ca lồng (`<ca>/vendor/bed<n>`) chứ không phẳng (`<ca>_bed<n>`): xoá một ca là
`rm -rf <ca>`, và tên ca không còn phải né ký tự `_`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: Gốc cây mã — thư mục `D710/`. Suy từ `__file__`, nên đúng dù chạy từ đâu.
ROOT = Path(__file__).resolve().parent.parent

_BED_RE = re.compile(r"bed(\d+)$")


class NoOutputRoot(SystemExit):
    """Không biết ghi vào đâu. Là `SystemExit` để script chết gọn, không traceback."""

    def __init__(self) -> None:
        super().__init__(
            "error: không biết ghi đầu ra vào đâu.\n"
            "  đặt  export D710_OUT=~/UET/d710_out\n"
            "  hoặc truyền  --out <thư mục>\n"
            "  (cố ý không có mặc định: đầu ra không bao giờ được rơi vào cây mã)")


def out_root(explicit: str | os.PathLike | None = None) -> Path:
    """`--out` > `$D710_OUT` > lỗi."""
    p = explicit or os.environ.get("D710_OUT")
    if not p:
        raise NoOutputRoot()
    return Path(os.path.expanduser(str(p))).resolve()


class Case:
    """Một ca chụp, và mọi thư mục của nó. Chỉ là đường dẫn — không đọc đĩa."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root / name

    def __repr__(self) -> str:
        return f"Case({self.name!r}, {self.root})"

    # ---------------------------------------------------------- thư mục
    @property
    def decoded(self) -> Path:
        return self.root / "decoded"

    @property
    def vendor(self) -> Path:
        return self.root / "vendor"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def recon(self) -> Path:
        """Thể tích đã ghép, count/voxel đã quy về thời điểm tiêm.

        Nằm ở gốc ca chứ không trong `export/`: nó là **đầu vào** của bước xuất,
        không phải sản phẩm xuất ra. `d710 osem` ghi, `d710 export` đọc.
        """
        return self.root / "recon.npz"

    def vendor_bed(self, n: int) -> Path:
        return self.vendor / f"bed{n}"

    def work_bed(self, n: int) -> Path:
        return self.work / f"bed{n}"

    def prompt(self, n: int) -> Path:
        """`.hs` prompt thô của bed n — cũng là template header của mọi số hạng."""
        return self.decoded / f"bed{n}.hs"

    def header(self, n: int) -> dict:
        """Sidecar header RDF của bed n, do `d710 decode` ghi."""
        with open(self.decoded / f"bed{n}.json") as f:
            return json.load(f)

    def mkdirs(self) -> Case:
        for d in (self.decoded, self.vendor, self.work, self.export,
                  self.scratch, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------- dò từ đĩa
    def decoded_beds(self) -> list[int]:
        """Bed đã qua bước 1 (có `bed<n>.hs` và sidecar)."""
        out = []
        for hs in self.decoded.glob("bed*.hs"):
            m = _BED_RE.match(hs.stem)
            if m and hs.with_suffix(".json").exists():
                out.append(int(m.group(1)))
        return sorted(out)

    def beds(self, terms=("background", "normdt")) -> list[int]:
        """Bed đã qua đủ cả ba bước — cái mà OSEM ăn được.

        Dò từ đĩa, đừng gõ tay: một bed hỏng ở bước 2 vẫn còn `bed<n>.hs` của
        bước 1, và ghép nó vào là ghép một bed không có số hạng nền.
        """
        return [n for n in self.decoded_beds()
                if all((self.work_bed(n) / f"{t}.hs").exists() for t in terms)]


def case(name: str, out: str | os.PathLike | None = None) -> Case:
    """Ca `name` dưới `--out` / `$D710_OUT`.

        from utils.paths import case
        C = case("ped")
        C.prompt(4), C.work_bed(4), C.export
    """
    return Case(name, out_root(out))


def cases(out: str | os.PathLike | None = None) -> list[Case]:
    """Mọi ca đã có mặt dưới gốc đầu ra."""
    root = out_root(out)
    if not root.is_dir():
        return []
    return [Case(d.name, root) for d in sorted(root.iterdir())
            if d.is_dir() and (d / "decoded").is_dir()]
