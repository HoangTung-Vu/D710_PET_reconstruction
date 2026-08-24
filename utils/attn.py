"""Hệ số suy giảm theo bed — số hạng DUY NHẤT không lấy từ kernel của GE.

`vendor/estimate.py` có dùng mu-map (nó phải có, để mô phỏng scatter), nhưng
cái nó xuất ra là **scatter**, không phải hệ số suy giảm. Nên `af` dựng lại ở
đây từ cùng series CT, bằng `utils/attenuation.py`.

Cache vào `work/bed<n>/attn.hs` để nằm cùng chỗ với ba số hạng kia (randoms /
scatter / normdt) — sau lần chạy đầu thì cả bốn số hạng của mô hình đều có mặt
trên đĩa, mở ra xem được bằng bất cứ công cụ STIR nào. Tính lại mất ~16 s/bed.

⚠ Xoá `attn.hs`/`attn.s` nếu đổi CT hoặc đổi lưới ảnh — file không tự biết.
"""

from __future__ import annotations

from . import attenuation


def check_same_exam(ct, hdr) -> None:
    """CT nào đi với exam nào là do UID quyết định, không phải do nằm cạnh nhau.

    Đây là đồng nhất thức chứ không phải phép so gần đúng: thư mục ảnh nằm cạnh
    thư mục raw **không** đảm bảo cùng exam (`11082026/` chứa ảnh của hai exam
    khác nhau).
    """
    got, want = ct.meta["frame_of_reference_uid"], hdr["sop_instance_uid"]
    if got != want:
        raise SystemExit(
            "error: CT không thuộc cùng exam với bed này\n"
            f"  CT  FrameOfReferenceUID {got}\n"
            f"  RDF sop_instance_uid    {want}")


class Attenuation:
    """`af` cho từng bed của một ca, cache trên đĩa và trong RAM.

        at = Attenuation(case, ct_dir, template_image, template_acq)
        af4 = at.af(4)                 # (1, 553, 288, 381) mảng numpy
    """

    def __init__(self, case, ct_dir: str, image, acq_template, verbose: bool = True):
        self.case = case
        self.ct = attenuation.load(ct_dir)
        self.image = image
        self.acq = acq_template
        self.verbose = verbose
        self._cache: dict[int, object] = {}

    def describe(self) -> str:
        return self.ct.describe()

    def af(self, n: int):
        """Hệ số suy giảm của bed `n` — xác suất sống sót ∈ (0, 1]."""
        import sirf.STIR as pet

        if n in self._cache:
            return self._cache[n]

        hdr = self.case.header(n)
        check_same_exam(self.ct, hdr)

        path = self.case.work_bed(n) / "attn.hs"
        if path.exists():
            self._cache[n] = pet.AcquisitionData(str(path)).as_array()
            if self.verbose:
                print(f"  bed {n}: attn.hs có sẵn         "
                      f"af mean {self._cache[n].mean():.4f}")
            return self._cache[n]

        path.parent.mkdir(parents=True, exist_ok=True)
        mu = attenuation.mu_image(self.ct, hdr["table_position_mm"], self.image)
        af, _acf = attenuation.factors(self.acq, mu)
        af.write(str(path))                          # -> attn.hs + attn.s
        self._cache[n] = af.as_array()
        if self.verbose:
            m = mu.as_array()
            print(f"  bed {n}: table {hdr['table_position_mm']:>8.2f} mm  "
                  f"mu max {m.max():.4f} 1/cm  "
                  f"af mean {self._cache[n].mean():.4f}  -> ghi attn.hs")
        return self._cache[n]

    def all(self, beds) -> dict:
        return {n: self.af(n) for n in beds}
