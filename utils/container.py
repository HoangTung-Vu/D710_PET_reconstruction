"""Chạy một lệnh trong `d710:full`. Chỗ DUY NHẤT biết cách gọi docker từ Python.

Image mang sẵn toàn bộ phần mềm của hãng: `/usr/PET` (librdf, pet_recon, các
công cụ list-mode), `/usr/g`, và bộ giải mã ở `/opt/custom_tool`. Nhờ vậy
`D710/` **không còn tham chiếu nào tới `custom_tool/` trên host** — thứ nó cần
đều đã ở trong image:

| trước ở host | trong image |
|---|---|
| `custom_tool/ge_rdf_tool.py` | `/opt/custom_tool/ge_rdf_tool.py` |
| `custom_tool/petsw/usr/PET/systemConfig/cal/*.3dnorm` | `/usr/PET/systemConfig/cal/*.3dnorm` |
| `custom_tool/petsw/.../cal/*.3dwcc` | cùng thư mục đó |

Hệ quả: bước decode / estimate / tostir chỉ cần **bash + docker + python3
stdlib** trên host. Không conda, không numpy, không pydicom, không i386
multiarch. Chỉ `osem` và `export` còn cần môi trường project, vì SIRF không có
trong image.

Mã của `D710/` **không** bake vào image: `d710` mount cây này vào lúc chạy, nên
sửa một dòng Python không phải dựng lại 7 GB.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Thư mục gốc của cây mã, nhìn từ bên trong container.
D710_IN = "/d710"

#: `vendor/` cũng được mount riêng ở đây, vì các script gdb `source /vendor/...`
#: theo đường tuyệt đối và đó là quy ước đã được kiểm chứng.
VENDOR_IN = "/vendor"


def image() -> str:
    return os.environ.get("D710_IMAGE", "d710:full")


def ensure_image() -> None:
    """Chết sớm với lệnh nạp image, thay vì chết muộn với lỗi của docker.

    Image được **bàn giao nguyên con**, không dựng lại: dựng nó cần cây
    `custom_tool/petsw/` 18 GB không nằm trong repo. `D710/Dockerfile` chỉ ghi
    lại image chứa gì.
    """
    p = subprocess.run(["docker", "image", "inspect", image()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise SystemExit(
            "error: không có image '%s'. Nạp nó:\n"
            "  docker load -i d710_full.tar\n"
            "(D710/Dockerfile ghi lại image chứa gì.)" % image())


def docker_argv(mounts=(), env=None, tty=False, extra=(), interactive=True) -> list:
    """Phần `docker run ...` đứng trước tên image.

    `--user` để mọi file sinh ra thuộc về người chạy. Không có nó thì tất cả
    dưới thư mục đầu ra quay về sở hữu của root — bind mount chuyển thẳng uid
    của container ra host, và người dùng của container chỉ có root.

    ⚠ Đường `pet_recon` (`vendor/run.sh`) là NGOẠI LỆ DUY NHẤT: nó ghi vào
    `/usr/PET/systemConfig`, `/usr/g/service/log` và `/petRDFS/OVLFILES`, cả ba
    chỉ root ghi được trong image, mà image thì không dựng lại. Nên nó chạy
    root. Mọi thứ khác — giải mã, ct_to_pifa, đọc cal, to_stir — chạy `--user`.
    """
    argv = ["docker", "run", "--rm"]
    if interactive:
        argv.append("-i")
    if tty:
        argv.append("-t")
    argv += ["--user", "%d:%d" % (os.getuid(), os.getgid()), "-e", "HOME=/tmp"]
    for host, dest, mode in mounts:
        argv += ["-v", "%s:%s%s" % (os.path.abspath(str(host)), dest,
                                    ":ro" if mode == "ro" else "")]
    for k, v in (env or {}).items():
        argv += ["-e", "%s=%s" % (k, v)]
    argv += list(extra)
    return argv


def run(argv, mounts=(), env=None, capture=False, check=True, tty=False,
        extra=(), verbose=True):
    """Chạy `argv` trong container. Trả `CompletedProcess`."""
    cmd = docker_argv(mounts, env, tty, extra) + [image()] + [str(a) for a in argv]
    if verbose:
        print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=capture)


def python(argv, **kw):
    """`python3 <argv>` trong container, với `D710/` trên `PYTHONPATH`.

    Người gọi phải tự mount `D710/` (dùng `d710_mount()`) nếu script cần nó.
    """
    env = dict(kw.pop("env", None) or {})
    env.setdefault("PYTHONPATH", "/opt/custom_tool:" + D710_IN)
    return run(["python3"] + list(argv), env=env, **kw)


def d710_mounts(root) -> list:
    """Mount cây mã hai lần: `/d710` cho Python, `/vendor` cho các script gdb."""
    return [(root, D710_IN, "ro"), (os.path.join(str(root), "vendor"), VENDOR_IN, "ro")]


def rdf_info(raw_file, verbose=False) -> str:
    """`ge_rdf_tool.py info <raw>` — nguyên văn, chạy trong container.

    Nguyên văn chứ không phải `--json` là có chủ ý: mọi thứ đọc kết quả này đều
    đã hiệu chuẩn theo **đúng** bản in đó. Đã đối chứng host ↔ container: khác
    duy nhất một dòng, dòng in lại đường dẫn file, mà không ai đọc.
    """
    raw_file = os.path.abspath(str(raw_file))
    d, name = os.path.dirname(raw_file), os.path.basename(raw_file)
    p = python(["/opt/custom_tool/ge_rdf_tool.py", "info", "/raw/" + name],
               mounts=[(d, "/raw", "ro")], capture=True, check=False,
               verbose=verbose)
    # Giữ cả stderr: khi bộ giải mã không mở được file, nó thoát khác 0 với
    # stdout rỗng, và báo mỗi stdout thì ra "công cụ nói:" rồi không gì cả.
    return p.stdout + (("\n" + p.stderr) if p.stderr else "")


def cal_tags(uid: str, suffix: str, tags, verbose=False):
    """Đọc vài tag DICOM của một file hiệu chuẩn trong `/usr/PET/systemConfig/cal/`.

    `tags` là danh sách `(tên, tag)` với tag là số nguyên, ví dụ
    `[("kind", 0x00171005), ("src", 0x00171007)]`. Trả dict, hoặc `None` nếu
    không có file.

    Ở trong container vì đó là nơi cây hiệu chuẩn của hãng sống bây giờ. Image
    có sẵn pydicom, nên host không cần.
    """
    import json

    code = (
        "import json,os,sys,pydicom\n"
        "p='/usr/PET/systemConfig/cal/%s.%s'\n"
        "if not os.path.exists(p):\n"
        "    print(json.dumps(None)); sys.exit(0)\n"
        "d=pydicom.dcmread(p, force=True)\n"
        "want=%r\n"
        "out={}\n"
        "for name,tag in want:\n"
        "    out[name]=str(d[tag].value) if tag in d else None\n"
        "print(json.dumps(out))\n" % (uid, suffix, list(tags)))
    p = python(["-c", code], capture=True, check=False, verbose=verbose)
    if p.returncode != 0:
        print("   không đọc được %s.%s trong container:\n%s"
              % (uid, suffix, p.stderr.strip()[:500]), file=sys.stderr)
        return None
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print("   %s.%s: đầu ra không phải JSON:\n%s"
              % (uid, suffix, p.stdout[:500]), file=sys.stderr)
        return None
