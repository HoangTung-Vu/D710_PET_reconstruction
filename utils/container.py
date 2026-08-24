"""Run a command inside `d710:full`. The ONLY place that knows how to call docker from Python.

The image carries the entire vendor software stack: `/usr/PET` (librdf,
pet_recon, the list-mode tools), `/usr/g`, and the decoder at
`/opt/custom_tool`. Because of that, `D710/` **no longer references
`custom_tool/` on the host at all** — everything it needs is inside the image:

| formerly on the host | inside the image |
|---|---|
| `custom_tool/ge_rdf_tool.py` | `/opt/custom_tool/ge_rdf_tool.py` |
| `custom_tool/petsw/usr/PET/systemConfig/cal/*.3dnorm` | `/usr/PET/systemConfig/cal/*.3dnorm` |
| `custom_tool/petsw/.../cal/*.3dwcc` | the same directory |

Consequence: the decode / estimate / tostir steps need only **bash + docker +
the python3 stdlib** on the host. No conda, no numpy, no pydicom, no i386
multiarch. Only `osem` and `export` still need the project environment, because
SIRF is not in the image.

The `D710/` code is **not** baked into the image: `d710` mounts this tree at run
time, so changing one line of Python does not mean rebuilding 7 GB.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Root of the code tree, as seen from inside the container.
D710_IN = "/d710"

#: `vendor/` is mounted separately here as well, because the gdb scripts
#: `source /vendor/...` by absolute path and that is the convention that has
#: been validated.
VENDOR_IN = "/vendor"


def image() -> str:
    return os.environ.get("D710_IMAGE", "d710:full")


def ensure_image() -> None:
    """Fail early with the command to load the image, rather than late with a docker error.

    The image is **handed over whole**, never rebuilt: building it needs the
    18 GB `custom_tool/petsw/` tree, which is not in the repo. `D710/Dockerfile`
    only documents what the image contains.
    """
    p = subprocess.run(["docker", "image", "inspect", image()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise SystemExit(
            "error: không có image '%s'. Nạp nó:\n"
            "  docker load -i d710_full.tar\n"
            "(D710/Dockerfile ghi lại image chứa gì.)" % image())


def docker_argv(mounts=(), env=None, tty=False, extra=(), interactive=True) -> list:
    """The `docker run ...` part that precedes the image name.

    `--user` makes every generated file belong to whoever ran the command.
    Without it everything under the output directory ends up owned by root — a
    bind mount passes the container uid straight through to the host, and the
    container's only user is root.

    ⚠ The `pet_recon` path (`vendor/run.sh`) is the ONE EXCEPTION: it writes to
    `/usr/PET/systemConfig`, `/usr/g/service/log` and `/petRDFS/OVLFILES`, all
    three writable only by root in the image, and the image is not rebuilt. So
    it runs as root. Everything else — decoding, ct_to_pifa, reading cal files,
    to_stir — runs with `--user`.
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
    """Run `argv` inside the container. Returns a `CompletedProcess`."""
    cmd = docker_argv(mounts, env, tty, extra) + [image()] + [str(a) for a in argv]
    if verbose:
        print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=capture)


def python(argv, **kw):
    """`python3 <argv>` inside the container, with `D710/` on `PYTHONPATH`.

    The caller must mount `D710/` itself (via `d710_mount()`) if the script
    needs it.
    """
    env = dict(kw.pop("env", None) or {})
    env.setdefault("PYTHONPATH", "/opt/custom_tool:" + D710_IN)
    return run(["python3"] + list(argv), env=env, **kw)


def d710_mounts(root) -> list:
    """Mount the code tree twice: `/d710` for Python, `/vendor` for the gdb scripts."""
    return [(root, D710_IN, "ro"), (os.path.join(str(root), "vendor"), VENDOR_IN, "ro")]


def rdf_info(raw_file, verbose=False) -> str:
    """`ge_rdf_tool.py info <raw>` — verbatim, run inside the container.

    Verbatim rather than `--json` is deliberate: everything that reads this
    output has been calibrated against **that exact** printout. Host ↔ container
    has been compared: the only difference is one line, the one echoing the file
    path, which nothing reads.
    """
    raw_file = os.path.abspath(str(raw_file))
    d, name = os.path.dirname(raw_file), os.path.basename(raw_file)
    p = python(["/opt/custom_tool/ge_rdf_tool.py", "info", "/raw/" + name],
               mounts=[(d, "/raw", "ro")], capture=True, check=False,
               verbose=verbose)
    # Keep stderr too: when the decoder cannot open the file it exits non-zero
    # with an empty stdout, and reporting stdout alone gives "the tool said:"
    # followed by nothing.
    return p.stdout + (("\n" + p.stderr) if p.stderr else "")


def cal_tags(uid: str, suffix: str, tags, verbose=False):
    """Read a few DICOM tags of a calibration file in `/usr/PET/systemConfig/cal/`.

    `tags` is a list of `(name, tag)` with the tag as an integer, e.g.
    `[("kind", 0x00171005), ("src", 0x00171007)]`. Returns a dict, or `None` if
    there is no such file.

    Done inside the container because that is where the vendor calibration tree
    lives now. The image ships pydicom, so the host does not need it.
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
