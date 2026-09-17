"""Running commands inside the `d710:full` container."""

from __future__ import annotations

import os
import subprocess
import sys

D710_IN = "/d710"

VENDOR_IN = "/vendor"


def image() -> str:
    return os.environ.get("D710_IMAGE", "d710:full")


def ensure_image() -> None:
    """Fail early, naming the command that loads the image, rather than on a docker error."""
    p = subprocess.run(["docker", "image", "inspect", image()],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise SystemExit(
            "error: no image '%s'. Load it:\n"
            "  docker load -i d710_full.tar\n"
            "(D710/Dockerfile records what the image contains.)" % image())


def docker_argv(mounts=(), env=None, tty=False, extra=(), interactive=True) -> list:
    """The `docker run ...` prefix that precedes the image name."""
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
    """Run `argv` inside the container."""
    cmd = docker_argv(mounts, env, tty, extra) + [image()] + [str(a) for a in argv]
    if verbose:
        print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check,
                          capture_output=capture, text=capture)


def python(argv, **kw):
    """Run `python3 <argv>` inside the container, with `D710/` on `PYTHONPATH`."""
    env = dict(kw.pop("env", None) or {})
    env.setdefault("PYTHONPATH", "/opt/custom_tool:" + D710_IN)
    return run(["python3"] + list(argv), env=env, **kw)


def d710_mounts(root) -> list:
    """Mount the code tree twice: `/d710` for Python and `/vendor` for the gdb scripts."""
    return [(root, D710_IN, "ro"), (os.path.join(str(root), "vendor"), VENDOR_IN, "ro")]


def rdf_info(raw_file, verbose=False) -> str:
    """Run `ge_rdf_tool.py info <raw>` verbatim inside the container."""
    raw_file = os.path.abspath(str(raw_file))
    d, name = os.path.dirname(raw_file), os.path.basename(raw_file)
    p = python(["/opt/custom_tool/ge_rdf_tool.py", "info", "/raw/" + name],
               mounts=[(d, "/raw", "ro")], capture=True, check=False,
               verbose=verbose)
    return p.stdout + (("\n" + p.stderr) if p.stderr else "")


def cal_tags(uid: str, suffix: str, tags, verbose=False):
    """Read DICOM tags of a calibration file in `/usr/PET/systemConfig/cal/`."""
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
        print("   could not read %s.%s inside the container:\n%s"
              % (uid, suffix, p.stderr.strip()[:500]), file=sys.stderr)
        return None
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print("   %s.%s: output is not JSON:\n%s"
              % (uid, suffix, p.stdout[:500]), file=sys.stderr)
        return None
