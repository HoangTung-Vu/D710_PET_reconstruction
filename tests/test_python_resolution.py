"""`d710` must find a usable interpreter on any machine, with or without conda.

`python3` is not a safe default and never was. On this machine `/usr/bin` comes
before conda on `PATH`, so `python3` is the SYSTEM interpreter even inside an
activated `petct_reconstruction` -- while `python` is conda's. A script that
tries only one of the two picks the wrong interpreter about half the time, and
the failure lands deep inside an import rather than at the front door.

So `d710` tries a list: an explicit `$D710_PYTHON` first and alone, then
`python3`, `python`, the active venv's, the active conda env's, and the usual
absolute paths. The steps that need only the standard library (decode, estimate,
tostir, exam) must work on any of them; the rest report what they tried.

Tested by running the real functions out of the real script, so this cannot pass
against a copy that has drifted.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT

D710 = ROOT / "d710"


def bash(snippet: str, env=None, path=None) -> subprocess.CompletedProcess:
    """Run `snippet` with `d710`'s interpreter-resolution functions in scope."""
    src = D710.read_text()
    funcs = "\n".join(
        m.group(0) for m in re.finditer(
            r"^(?:py_candidates|resolve_py|need_py)\(\)\s*\{.*?^\}",
            src, re.S | re.M))
    assert "py_candidates" in funcs and "resolve_py" in funcs, (
        "d710 no longer defines py_candidates/resolve_py; this test is stale")
    prelude = 'die(){ echo "DIE: $*" >&2; return 1; }\n'
    return subprocess.run(["bash", "-c", prelude + funcs + "\n" + snippet],
                          capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def bare_env():
    """A PATH with no conda and no venv -- the machine `d710` has to work on."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}


def test_an_interpreter_is_found_without_conda(bare_env):
    """The stdlib-only steps must run on a machine that has only /usr/bin/python3."""
    if not shutil.which("python3", path="/usr/bin:/bin"):
        pytest.skip("no /usr/bin/python3 on this machine")
    r = bash("resolve_py", env=bare_env)
    assert r.returncode == 0, f"resolve_py failed: {r.stderr}"
    assert r.stdout.strip(), "resolve_py printed nothing"


def test_the_resolved_interpreter_actually_runs_python_3(bare_env):
    r = bash('c="$(resolve_py)"; "$c" -c "import sys; print(sys.version_info[0])"',
             env=bare_env)
    assert r.stdout.strip().endswith("3"), r.stdout + r.stderr


def test_an_explicit_choice_is_used_alone_and_never_replaced(bare_env):
    """`D710_PYTHON` is a decision, so a wrong one must fail loudly, not fall back."""
    env = dict(bare_env, D710_PYTHON="/nonexistent/python")
    r = bash("py_candidates", env=env)
    assert r.stdout.split() == ["/nonexistent/python"], r.stdout

    r = bash('PY=x; need_py "os, sys" "x"', env=env)
    assert r.returncode != 0, "a broken D710_PYTHON was silently replaced"


def test_the_candidate_list_has_no_duplicates_and_no_empties():
    r = bash("py_candidates", env={"PATH": "/usr/bin:/bin", "CONDA_PREFIX": "/usr"})
    got = r.stdout.split()
    assert got == list(dict.fromkeys(got)), f"duplicates: {got}"
    assert all(got), "empty candidate emitted"
    assert "/usr/bin/python" in got, f"CONDA_PREFIX not honoured: {got}"


def test_the_running_interpreter_would_be_found_for_the_modules_it_has():
    """Whatever is running this test can import numpy, so `need_py` must find it."""
    r = bash('PY=x; need_py "numpy" "x"; echo "PY=$PY"',
             env={"PATH": f"{sys.prefix}/bin:/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert "PY=" in r.stdout
