"""No docstring or document under `D710/` may name a file that is not there."""

from __future__ import annotations

import functools
import re

import pytest

from conftest import ROOT

REPO = ROOT.parent

NAMED = re.compile(r"`([^`\s]+\.md)`|(?<![\w`/])((?:[A-Za-z0-9_][\w./-]*/)?"
                   r"[A-Z][A-Za-z0-9_]*\.md)\b")

EXCLUDE = {"tests/test_docs_are_not_stale.py"}

NOT_OURS = {".venv", "venv", "site-packages", "__pycache__", ".git",
            "node_modules", ".pytest_cache", "build", "dist"}


def sources():
    for pattern in ("*.py", "*.sh", "*.md"):
        for p in ROOT.rglob(pattern):
            if NOT_OURS & set(p.parts):
                continue
            if str(p.relative_to(ROOT)) in EXCLUDE:
                continue
            yield p
    yield ROOT / "d710"


@functools.lru_cache(maxsize=1)
def our_documents() -> frozenset[str]:
    """Basenames of every `.md` in this tree and one level below the repository root."""
    names = {p.name for p in ROOT.rglob("*.md") if not NOT_OURS & set(p.parts)}
    names |= {p.name for p in REPO.glob("*.md")}
    names |= {p.name for p in REPO.glob("*/*.md")}
    return frozenset(names)


def resolves(name: str, origin) -> bool:
    """True when `name` names a file that exists somewhere sensible."""
    if any((base / name).exists() for base in (origin.parent, ROOT, REPO)):
        return True
    return name.split("/")[-1] in our_documents()


@pytest.mark.parametrize("path", sorted(sources(), key=str),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_every_document_a_file_names_exists(path):
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        pytest.skip(f"cannot read {path}")

    missing = sorted({(m.group(1) or m.group(2)) for m in NAMED.finditer(text)
                      if not resolves(m.group(1) or m.group(2), path)})
    assert not missing, (
        f"{path.relative_to(ROOT)} names {', '.join(missing)}, which "
        f"does not exist. Either restore the document or repoint the reference "
        f"-- a dead pointer is worse than none, and one of these used to be "
        f"printed to the user at run time.")
