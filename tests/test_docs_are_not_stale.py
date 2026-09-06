"""No comment or docstring under `D710/` may name a document that is not there.

Five separate places pointed at `TOF_SCATTER_REVERSE.md`, `TOF_PLAN.md` and
`LISTMODE_TOF_PLAN.md` after those files were deleted -- one of them printed the
name to the user at run time, telling them to read something that does not
exist. Docs rot silently because nothing executes them; this is the one thing
that can.

Deliberately narrow: it checks that a named `.md` **resolves somewhere**, not
that the section it cites still says what the citation claims. Cheap enough to
run every time, and it catches the whole class of failure that actually happened.
"""

from __future__ import annotations

import re

import pytest

from conftest import ROOT

REPO = ROOT.parent

#: A markdown file named inside backticks, or bare after "see"/"See".
NAMED = re.compile(r"`([^`\s]+\.md)`|(?<![\w`/])((?:[A-Za-z0-9_][\w./-]*/)?"
                   r"[A-Z][A-Za-z0-9_]*\.md)\b")

#: Their whole purpose is to record which documents are missing.
EXCLUDE = {"tests/audit_petsw.md", "tests/audit_decode.md",
           "tests/audit_frameworks.md", "tests/test_docs_are_not_stale.py"}


def sources():
    for pattern in ("*.py", "*.sh", "*.md"):
        for p in ROOT.rglob(pattern):
            if "__pycache__" in p.parts or ".git" in p.parts:
                continue
            if str(p.relative_to(ROOT)) in EXCLUDE:
                continue
            yield p
    yield ROOT / "d710"


def resolves(name: str, origin) -> bool:
    """True when `name` names a file that exists somewhere sensible.

    Tried against the citing file's own directory, `D710/`, and the repo root,
    then by bare filename anywhere in the tree -- a citation is a pointer for a
    human, so finding the document at all is enough.
    """
    tail = name.split("/")[-1]
    if any((base / name).exists()
           for base in (origin.parent, ROOT, REPO)):
        return True
    return any(True for _ in REPO.glob(f"*/{tail}")) or (REPO / tail).exists() \
        or any(True for _ in ROOT.rglob(tail))


@pytest.mark.parametrize("path", sorted(sources(), key=str),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_every_document_a_file_names_exists(path):
    try:
        text = path.read_text(errors="ignore")
    except OSError:                                  # pragma: no cover
        pytest.skip(f"cannot read {path}")

    missing = sorted({(m.group(1) or m.group(2)) for m in NAMED.finditer(text)
                      if not resolves(m.group(1) or m.group(2), path)})
    assert not missing, (
        f"{path.relative_to(ROOT)} names {', '.join(missing)}, which "
        f"does not exist. Either restore the document or repoint the reference "
        f"-- a dead pointer is worse than none, and one of these used to be "
        f"printed to the user at run time.")
