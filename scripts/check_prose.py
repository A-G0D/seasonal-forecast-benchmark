"""Writing-hygiene check for tracked markdown and Python docstrings/comments.

Scans for a few things that shouldn't end up in a public repo:

  * em-dashes (a common AI-writing tell)
  * leaked tool-call artifacts (e.g. `</invoke>`, `<function>`, `tool_use`)
  * employer / internal-tool references that don't belong in a portfolio repo

Stdlib only, no dependencies. Exits non-zero and prints `path:line: reason`
for every hit.

Usage:
    python scripts/check_prose.py [path ...]

With no arguments it scans every tracked `*.md` and `*.py` file under the
repo root (via `git ls-files`, falling back to a filesystem walk if git isn't
available). This is not wired into a git hook or CI, run it by hand when you
want a check.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EM_DASH = "—"

ARTIFACT_MARKERS = (
    "</invoke>",
    "<invoke",
    "</function>",
    "<function>",
    "<parameter>",
    "</parameter>",
    "tool_use",
    "</content>",
)

# Employer / internal-tool references that shouldn't leak into a public repo.
BLOCKED_KEYWORDS = (
    "gss",
    "global shop",
    "global shop solutions",
    "gssmail",
    "fact graph",
    "brain vault",
    "livefire",
)

SCAN_SUFFIXES = (".md", ".py")


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = [ROOT / line for line in out.splitlines() if line.strip()]
        if paths:
            return [p for p in paths if p.suffix in SCAN_SUFFIXES and p.is_file()]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for p in ROOT.rglob("*") if p.suffix in SCAN_SUFFIXES and p.is_file()]


def _check_file(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return problems

    rel = path.relative_to(ROOT)
    lower_keywords = [kw.lower() for kw in BLOCKED_KEYWORDS]

    for lineno, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            problems.append(f"{rel}:{lineno}: em-dash found")

        for marker in ARTIFACT_MARKERS:
            if marker in line:
                problems.append(f"{rel}:{lineno}: leaked tool-call artifact ({marker!r})")

        low = line.lower()
        for kw in lower_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                problems.append(f"{rel}:{lineno}: blocked keyword ({kw!r})")

    return problems


def main(argv: list[str]) -> int:
    if argv:
        targets = [Path(a).resolve() for a in argv]
    else:
        targets = _tracked_files()

    problems: list[str] = []
    for path in targets:
        problems.extend(_check_file(path))

    if problems:
        for p in problems:
            print(p)
        print(f"\n{len(problems)} problem(s) found")
        return 1

    print("no issues found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
