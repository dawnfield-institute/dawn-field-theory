#!/usr/bin/env python3
"""Report unresolved relative markdown links across tracked .md files.

The repository carries known link rot, mostly inside archived documents whose targets
moved or were never committed. That is recorded rather than hidden. What this tool exists
to prevent is *new* rot: run with --max to fail when the count rises above a pinned
ceiling.

    python tools/check_links.py                 # report, grouped by area
    python tools/check_links.py --max 200       # exit 1 if above the ceiling (CI)
    python tools/check_links.py --list          # every unresolved link
    python tools/check_links.py --changelog     # include .changelog/ (excluded by default)

.changelog/ is excluded by default: a changelog is a dated record of what was true when it
was written, so a path that has since moved is correct history, not a broken link.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
EXTERNAL = ("http://", "https://", "mailto:", "ftp://", "//")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-z", "*.md"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [REPO / p for p in out.split("\0") if p]


def unresolved(include_changelog: bool) -> list[tuple[Path, str, str]]:
    """Return (source_file, link_text, target) for each link that does not resolve."""
    bad: list[tuple[Path, str, str]] = []
    for path in tracked_markdown():
        rel = path.relative_to(REPO).as_posix()
        if not include_changelog and rel.startswith(".changelog/"):
            continue
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, target in LINK_RE.findall(text):
            link = target.split("#")[0].strip().strip("<>")
            if not link or link.lower().startswith(EXTERNAL):
                continue
            decoded = urllib.parse.unquote(link)
            resolved = (path.parent / decoded).resolve()
            if not resolved.exists():
                bad.append((path, label, link))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=None,
                    help="fail if unresolved count exceeds this ceiling")
    ap.add_argument("--list", action="store_true", help="print every unresolved link")
    ap.add_argument("--changelog", action="store_true",
                    help="include .changelog/ (dated records, excluded by default)")
    args = ap.parse_args()

    bad = unresolved(args.changelog)
    by_area = Counter(p.relative_to(REPO).as_posix().split("/")[0] for p, _, _ in bad)

    print(f"unresolved relative links: {len(bad)}"
          f"{'' if args.changelog else '  (.changelog/ excluded)'}")
    for area, count in by_area.most_common():
        print(f"  {area:<16} {count}")

    if args.list:
        print()
        for path, label, link in sorted(bad, key=lambda b: b[0].as_posix()):
            print(f"  {path.relative_to(REPO).as_posix()}")
            print(f"      [{label}] -> {link}")

    if args.max is not None and len(bad) > args.max:
        print(f"\nFAIL: {len(bad)} unresolved links exceeds ceiling of {args.max}.",
              file=sys.stderr)
        print("New link rot was introduced. Fix the new links, or if the rise is "
              "deliberate, raise the ceiling in .github/workflows/repo-standards.yml "
              "in the same commit.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
