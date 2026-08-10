#!/usr/bin/env python3
"""Generate experiments/EXPERIMENTS.md from meta.yaml.

The index is generated, never hand-maintained. Five documents in this repo once
claimed five different experiment counts (51 / 61+ / 117+ / 130+ / 170+) against a
real count of 73; deriving the index from metadata removes that failure mode.

Output is deterministic: regenerating without source changes produces a byte-identical
file, so a dirty `git status` after a run means the index was stale.

    python tools/generate_experiment_index.py [--check]

--check exits 1 if the file on disk differs from what would be generated.
"""
from __future__ import annotations

import os
import re
import sys

import yaml

# Scores live in README headers, in several shapes that all appear in the corpus:
#   "## Score: 64/71 (90%)"
#   "**Status**: Active | **Score**: 22/32"
#   "## Status: Active | Score: 60/112 (54%)"
SCORE_RE = re.compile(r"Score\**\s*:\s*\**\s*(\d+\s*/\s*\d+)", re.IGNORECASE)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(REPO, "experiments")
ARCHIVE = os.path.join(REPO, "archive")
OUT = os.path.join(EXPERIMENTS, "EXPERIMENTS.md")

ERA_TITLES = {
    "era1-symbolic-collapse-2025h1": "Era 1 — Symbolic Collapse (2025-06 → 2025-08)",
    "era2-prefield-infodynamics-2025h2": "Era 2 — Pre-field / Infodynamics (2025-09 → 2025-12)",
    "era3-pac-formalization-2026q1": "Era 3 — PAC Formalization (2026-01 → 2026-04)",
    "era4-milestone-stack-2026q2": "Era 4 — Milestone Stack (2026-04 → present)",
}
ERA_ORDER = list(ERA_TITLES)


def read_meta(d: str) -> dict:
    p = os.path.join(d, "meta.yaml")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def read_score(d: str, meta: dict) -> str:
    """meta.yaml wins; otherwise recover the score from the README header."""
    if meta.get("score"):
        return str(meta["score"]).strip()
    r = os.path.join(d, "README.md")
    if not os.path.exists(r):
        return ""
    try:
        with open(r, encoding="utf-8", errors="replace") as f:
            head = f.read(4000)
    except OSError:
        return ""
    m = SCORE_RE.search(head)
    return re.sub(r"\s+", "", m.group(1)) if m else ""


def collect() -> tuple[list[dict], list[dict]]:
    live, archived = [], []
    for group in ("milestones", "sidecars", "studies"):
        gd = os.path.join(EXPERIMENTS, group)
        if not os.path.isdir(gd):
            continue
        for name in sorted(os.listdir(gd)):
            d = os.path.join(gd, name)
            if not os.path.isdir(d) or name.startswith("."):
                continue
            m = read_meta(d)
            live.append({
                "dir": name,
                "kind": group,
                "path": f"{group}/{name}",
                "title": (m.get("title") or name).strip(),
                "status": m.get("status", "?"),
                "era": m.get("era", "?"),
                "score": read_score(d, m),
            })
    if os.path.isdir(ARCHIVE):
        for era in sorted(os.listdir(ARCHIVE)):
            ed = os.path.join(ARCHIVE, era)
            if not os.path.isdir(ed):
                continue
            for name in sorted(os.listdir(ed)):
                d = os.path.join(ed, name)
                if not os.path.isdir(d) or name.startswith("cross_experiment_"):
                    continue
                if not os.path.exists(os.path.join(d, "meta.yaml")):
                    continue
                m = read_meta(d)
                archived.append({
                    "dir": name,
                    "path": f"../archive/{era}/{name}",
                    "title": (m.get("title") or name).strip(),
                    "status": m.get("status", "?"),
                    "era": m.get("era", era),
                    "score": read_score(d, m),
                })
    return live, archived


def table(rows: list[dict]) -> list[str]:
    has_kind = any(r.get("kind") for r in rows)
    head = "| Experiment | Kind | Title | Score |" if has_kind else "| Experiment | Title | Score |"
    sep = "|---|---|---|---|" if has_kind else "|---|---|---|"
    out = [head, sep]
    for r in rows:
        score = r["score"] or "—"
        if has_kind:
            out.append(f"| [`{r['dir']}`]({r['path']}/) | {r.get('kind','—')} | {r['title']} | {score} |")
        else:
            out.append(f"| [`{r['dir']}`]({r['path']}/) | {r['title']} | {score} |")
    out.append("")
    return out


def build() -> str:
    live, archived = collect()
    active = [r for r in live if r["status"] == "active"]
    completed = [r for r in live if r["status"] == "completed"]
    other = [r for r in live if r["status"] not in ("active", "completed")]

    L: list[str] = []
    L.append("# Experiment Index")
    L.append("")
    L.append(
        "**Generated file — do not edit by hand.** "
        "Regenerate with `python tools/generate_experiment_index.py`."
    )
    L.append("")
    L.append(
        f"{len(live) + len(archived)} experiments: {len(live)} live "
        f"({len(active)} active, {len(completed)} completed"
        + (f", {len(other)} other" if other else "")
        + f") and {len(archived)} archived."
    )
    L.append("")
    L.append(
        "Eras, lifecycle values, and the archival rule are defined in "
        "[`STANDARDS.md`](../STANDARDS.md) §2.2–2.3. Archived work is preserved "
        "lineage, not deprecated work — see "
        "[`../archive/README.md`](../archive/README.md)."
    )
    L.append("")

    if active:
        L.append("## Active")
        L.append("")
        L.append("Currently being worked.")
        L.append("")
        L += table(active)
    if completed:
        L.append("## Completed")
        L.append("")
        L.append("Validated and documented; not being extended.")
        L.append("")
        L += table(completed)
    if other:
        L.append("## Archived in place")
        L.append("")
        L.append(
            "Individually superseded, but not part of an archived era, so they remain "
            "alongside their era peers. Status and era are independent.")
        L.append("")
        L += table(other)

    L.append("## Archive")
    L.append("")
    by_era: dict[str, list[dict]] = {}
    for r in archived:
        by_era.setdefault(r["era"], []).append(r)
    for era in ERA_ORDER:
        rows = by_era.get(era)
        if not rows:
            continue
        L.append(f"### {ERA_TITLES[era]}")
        L.append("")
        L += table(rows)

    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    content = build()
    if "--check" in sys.argv:
        existing = ""
        if os.path.exists(OUT):
            with open(OUT, encoding="utf-8") as f:
                existing = f.read()
        if existing != content:
            print("EXPERIMENTS.md is out of date. Run tools/generate_experiment_index.py")
            import difflib
            diff = difflib.unified_diff(existing.splitlines(), content.splitlines(),
                                        fromfile="committed", tofile="regenerated",
                                        lineterm="", n=1)
            for line in list(diff)[:60]:
                print(line)
            return 1
        print("EXPERIMENTS.md is up to date.")
        return 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"wrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
