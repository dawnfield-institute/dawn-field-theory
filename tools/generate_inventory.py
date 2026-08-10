#!/usr/bin/env python3
"""Generate INVENTORY.md — the whole corpus at a glance, by layer and lifecycle.

Complements EXPERIMENTS.md, which covers experiments only. This spans all four layers
plus the archive, so a reader arriving cold can see the shape of the thing before
choosing a door.

Generated, never hand-maintained: five documents in this repository once claimed five
different experiment counts.

    python tools/generate_inventory.py [--check]
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "INVENTORY.md")

ERA_TITLES = {
    "era1-symbolic-collapse-2025h1": "Era 1 — Symbolic Collapse (2025-06 → 2025-08)",
    "era2-prefield-infodynamics-2025h2": "Era 2 — Pre-field / Infodynamics (2025-09 → 2025-12)",
    "era3-pac-formalization-2026q1": "Era 3 — PAC Formalization (2026-01 → 2026-04)",
    "era4-milestone-stack-2026q2": "Era 4 — Milestone Stack (2026-04 → present)",
}


def tracked():
    # -z and an explicit UTF-8 decode: text=True uses the locale codec (cp1252 on
    # Windows) and mangles non-ASCII paths, which made this tool's counts depend on the
    # platform it ran on.
    out = subprocess.run(["git", "-c", "core.quotePath=false", "ls-files", "-z"],
                         cwd=REPO, capture_output=True, check=True
                         ).stdout.decode("utf-8", "surrogateescape")
    return [p for p in out.split("\0") if p]


def warn_untracked():
    """Warn about untracked, non-ignored files in counted layers.

    Counts come from `git ls-files`, so this generator must run AFTER `git add`. Running
    it before staging silently undercounts new files, produces an INVENTORY.md that looks
    correct locally, and fails CI on the next push with no obvious cause.
    """
    out = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-z", "--others",
         "--exclude-standard"],
        cwd=REPO, capture_output=True, check=True
        ).stdout.decode("utf-8", "surrogateescape")
    layers = ("theory/", "formal/", "experiments/", "papers/", "archive/", "tools/")
    stray = [p for p in out.split("\0") if p and p.startswith(layers)]
    if stray:
        print(f"warning: {len(stray)} untracked file(s) in counted layers are NOT included "
              f"in these counts. Stage them and re-run, or CI will report this file stale:",
              file=sys.stderr)
        for p in stray[:10]:
            print(f"  {p}", file=sys.stderr)


def count_under(files, prefix):
    return sum(1 for f in files if f == prefix or f.startswith(prefix.rstrip("/") + "/"))


def read_meta(d):
    p = os.path.join(d, "meta.yaml")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def experiments():
    rows = []
    base = os.path.join(REPO, "experiments")
    for kind in ("milestones", "sidecars", "studies"):   # spikes are exempt (STANDARDS 3)
        d = os.path.join(base, kind)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if not os.path.isdir(p):
                continue
            m = read_meta(p)
            rows.append({
                "name": n, "kind": kind,
                "status": m.get("status", "—"),
                "era": m.get("era", "—"),
                "title": (m.get("title") or n).strip(),
            })
    return rows


def archived():
    rows = []
    base = os.path.join(REPO, "archive")
    if not os.path.isdir(base):
        return rows
    for era in sorted(os.listdir(base)):
        d = os.path.join(base, era)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "meta.yaml")):
                continue
            m = read_meta(p)
            rows.append({"name": n, "era": m.get("era", era),
                         "title": (m.get("title") or n).strip()})
    return rows


def lexicon_stats():
    p = os.path.join(REPO, "theory", "lexicon.yaml")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return Counter(t.get("status", "?") for t in d.get("terms", []))


def build():
    files = tracked()
    exp = experiments()
    arc = archived()
    lex = lexicon_stats()

    active = [e for e in exp if e["status"] == "active"]
    settled = [e for e in exp if e["status"] == "completed"]
    other = [e for e in exp if e["status"] not in ("active", "completed")]

    L = ["# Inventory", "",
         "**Generated — do not edit by hand.** `python tools/generate_inventory.py`", "",
         "The whole corpus at a glance. Per-experiment detail is in "
         "[`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md); the argument itself "
         "is in [`THEORY_MAP.md`](THEORY_MAP.md).", "",
         "## Layers", "",
         "| Layer | Holds | Files |", "|---|---|---|"]
    for path, holds in (
        ("theory/", "what is claimed — framework, constants, lexicon, corrections, essays"),
        ("formal/", "why it holds — theorems, derivations, conjectures"),
        ("experiments/", "what was measured"),
        ("papers/", "what was published"),
        ("archive/", "lineage, by era — terminal"),
        ("tools/", "generators and validators"),
    ):
        L.append(f"| [`{path}`]({path}) | {holds} | {count_under(files, path)} |")
    L += ["", f"Tracked files: **{len(files)}**", "",
          f"**{len(exp) + len(arc)} experiments** — {len(exp)} live, {len(arc)} archived. "
          f"Plus 2 spikes, exempt from the experiment standard (STANDARDS.md §3).", ""]

    L += ["## On deck", "", f"{len(active)} experiments being worked now.", "",
          "| Experiment | Kind | Title |", "|---|---|---|"]
    for e in active:
        L.append(f"| [`{e['name']}`](experiments/{e['kind']}/{e['name']}/) | {e['kind']} | {e['title']} |")

    L += ["", "## Settled", "",
          f"{len(settled)} experiments validated and documented, not being extended.", "",
          "| Experiment | Kind | Title |", "|---|---|---|"]
    for e in settled:
        L.append(f"| [`{e['name']}`](experiments/{e['kind']}/{e['name']}/) | {e['kind']} | {e['title']} |")

    if other:
        L += ["", "## Other status", "", "| Experiment | Kind | Status |", "|---|---|---|"]
        for e in other:
            L.append(f"| [`{e['name']}`](experiments/{e['kind']}/{e['name']}/) | {e['kind']} | {e['status']} |")

    L += ["", "## Legacy", "",
          f"{len(arc)} archived experiments. Archived is **not** deprecated — this work is "
          "lineage, preserved in its original shape. See "
          "[`archive/README.md`](archive/README.md).", ""]
    by_era = {}
    for a in arc:
        by_era.setdefault(a["era"], []).append(a)
    for era in ERA_TITLES:
        rows = by_era.get(era)
        if not rows:
            continue
        L += [f"### {ERA_TITLES[era]}", "", "| Experiment | Title |", "|---|---|"]
        for a in rows:
            L.append(f"| `{a['name']}` | {a['title']} |")
        L.append("")

    if lex:
        L += ["## Vocabulary", "",
              f"[`theory/lexicon.yaml`](theory/lexicon.yaml) — {sum(lex.values())} terms, "
              "each carrying the era it was coined in.", "",
              "| Status | Terms |", "|---|---|"]
        for k in ("core", "superseded", "historical"):
            if lex.get(k):
                L.append(f"| {k} | {lex[k]} |")
        L.append("")

    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    warn_untracked()
    content = build()
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != content:
            print("INVENTORY.md is out of date. Run tools/generate_inventory.py")
            # Print the diff. A gate that only says "stale" is undiagnosable from a CI
            # log, which matters most when the staleness is platform-dependent and the
            # committer's machine reports it as current.
            import difflib
            diff = difflib.unified_diff(cur.splitlines(), content.splitlines(),
                                        fromfile="committed", tofile="regenerated",
                                        lineterm="", n=1)
            for line in list(diff)[:60]:
                print(line)
        print("INVENTORY.md is up to date.")
        return 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"wrote {os.path.relpath(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
