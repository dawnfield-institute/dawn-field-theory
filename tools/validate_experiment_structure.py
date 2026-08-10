#!/usr/bin/env python3
"""Validate experiments against STANDARDS.md.

Checks the structure and metadata rules in STANDARDS.md sections 2 and 5.
Archived experiments (Eras 1-2) are exempted from *structure* rules by their
lifecycle, not by special-casing: they predate the current layout and are
preserved as-is. Their metadata is still checked.

Exit 0 if no errors. Warnings never fail the build.

    python tools/validate_experiment_structure.py [--warnings-as-errors]
"""
from __future__ import annotations

import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(REPO, "foundational", "experiments")
ARCHIVE = os.path.join(EXPERIMENTS, "archive")

VALID_STATUS = {"active", "completed", "archived", "falsified"}
VALID_ERA = {
    "era1-symbolic-collapse-2025h1",
    "era2-prefield-infodynamics-2025h2",
    "era3-pac-formalization-2026q1",
    "era4-milestone-stack-2026q2",
}

# exp_NN_name.py, plus the lettered sub-experiment convention used systematically
# across exp_30..exp_33 and the milestones (exp_30a_, exp_01b_refined, ...).
SCRIPT_RE = re.compile(r"^exp_\d{2,}[a-z]?_[a-z0-9_]+\.py$")
# Shared helpers legitimately live alongside the numbered scripts.
HELPER_RE = re.compile(r"^(_[a-z0-9_]+|constants|run_all(_experiments)?|conftest)\.py$")
# The name part is optional: exp_01_20260122_161159.json is in wide use.
RESULT_RE = re.compile(r"^exp_\d{2,}[a-z]?(_[a-z0-9_]+)?_\d{8}_\d{6}\.json$")

errors: list[str] = []
warnings: list[str] = []


def err(exp: str, msg: str) -> None:
    errors.append(f"{exp}: {msg}")


def warn(exp: str, msg: str) -> None:
    warnings.append(f"{exp}: {msg}")


def load_meta(path: str):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        return e


def check_metadata(exp: str, d: str) -> dict:
    """STANDARDS.md sections 5.2 and 5.3. Applies to every experiment, archived included."""
    meta_path = os.path.join(d, "meta.yaml")
    if not os.path.exists(meta_path):
        err(exp, "missing meta.yaml")
        return {}

    meta = load_meta(meta_path)
    if isinstance(meta, Exception):
        err(exp, f"unparseable meta.yaml ({type(meta).__name__})")
        return {}
    if not isinstance(meta, dict):
        err(exp, "meta.yaml is not a mapping")
        return {}

    for field in ("schema_version", "description"):
        if not meta.get(field):
            err(exp, f"meta.yaml missing required field '{field}'")
    # `name` is an accepted alias for `directory_name` (STANDARDS.md 5.2).
    if not (meta.get("directory_name") or meta.get("name")):
        err(exp, "meta.yaml missing 'directory_name' (or alias 'name')")

    status = meta.get("status")
    if not status:
        err(exp, "meta.yaml missing 'status'")
    elif status not in VALID_STATUS:
        err(exp, f"invalid status '{status}' (expected one of {sorted(VALID_STATUS)})")

    era = meta.get("era")
    if not era:
        err(exp, "meta.yaml missing 'era'")
    elif era not in VALID_ERA:
        err(exp, f"invalid era '{era}'")

    if not meta.get("title"):
        err(exp, "meta.yaml missing required field 'title'")
    if not meta.get("tags"):
        warn(exp, "meta.yaml has no 'tags'")
    return meta


def check_structure(exp: str, d: str) -> None:
    """STANDARDS.md 2.1 and 2.4. Skipped for archived experiments."""
    if not os.path.exists(os.path.join(d, "README.md")):
        err(exp, "missing README.md")

    scripts = os.path.join(d, "scripts")
    if not os.path.isdir(scripts):
        warn(exp, "no scripts/ directory")
    else:
        for f in os.listdir(scripts):
            if f.endswith(".py") and not (SCRIPT_RE.match(f) or HELPER_RE.match(f)):
                warn(exp, f"script name off-convention: scripts/{f}")

    results = os.path.join(d, "results")
    if not os.path.isdir(results):
        warn(exp, "no results/ directory")
    else:
        stray = [
            f for f in os.listdir(results)
            if f.endswith(".json") and not RESULT_RE.match(f)
        ]
        if stray:
            warn(exp, f"{len(stray)} result file(s) off-convention, e.g. {stray[0]}")

    # Loose scripts at the experiment root belong in scripts/.
    loose = [
        f for f in os.listdir(d)
        if f.endswith(".py") and os.path.isfile(os.path.join(d, f))
    ]
    if loose:
        warn(exp, f"{len(loose)} loose .py at root (belongs in scripts/): {loose[0]}")


def iter_experiments():
    for name in sorted(os.listdir(EXPERIMENTS)):
        d = os.path.join(EXPERIMENTS, name)
        if not os.path.isdir(d) or name == "archive" or name.startswith("."):
            continue
        yield name, d, False
    if os.path.isdir(ARCHIVE):
        for era in sorted(os.listdir(ARCHIVE)):
            era_dir = os.path.join(ARCHIVE, era)
            if not os.path.isdir(era_dir):
                continue
            for name in sorted(os.listdir(era_dir)):
                d = os.path.join(era_dir, name)
                if not os.path.isdir(d):
                    continue
                # Relocated cross-experiment artifacts, not experiments.
                if name.startswith("cross_experiment_"):
                    continue
                yield f"{era}/{name}", d, True


def main() -> int:
    total = archived = 0
    for exp, d, is_archived in iter_experiments():
        total += 1
        meta = check_metadata(exp, d)
        if is_archived:
            archived += 1
            if meta and meta.get("status") != "archived":
                err(exp, f"under archive/ but status is '{meta.get('status')}'")
            continue
        check_structure(exp, d)

    print(f"Validated {total} experiments ({archived} archived, {total - archived} live)")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARN  {w}")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  ERROR {e}")
        return 1
    print("\nNo errors.")
    if warnings and "--warnings-as-errors" in sys.argv:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
