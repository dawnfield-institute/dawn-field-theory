---
name: dft-repo-gates
description: Run the dawn-field-theory CI gates and regenerate generated artifacts in the correct order. Use before committing or opening a PR in this repo, or when CI reports a generated file is stale, link rot grew, or the structure validator failed.
---

# The repository gates

CI (`.github/workflows/repo-standards.yml`) verifies; it never writes to the repo. Four
gates, all runnable locally.

```bash
python tools/validate_experiment_structure.py      # STANDARDS §2/§5
python tools/generate_experiment_index.py --check  # experiments/EXPERIMENTS.md
python tools/generate_inventory.py --check         # INVENTORY.md
python tools/check_links.py --max 201              # link rot ceiling
```

Plus a freshness check that regenerates `map.yaml` and the `meta.yaml` generated zone and
fails if anything differs.

## The ordering trap — read this before regenerating

**The generators read `git ls-files`. They must run AFTER `git add`.**

Run them on unstaged work and they silently undercount new files. The result looks correct
locally, passes `--check` on the *next* run, and fails CI with no visible cause. This
happened: two new layer READMEs were missed, `INVENTORY.md` recorded `theory/` as 15 files
instead of 16, and it cost a CI round-trip to find.

Correct order — stage first, regenerate, stage again:

```bash
git add -A
python tools/update_meta_yamls.py
python tools/generate_path.py
python tools/generate_inventory.py
python tools/generate_experiment_index.py
git add -A
git diff --quiet && echo "converged"      # must be clean
```

Adding a file can change `map.yaml` and a `meta.yaml`, which is why the second `git add`
matters. `generate_inventory.py` warns when untracked files sit in counted layers — heed it.

## Generated files are never hand-edited

`map.yaml`, `experiments/EXPERIMENTS.md`, `INVENTORY.md`, and the `files` /
`child_directories` keys of any `meta.yaml`. Editing them by hand produces a diff CI
immediately reverts you on.

## Link rot is ceilinged, not fixed

`check_links.py --max 201` — the count may **fall, never rise**. The repo carries known rot
inside archived documents; the gate exists to stop new rot, not to force old cleanup. If a
change deliberately raises the count, raise the ceiling in the workflow *in the same
commit* and say why.

`.changelog/` is excluded by default: a changelog is a dated record of what was true when
written, so a path that has since moved is correct history.

## Reading git from a tool: two rules

Both of these caused real bugs here.

1. **Decode as UTF-8 explicitly.** `subprocess.run(..., text=True)` uses the locale codec —
   cp1252 on Windows — which mangles non-ASCII paths. The link checker then looked for a
   file that did not exist under the mangled name, skipped it, and disagreed with CI by
   exactly one link. Use `capture_output=True` and `.stdout.decode("utf-8", "surrogateescape")`.
2. **Enumerate from git, not the filesystem.** `generate_path.py` once walked the disk and
   swept the gitignored 600 MB `internal/` tree — containing private material — into
   `map.yaml`, a file committed to a **public** repo. Use `git ls-files -z`.

## When a gate fails

`--check` gates print a unified diff of committed vs regenerated. Read it — it names the
drift. If the diff is only counts, you hit the ordering trap above.