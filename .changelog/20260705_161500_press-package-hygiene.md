# Package hygiene: stop tracking preprint build products

Part of the Press rebuild (bert ADR-0029). `foundational/docs/preprints/packages/`
(~170MB of timestamped package dirs + zips), `preprints/.build/`, and the root
`PACSeries.zip` are now gitignored and removed from the index (working tree
untouched — files remain on disk; no history rewrite, so DOI'd artifacts stay
reachable at old commits).

Reproducibility is preserved a better way: the forthcoming
`preprints/RELEASES.yaml` ledger records git_commit + source_hash + package
sha256 per release, so any published zip is rebuildable from source at the
recorded commit — and the canonical copy lives on Zenodo.

`packages/UPLOAD_CHECKLIST.md` (the old manual upload checklist) goes untracked
with the rest; it is superseded by the press validation suite + ledger.
