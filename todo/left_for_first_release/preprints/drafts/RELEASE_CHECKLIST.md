# Dawn Field Theory – Release 1.0 Checklist

Scope: lock claims to artifacts, pin environment, and ship preprints with tag/DOI by Sept 1, 2025.

Pre-freeze (week 1)
- [ ] Cut 1.0-rc branch; content/code freeze date set
- [ ] ENVIRONMENT.md updated (Python, OS, CUDA notes, setup steps)
- [ ] Evidence Map updated (claims → CSV/PNG/script paths)
- [ ] Link-check and smoke-test CI green (Windows + Ubuntu)

Preprints polish (week 2)
- [ ] AIX, SEC, DFT drafts: “no offline training; online adaptation” language verified
- [ ] Each figure cites repo-relative paths; seeds noted
- [ ] Shared glossary terms match foundational/lexicon.md

Roadmaps & policy (week 3)
- [ ] timeline.md in quarterly format (Q3 2025 block + Discord-only note)
- [ ] roadmaps/README.md and roadmaps/core_project_roadmap.md refreshed
- [ ] CONTRIBUTION.md updated; contribution freeze lifting policy queued for release

Tag & archive (week 4)
- [ ] Re-run flagship demos (TinyCIMM-Euler + SCBF) to confirm artifacts
- [ ] Tag v1.0.0; generate Zenodo DOI
- [ ] Preprints cite tag + DOI in Reproducibility sections
- [ ] Discord announcement posted

Definition of Done
- [ ] README links: ENVIRONMENT.md, EVIDENCE_MAP.md, RELEASE_CHECKLIST.md
- [ ] Seeds + configs stored with run directories
- [ ] CI: link-check + short SCBF test job green