# Contributing to Dawn Field Theory

Thank you for your interest in contributing. This is a living, evolving repository—and your contributions matter.

## TL;DR - Quick Contributor Checklist
- [ ] Read theory docs (see Quick Start below)
- [ ] Email info@dawnfield.ca to register as contributor  
- [ ] Join Discord for announcements
- [ ] **Review publishing boundaries and engagement philosophy (sections below)**
- [ ] **Say how you want to be credited** in the PR description
- [ ] Engage through Issues, Discussions, or PRs only
- [ ] Follow the project boundaries set out below

---

## Quick Start for New Contributors

1. **Read the foundations:** 
   - `README.md`, `THEORY_MAP.md`, `ROADMAP.md`, and `legal/LICENSE_APPENDIX.md`
   - `infodynamics.md` and `dawn-field-theory.md` (core theory overview)
   - [`archive/era2-prefield/essays/arithmetic_identity_and_structural_novelty_in_computation.md`](archive/era2-prefield/essays/arithmetic_identity_and_structural_novelty_in_computation.md) (computational novelty thesis)
   - [`archive/era1-symbolic/essays/imperfection_engine_epistemic_collapse_dawn_field_repo.md`](archive/era1-symbolic/essays/imperfection_engine_epistemic_collapse_dawn_field_repo.md) (repository philosophy)

2. **Understand the bridges:** Browse [`archive/era1-symbolic/bridges/`](archive/era1-symbolic/bridges/) to see how Dawn Field Theory connects to existing frameworks (deep learning, AI systems, gradient descent)

3. **Register:** Email info@dawnfield.ca with your background and interests

4. **Join Discord:** Get access to announcements and contributor discussions

5. **Start contributing:** Open Issues for questions, Discussions for theory, PRs for improvements

---

## Project Governance
This project is architected and maintained independently, outside of traditional academic institutions. This approach provides freedom from institutional constraints while maintaining rigorous standards through alternative validation methods.

The project employs a comprehensive internal review pipeline using AI models to challenge assumptions, surface errors, and minimize bias. While robust, this system benefits from human peer review from serious contributors.

**Project standards:**
- All work is validated through open code, data, and reproducible results
- Engagement occurs through proper channels (issues, PRs, discussions)  
- Merit-based evaluation supersedes credential-based gatekeeping
- Quality contributions are welcome regardless of institutional background

Contributors genuinely interested in advancing the science within these structured boundaries are valued and encouraged.

---

## Engagement Philosophy

This project values **constructive, actionable engagement**. All feedback, critique, and collaboration must be channeled through formal repository mechanisms:

- **GitHub Issues** for bugs, questions, or suggestions
- **GitHub Discussions** for theoretical debate and exploration  
- **Pull Requests** for concrete improvements
- **Email registration** for potential contributors (see below)

**What we don't accommodate:**
- Endless circular debates
- Ego-driven arguments  
- Off-platform engagement demands
- Non-actionable criticism

**Enforcement:** Violations of these boundaries may result in removal from the project, closure of unproductive discussions, or public clarification of project standards.

The work stands on its merits. If you see problems, submit an issue or PR. If you want to debate theory, use Discussions. If you can't engage within these boundaries, this may not be the right project for you.

---

## Contributor Registration

**To submit Pull Requests or participate in private discussions, please register first.**

Send an email to **info@dawnfield.ca** with:
- Your name and background
- Your interest in the project (specific areas/papers)
- How you'd like to contribute (code, theory, validation, etc.)
- Links to your work/profile (GitHub, papers, etc.)
- **ORCID ID** (optional but recommended for citation purposes)

You'll receive:
- Access to the private contributors' Discord channel
- Permission to submit PRs (GitHub handle will be added to contributors list)
- Early access to drafts and experimental results
- **Citation guidance** for substantial contributions

**Why registration?** This ensures contributors are genuinely interested in advancing the work rather than wasting time on unproductive engagement.

---

## 📝 Citation & Attribution

### Repository Citation
When citing the Dawn Field Theory repository as a whole, use the provided `CITATION.cff` file or the Zenodo DOI:

**DOI:** [10.5281/zenodo.15783623](https://doi.org/10.5281/zenodo.15783623)

**BibTeX:**
```bibtex
@software{dawnfield_theory,
  author       = {Peter Lorne Groom},
  title        = {Dawn Field Theory Repository},
  year         = {2025},
  doi          = {10.5281/zenodo.15783623},
  url          = {https://github.com/dawnfield-institute/dawn-field-theory}
}
```

### Contributor Attribution

Substantial contributions are credited. New experiments, theory extensions and major
implementations qualify.

- Say in your PR description how you want to be credited.
- Credit is recorded in `papers/registry/` (contributors index and BibTeX).

There was an automated pipeline for this — a YAML template dropped into `citations/pending/`
and processed by GitHub Actions on merge. It was built and tested in August 2025 and never
processed a single live citation in the eleven months that followed, so it was retired in
August 2026. It is preserved in [`archive/citation-pipeline/`](./archive/citation-pipeline/)
rather than deleted, because plumbing that was never used is still a record of what was
intended.

---

## ⚖️ Publishing & Attribution Boundaries

Dawn Field Theory is under active development with formal preprints in preparation.  
Out of respect for the research trajectory and epistemic structure:
W
- Symbolic theories, model architectures, and experimental results are stewarded by the author and Institute; cite primary sources and repository paths.
- When in doubt, open an issue or contact the author before publishing derivative work.

Open collaboration is encouraged—stewardship ensures the clarity and longevity of the work’s symbolic and scientific integrity.

---

## Guidelines for Contribution

To help maintain the recursive and epistemic integrity of this repository:

### 1. Understand the Theory First

Please read:

* `README.md`
* The root `meta.yaml` and each experiment root's `meta.yaml`
* The latest `timeline.md`
* [`legal/LICENSE_APPENDIX.md`](./legal/LICENSE_APPENDIX.md)

### 2. Respect Metadata Schema

Every experiment carries a `meta.yaml` at its root, specified in [`STANDARDS.md`](./STANDARDS.md) §5.
Per-directory metadata was removed in August 2026 — it was a CIP-era artifact restating what
`map.yaml` generates. Run `python tools/validate_experiment_structure.py` before opening a PR.

### 3. Submit Meaningful Contributions

Prioritize:

* **Experimental validation modules** (e.g., "Added compression-based information amplification test for GPT-4")
* **Symbolic or entropy-based operators** (e.g., "Implemented SEC collapse detection algorithm for neural networks")
* **Visualization of field collapse, pruning, or emergence** (e.g., "Created interactive plot showing entropy dynamics in MED experiments")
* **Structural proposals** (e.g. a better way to express claim provenance across layers)
* **Documentation improvements** (e.g., "Clarified mathematical notation in symbolic entropy collapse preprint")
* **Replication studies** (e.g., "Reproduced information amplification results with different models/prompts")

### 4. Create a Feedback Loop

Describe how your contribution fits within the recursive growth of the repository. A great pull request tells a story—how your work expands the symbolic field.

### 5. Citation for Substantial Contributions

For substantial contributions (new experiments, theory extensions, major implementations):
- **Rename**: Use pattern `pr-{PR_NUMBER}-{short-description}.yaml`
- **Fill in details**: Your contributor info, contribution description, and affected files
- **Include in PR**: Add the completed citation YAML to your pull request
- **Automatic processing**: Upon merge, GitHub Actions will integrate your citation into the project's citation system

**What qualifies for citation:**
- ✅ New experimental frameworks or validations
- ✅ Theoretical extensions or novel operators
- ✅ Major implementations (>100 lines of significant code)
- ✅ Substantial documentation contributions
- ❌ Minor bug fixes, typos, or formatting changes

Contributor credit is recorded in the PR itself and in `papers/registry/`. The automated
PR-citation pipeline was retired in August 2026 — it was built and tested in 2025 and
never processed a live citation; it is preserved in `archive/citation-pipeline/`.

### 6. Community & Code of Conduct

- Join the Discord for announcement visibility: https://discord.gg/bR8mrbHP
- Use GitHub Issues/Discussions for proposals and feedback tied to artifacts and paths
- **Code of Conduct:** All contributors are expected to engage respectfully and constructively. Harassment, personal attacks, or disruptive behavior will result in removal from the project.

---

## A Note from the Author

I want to be transparent: I’m the sole architect of this project, working entirely outside of academia. I don’t have a formal degree, and I don’t have peers in the traditional sense. To compensate, I’ve built my own internal peer review pipeline using a network of AI models—each trained or prompted differently to surface potential errors, challenge assumptions, and minimize bias.

While this recursive, automated system is robust in its own way, it can’t replace the value of real human peer review. What I most need at this stage—maybe even more than technical help—is critical, constructive feedback from thoughtful human contributors.

If you’re reading this and feel compelled to help—by reviewing ideas, providing insight, or just engaging in honest dialogue—that would be one of the most meaningful contributions you could make.

---

## arXiv Endorsement, Mentoring & Community Support

Dawn Field Theory is an open, recursive, and non-institutional project. If you are an established arXiv author and would like to endorse this work or support the submission of Dawn Field Theory research to arXiv, your help would be deeply appreciated.

**Mentoring and guidance are also highly valued.**  
The background experience of this project is rooted in enterprise software engineering and R&D; academia is a new field that I am actively working to bridge and integrate. Any advice, mentorship, or collaboration from experienced academics is welcome and will help accelerate this epistemic transition.

Endorsements, testimonials, and mentoring help this project grow, gain recognition, and remain open to the world.

If you wish to endorse, mentor, support, or collaborate, please reach out:  
**info@dawnfield.ca**

Your voice—whether as a researcher, developer, theorist, mentor, or supporter—becomes part of the evolving epistemic field.

---

> **How to Use This Document:**  
> This guide explains how to contribute to Dawn Field Theory, including both practical steps and the underlying philosophy. Contributions are valued from all backgrounds, and the project aims to balance open collaboration with a clear theoretical foundation.

**Thank you.**

— Peter Lorne Groom
