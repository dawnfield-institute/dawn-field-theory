# Preprint Drafts Iteration Plan (Cycle 2)

Date: 2025-08-13  
Scope: Second refinement pass on all preprint drafts before external pre-review.  
Drafts in scope:
- AIXPreprint_draft.md
- CIPPreprint_draft.md
- DFT_Infodynamics_preprint_draft.md
- RecursiveMathematicalPlasticity_EntropyAwareArchitecture_draft.md
- ResonantSymbolicConvergencePreprint_draft.md
- SECPreprint_draft.md

## 1. Objectives
1. Structural consistency (shared template + section harmonization).  
2. Evidence + asset traceability into repository (#codebase mapping).  
3. Terminology alignment with `foundational/lexicon.*`.  
4. Citation normalization (APA + BibTeX sync).  
5. Gap closure (explicit TODO tags converted into actionable items or resolved).  
6. Pre-flight quality gates for release checklist.

## 2. Standard Section Template (IMRaD+)
1. Title & Abstract  
2. Keywords (aligned to controlled vocabulary in lexicon)  
3. Introduction (problem framing + prior work with citations)  
4. Background / Theoretical Framework  
5. Methods / Architecture  
6. Experiments / Evaluation  
7. Results  
8. Discussion (limitations, ablations, failure modes)  
9. Alignment & Ethics (as applicable)  
10. Roadmap & Future Work (link to `roadmaps/` docs)  
11. Conclusion  
12. References  
13. Appendix (expanded math, proofs, derivations, tables)  

## 3. Repository Mapping Matrix (Draft → Assets)
Legend: (T)=Theory, (M)=Model code/spec, (B)=Blueprint, (R)=Roadmap, (U)=Utility/Tool, (C)=Citation/Evidence.

### AIXPreprint
- Core Focus: AI explainability / detection / provenance / entropy pipeline.  
- Assets:  
  - (B) `blueprints/AI_detection/*` (pipeline + protocol)  
  - (M) `models/GAIA/`, `models/CIMM/` (interfaces / cognition modules when referenced)  
  - (T) `infodynamics.md`, `origin_of_infodynamics.md` (entropy framing)  
  - (U) `tools/` (meta + converter scripts referencing provenance)  
  - (R) `roadmaps/xai_scbf_roadmap.md`  
  - (C) `citations/*`  
  - (Foundational) `foundational/lexicon.*`  

### CIPPreprint (Cognition Index Protocol)
- Assets:  
  - (B) `blueprints/cognition_index_protocol.md`  
  - (Dir) `cognition_index_protocol/` (architecture + README)  
  - (R) `roadmaps/cip_protocol_roadmap.md`  
  - (M) `models/TinyCIMM/`, `models/CIMM/` (index computation references)  
  - (T) `foundational/` (cognitive primitives)  
  - (C) `citations/*`  

### DFT_Infodynamics Preprint
- Assets:  
  - (T) `infodynamics.md`, `origin_of_infodynamics.md`, `dawn-field-theory.md`  
  - (Foundational) `foundational/lexicon.*`  
  - (R) `roadmaps/core_project_roadmap.md`, `roadmaps/general_experiments_roadmap.md`  
  - (Experiments) `foundational/experiments/` (if populated)  
  - (M) Any model subdirs cited (`models/GAIA/`, `models/scbf/`)  
  - (C) `citations/*`  

### RecursiveMathematicalPlasticity & Entropy-Aware Architecture
- Assets:  
  - (B) Recursive fractal entropy provenance pipeline: `blueprints/AI_detection/*`  
  - (M) `models/scbf/`, `models/CIMM/` (plasticity / adaptation)  
  - (U) `devkit/compression/`, `devkit/random_number/` (entropy measurement tooling)  
  - (T) `infodynamics.md` (definitions), `foundational/arithmetic/` (symbolic math ops)  
  - (R) `roadmaps/symbolic_entropy_roadmap.md`  
  - (C) `citations/*`  

### ResonantSymbolicConvergence Preprint
- Assets:  
  - (T) Symbolic convergence constructs: `foundational/lexicon.*`, `infodynamics.md`  
  - (M) `models/scbf/`  
  - (R) `roadmaps/symbolic_entropy_roadmap.md`  
  - (B) Any resonance / convergence blueprint (if missing → create)  
  - (U) `tools/` (symbolic converters)  
  - (C) `citations/*`  

### SECPreprint
- (Assumed) SEC = Symbolic Entropy Convergence (confirm).  
- Assets:  
  - (Backlog Plan) `backlog/[m][D][v0.1][C6][I1]_cimm_ml_vs_sec_benchmark_plan.md`  
  - (R) `roadmaps/symbolic_entropy_roadmap.md`  
  - (M) `models/CIMM/`, `models/scbf/`  
  - (T) `infodynamics.md`, `foundational/lexicon.*`  
  - (U) `devkit/` (benchmark harness pieces)  
  - (C) `citations/*`  
  - Action: Validate acronym expansion + define in lexicon if absent.

## 4. Iteration Workflow
Phase 0 (Prep): Extract current heading structure & add TODO tags where sections absent.  
Phase 1 (Normalization): Apply template ordering; insert placeholder headings (NO content inflation yet).  
Phase 2 (Traceability Pass): For each factual / claim statement add inline tag: `[TRACE: path/to/source#anchor]` (temporary markup); later replaced by formal citations or appendix callouts.  
Phase 3 (Citation Sync): Crosswalk all added references into `citations_bibtex.bib` & `citations_apa.txt`; prune unused.  
Phase 4 (Terminology): Run lexicon diff: ensure every capitalized domain term has lexicon entry or add.  
Phase 5 (Quant / Methods Rigor): Ensure equations & algorithm pseudocode enumerated (Eq. 1, Alg. 1).  
Phase 6 (Cross-Doc Consistency): Check duplicated conceptual paragraphs; consolidate into canonical location (usually DFT or foundational).  
Phase 7 (Quality Gates): Lint (format, heading levels), link-check internal refs, word count, abstract length, missing figure alt text, unresolved TRACE tags=0.  
Phase 8 (Freeze): Tag git annotated tag `preprint-iter2-freeze`.

## 5. Task Breakdown (Per Draft)
For each draft create a checklist block appended at file end under `<!-- ITER2_CHECKLIST -->`.

Checklist Template:
```
<!-- ITER2_CHECKLIST -->
- [ ] Template normalized
- [ ] All sections present (Intro, Methods, Results, Discussion, Conclusion)
- [ ] TRACE tags resolved
- [ ] Citations updated
- [ ] Terminology validated
- [ ] Equations numbered
- [ ] Figures / diagrams referenced
- [ ] Acronyms defined on first use
- [ ] Abstract ≤ 250 words
- [ ] Limitations section present
```

## 6. Automation Hooks (Future Optional)
- Script idea: `tools/converter.py` extension to scan drafts for `TRACE:` patterns and build an evidence map update feeding into `EVIDENCE_MAP.md`.  
- Add simple markdown linter config (if absent) to enforce heading increments.  
- Potential: Python script to inject checklist if missing.

## 7. Version Control Conventions
- Branch naming: `docs/preprints/iter2/<slug>`  
- Singular squash merge per draft into `main`.  
- Commit prefixes: `[ITER2][AIX] Normalize template`, `[ITER2][SEC] Citations pass`, etc.  
- Tag after all merges: `preprint-iter2-freeze`.

## 8. Quality Gate Criteria
| Gate | Tooling / Method | Pass Condition |
|------|------------------|----------------|
| Structure | Manual diff / heading extractor | Matches template |
| Traceability | Grep for `TRACE:` | Zero unresolved |
| Citations | Compare bib vs in-text keys | No missing references |
| Terminology | Lexicon crosswalk script (future) | All domain terms defined |
| Figures | Search `![` patterns | Each has alt text + caption reference |
| Equations | Regex `(Eq\. ?\d+)` continuity | No gaps / duplicates |
| Word Count | Simple script | Abstract ≤ 250 words |

## 9. Open Questions / Assumptions
1. SEC acronym final expansion unconfirmed (assumed Symbolic Entropy Convergence).  
2. Some model subdirectories may still be placeholders; methods referencing them will label as prospective if code incomplete.  
3. If resonance blueprint missing, create `blueprints/resonant_symbolic_convergence.md` (flagged).  

## 10. Immediate Next Actions
1. Inject checklist block into each draft if absent.  
2. Normalize headings order (Phase 1).  
3. Insert TRACE placeholders where claims lack explicit repository pointer.  
4. Prepare simple grep-based scripts (optional) for gates.  

---
Owner: (assign)  
Cycle End Target: 2025-08-20  
Prepared by: GitHub Copilot
