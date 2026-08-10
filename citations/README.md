# Citation System for Dawn Field Theory Repository

This folder contains the automated citation infrastructure for tracking and attributing contributions to the Dawn Field Theory repository.

## Repository Citation

**Primary Citation:** Use the root `CITATION.cff` file for citing the overall repository and Peter Groom's theory work.

**BibTeX format:**
```bibtex
@software{dawnfield_theory,
  author       = {Peter Lorne Groom},
  title        = {Dawn Field Theory Repository},
  year         = {2025},
  url          = {https://github.com/dawnfield-institute/dawn-field-theory},
  note         = {For component-specific citations, see contributors index}
}
```

## Automated Citation Workflow

### For Contributors:

1. **Copy the template**: `pr-citation-template.yaml` to `pending/`
2. **Rename**: Use pattern `pr-{PR_NUMBER}-{short-description}.yaml`
3. **Fill in details**: Your contributor info, contribution description, and affected files
4. **Include in PR**: Add the completed citation YAML to your pull request
5. **Automatic validation**: When you open the PR, citation files are automatically validated
6. **Review feedback**: Address any validation issues shown in the automated PR comments
7. **Automatic processing**: Upon merge, GitHub Actions processes and integrates your citation

**Two-stage workflow:**
- 🔍 **PR Validation**: Files validated on PR open/update with feedback via comments
- ✅ **Merge Processing**: Valid citations processed and integrated when PR is merged

### What qualifies as citable:
- ✅ New experimental modules or validation frameworks
- ✅ Theoretical extensions or mathematical contributions  
- ✅ Major algorithm implementations (>100 lines of significant code)
- ✅ Substantial documentation contributions (e.g., comprehensive tutorials)
- ❌ Bug fixes, typos, minor formatting changes
- ❌ Code refactoring without functional changes

### Automated Outputs:
When your PR is merged with a citation file, the system automatically:
- Updates `contributors-index.json` with your information
- Generates BibTeX entries in `contributors_bibtex.bib`
- Updates the main `CITATION.cff` file with contributor info
- Moves your citation file to `processed/` for record-keeping

## File Structure

```
citations/
├── README.md                    # This file
├── pr-citation-template.yaml    # Template for contributors
├── contributors-index.json      # Auto-generated contributor index
├── contributors_bibtex.bib      # Auto-generated contributor BibTeX entries
├── pending/                     # Submit citation files here
│   ├── README.md
│   └── example-pr-123-entropy-operator.yaml
├── processed/                   # Auto-archived processed citations
│   ├── README.md
│   └── [YYYYMMDD-processed-files]
└── external_citations/          # External/theory references
    ├── README.md
    ├── external_citations.md
    ├── citations_bibtex.bib     # External reference BibTeX
    └── citations_apa.txt        # External reference APA format
```

## Technical Details

The citation system is powered by two GitHub Actions workflows:
- **Validation**: `.github/workflows/validate-citations.yml` - Validates citation files on PR open/update
- **Processing**: `.github/workflows/process-citations.yml` - Processes citations on PR merge
- **Processing Script**: `tools/process_citations.py` - Core citation processing logic
- **Validation Script**: `tools/validate_citations.py` - Citation validation and error checking

## External Citations

For theory and external references that inform Dawn Field Theory, see [`external_citations/README.md`](external_citations/README.md). This includes classical papers in information theory, entropy, complex systems, and related fields that provide theoretical context for the project.

## Integration with Registration Process

As part of the contributor registration process (see `CONTRIBUTION.md`), contributors provide their citation preferences and ORCID information, which streamlines the citation process for their PRs.

## Questions?

For questions about the citation system, contact info@dawnfield.ca or open a GitHub Issue.
