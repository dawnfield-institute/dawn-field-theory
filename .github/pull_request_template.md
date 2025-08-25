# 📝 Pull Request Template — Dawn Field Theory

## Summary

<!-- Briefly describe your contribution and its purpose. -->

---

## Checklist

- [ ] I have read the [CONTRIBUTION.md](../CONTRIBUTION.md) guidelines
- [ ] I have registered as a contributor (if required)
- [ ] I have reviewed the publishing boundaries and citation process
- [ ] I have added/updated documentation as needed
- [ ] I have included a citation YAML in `citations/pending/` for substantial contributions
- [ ] Citation validation has passed (automatically checked when PR is opened)

---

## Description of Changes

<!-- List major files, modules, or docs changed. -->

---

## Testing

<!-- Describe how you tested your changes and/or add test instructions. -->

---

## Citation Metadata (for substantial contributions)

If your PR represents a substantial theoretical, experimental, or code contribution, please:

1. **Copy the template**: `citations/pr-citation-template.yaml` to `citations/pending/`
2. **Rename it**: Use pattern `pr-{PR_NUMBER}-{short-description}.yaml`
3. **Fill in your details**: Contributor info, contribution description, and affected files
4. **Validate locally** (optional): Run `python tools/validate_citations.py` to check formatting
5. **Include in this PR**: The citation YAML will be automatically validated when you open the PR
6. **Review feedback**: Address any validation issues shown in the automated comment
7. **Merge**: Your citation will be automatically processed and integrated upon merge

**Automated Workflow:**
- 🔍 **On PR open/update**: Citation files are validated and results posted as comments
- ✅ **On PR merge**: Valid citations are processed, integrated, and archived automatically


**What qualifies for citation:**
- ✅ New experimental frameworks or validations
- ✅ Theoretical extensions or novel operators
- ✅ Major implementations (>100 lines of significant code)
- ✅ Substantial documentation contributions
- ❌ Minor bug fixes, typos, or formatting changes

See [`citations/README.md`](../citations/README.md) for complete guidelines.

---

## Additional Notes

<!-- Add any other context, links, or information here. -->
