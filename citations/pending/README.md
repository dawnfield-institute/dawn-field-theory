# Pending Citations

This folder is for contributors to submit citation data as part of their Pull Requests.

## How to Submit

1. Copy `../pr-citation-template.yaml` to this folder
2. Rename it to `pr-{PR_NUMBER}-{short-description}.yaml`
3. Fill in your contribution details
4. Include it in your PR

## Example Filename
```
pr-123-entropy-operator.yaml
pr-124-validation-framework.yaml
pr-125-documentation-update.yaml
```

**Note:** Files starting with `example-` are blacklisted from processing and serve as reference templates only.

## Processing

When your PR is merged:
- Citation data is automatically processed and integrated
- Your file is moved to `../processed/` for record-keeping
- Citation indexes and files are updated automatically

## Need Help?

See `../README.md` for complete citation guidelines or open an issue if you have questions.
