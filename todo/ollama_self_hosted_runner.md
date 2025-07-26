# TODO: Self-hosted Runner with Ollama Integration for meta.yaml Automation

- [ ] Set up a VM or server to act as a self-hosted GitHub Actions runner
    - Install Python, Ollama, and all required dependencies
    - Register the runner with the dawn-field-theory repository
- [ ] Update workflow YAML to use `runs-on: [self-hosted]`
- [ ] Enhance `tools/update_meta_yamls.py` to call Ollama for semantic scope and description generation
- [ ] Test the workflow end-to-end (push, update, commit, and push changes)
- [ ] Document the setup and usage for future maintainers
