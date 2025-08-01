# MCP Server README

## Overview
The MCP (Model Context Protocol) server provides experimental agentic navigation and resource access for the Dawn Field Theory codebase. This is an **experimental offering** designed to help researchers, AI systems, and collaborators connect to the repository on their own terms, using intelligent, context-aware exploration rather than manual file browsing.

It natively integrates with the Cognition Index Protocol (CIP) and the GPT Resource Guide, enabling instruction-driven exploration and automation that understands the theoretical structure of the codebase.

## Experimental Status
⚠️ **This is experimental infrastructure.** We're exploring how AI systems can intelligently navigate research repositories using semantic protocols. The server may evolve rapidly as we learn what works best for different use cases and user needs.

Our goal is to make research codebases more accessible and discoverable, allowing people to engage with complex theoretical work on their own terms rather than requiring deep familiarity with the file structure.

## Key Features
- **CIP Integration:** All endpoints return actionable instructions from the CIP resource guide, ensuring every resource or query is contextually mapped to the relevant theory, blueprint, or protocol.
- **Agentic Endpoints:** Tools and resources are exposed for agentic workflows, including file listing, reading, and repo search.
- **Security:** All file and directory access is sandboxed to the repo root.
- **Batch Operations:** Read multiple files at once for efficient exploration.
- **CIP Enhancement Tools:** Experimental validation, cross-reference discovery, and metadata extraction.

## Endpoints
### Resource Endpoint
- `repo_resource(path: str)`
  - Browse files or directories in the repo.
  - Returns file contents or directory listing, plus a `cip_instruction` field with guidance from the resource guide.

### Core Tools
- `list_files(path: str)`
  - Lists files and directories under a given path.
  - Returns a list of items and a `cip_instruction` for context.

- `read_file(path: str)`
  - Reads the contents of a file.
  - Returns file content and a `cip_instruction` for context.

- `read_files(paths: List[str])`
  - Batch read multiple files at once.
  - Returns a dictionary mapping each path to its content and CIP instruction.

- `search_repo(query: str, path: str)`
  - Searches for a literal string in the repo.
  - Returns file:line matches and a `cip_instruction` relevant to the query.

### Experimental CIP Enhancement Tools
- `validate_cip_compliance(path: str)`
  - Checks if a file or directory follows CIP standards.
  - Returns compliance status, issues, and suggestions for improvement.

- `find_related_content(path: str)`
  - Discovers related files based on CIP metadata connections.
  - Returns documents and experiments from the same theoretical framework.

- `extract_metadata(path: str)`
  - Extracts CIP metadata, YAML frontmatter, and structural information from files.
  - Returns parsed metadata and content analysis.

## How It Works
- The server loads the CIP resource guide from `cognition_index_protocol/gpt/gpt_resource_guide.yaml`.
- For any file path or query, the server maps it to the most relevant instruction in the guide, surfacing actionable context for agents and users.
- All outputs include a `cip_instruction` field, making agentic workflows seamless and discoverable.
- Enhancement tools provide deeper insight into the semantic structure and compliance of repository content.

## Usage Examples
```python
# List files in a directory with context
result = list_files('foundational/docs')
print(result['items'])
print(result['cip_instruction'])

# Read a file with guidance
result = read_file('foundational/lexicon.md')
print(result['content'])
print(result['cip_instruction'])

# Batch read multiple files
result = read_files(['README.md', 'foundational/lexicon.md'])
for path, data in result.items():
    print(f"{path}: {data['cip_instruction']}")

# Search with contextual results
result = search_repo('collapse', 'foundational/docs')
print(result['results'])
print(result['cip_instruction'])

# Check CIP compliance (experimental)
result = validate_cip_compliance('foundational/docs/some_file.md')
print(f"Compliant: {result['compliant']}")
print(f"Issues: {result['issues']}")
print(f"Suggestions: {result['suggestions']}")

# Find related content (experimental)
result = find_related_content('foundational/docs/[m][F][v1.0][C4][I5]_recursive_balance_field.md')
print(f"Theory: {result['current_theory']}")
for related in result['related']:
    print(f"Related: {related['path']} ({related['relation']})")
```

## Philosophy
This server embodies our belief that **research repositories should be intelligent, not just collections of files.** By integrating semantic protocols and contextual guidance, we're exploring how AI systems can help humans navigate complex theoretical work more effectively.

Rather than requiring people to understand our specific organizational structure, the server provides contextual guidance that helps users engage with the content on their own terms.

## Extending the Server
- To add new endpoints, ensure outputs include a `cip_instruction` field for consistency.
- Update the resource guide as new theories, blueprints, or protocols are added.
- Consider the semantic relationships when designing new enhancement tools.
- Test extensively with different AI systems and user workflows.

## Future Directions

### GAIA Memory Architecture
This MCP server pattern represents a potential foundation for how AI systems like GAIA could implement **long-term memory without traditional training data**:

- **Dynamic Knowledge Access:** Instead of frozen parameters, AI systems could live-connect to intelligent repositories that evolve over time
- **Contextual Learning:** CIP instructions provide semantic guidance for each interaction, enabling understanding without parameter updates
- **Distributed Intelligence:** Knowledge lives in structured, semantically-aware repositories rather than monolithic training sets
- **Network Effect Memory:** Each CIP-compliant repository becomes a specialized memory module that knows how to teach itself

### Model Repos as Memory Modules
**Vision:** Replace static training datasets with networks of intelligent repositories that:
- Provide **contextual guidance** for every piece of information
- **Evolve automatically** as research progresses
- **Teach themselves** through semantic protocols
- **Connect dynamically** to AI systems as needed

### Research Repository Networks
This approach could enable:
- **Cross-repository knowledge discovery** through shared CIP protocols
- **Collaborative research memory** where repositories can reference and build on each other
- **Semantic interoperability** between different research domains
- **AI systems that grow smarter** by connecting to more repositories rather than requiring retraining

The goal is **repositories that actively participate in knowledge creation** rather than passively storing information.

## Contributing & Feedback
This is experimental infrastructure, and we welcome feedback on:
- What works well for your use case
- What's missing or confusing
- Ideas for new enhancement tools
- Suggestions for better semantic mapping
- **Thoughts on the GAIA memory architecture vision**
- **Ideas for repository-to-repository communication protocols**

## Contact
For questions, contributions, or feedback on this experimental infrastructure, see the main Dawn Field Theory repository or contact the maintainers. We're particularly interested in hearing from researchers who might want to adopt similar approaches for their own repositories.
