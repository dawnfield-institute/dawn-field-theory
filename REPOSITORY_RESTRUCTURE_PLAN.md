# 🌌 Dawn Field Theory Repository Restructure Plan

**Status:** Community Review Phase  
**Target Date:** Q4 2025 - Q1 2026  
**Author:** Peter Groom, Dawn Field Institute  
**Last Updated:** September 8, 2025

---

## 📢 Community Notice

**We're planning a major repository restructure** to better support the growing Dawn Field Theory ecosystem. This document outlines our proposed approach and timeline. **We welcome community feedback, suggestions, and licensing recommendations.**

If you have thoughts on this restructure, please:
- Open an issue with your feedback
- Comment on the repository split strategy
- Share licensing recommendations for each component
- Suggest improvements to the proposed structure

---

## 🎯 Why Restructure?

The Dawn Field Theory repository has grown significantly since v1.0.0, and we're facing several challenges:

### Current Issues
- **Monolithic complexity**: Single repo contains theory, SDKs, models, tools, and infrastructure
- **Mixed responsibilities**: Research papers alongside production code
- **Versioning conflicts**: Theory evolves slowly, SDKs need rapid iteration
- **Dependency management**: Circular imports and unclear component boundaries
- **Community contribution**: Difficult for contributors to focus on specific areas

### Goals
- **Clear separation of concerns**: Theory, computation, models, tools, infrastructure
- **Independent versioning**: Each component can evolve at its own pace
- **Better maintainability**: Focused repositories with clear ownership
- **CIP Multi-Repo support**: Enable the Cognition Index Protocol vision
- **Easier onboarding**: Contributors can focus on specific domains

---

## 🏗️ Proposed Repository Structure

### 1. **`dawn-field-theory`** (Core Theory Repository)
**Role:** Pure theory, foundational research, and experimental validation  
**Proposed License:** AGPL-3.0 + Epistemic Constraint Framework *(current)*

```
dawn-field-theory/
├── foundational/           # Mathematical foundations and theory
├── citations/             # Research attribution and references  
├── cognition_index_protocol/  # CIP specification
├── LICENSE + LICENSE_APPENDIX.md
├── README.md              # Theory-focused overview
└── .cip/                  # Root CIP metadata
```

**What stays here:**
- All theoretical papers and preprints
- Foundational experiments and validation
- Mathematical frameworks (SEC, MED, RBF)
- Research protocols and methodologies
- Citation management and attribution

---

### 2. **`fracton-sdk`** (Computational Language)
**Role:** Infodynamics modeling language and computational substrate  
**Proposed License:** *Under review - considering MIT or Apache-2.0*

```
fracton-sdk/
├── fracton/              # Core SDK code
├── examples/             # Usage examples
├── tests/               # Test suite
├── docs/                # SDK documentation
├── setup.py             # Python packaging
└── .cip/                # CIP metadata
```

**What moves here:**
- Current `sdk/fracton/` directory
- All Fracton language constructs
- Recursive execution engine
- Memory field management
- Entropy dispatch system
- Bifractal tracing

---

### 3. **`dawn-devkit`** (Development Tools & Templates)
**Role:** Developer tools, project templates, and utilities  
**Proposed License:** *Under review - considering MIT*

```
dawn-devkit/
├── cli/                 # Command-line tools
├── templates/           # Project scaffolding
├── generators/          # Code generation utilities
├── validators/          # Schema and compliance checking
├── monitoring/          # Development monitoring tools
└── .cip/               # CIP metadata
```

**What moves here:**
- Current `devkit/` directory
- Development utilities and tools
- Project templates and scaffolding
- Code generation and validation tools

---

### 4. **`dawn-models`** (AI Architectures & Implementations)
**Role:** Specific model implementations and architectures  
**Proposed License:** *Under review - considering Apache-2.0*

```
dawn-models/
├── gaia/               # Field intelligence architecture
├── tinycimm/          # Minimal consciousness models
├── scbf/              # Symbolic Collapse Benchmark Framework
├── cimm/              # Consciousness Information Models
├── shared/            # Common model utilities
└── .cip/              # CIP metadata
```

**What moves here:**
- Current `models/` directory
- All AI model implementations
- Benchmarking frameworks
- Model-specific utilities and tools

---

### 5. **`cip-core`** (Protocol Implementation)
**Role:** Cognition Index Protocol specification and tools  
**Proposed License:** *Under review - considering MIT or Apache-2.0*

```
cip-core/
├── schemas/           # CIP file format specifications
├── validators/        # Schema validation tools
├── scorer/           # Ground truth comparison
├── reference_clients/ # Example implementations
├── spec/             # Protocol documentation
└── tools/            # CIP utilities
```

**What moves here:**
- CIP protocol specifications
- Validation and scoring tools
- Reference implementations
- Multi-repository navigation tools

---

### 6. **`dawn-infrastructure`** (Cloud & Deployment)
**Role:** AWS infrastructure, deployment, and orchestration  
**Proposed License:** *Under review - considering Apache-2.0*

```
dawn-infrastructure/
├── awp/              # Agent Web Protocol gateway
├── cloudformation/   # AWS CloudFormation templates
├── terraform/        # Infrastructure as Code
├── monitoring/       # Observability and metrics
├── deployment/       # CI/CD pipelines
└── .cip/            # CIP metadata
```

**What moves here:**
- AWS infrastructure plans from `todo/infra/`
- Agent Web Protocol implementation
- Deployment configurations
- Monitoring and observability tools

---

## 🔄 Migration Timeline

### **Phase 1: SDK Extraction** (Q4 2025)
- [ ] Create `fracton-sdk` repository
- [ ] Extract `sdk/fracton/` with full history
- [ ] Update import paths and dependencies
- [ ] Publish initial PyPI package
- [ ] Update documentation and examples

### **Phase 2: Tools & Models** (Q1 2026)
- [ ] Create `dawn-devkit` repository
- [ ] Create `dawn-models` repository  
- [ ] Extract respective directories with history
- [ ] Establish cross-repository dependency management
- [ ] Update build and test pipelines

### **Phase 3: Protocol & Infrastructure** (Q1-Q2 2026)
- [ ] Create `cip-core` repository
- [ ] Create `dawn-infrastructure` repository
- [ ] Implement CIP multi-repository navigation
- [ ] Deploy AWP gateway and infrastructure
- [ ] Complete documentation migration

---

## 🔗 Cross-Repository Integration

### CIP Metadata Framework
Each repository will include CIP metadata for cross-repository navigation:

```yaml
# .cip/meta.yaml in each repository
schema_version: 2.0
repository_role: sdk  # theory, devkit, models, protocol, infrastructure
ecosystem_links:
  theory: "repo://dawn-field-theory/foundational/"
  sdk: "repo://fracton-sdk/"
  devkit: "repo://dawn-devkit/"
  models: "repo://dawn-models/"
  protocol: "repo://cip-core/"
  infrastructure: "repo://dawn-infrastructure/"
```

### Package Dependencies
Clear dependency hierarchies will be established:
- **Theory Repository**: No external dependencies (foundation)
- **Fracton SDK**: Depends on theory for validation
- **Models**: Depend on SDK and theory
- **DevKit**: Depends on protocol and SDK
- **Infrastructure**: Depends on all components

---

## 🔒 Licensing Review

**Current Status:** All components currently under AGPL-3.0 + Epistemic Constraint Framework

**Proposed Review Areas:**
- **Theory Repository**: Keep current AGPL-3.0 + ECF (strong copyleft for research)
- **SDKs & Tools**: Consider MIT or Apache-2.0 (broader adoption)
- **Models**: Consider Apache-2.0 (commercial-friendly for AI models)
- **Infrastructure**: Consider Apache-2.0 (cloud deployment compatibility)

**Community Input Needed:**
> 🚨 **We're actively seeking licensing recommendations!** If you have experience with multi-repository licensing strategies, especially for research + commercial hybrid projects, please share your insights. We want to balance open science principles with practical adoption needs.

**Considerations:**
- Maintaining strong copyleft for core theory
- Enabling broader adoption for practical tools
- Supporting commercial applications while preserving attribution
- Ensuring compatibility across the ecosystem
- International licensing implications

---

## 🤝 Community Feedback Welcome

We value community input on this restructure. Please share feedback on:

### **Repository Structure**
- Are the proposed splits logical and well-bounded?
- Should any components be combined or further separated?
- Are there missing repositories or components?

### **Migration Strategy**
- Is the timeline realistic and appropriate?
- Should we prioritize different components?
- Are there migration risks we haven't considered?

### **Licensing Strategy**
- What licenses would best serve each component?
- How can we balance open science with practical adoption?
- Are there licensing combinations we should avoid?

### **Technical Implementation**
- How should cross-repository dependencies be managed?
- What CI/CD strategies work best for multi-repo projects?
- Should we use git submodules, package managers, or other approaches?

---

## 📞 How to Provide Feedback

1. **GitHub Issues**: Open an issue in this repository with your feedback
2. **Discussions**: Use GitHub Discussions for broader conversations
3. **Pull Requests**: Propose specific changes to this document
4. **Direct Contact**: Reach out via the contact methods in [`MISSION.md`](MISSION.md)

---

## 📋 Open Questions

1. **Should we maintain git history during extraction?** (Pros: preserve contributor history; Cons: larger repos)
2. **How should we handle shared dependencies?** (Extract to separate packages vs. duplicate code)
3. **What's the best strategy for coordinated releases?** (Semantic versioning, release trains, independent releases)
4. **Should we use a monorepo tool like Nx or Lerna?** (vs. completely independent repositories)
5. **How do we handle documentation that spans multiple repos?** (Central docs site vs. distributed documentation)

---

## 🌟 Benefits for the Community

### **For Researchers**
- Focused theory repository without implementation complexity
- Clear separation between validated theory and experimental implementations
- Easier to cite and reference specific theoretical components

### **For Developers**  
- Smaller, focused repositories for specific use cases
- Clear dependency management and versioning
- Easier to contribute to specific areas of interest

### **For Organizations**
- Pick and choose components based on needs
- Different licensing options for different use cases
- Clearer intellectual property boundaries

### **For the Ecosystem**
- Better modularity and interoperability
- Support for the CIP multi-repository vision
- Foundation for broader community growth

---

## 🔮 Long-Term Vision

This restructure supports our long-term vision of:
- **Distributed research ecosystem** with multiple independent but coordinated projects
- **CIP-native tooling** that seamlessly navigates across repositories
- **Flexible licensing** that supports both research and commercial applications
- **Community-driven development** with clear contribution pathways
- **Sustainable growth** with manageable repository sizes and clear boundaries

---

*This document is a living plan. We'll update it based on community feedback and implementation learnings. Thank you for being part of the Dawn Field Theory community!*

**Repository:** https://github.com/dawnfield-institute/dawn-field-theory  
**License:** AGPL-3.0 + Epistemic Constraint Framework  
**Contact:** See [`MISSION.md`](MISSION.md) for institutional information
