# Fractal Document Object (FDO) Specification v0.1

## Overview

The Fractal Document Object (FDO) specification defines a YAML-based format for creating AI-native documents with embedded semantic structure, temporal indexing, and recursive contextual awareness. FDO documents become active knowledge nodes that support bifractal emergence, cross-document resolution, and intelligent context loading.

## Design Principles

### 1. AI-Native Architecture
FDO documents are designed from the ground up for AI consumption and interaction, with semantic vectors, typed relationships, and machine-readable metadata as first-class citizens.

### 2. Bifractal Temporal Structure
Every FDO document embeds both forward and backward temporal references, enabling traceability of idea emergence and influence propagation across time.

### 3. Semantic Atomization
Documents decompose into addressable semantic chunks that can be independently reasoned about, linked, and evolved while maintaining contextual relationships.

### 4. Cross-Document Intelligence
FDO documents form semantic webs that transcend individual document boundaries, enabling trans-document cognitive networks and unified knowledge graphs.

### 5. Provenance and Evolution
Every element tracks its source, authorship, and evolution history, supporting collaborative AI-human document development and delta resolution.

## File Structure and Naming

### File Extensions
- `.fdo.yaml` - Primary FDO document format
- `.fdo.json` - Alternative JSON format for programmatic generation
- `.fdo.schema.yaml` - Schema definitions for validation

### Naming Conventions
```
document-name.fdo.yaml
project-overview.fdo.yaml
research-findings-2025.fdo.yaml
```

### Directory Structure
```
project/
├── docs/
│   ├── overview.fdo.yaml
│   ├── architecture.fdo.yaml
│   └── implementation.fdo.yaml
├── schemas/
│   └── fdo.schema.yaml
└── templates/
    └── basic-template.fdo.yaml
```

## Core Schema Definition

### Document Header
```yaml
# Required document metadata
doc_id: string                    # Unique identifier for document
version: string                   # FDO format version (e.g., "fdo-v0.1")
title: string                     # Human-readable document title
authors: [string]                 # List of document authors
doc_type: string                  # Document type classification
created: datetime                 # Document creation timestamp
modified: datetime                # Last modification timestamp

# Optional document metadata
audience: [string]                # Target audience classifications
keywords: [string]                # Document keywords for discovery
complexity_level: string          # Complexity classification
license: string                   # Document license
source_document: string           # Original source file reference
```

### Metadata Structure
```yaml
metadata:
  # Source and provenance
  source_document: string         # Original document path/URL
  license: string                  # License information
  keywords: [string]              # Search keywords
  complexity_level: string        # "basic" | "intermediate" | "advanced" | "expert"
  
  # Quality and validation
  quality_score: float            # Overall document quality (0.0-1.0)
  validation_status: string       # "validated" | "pending" | "error"
  last_validated: datetime        # Last validation timestamp
  
  # Processing metadata
  processing_history: [ProcessingStep]
  embedding_metadata: EmbeddingMetadata
  relationship_stats: RelationshipStats
```

### Chunk Definition
```yaml
chunks:
  - id: string                    # Unique chunk identifier (hierarchical: "1.0", "1.1", "2.3.1")
    type: string                  # Chunk type from ontology
    title: string                 # Optional chunk title
    content: string               # Primary chunk content
    
    # Optional chunk metadata
    tone: string                  # "neutral" | "speculative" | "assertive" | "questioning"
    confidence: float             # Author confidence in content (0.0-1.0)
    reading_level: string         # "elementary" | "middle" | "high" | "graduate"
    word_count: integer           # Automatic word count
    
    # Semantic and AI metadata
    embedding: [float]            # Optional inline embedding vector
    entities: [NamedEntity]       # Extracted named entities
    concepts: [string]            # Key concepts in chunk
    
    # Relationship definitions
    links_to: [ChunkRelationship] # Outgoing relationships
    
    # Provenance and evolution
    provenance: ProvenanceInfo    # Author and source information
    metadata: ChunkMetadata       # Additional chunk-specific metadata
```

### Relationship Definition
```yaml
# Individual chunk relationship
links_to:
  - chunk: string                 # Target chunk ID
    doc: string                   # Target document ID (for cross-doc refs)
    type: string                  # Relationship type from ontology
    weight: float                 # Relationship strength (0.0-1.0)
    description: string           # Optional relationship description
    inverse: boolean              # Whether this is an inverse relationship
    created: datetime             # When relationship was created
    validated: boolean            # Whether relationship has been validated
```

## Chunk Type Ontology

### Primary Content Types
```yaml
chunk_types:
  # Structural types
  thesis:
    description: "Primary thesis or main argument"
    color: "#FF6B6B"
    icon: "thesis"
    
  evidence:
    description: "Supporting evidence or data"
    color: "#4ECDC4"
    icon: "evidence"
    
  analysis:
    description: "Analytical reasoning or interpretation"
    color: "#45B7D1"
    icon: "analysis"
    
  conclusion:
    description: "Conclusions or synthesis"
    color: "#96CEB4"
    icon: "conclusion"
    
  # Argument types
  claim:
    description: "Factual claim or assertion"
    color: "#FECA57"
    icon: "claim"
    
  counterpoint:
    description: "Opposing view or counterargument"
    color: "#FF9FF3"
    icon: "counterpoint"
    
  synthesis:
    description: "Synthesis of multiple perspectives"
    color: "#54A0FF"
    icon: "synthesis"
    
  # Metadata types
  definition:
    description: "Definition or explanation of terms"
    color: "#5F27CD"
    icon: "definition"
    
  example:
    description: "Illustrative example or case study"
    color: "#00D2D3"
    icon: "example"
    
  question:
    description: "Research question or inquiry"
    color: "#FF6348"
    icon: "question"
    
  methodology:
    description: "Methodological approach or process"
    color: "#2ED573"
    icon: "methodology"
    
  future_work:
    description: "Future research directions"
    color: "#FFA502"
    icon: "future"
```

## Relationship Type Ontology

### Core Relationship Types
```yaml
relationship_types:
  # Support relationships
  supports:
    description: "Provides evidence or logical support"
    inverse: "supported_by"
    weight_range: [0.0, 1.0]
    color: "#2ECC71"
    
  evidence_for:
    description: "Serves as evidence for claim"
    inverse: "evidenced_by"
    weight_range: [0.0, 1.0]
    color: "#27AE60"
    
  # Opposition relationships
  contradicts:
    description: "Logically contradicts or challenges"
    inverse: "contradicted_by"
    weight_range: [0.0, 1.0]
    color: "#E74C3C"
    
  challenges:
    description: "Questions or challenges assumption"
    inverse: "challenged_by"
    weight_range: [0.0, 1.0]
    color: "#C0392B"
    
  # Reference relationships
  refers_to:
    description: "General reference or citation"
    inverse: "referenced_by"
    weight_range: [0.0, 1.0]
    color: "#3498DB"
    
  builds_on:
    description: "Builds upon or extends idea"
    inverse: "extended_by"
    weight_range: [0.0, 1.0]
    color: "#2980B9"
    
  # Synthesis relationships
  synthesizes:
    description: "Combines or synthesizes multiple ideas"
    inverse: "synthesized_by"
    weight_range: [0.0, 1.0]
    color: "#9B59B6"
    
  resolves:
    description: "Resolves conflict or tension"
    inverse: "resolved_by"
    weight_range: [0.0, 1.0]
    color: "#8E44AD"
    
  # Dependency relationships
  depends_on:
    description: "Logically depends on another chunk"
    inverse: "depended_on_by"
    weight_range: [0.0, 1.0]
    color: "#F39C12"
    
  enables:
    description: "Enables or makes possible"
    inverse: "enabled_by"
    weight_range: [0.0, 1.0]
    color: "#E67E22"
```

## Provenance and Authorship

### Provenance Information
```yaml
provenance:
  source: string                  # "human" | "ai" | "collaborative" | "system"
  author: string                  # Author identifier or name
  author_type: string             # "human" | "ai" | "system"
  
  # For AI-generated content
  model: string                   # AI model used (e.g., "gpt-4")
  model_version: string           # Model version
  prompt_context: string          # Context or prompt used
  
  # Delta and conflict resolution
  delta_status: string            # "original" | "modified" | "reconciled" | "conflicted"
  conflict_resolution: string     # "synthesis" | "override" | "merge" | "flag"
  resolution_strategy: string     # Strategy used for conflict resolution
  
  # Timestamps and versioning
  created: datetime               # When chunk was created
  last_modified: datetime         # Last modification time
  version: string                 # Chunk version identifier
  
  # Quality and validation
  confidence_score: float         # Confidence in content (0.0-1.0)
  validation_status: string       # "validated" | "pending" | "rejected"
  notes: string                   # Additional provenance notes
```

## Embeddings and Semantic Metadata

### Embedding Configuration
```yaml
embeddings:
  model: string                   # Embedding model used
  dimension: integer              # Vector dimension
  generated: datetime             # When embeddings were generated
  index_status: string            # "indexed" | "pending" | "error"
  
  # Model-specific configuration
  model_config:
    normalize: boolean            # Whether vectors are normalized
    max_length: integer           # Maximum input length
    batch_size: integer           # Batch size used for generation
    
  # Quality metrics
  quality_metrics:
    coherence_score: float        # Embedding coherence (0.0-1.0)
    similarity_distribution: object # Statistics on similarity scores
```

### Named Entity Structure
```yaml
entities:
  - text: string                  # Entity text
    label: string                 # Entity type (PERSON, ORG, CONCEPT, etc.)
    start: integer                # Start position in text
    end: integer                  # End position in text
    confidence: float             # Extraction confidence (0.0-1.0)
    
    # Optional entity metadata
    canonical_form: string        # Canonical entity name
    external_ids: object          # External knowledge base IDs
    description: string           # Entity description
```

## Cross-Document References

### External Reference Format
```yaml
# Reference to chunk in another document
links_to:
  - doc: "symbolic-collapse-theory"     # Target document ID
    chunk: "1.2"                        # Target chunk ID
    type: "supports"                    # Relationship type
    weight: 0.9                         # Relationship strength
    
    # Resolution metadata
    resolution_path: string             # How to resolve document
    access_requirements: [string]       # Required permissions
    cache_duration: integer             # How long to cache resolution
```

### Document Registry Integration
```yaml
# Document references for resolution
document_registry:
  local_documents: [string]             # Local document paths
  remote_repositories: [RepositoryRef] # Remote repository references
  resolution_strategies: [ResolutionStrategy]
```

## Backlink Generation

### Automatic Backlink Structure
```yaml
# Auto-generated during compilation
backlinks:
  - from_doc: string              # Source document ID
    from_chunk: string            # Source chunk ID
    to_chunk: string              # Target chunk ID (this document)
    type: string                  # Relationship type
    weight: float                 # Relationship weight
    
    # Metadata
    created: datetime             # When backlink was created
    last_verified: datetime       # Last verification time
    status: string                # "active" | "broken" | "pending"
```

## Graph Metrics and Analytics

### Document-Level Metrics
```yaml
graph_metrics:
  # Basic graph statistics
  node_count: integer             # Number of chunks
  edge_count: integer             # Number of relationships
  
  # Network analysis metrics
  clustering_coefficient: float   # How clustered the graph is
  average_path_length: float      # Average shortest path between chunks
  density: float                  # Graph density
  
  # Centrality measures
  centrality_scores:
    "1.0": float                  # Betweenness centrality per chunk
    "1.1": float
    # ... etc
    
  # Quality metrics
  relationship_quality: float     # Average relationship quality
  semantic_coherence: float       # Overall semantic coherence
  completeness_score: float       # How complete the document is
```

## Validation and Quality Assurance

### Validation Results
```yaml
validation_result:
  is_valid: boolean               # Overall validation status
  
  # Specific validation checks
  schema_compliance: boolean      # Schema validation passed
  reference_resolution: boolean   # All references resolve
  temporal_consistency: boolean   # Temporal relationships valid
  semantic_coherence: boolean     # Semantic consistency check
  
  # Error and warning details
  errors: [ValidationError]       # Critical validation errors
  warnings: [ValidationWarning]   # Non-critical warnings
  suggestions: [ValidationSuggestion] # Improvement suggestions
  
  # Quality scores
  overall_quality: float          # Overall quality score (0.0-1.0)
  information_density: float      # Information density metric
  readability_score: float        # Readability assessment
```

## Example FDO Document

### Complete Example
```yaml
doc_id: "entropy-abstraction-emergence"
version: "fdo-v0.1"
title: "Entropy as Field Operator for Abstraction Emergence"
authors: ["Peter Groom"]
doc_type: "theory-paper"
created: 2025-08-15T10:30:00Z
modified: 2025-08-15T15:45:00Z

metadata:
  source_document: "entropy_paper.md"
  license: "MIT"
  keywords: ["entropy", "abstraction", "emergence", "field-theory"]
  complexity_level: "advanced"
  quality_score: 0.89
  validation_status: "validated"

chunks:
  - id: "1.0"
    type: "thesis"
    title: "Core Thesis: Entropy as Field Operator"
    content: "Entropy is not disorder; it is a field operator for abstraction emergence."
    tone: "assertive"
    confidence: 0.85
    
    embedding: [0.24, -0.38, 0.71, 0.15, ...]
    entities:
      - text: "entropy"
        label: "CONCEPT"
        start: 0
        end: 7
        confidence: 0.95
        
    links_to:
      - chunk: "2.1"
        type: "supported_by"
        weight: 0.9
        description: "Supported by field theory evidence"
        
      - chunk: "2.3"
        type: "evidenced_by"
        weight: 0.8
        description: "Evidenced by emergence patterns"
        
    provenance:
      source: "human"
      author: "Peter Groom"
      author_type: "human"
      created: 2025-08-15T10:35:00Z
      confidence_score: 0.85
      validation_status: "validated"

  - id: "2.1"
    type: "evidence"
    title: "Field Theory Foundation"
    content: "Field theory demonstrates that entropy gradients create potential landscapes for abstraction emergence."
    
    links_to:
      - chunk: "1.0"
        type: "supports"
        weight: 0.9
        inverse: true
        
      - doc: "field-dynamics-theory"
        chunk: "3.2"
        type: "builds_on"
        weight: 0.7
        description: "Builds on field dynamics research"
        
    provenance:
      source: "human"
      author: "Peter Groom"
      author_type: "human"
      created: 2025-08-15T11:15:00Z

  - id: "2.3"
    type: "analysis"
    title: "Emergence Pattern Analysis"
    content: "Analysis of emergence patterns reveals consistent entropy-abstraction coupling across multiple domains."
    
    links_to:
      - chunk: "1.0"
        type: "evidence_for"
        weight: 0.8
        
    provenance:
      source: "collaborative"
      author: "Peter Groom & GPT-4"
      author_type: "human"
      model: "gpt-4"
      delta_status: "reconciled"
      created: 2025-08-15T14:22:00Z

relationships:
  ontology:
    - type: "supports"
      inverse: "supported_by"
      description: "Provides logical or evidential support"
      weight_range: [0.0, 1.0]
      
    - type: "evidenced_by"
      inverse: "evidence_for"
      description: "Provides empirical evidence"
      weight_range: [0.0, 1.0]

embeddings:
  model: "text-embedding-ada-002"
  dimension: 1536
  generated: 2025-08-15T10:35:00Z
  index_status: "indexed"
  
graph_metrics:
  node_count: 3
  edge_count: 4
  clustering_coefficient: 0.67
  centrality_scores:
    "1.0": 0.85
    "2.1": 0.72
    "2.3": 0.58
    
validation_result:
  is_valid: true
  schema_compliance: true
  reference_resolution: true
  temporal_consistency: true
  overall_quality: 0.89
```

## Implementation Guidelines

### Best Practices
1. **Start Simple**: Begin with basic chunk structure and relationships
2. **Iterative Refinement**: Gradually add complexity and metadata
3. **Consistent Ontology**: Use standardized chunk and relationship types
4. **Quality Over Quantity**: Focus on meaningful relationships over exhaustive linking
5. **Validation First**: Always validate documents before processing

### Common Patterns
- **Hierarchical Chunking**: Use dotted notation for nested concepts (1.0, 1.1, 1.1.1)
- **Cross-References**: Prefix external documents with clear identifiers
- **Temporal Ordering**: Maintain creation timestamps for causality verification
- **Bidirectional Links**: Always consider inverse relationships
- **Provenance Tracking**: Document authorship and AI involvement

### Migration Strategy
1. **Analysis**: Analyze existing documents for structure and relationships
2. **Chunking**: Break documents into semantic chunks with clear boundaries
3. **Relationship Mapping**: Identify and map relationships between chunks
4. **Metadata Addition**: Add provenance, embeddings, and quality metrics
5. **Validation**: Validate FDO structure and resolve issues
6. **Integration**: Integrate with Kronos processing pipeline

## Version History

### v0.1 (Initial Specification)
- Basic FDO structure and chunk definition
- Core relationship ontology
- Provenance and authorship tracking
- Cross-document reference support
- Validation framework
- Graph metrics and analytics

### Future Versions
- v0.2: Enhanced temporal metadata and causality tracking
- v0.3: Advanced AI interaction patterns
- v1.0: Complete specification with full ecosystem integration
