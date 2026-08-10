# CIP Middleware & Tools Module

## Overview

The CIP Middleware & Tools module provides comprehensive integration with the Cognition Index Protocol (CIP), enabling protocol enforcement, multi-repository navigation, and agentic workflows. This module serves as the bridge between the MCP Server and CIP-compliant repositories, ensuring all data transactions are validated and auditable.

## Core Responsibilities

### Protocol Enforcement Middleware
- Validate all incoming and outgoing requests against CIP specifications
- Enforce CIP context routing (public_release, internal_only, embargoed)
- Apply permission labels and access controls based on CIP metadata
- Generate audit trails for all CIP-related operations

### Multi-Repository Navigation
- Provide unified access to multiple CIP-compliant repositories
- Resolve cross-repository links and dependencies
- Manage repository registry and trust policies
- Handle repository discovery and capability negotiation

### CIP Tools and Resources
- Expose CIP-specific tools for repository inspection and navigation
- Provide semantic search across CIP-indexed content
- Enable batch operations for efficient context loading
- Support validation and compliance checking

## Technical Architecture

### CIP Request Middleware

```python
class CIPRequestMiddleware:
    def __init__(self, cip_config: CIPConfig):
        self.context_validator = CIPContextValidator()
        self.permission_enforcer = CIPPermissionEnforcer()
        self.audit_logger = CIPAuditLogger()
        self.routing_engine = CIPRoutingEngine()
    
    async def process_request(self, request: MCPRequest, 
                            context: RequestContext) -> CIPProcessingResult:
        """Process incoming request through CIP validation pipeline"""
        
        # Extract CIP context from request
        cip_context = self._extract_cip_context(request)
        
        # Validate CIP context and permissions
        validation_result = await self.context_validator.validate_context(
            context=cip_context,
            user_permissions=context.user_permissions,
            resource_requirements=request.resource_requirements
        )
        
        if not validation_result.is_valid:
            return CIPProcessingResult(
                success=False,
                error_type="CIP_VALIDATION_FAILED",
                error_details=validation_result.errors,
                audit_event=self._create_audit_event(request, validation_result)
            )
        
        # Apply permission enforcement
        permission_result = await self.permission_enforcer.enforce_permissions(
            request=request,
            cip_context=cip_context,
            validation_result=validation_result
        )
        
        # Route request based on CIP context
        routing_result = await self.routing_engine.route_request(
            request=request,
            cip_context=cip_context,
            permission_result=permission_result
        )
        
        # Log audit event
        audit_event = await self.audit_logger.log_cip_event(
            request=request,
            cip_context=cip_context,
            routing_result=routing_result
        )
        
        return CIPProcessingResult(
            success=True,
            processed_request=routing_result.routed_request,
            cip_metadata=cip_context,
            audit_event=audit_event
        )
```

### Repository Registry & Trust Management

```python
class CIPRepositoryRegistry:
    def __init__(self, registry_config: RepositoryRegistryConfig):
        self.trust_manager = RepositoryTrustManager()
        self.policy_engine = RepositoryPolicyEngine()
        self.discovery_service = RepositoryDiscoveryService()
        self.cache_manager = RepositoryCache()
    
    async def register_repository(self, repo_spec: RepositorySpecification) -> RegistrationResult:
        """Register a new CIP-compliant repository"""
        
        # Validate repository CIP compliance
        compliance_check = await self._validate_cip_compliance(repo_spec)
        if not compliance_check.is_compliant:
            return RegistrationResult(
                success=False,
                compliance_issues=compliance_check.issues
            )
        
        # Assess repository trust level
        trust_assessment = await self.trust_manager.assess_trust(
            repository=repo_spec,
            trust_criteria=self._get_trust_criteria()
        )
        
        # Apply repository policies
        policy_result = await self.policy_engine.apply_policies(
            repository=repo_spec,
            trust_level=trust_assessment.trust_level
        )
        
        # Register in discovery service
        discovery_registration = await self.discovery_service.register(
            repository=repo_spec,
            trust_metadata=trust_assessment,
            policy_constraints=policy_result.constraints
        )
        
        return RegistrationResult(
            success=True,
            repository_id=discovery_registration.repository_id,
            trust_level=trust_assessment.trust_level,
            policy_constraints=policy_result.constraints
        )
```

## CIP Tools API

### Core Tools

```python
# Tool specifications for MCP transport
CIP_TOOLS = [
    {
        "name": "cip.get_meta",
        "description": "Read and validate meta.yaml at specified repository path",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository identifier"},
                "path": {"type": "string", "description": "Path to meta.yaml file"}
            },
            "required": ["repo", "path"]
        }
    },
    {
        "name": "cip.get_map", 
        "description": "Read root map.yaml for repository structure overview",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository identifier"}
            },
            "required": ["repo"]
        }
    },
    {
        "name": "cip.list_meta",
        "description": "List all meta.yaml files under specified path",
        "input_schema": {
            "type": "object", 
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": false}
            },
            "required": ["repo", "path"]
        }
    },
    {
        "name": "cip.search",
        "description": "Semantic search across repository content",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "query": {"type": "string"},
                "path": {"type": "string", "default": ""},
                "semantic": {"type": "boolean", "default": true}
            },
            "required": ["repo", "query"]
        }
    },
    {
        "name": "cip.resolve_links",
        "description": "Resolve cross-repository links from meta.yaml",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "meta_path": {"type": "string"}
            },
            "required": ["repo", "meta_path"]
        }
    },
    {
        "name": "cip.batch_fetch",
        "description": "Efficiently fetch multiple files for reasoning windows",
        "input_schema": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "path": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["targets"]
        }
    },
    {
        "name": "cip.validate",
        "description": "Validate file against CIP schema specifications",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "schema_version": {"type": "string", "default": "2.0"}
            },
            "required": ["repo", "path"]
        }
    }
]
```

## Resources API

### Repository Resource Handler

```python
class CIPRepositoryResource:
    """Handles repo:// URI resolution and content access"""
    
    def __init__(self, registry: CIPRepositoryRegistry):
        self.registry = registry
        self.git_manager = GitRepositoryManager()
        self.access_controller = ResourceAccessController()
    
    async def get_resource(self, uri: str, context: RequestContext) -> ResourceResult:
        """Resolve repo:// URI and return content with CIP validation"""
        
        # Parse repo:// URI
        parsed_uri = self._parse_repo_uri(uri)  # repo://repo-id/path
        
        # Validate repository access
        access_check = await self.access_controller.check_access(
            repository_id=parsed_uri.repo_id,
            path=parsed_uri.path,
            user_context=context.user_context
        )
        
        if not access_check.access_granted:
            return ResourceResult(
                success=False,
                error_type="ACCESS_DENIED",
                error_details=access_check.denial_reason
            )
        
        # Resolve repository and fetch content
        repository = await self.registry.get_repository(parsed_uri.repo_id)
        content = await self.git_manager.get_file_content(
            repository=repository,
            path=parsed_uri.path,
            ref=parsed_uri.ref or repository.default_branch
        )
        
        # Validate content against CIP if applicable
        if parsed_uri.path.endswith(('.yaml', '.yml')):
            validation_result = await self._validate_cip_content(content)
            if not validation_result.is_valid:
                return ResourceResult(
                    success=False,
                    error_type="CIP_VALIDATION_FAILED",
                    error_details=validation_result.errors
                )
        
        return ResourceResult(
            success=True,
            content=content,
            metadata=ResourceMetadata(
                repository_id=parsed_uri.repo_id,
                path=parsed_uri.path,
                content_type=self._detect_content_type(content),
                cip_validated=True
            )
        )
```

## CIP Context Routing

### Context Enforcement

```python
class CIPContextRouter:
    """Enforces CIP context routing for public/internal/embargoed content"""
    
    def __init__(self, routing_config: ContextRoutingConfig):
        self.context_classifier = ContextClassifier()
        self.permission_matrix = PermissionMatrix()
        self.embargo_manager = EmbargoManager()
    
    async def route_request(self, request: MCPRequest, 
                          cip_context: CIPContext) -> RoutingResult:
        """Route request based on CIP context and permissions"""
        
        # Classify content context
        context_classification = await self.context_classifier.classify(
            content_metadata=cip_context.metadata,
            file_path=request.resource_path,
            repository_context=cip_context.repository_context
        )
        
        # Check embargo status
        embargo_check = await self.embargo_manager.check_embargo(
            context_classification=context_classification,
            request_timestamp=request.timestamp
        )
        
        if embargo_check.is_embargoed:
            return RoutingResult(
                success=False,
                routing_decision="EMBARGO_BLOCKED",
                embargo_until=embargo_check.embargo_until,
                alternative_resources=embargo_check.public_alternatives
            )
        
        # Apply permission matrix
        permission_result = await self.permission_matrix.check_permissions(
            user_context=request.user_context,
            required_context=context_classification.required_context,
            resource_sensitivity=context_classification.sensitivity_level
        )
        
        return RoutingResult(
            success=permission_result.access_granted,
            routing_decision=permission_result.decision,
            allowed_operations=permission_result.allowed_operations,
            audit_requirements=permission_result.audit_requirements
        )
```

## Data Structures

```python
@dataclass
class CIPContext:
    context_type: str  # public_release, internal_only, embargoed
    permissions: List[str]
    embargo_until: Optional[datetime]
    routing_rules: List[RoutingRule]
    metadata: Dict[str, Any]
    repository_context: RepositoryContext

@dataclass
class RepositorySpecification:
    repository_id: str
    remote_url: str
    default_branch: str
    schema_version: str
    trust_level: str
    license: str
    access_policies: List[AccessPolicy]

@dataclass
class CIPProcessingResult:
    success: bool
    processed_request: Optional[MCPRequest]
    cip_metadata: Optional[CIPContext]
    audit_event: Optional[AuditEvent]
    error_type: Optional[str]
    error_details: Optional[Dict[str, Any]]
```

## Integration Points

### MCP Server Integration
- **Protocol Handler Layer**: CIP middleware sits in the request/response pipeline
- **Context Processing Engine**: Enhanced with CIP metadata and semantic understanding
- **Session State Manager**: Maintains CIP context across session lifecycle

### External Integration
- **CIP Core Repository**: Schema validation and compliance checking
- **Git Repositories**: Multi-repository content access and version control
- **SCBF Framework**: Audit logging and provenance tracking
- **Brainstem UI**: Repository visualization and navigation interface

## Deployment Configuration

### Registry Configuration
```yaml
repository_registry:
  repositories:
    - id: field-theory
      remote: https://github.com/dawnfield-institute/dawn-field-theory
      default_branch: main
      schema_version: "2.0"
      trust_level: "trusted"
      license: "Dawn Field Theory License"
    - id: cip-core
      remote: https://github.com/dawnfield-institute/cip-core
      default_branch: main
      schema_version: "2.0" 
      trust_level: "trusted"
      license: "MIT"
  
  policies:
    read_only_mode: true
    max_batch_size: 50
    cache_ttl: 3600
    audit_all_access: true
```

## Example Request Flow

```python
# Example: Agent traversing from Field Theory to CIP Core
async def example_agent_flow():
    # 1. Start audit session
    session = await audit.start_session(
        repo="field-theory",
        query="Explain core infodynamics concepts"
    )
    
    # 2. Get repository metadata
    meta = await cip.get_meta(
        repo="field-theory", 
        path=".cip/meta.yaml"
    )
    
    # 3. Get structure overview
    map_data = await cip.get_map(repo="field-theory")
    
    # 4. List relevant metadata files
    meta_files = await cip.list_meta(
        repo="field-theory",
        path="docs/infodynamics",
        recursive=True
    )
    
    # 5. Batch fetch relevant content
    content = await cip.batch_fetch(targets=[
        {"repo": "field-theory", "path": "docs/infodynamics/overview.md"},
        {"repo": "cip-core", "path": "spec/overview.md"}
    ])
    
    # 6. Resolve cross-repository links
    links = await cip.resolve_links(
        repo="field-theory",
        meta_path="docs/infodynamics/meta.yaml"
    )
    
    # 7. End audit session
    await audit.end_session(session_id=session.id, status="success")
    
    return {
        "content": content,
        "links": links,
        "audit_trail": session.events
    }
```

This CIP Middleware & Tools module provides the theory infrastructure for protocol-compliant repository navigation, multi-repository traversal, and agentic workflows while maintaining full auditability and security.
