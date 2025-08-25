#!/usr/bin/env python3
"""
Citation Validation Script
Validates citation YAML files for proper structure and required fields.
Can be used in PR workflows or locally before submission.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_yaml_safe(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Load and parse a YAML file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}, "Empty YAML file"
            return data, ""
    except yaml.YAMLError as e:
        return {}, f"YAML parsing error: {e}"
    except Exception as e:
        return {}, f"Error loading file: {e}"

def validate_citation_structure(data: Dict[str, Any], filename: str) -> List[str]:
    """Validate the structure and required fields of a citation YAML."""
    errors = []
    
    # Check required top-level fields
    required_fields = ['contributor', 'contribution', 'type']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
    
    # Validate contributor section
    if 'contributor' in data:
        contributor = data['contributor']
        if not isinstance(contributor, dict):
            errors.append("'contributor' must be a dictionary")
        else:
            # Check required contributor fields
            required_contributor_fields = ['name']
            for field in required_contributor_fields:
                if field not in contributor:
                    errors.append(f"Missing required contributor field: '{field}'")
                elif not contributor[field] or not contributor[field].strip():
                    errors.append(f"Contributor field '{field}' cannot be empty")
            
            # Validate optional contributor fields
            optional_fields = ['email', 'github', 'orcid', 'affiliation']
            for field in optional_fields:
                if field in contributor and contributor[field]:
                    if field == 'email' and '@' not in str(contributor[field]):
                        errors.append(f"Invalid email format: '{contributor[field]}'")
                    elif field == 'orcid' and not str(contributor[field]).startswith(('https://orcid.org/', '0000-')):
                        errors.append(f"ORCID should start with 'https://orcid.org/' or '0000-': '{contributor[field]}'")
    
    # Validate contribution section
    if 'contribution' in data:
        contribution = data['contribution']
        if not isinstance(contribution, dict):
            errors.append("'contribution' must be a dictionary")
        else:
            # Check required contribution fields
            required_contribution_fields = ['title']
            for field in required_contribution_fields:
                if field not in contribution:
                    errors.append(f"Missing required contribution field: '{field}'")
                elif not contribution[field] or not contribution[field].strip():
                    errors.append(f"Contribution field '{field}' cannot be empty")
            
            # Validate files field if present
            if 'files' in contribution:
                files = contribution['files']
                if not isinstance(files, list):
                    errors.append("'contribution.files' must be a list")
                elif len(files) == 0:
                    errors.append("'contribution.files' cannot be empty if specified")
    
    # Validate type field
    if 'type' in data:
        valid_types = ['implementation', 'theory', 'experiment', 'documentation', 'validation']
        if data['type'] not in valid_types:
            errors.append(f"Invalid type: '{data['type']}'. Must be one of: {', '.join(valid_types)}")
    
    # Validate optional fields
    if 'keywords' in data:
        keywords = data['keywords']
        if not isinstance(keywords, list):
            errors.append("'keywords' must be a list")
        elif any(not isinstance(kw, str) or not kw.strip() for kw in keywords):
            errors.append("All keywords must be non-empty strings")
    
    # Check filename convention
    if not filename.startswith('pr-') or not filename.endswith('.yaml'):
        errors.append(f"Filename should follow pattern 'pr-{{PR_NUMBER}}-{{description}}.yaml', got: '{filename}'")
    
    return errors

def validate_files(file_paths: List[Path]) -> Tuple[List[str], List[str]]:
    """Validate multiple citation files."""
    all_errors = []
    valid_files = []
    
    # Blacklisted files that should be skipped
    blacklisted_patterns = ['example-', 'README.md']
    
    for file_path in file_paths:
        filename = file_path.name
        
        # Skip blacklisted files
        if any(filename.startswith(pattern) or filename == pattern for pattern in blacklisted_patterns):
            print(f"⏭️  Skipping blacklisted file: {filename}")
            continue
        
        print(f"🔍 Validating: {filename}")
        
        # Load and validate YAML
        data, load_error = load_yaml_safe(file_path)
        if load_error:
            all_errors.append(f"{filename}: {load_error}")
            continue
        
        # Validate structure
        validation_errors = validate_citation_structure(data, filename)
        if validation_errors:
            for error in validation_errors:
                all_errors.append(f"{filename}: {error}")
        else:
            valid_files.append(filename)
            print(f"✅ Valid: {filename}")
    
    return all_errors, valid_files

def main():
    """Main validation function."""
    if len(sys.argv) > 1:
        # Validate specific files provided as arguments
        file_paths = [Path(arg) for arg in sys.argv[1:]]
        # Filter to only existing files
        file_paths = [p for p in file_paths if p.exists()]
    else:
        # Validate all YAML files in citations/pending/
        pending_dir = Path('citations/pending')
        if not pending_dir.exists():
            print("❌ citations/pending directory not found")
            return 1
        
        file_paths = list(pending_dir.glob('*.yaml'))
    
    if not file_paths:
        print("ℹ️  No citation files found to validate")
        return 0
    
    print(f"🎯 Validating {len(file_paths)} citation files...\n")
    
    errors, valid_files = validate_files(file_paths)
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"  ✅ Valid files: {len(valid_files)}")
    print(f"  ❌ Files with errors: {len(errors)}")
    
    if valid_files:
        print(f"\n✅ Valid files:")
        for filename in valid_files:
            print(f"  - {filename}")
    
    if errors:
        print(f"\n❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print(f"\n🎉 All citation files are valid!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
