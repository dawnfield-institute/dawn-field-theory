#!/usr/bin/env python3
"""
Citation Processing Script
Processes pending citation YAML files and integrates them into the project citation system.
"""

import os
import sys
import json
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def save_yaml(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def validate_citation_yaml(data: Dict[str, Any]) -> bool:
    """Validate citation YAML structure."""
    required_fields = ['contributor', 'contribution', 'type']
    
    for field in required_fields:
        if field not in data:
            print(f"Missing required field: {field}")
            return False
    
    # Validate contributor fields
    if 'name' not in data['contributor']:
        print("Missing contributor.name field")
        return False
    
    return True

def process_citation_file(file_path: Path, contributors_index: Dict[str, Any]) -> bool:
    """Process a single citation file."""
    print(f"Processing: {file_path.name}")
    
    citation_data = load_yaml(file_path)
    if not citation_data:
        print(f"Failed to load {file_path}")
        return False
    
    if not validate_citation_yaml(citation_data):
        print(f"Invalid citation format in {file_path}")
        return False
    
    # Extract contributor info
    contributor = citation_data['contributor']
    contributor_id = contributor['name'].lower().replace(' ', '_')
    
    # Ensure contributors_index has the expected structure
    if 'contributors' not in contributors_index:
        contributors_index['contributors'] = {}
    
    # Convert existing list format to dict format if needed
    if isinstance(contributors_index['contributors'], list):
        existing_contributors = contributors_index['contributors']
        contributors_index['contributors'] = {}
        for contrib in existing_contributors:
            if isinstance(contrib, dict) and 'name' in contrib:
                key = contrib['name'].lower().replace(' ', '_')
                # Convert old format to new format
                contributors_index['contributors'][key] = {
                    'name': contrib['name'],
                    'email': contrib.get('email', ''),
                    'orcid': contrib.get('orcid', ''),
                    'github': contrib.get('github', ''),
                    'affiliation': contrib.get('affiliation', ''),
                    'contributions': []  # Reset to new format
                }
    
    # Add to contributors index
    if contributor_id not in contributors_index['contributors']:
        contributors_index['contributors'][contributor_id] = {
            'name': contributor['name'],
            'email': contributor.get('email', ''),
            'orcid': contributor.get('orcid', ''),
            'github': contributor.get('github', ''),
            'affiliation': contributor.get('affiliation', ''),
            'contributions': []
        }
    
    # Add this contribution
    contribution_entry = {
        'title': citation_data['contribution']['title'],
        'type': citation_data['type'],
        'date': citation_data.get('date', datetime.now().strftime('%Y-%m-%d')),
        'description': citation_data['contribution'].get('description', ''),
        'files': citation_data['contribution'].get('files', []),
        'pr_number': citation_data.get('pr_number', ''),
        'doi': citation_data.get('doi', ''),
        'keywords': citation_data.get('keywords', []),
        'notes': citation_data.get('notes', '')
    }
    
    # Remove empty optional fields for cleaner output
    contribution_entry = {k: v for k, v in contribution_entry.items() if v}
    
    contributors_index['contributors'][contributor_id]['contributions'].append(contribution_entry)
    
    print(f"✅ Added contribution: {contribution_entry['title']}")
    return True

def generate_bibtex_entries(contributors_index: Dict[str, Any]) -> str:
    """Generate BibTeX entries for all contributors."""
    bibtex_entries = []
    
    # Handle both dict and list structures for backwards compatibility
    contributors = contributors_index.get('contributors', {})
    if isinstance(contributors, list):
        # Convert list to dict format
        contributors_dict = {}
        for i, contributor in enumerate(contributors):
            key = contributor.get('name', f'contributor_{i}').lower().replace(' ', '_')
            contributors_dict[key] = contributor
        contributors = contributors_dict
    
    for contributor_id, contributor in contributors.items():
        for i, contribution in enumerate(contributor.get('contributions', [])):
            entry_id = f"dawnfield_{contributor_id}_{i+1}"
            
            # Build BibTeX entry with optional fields
            bibtex_lines = [
                f"@misc{{{entry_id},",
                f"  author       = {{{contributor['name']}}},",
                f"  title        = {{{contribution['title']}}},",
                f"  year         = {{{contribution['date'][:4]}}},",
                f"  note         = {{Contribution to Dawn Field Theory Repository}},",
                f"  url          = {{https://github.com/dawnfield-institute/dawn-field-theory}},",
                f"  type         = {{{contribution['type']}}}"
            ]
            
            # Add optional fields if present
            if contribution.get('doi'):
                bibtex_lines.append(f"  doi          = {{{contribution['doi']}}},")
            if contribution.get('keywords'):
                keywords_str = ', '.join(contribution['keywords'])
                bibtex_lines.append(f"  keywords     = {{{keywords_str}}},")
            if contribution.get('pr_number'):
                bibtex_lines.append(f"  note         = {{PR #{contribution['pr_number']}, Dawn Field Theory}},")
            
            # Remove trailing comma from last line and close entry
            bibtex_lines[-1] = bibtex_lines[-1].rstrip(',')
            bibtex_lines.append("}")
            
            bibtex_entry = '\n'.join(bibtex_lines)
            bibtex_entries.append(bibtex_entry)
    
    return '\n\n'.join(bibtex_entries)

def update_citation_cff(contributors_index: Dict[str, Any], citation_cff_path: Path) -> None:
    """Update the CITATION.cff file with contributor information."""
    try:
        # Load existing CITATION.cff
        citation_cff = load_yaml(citation_cff_path)
        
        # Update authors list
        authors = [citation_cff.get('authors', [{}])[0]]  # Keep main author first
        
        # Handle both dict and list structures for backwards compatibility
        contributors = contributors_index.get('contributors', {})
        if isinstance(contributors, list):
            # Convert list to dict format
            contributors_dict = {}
            for i, contributor in enumerate(contributors):
                key = contributor.get('name', f'contributor_{i}').lower().replace(' ', '_')
                contributors_dict[key] = contributor
            contributors = contributors_dict
        
        for contributor_id, contributor in contributors.items():
            if contributor['name'] != citation_cff.get('authors', [{}])[0].get('given-names', '') + ' ' + citation_cff.get('authors', [{}])[0].get('family-names', ''):
                author_entry = {
                    'given-names': contributor['name'].split()[0],
                    'family-names': ' '.join(contributor['name'].split()[1:]) if len(contributor['name'].split()) > 1 else '',
                    'email': contributor.get('email', ''),
                    'orcid': contributor.get('orcid', '').replace('https://orcid.org/', '') if contributor.get('orcid') else ''
                }
                # Remove empty fields
                author_entry = {k: v for k, v in author_entry.items() if v}
                authors.append(author_entry)
        
        citation_cff['authors'] = authors
        save_yaml(citation_cff, citation_cff_path)
        print(f"✅ Updated CITATION.cff with {len(authors)} authors")
        
    except Exception as e:
        print(f"Error updating CITATION.cff: {e}")

def main():
    """Main processing function."""
    repo_root = Path(__file__).parent.parent
    citations_dir = repo_root / 'citations'
    pending_dir = citations_dir / 'pending'
    processed_dir = citations_dir / 'processed'
    
    # Load or create contributors index
    contributors_index_path = citations_dir / 'contributors-index.json'
    if contributors_index_path.exists():
        with open(contributors_index_path, 'r') as f:
            contributors_index = json.load(f)
    else:
        contributors_index = {
            'last_updated': datetime.now().isoformat(),
            'contributors': {}
        }
    
    # Process all pending citation files
    processed_files = []
    failed_files = []
    
    # Blacklist for files that should not be processed
    blacklisted_files = {
        'example-pr-123-entropy-operator.yaml',
        'README.md'
    }
    
    for yaml_file in pending_dir.glob('*.yaml'):
        if yaml_file.name in blacklisted_files:
            print(f"⏭️  Skipping blacklisted file: {yaml_file.name}")
            continue
            
        try:
            if process_citation_file(yaml_file, contributors_index):
                processed_files.append(yaml_file)
            else:
                failed_files.append(yaml_file)
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            failed_files.append(yaml_file)
    
    if not processed_files and not failed_files:
        print("No citation files to process.")
        return
    
    # Update timestamp
    contributors_index['last_updated'] = datetime.now().isoformat()
    
    # Save updated contributors index
    save_json(contributors_index, contributors_index_path)
    print(f"✅ Updated contributors index: {len(contributors_index['contributors'])} contributors")
    
    # Generate BibTeX file
    bibtex_content = generate_bibtex_entries(contributors_index)
    bibtex_path = citations_dir / 'contributors_bibtex.bib'
    with open(bibtex_path, 'w') as f:
        f.write(bibtex_content)
    print(f"✅ Generated contributors BibTeX file with {len(processed_files)} entries")
    
    # Update CITATION.cff
    citation_cff_path = repo_root / 'CITATION.cff'
    if citation_cff_path.exists():
        update_citation_cff(contributors_index, citation_cff_path)
    
    # Move processed files to processed directory
    for file_path in processed_files:
        dest_path = processed_dir / f"{datetime.now().strftime('%Y%m%d')}-{file_path.name}"
        shutil.move(str(file_path), str(dest_path))
        print(f"✅ Moved {file_path.name} to processed/")
    
    # Report results
    print(f"\n📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {len(processed_files)} files")
    print(f"  ❌ Failed to process: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for file_path in failed_files:
            print(f"  - {file_path.name}")
        sys.exit(1)

if __name__ == '__main__':
    main()
