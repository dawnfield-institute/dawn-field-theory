#!/usr/bin/env python3
"""
Zenodo Package Preparation Tool
Creates upload-ready packages for Dawn Field Theory preprints.

Usage:
    python prepare_zenodo_packages.py [paper_slug] [--all-pending] [--all-updates]
    
Examples:
    python prepare_zenodo_packages.py cellular_automata_xi_clustering
    python prepare_zenodo_packages.py --all-pending
    python prepare_zenodo_packages.py --all-updates
"""

import os
import sys
import shutil
import json
import yaml
from datetime import datetime
from pathlib import Path
import zipfile
import hashlib

# Configuration
PREPRINTS_DIR = Path(__file__).parent
PACKAGES_DIR = PREPRINTS_DIR / "packages"
REGISTRY_FILE = PREPRINTS_DIR / "ZENODO_REGISTRY.yaml"

def load_registry():
    """Load the Zenodo registry."""
    with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_paper_path(slug: str) -> Path:
    """Get the path to a paper folder."""
    # Check if it's in PACSeries
    pac_path = PREPRINTS_DIR / "PACSeries" / slug
    if pac_path.exists():
        return pac_path
    # Otherwise it's a standalone paper
    return PREPRINTS_DIR / slug

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def create_manifest(paper_path: Path) -> dict:
    """Create a manifest of all files in the package."""
    manifest = {
        'created': datetime.now().isoformat(),
        'files': []
    }
    
    for filepath in sorted(paper_path.rglob('*')):
        if filepath.is_file() and not filepath.name.startswith('.'):
            rel_path = filepath.relative_to(paper_path)
            manifest['files'].append({
                'path': str(rel_path),
                'size': filepath.stat().st_size,
                'sha256': calculate_file_hash(filepath)
            })
    
    return manifest

def create_zenodo_metadata(paper_path: Path, slug: str) -> dict:
    """Create Zenodo-compatible metadata from paper meta.yaml."""
    meta_file = paper_path / 'meta.yaml'
    
    # Default metadata
    metadata = {
        'title': slug.replace('_', ' ').title(),
        'upload_type': 'publication',
        'publication_type': 'preprint',
        'access_right': 'open',
        'license': 'AGPL-3.0',
        'creators': [{'name': 'Groom, Peter'}],
        'keywords': ['Dawn Field Theory', 'infodynamics', 'PAC framework'],
        'related_identifiers': [
            {
                'identifier': 'https://github.com/dawn-field-institute/dawn-field-theory',
                'relation': 'isSupplementTo',
                'resource_type': 'software'
            }
        ]
    }
    
    # Load paper meta.yaml if exists
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            paper_meta = yaml.safe_load(f)
            if paper_meta:
                if 'title' in paper_meta:
                    metadata['title'] = paper_meta['title']
                if 'description' in paper_meta:
                    metadata['description'] = paper_meta['description']
                if 'keywords' in paper_meta:
                    metadata['keywords'].extend(paper_meta.get('keywords', []))
                if 'version' in paper_meta:
                    metadata['version'] = paper_meta['version']
    
    # Try to extract abstract from paper.md
    paper_file = paper_path / 'paper.md'
    if paper_file.exists():
        content = paper_file.read_text(encoding='utf-8')
        # Look for abstract section
        if '## Abstract' in content:
            start = content.find('## Abstract')
            end = content.find('\n## ', start + 1)
            if end == -1:
                end = content.find('\n---', start + 1)
            if end > start:
                abstract = content[start + len('## Abstract'):end].strip()
                metadata['description'] = abstract[:2000]  # Zenodo limit
    
    return metadata

def create_package(slug: str, version: str = None) -> Path:
    """Create a Zenodo-ready package for a paper."""
    paper_path = get_paper_path(slug)
    
    if not paper_path.exists():
        raise ValueError(f"Paper not found: {slug}")
    
    # Determine version
    if version is None:
        meta_file = paper_path / 'meta.yaml'
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f)
                version = meta.get('version', '1.0')
        else:
            version = '1.0'
    
    # Create package directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    package_name = f"{slug}_v{version}_{timestamp}"
    package_dir = PACKAGES_DIR / slug / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating package: {package_name}")
    print(f"  Source: {paper_path}")
    print(f"  Target: {package_dir}")
    
    # Copy all paper files
    for item in paper_path.iterdir():
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            shutil.copytree(item, package_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, package_dir / item.name)
    
    # Create manifest
    manifest = create_manifest(package_dir)
    manifest_file = package_dir / 'MANIFEST.json'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Created MANIFEST.json ({len(manifest['files'])} files)")
    
    # Create Zenodo metadata
    zenodo_meta = create_zenodo_metadata(paper_path, slug)
    zenodo_meta['version'] = version
    zenodo_file = package_dir / '.zenodo.json'
    with open(zenodo_file, 'w', encoding='utf-8') as f:
        json.dump(zenodo_meta, f, indent=2)
    print(f"  Created .zenodo.json")
    
    # Create zip file
    zip_path = package_dir.parent / f"{package_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filepath in package_dir.rglob('*'):
            if filepath.is_file():
                arcname = filepath.relative_to(package_dir)
                zf.write(filepath, arcname)
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Created {zip_path.name} ({zip_size_mb:.2f} MB)")
    
    return zip_path

def get_pending_papers() -> list:
    """Get list of papers pending Zenodo upload."""
    registry = load_registry()
    return [p['slug'] for p in registry.get('pending_upload', []) if p.get('status') == 'ready']

def get_update_papers() -> list:
    """Get list of published papers needing updates."""
    registry = load_registry()
    return [p['slug'] for p in registry.get('published', []) if p.get('status') == 'needs_update']

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCurrent status:")
        registry = load_registry()
        
        print("\n📦 Ready for first upload:")
        for p in registry.get('pending_upload', []):
            if p.get('status') == 'ready':
                print(f"  ✓ {p['slug']}")
            else:
                print(f"  ○ {p['slug']} ({p.get('status', 'unknown')})")
        
        print("\n🔄 Published papers needing updates:")
        for p in registry.get('published', []):
            if p.get('status') == 'needs_update':
                print(f"  ↻ {p['slug']}")
        
        print(f"\nRun with --all-pending to package all ready papers")
        return
    
    # Ensure packages directory exists
    PACKAGES_DIR.mkdir(exist_ok=True)
    
    if sys.argv[1] == '--all-pending':
        papers = get_pending_papers()
        print(f"Packaging {len(papers)} pending papers...")
        for slug in papers:
            try:
                create_package(slug)
                print()
            except Exception as e:
                print(f"  ERROR: {e}\n")
    
    elif sys.argv[1] == '--all-updates':
        papers = get_update_papers()
        print(f"Packaging {len(papers)} papers needing updates...")
        for slug in papers:
            try:
                create_package(slug)
                print()
            except Exception as e:
                print(f"  ERROR: {e}\n")
    
    else:
        slug = sys.argv[1]
        version = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            create_package(slug, version)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    print("Done! Packages are in:", PACKAGES_DIR)

if __name__ == '__main__':
    main()
