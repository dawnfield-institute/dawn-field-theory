#!/usr/bin/env python3
"""
Migration Script: Restructure Preprints to New Schema
======================================================

Moves from:
  drafts/[tag][D][vX.X]..._name_preprint.md

To:
  paper_slug/
    ├── meta.yaml
    ├── paper.md
    ├── README.md
    ├── CITATION.md
    ├── LICENSE
    ├── Code/
    │   ├── trace.yaml    # Links to source files
    │   └── ...
    └── Data/
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml


# Mapping of paper slugs to experiment code locations (relative to foundational/)
EXPERIMENT_MAP = {
    'cellular_automata_xi_clustering': 'experiments/cellular_automata_pac_attractors',
    'ml_validation_pythia_gpt2': '../dawn-models/research/GAIA/proof_of_concepts/poc_020_multi_model_pac',
    'golden_ratio_prime_distribution': 'experiments/sec_prime_manifold',
    'potential_actualization_conservation': 'arithmetic/PACEngine',
    'gaia_field_native_intelligence': '../dawn-models/research/GAIA',
    'pac_necessity_proof': 'experiments/pac_confluence_xi',
    'xi_bounded_invariant': 'experiments/pac_confluence_xi',
    'sec_med_framework': 'experiments/sec_prime_manifold',
    'qbe_pac_unification': 'experiments/pac_confluence_xi',
    'symbolic_entropy_collapse': 'experiments/sec_prime_manifold',
    'mobius_confluence_operator': 'experiments/pac_confluence_xi',
    'relativistic_mas_universal_frequency': 'experiments/pac_cosmology_validation',
}


class PrePrintMigrator:
    """Migrate preprints to new folder structure."""
    
    def __init__(self, preprints_path: Path):
        self.preprints_path = preprints_path
        self.drafts_path = preprints_path / "drafts"
        self.foundational_path = preprints_path.parent.parent  # foundational/
        self.repo_root = self.foundational_path.parent  # dawn-field-theory/
        
    def parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse paper filename into components."""
        match = re.match(
            r'\[(\w+)\]\[([DF])\]\[v([\d.]+)\]\[C(\d)\]\[I(\d)\](?:\[([ERAO])\])?_(.+?)(?:_preprint)?\.md$',
            filename
        )
        if match:
            return {
                'tag': match.group(1),
                'status': 'Draft' if match.group(2) == 'D' else 'Final',
                'version': match.group(3),
                'complexity': int(match.group(4)),
                'impact': int(match.group(5)),
                'type': match.group(6) or 'U',
                'slug': match.group(7),
                'name': match.group(7).replace('_', ' ').title()
            }
        return None
    
    def find_all_papers(self) -> List[Dict]:
        """Find all papers in drafts and PACSeries."""
        papers = []
        
        # Main drafts folder
        if self.drafts_path.exists():
            for file in self.drafts_path.glob("*.md"):
                meta = self.parse_filename(file.name)
                if meta:
                    meta['file'] = file
                    meta['source'] = 'drafts'
                    papers.append(meta)
        
        # PACSeries subfolder
        pac_series = self.drafts_path / "PACSeries"
        if pac_series.exists():
            for file in pac_series.glob("*.md"):
                meta = self.parse_filename(file.name)
                if meta:
                    meta['file'] = file
                    meta['source'] = 'PACSeries'
                    papers.append(meta)
        
        return papers
    
    def find_experiment_code(self, slug: str) -> Optional[Path]:
        """Find experiment code for a paper."""
        for key, rel_path in EXPERIMENT_MAP.items():
            if key in slug:
                code_path = self.foundational_path / rel_path
                if code_path.exists():
                    return code_path
        return None
    
    def create_trace_yaml(self, code_dest: Path, code_source: Optional[Path]) -> str:
        """Create trace.yaml content linking to source files."""
        trace = {
            'schema_version': '1.0',
            'created': datetime.now().isoformat(),
            'description': 'Traces code files back to their original repository locations',
            'source_repository': 'dawn-field-theory' if code_source and 'dawn-field-theory' in str(code_source) else 'dawn-models',
            'files': []
        }
        
        if code_source and code_source.exists():
            # Track all files copied
            for root, dirs, files in os.walk(code_source):
                for file in files:
                    if file.endswith(('.py', '.yaml', '.json', '.txt', '.md')):
                        source_file = Path(root) / file
                        # Get relative path from repo root
                        try:
                            rel_to_repo = source_file.relative_to(self.repo_root.parent)
                            trace['files'].append({
                                'local': str(Path(root).relative_to(code_source) / file),
                                'source': str(rel_to_repo),
                                'repo': 'dawn-field-theory' if 'dawn-field-theory' in str(source_file) else 'dawn-models'
                            })
                        except ValueError:
                            # File outside repo structure
                            trace['files'].append({
                                'local': str(Path(root).relative_to(code_source) / file),
                                'source': str(source_file),
                                'repo': 'unknown'
                            })
        
        return yaml.dump(trace, default_flow_style=False, sort_keys=False)
    
    def create_paper_folder(self, paper: Dict) -> Path:
        """Create the new folder structure for a paper."""
        slug = paper['slug']
        paper_dir = self.preprints_path / slug
        
        print(f"\n{'='*60}")
        print(f"Migrating: {paper['name']}")
        print(f"{'='*60}")
        
        # Create directory structure
        dirs = [
            '',
            'Code',
            'Code/core',
            'Code/experiments',
            'Data',
            'Data/results',
            'Figures',
        ]
        for d in dirs:
            (paper_dir / d).mkdir(parents=True, exist_ok=True)
        
        # Copy paper as paper.md
        source_paper = paper['file']
        dest_paper = paper_dir / "paper.md"
        shutil.copy(source_paper, dest_paper)
        print(f"  [OK] Copied paper.md")
        
        # Keep original filename as symlink/reference in meta
        original_filename = source_paper.name
        
        # Find and copy experiment code
        code_source = self.find_experiment_code(slug)
        if code_source:
            self.copy_experiment_code(code_source, paper_dir / "Code")
            print(f"  [OK] Copied code from: {code_source.name}")
        else:
            print(f"  [--] No experiment code found")
            self.create_placeholder_code(paper_dir / "Code")
        
        # Create trace.yaml
        trace_content = self.create_trace_yaml(paper_dir / "Code", code_source)
        (paper_dir / "Code" / "trace.yaml").write_text(trace_content, encoding='utf-8')
        print(f"  [OK] Created trace.yaml")
        
        # Generate metadata files
        self.create_meta_yaml(paper_dir, paper, original_filename)
        self.create_readme(paper_dir, paper)
        self.create_citation(paper_dir, paper)
        self.create_license(paper_dir)
        print(f"  [OK] Generated metadata files")
        
        return paper_dir
    
    def copy_experiment_code(self, source: Path, dest: Path):
        """Copy experiment code to package."""
        # Copy core modules
        core_source = source / "core"
        if core_source.exists():
            shutil.copytree(core_source, dest / "core", dirs_exist_ok=True)
        
        # Copy scripts
        scripts_source = source / "scripts"
        if scripts_source.exists():
            shutil.copytree(scripts_source, dest / "experiments", dirs_exist_ok=True)
        
        # Copy results
        results_source = source / "results"
        if results_source.exists():
            for json_file in results_source.glob("*.json"):
                shutil.copy(json_file, dest.parent / "Data" / "results" / json_file.name)
        
        # Copy or create requirements.txt
        req_file = source / "requirements.txt"
        if req_file.exists():
            shutil.copy(req_file, dest / "requirements.txt")
        else:
            (dest / "requirements.txt").write_text("numpy>=1.20.0\nscipy>=1.7.0\n", encoding='utf-8')
        
        # Create reproduce.py
        self.create_reproduce_script(dest)
    
    def create_placeholder_code(self, dest: Path):
        """Create placeholder code structure."""
        (dest / "requirements.txt").write_text("numpy>=1.20.0\nscipy>=1.7.0\n", encoding='utf-8')
        (dest / "core" / ".gitkeep").touch()
        (dest / "experiments" / ".gitkeep").touch()
        self.create_reproduce_script(dest)
    
    def create_reproduce_script(self, dest: Path):
        """Create reproduce.py script."""
        content = '''#!/usr/bin/env python3
"""
Reproduction Script
==================

Run all experiments to reproduce paper results.

Usage:
    python reproduce.py              # Run all experiments
    python reproduce.py 7            # Run experiment 07 only
    python reproduce.py --list       # List available experiments
"""

import sys
import subprocess
from pathlib import Path


def main():
    scripts_dir = Path(__file__).parent / "experiments"
    
    if not scripts_dir.exists():
        print("No experiments directory found.")
        return
    
    scripts = sorted(scripts_dir.glob("exp_*.py"))
    
    if not scripts:
        print("No experiment scripts found.")
        return
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            print("Available experiments:")
            for s in scripts:
                print(f"  - {s.name}")
            return
        
        # Run specific experiment
        exp_num = sys.argv[1].zfill(2)
        scripts = [s for s in scripts if f"exp_{exp_num}" in s.name]
        
        if not scripts:
            print(f"No experiment {exp_num} found.")
            return
    
    for script in scripts:
        print(f"\\n{'='*60}")
        print(f"Running: {script.name}")
        print('='*60)
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"\\n[WARNING] {script.name} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
'''
        (dest / "reproduce.py").write_text(content, encoding='utf-8')
    
    def create_meta_yaml(self, paper_dir: Path, paper: Dict, original_filename: str):
        """Create meta.yaml for the paper."""
        content = f"""schema_version: "2.0"
slug: "{paper['slug']}"
title: "{paper['name']}"
category: "{paper['tag']}"
status: "{paper['status']}"
version: "{paper['version']}"
complexity: {paper['complexity']}
impact: {paper['impact']}
evidence_type: "{paper['type']}"
created: "{datetime.now().isoformat()}"

original_filename: "{original_filename}"
source_location: "{paper['source']}"

files:
  paper: "paper.md"
  latex: "paper.tex"
  pdf: "paper.pdf"

code:
  has_experiments: true
  trace_file: "Code/trace.yaml"
  
keywords:
  - Dawn Field Theory
  - {paper['tag'].upper()}
"""
        (paper_dir / "meta.yaml").write_text(content, encoding='utf-8')
    
    def create_readme(self, paper_dir: Path, paper: Dict):
        """Create README.md."""
        content = f"""# {paper['name']}

**Category**: {paper['tag'].upper()}  
**Version**: {paper['version']}  
**Impact**: {paper['impact']}/5  
**Complexity**: {paper['complexity']}/5  
**Evidence Type**: {paper['type']}

## Overview

See `paper.md` for the full paper.

## Quick Start

```bash
# Install dependencies
pip install -r Code/requirements.txt

# Run all experiments
python Code/reproduce.py

# Run specific experiment
python Code/reproduce.py 7

# List available experiments
python Code/reproduce.py --list
```

## Contents

```
.
├── paper.md          # Main paper (Markdown)
├── paper.tex         # LaTeX version (if generated)
├── paper.pdf         # PDF version (if generated)
├── meta.yaml         # Paper metadata
├── Code/
│   ├── trace.yaml    # Links to original source files
│   ├── core/         # Reusable modules
│   ├── experiments/  # Numbered experiment scripts
│   └── reproduce.py  # Main entry point
└── Data/
    └── results/      # Generated results (JSON)
```

## Code Traceability

See `Code/trace.yaml` for links to the original source files in the repository.

## Citation

See `CITATION.md` for how to cite this work.

## License

MIT License (code), CC-BY-4.0 (paper). See `LICENSE`.

---

*This is exploratory research. Results require independent validation.*
"""
        (paper_dir / "README.md").write_text(content, encoding='utf-8')
    
    def create_citation(self, paper_dir: Path, paper: Dict):
        """Create CITATION.md."""
        slug_short = paper['slug'][:20].replace('_', '')
        
        content = f"""# Citation

## BibTeX

```bibtex
@misc{{dawnfield2025{slug_short},
    title = {{{paper['name']}}},
    author = {{Dawn Field Institute Research Team}},
    year = {{2025}},
    publisher = {{Zenodo}},
    doi = {{10.5281/zenodo.XXXXXXX}},
    url = {{https://doi.org/10.5281/zenodo.XXXXXXX}}
}}
```

## Plain Text

Dawn Field Institute Research Team. (2025). {paper['name']}. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

---

*Update DOI after Zenodo upload.*
"""
        (paper_dir / "CITATION.md").write_text(content, encoding='utf-8')
    
    def create_license(self, paper_dir: Path):
        """Create LICENSE."""
        content = """MIT License

Copyright (c) 2025 Dawn Field Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Papers are additionally licensed under CC-BY-4.0.
"""
        (paper_dir / "LICENSE").write_text(content, encoding='utf-8')
    
    def create_master_index(self, papers: List[Dict]):
        """Create master meta.yaml index."""
        index = {
            'schema_version': '2.0',
            'description': 'Dawn Field Theory Preprints Index',
            'updated': datetime.now().isoformat(),
            'papers': []
        }
        
        for paper in sorted(papers, key=lambda x: (-x['impact'], -x['complexity'])):
            index['papers'].append({
                'slug': paper['slug'],
                'title': paper['name'],
                'category': paper['tag'],
                'impact': paper['impact'],
                'complexity': paper['complexity'],
                'path': f"{paper['slug']}/"
            })
        
        content = yaml.dump(index, default_flow_style=False, sort_keys=False)
        (self.preprints_path / "meta.yaml").write_text(content, encoding='utf-8')
        print(f"\n[OK] Created master index: meta.yaml")
    
    def migrate_all(self):
        """Run the full migration."""
        print("=" * 60)
        print("Dawn Field Theory Preprints Migration")
        print("=" * 60)
        
        papers = self.find_all_papers()
        print(f"\nFound {len(papers)} papers to migrate")
        
        migrated = []
        for paper in papers:
            try:
                self.create_paper_folder(paper)
                migrated.append(paper)
            except Exception as e:
                print(f"  [ERROR] Failed to migrate {paper['slug']}: {e}")
        
        # Create master index
        self.create_master_index(migrated)
        
        print("\n" + "=" * 60)
        print(f"Migration complete: {len(migrated)}/{len(papers)} papers")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review the new structure")
        print("  2. Delete drafts/ folder when satisfied")
        print("  3. Update any external references")
        print("  4. Commit changes")


def main():
    script_path = Path(__file__).parent
    migrator = PrePrintMigrator(script_path)
    migrator.migrate_all()


if __name__ == "__main__":
    main()
