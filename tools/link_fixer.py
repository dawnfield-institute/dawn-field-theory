#!/usr/bin/env python3
"""
Link Fixer for Dawn Field Theory Preprints
Automatically fixes the most common broken link patterns identified by link_checker.py
"""

import os
import re
from pathlib import Path

def fix_file_links(file_path):
    """Fix broken links in a single markdown file"""
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    
    original_content = content
    fixes_made = 0
    
    # Fix 1: Add missing /experiments/ directory
    experiment_patterns = [
        r'(https://github\.com/dawnfield-institute/dawn-field-theory/blob/main/foundational)/(entropy_information_polarity_field|biology_experiments|quantum_validation|symbolic_superfluid_collapse_pi)',
        r'(https://github\.com/dawnfield-institute/dawn-field-theory/tree/main/foundational)/(entropy_information_polarity_field|biology_experiments|quantum_validation|symbolic_superfluid_collapse_pi)'
    ]
    
    for pattern in experiment_patterns:
        old_matches = re.findall(pattern, content)
        content = re.sub(pattern, r'\1/experiments/\2', content)
        new_matches = re.findall(pattern, content)
        fixes_made += len(old_matches) - len(new_matches)
    
    # Fix 2: Encode unmatched closing brackets at end of URLs
    bracket_pattern = r'(https://github\.com/[^)\s]+)\](?!\()'
    old_matches = re.findall(bracket_pattern, content)
    content = re.sub(bracket_pattern, r'\1%5D', content)
    new_matches = re.findall(bracket_pattern, content)
    fixes_made += len(old_matches) - len(new_matches)
    
    # Fix 3: Remove stray backticks from URLs
    backtick_patterns = [
        (r'(https://github\.com/[^)\s]*)`(\*\*:?)', r'\1\2'),  # Remove backtick before **
        (r'(https://github\.com/[^)\s]*)`(\])', r'\1\2'),      # Remove backtick before ]
        (r'(https://github\.com/[^)\s]*)`(\s)', r'\1\2'),      # Remove backtick before space
        (r'(https://github\.com/[^)\s]*)`$', r'\1'),           # Remove backtick at end of line
    ]
    
    for pattern, replacement in backtick_patterns:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_made += 1
    
    # Fix 4: Clean up malformed markdown links
    malformed_patterns = [
        (r'(https://github\.com/[^)\s]*)`\]\((https://github\.com/[^)]+)\)', r'(\1)'),  # `](url) pattern
        (r'\[`(https://github\.com/[^`]+)`\]\((https://github\.com/[^)]+)\)', r'[\1](\2)'),  # [`url`](url) pattern
    ]
    
    for pattern, replacement in malformed_patterns:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_made += 1
    
    # Fix 5: Handle common file extension issues in URLs
    # Remove double asterisks from URLs
    content = re.sub(r'(https://github\.com/[^)\s]*)\*\*', r'\1', content)
    
    # Fix comma issues in URLs
    content = re.sub(r'(https://github\.com/[^)\s]*),\s*', r'\1', content)
    
    # Write back if changes were made
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {fixes_made} issues in {Path(file_path).name}")
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return 0
    else:
        print(f"✨ No fixes needed in {Path(file_path).name}")
    
    return fixes_made

def main():
    """Main function to fix all preprint files"""
    drafts_dir = Path("c:/Users/peter/repos/dawn-field-theory/foundational/docs/preprints/drafts")
    
    if not drafts_dir.exists():
        print(f"Drafts directory not found: {drafts_dir}")
        return
    
    print("🔧 Starting automatic link fixes...")
    print("=" * 50)
    
    total_fixes = 0
    md_files = list(drafts_dir.glob("*.md"))
    
    for md_file in md_files:
        fixes = fix_file_links(md_file)
        total_fixes += fixes
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY: Fixed {total_fixes} total issues across {len(md_files)} files")
    print("\n🔍 Run link_checker.py again to verify fixes!")

if __name__ == "__main__":
    main()
