#!/usr/bin/env python3
"""
Link Checker for Dawn Field Theory Preprints
Scans all markdown files in the drafts folder, extracts GitHub links, 
tests them, and reports broken ones with suggested fixes.
Multithreaded for fast link validation.
"""

import os
import re
import requests
import time
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def extract_github_links(content):
    """Extract all GitHub links from markdown content"""
    # Pattern to match GitHub links in markdown
    patterns = [
        r'\[([^\]]*)\]\((https://github\.com/[^)]+)\)',  # [text](link)
        r'(https://github\.com/[^\s\)]+)',  # bare links
    ]
    
    links = []
    for pattern in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            if len(match.groups()) == 2:
                # Link with text
                text, url = match.groups()
                links.append({
                    'text': text,
                    'url': url,
                    'line_content': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
            else:
                # Bare link
                url = match.group(1)
                links.append({
                    'text': '',
                    'url': url,
                    'line_content': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
    
    return links

def test_url_worker(url_info):
    """Worker function to test a single URL with context info"""
    url = url_info['url']
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        is_working = response.status_code == 200
        status = response.status_code
    except requests.exceptions.RequestException as e:
        is_working = False
        status = str(e)
    
    return {
        **url_info,
        'is_working': is_working,
        'status': status,
        'suggested_fixes': suggest_fix(url) if not is_working else []
    }

def test_url(url, timeout=10):
    """Test if a URL returns 200 status code (kept for compatibility)"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200, response.status_code
    except requests.exceptions.RequestException as e:
        return False, str(e)

def suggest_fix(url):
    """Suggest potential fixes for broken URLs"""
    fixes = []
    
    # Common missing /experiments/ pattern
    if '/foundational/' in url and '/experiments/' not in url:
        # Check for specific patterns that need /experiments/
        experiment_patterns = [
            'entropy_information_polarity_field',
            'biology_experiments',
            'quantum_validation',
            'symbolic_superfluid_collapse_pi',
            'landauer_symbolic_erasure_energy_validation',
            'born_rule',
            'symbolic_entropy_collapse_vs_quantum_decoherence'
        ]
        
        for pattern in experiment_patterns:
            if pattern in url:
                fixed_url = url.replace('/foundational/', '/foundational/experiments/')
                fixes.append(f"Add /experiments/: {fixed_url}")
                break
    
    # Check for unencoded characters
    if '[' in url or ']' in url:
        fixed_url = url.replace('[', '%5B').replace(']', '%5D')
        fixes.append(f"Encode brackets: {fixed_url}")
    
    if ' ' in url:
        fixed_url = url.replace(' ', '%20')
        fixes.append(f"Encode spaces: {fixed_url}")
    
    return fixes

def scan_file(file_path):
    """Scan a single markdown file for GitHub links and prepare them for testing"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    links = extract_github_links(content)
    url_infos = []
    
    for link_info in links:
        # Find line number
        lines_before = content[:link_info['start']].count('\n')
        line_number = lines_before + 1
        
        url_info = {
            'file': file_path,
            'line': line_number,
            'url': link_info['url'],
            'text': link_info['text'],
            'line_content': link_info['line_content']
        }
        
        url_infos.append(url_info)
    
    return url_infos

def main():
    """Main function to scan all preprint files with multithreading"""
    drafts_dir = Path("c:/Users/peter/repos/dawn-field-theory/foundational/docs/preprints/drafts")
    
    if not drafts_dir.exists():
        print(f"Drafts directory not found: {drafts_dir}")
        return
    
    print("🔍 Scanning preprint files for GitHub links...")
    print("=" * 60)
    
    # Find all markdown files and collect all URL info
    md_files = list(drafts_dir.glob("*.md"))
    all_url_infos = []
    file_url_counts = {}
    
    for md_file in md_files:
        print(f"\n📄 Scanning: {md_file.name}")
        url_infos = scan_file(md_file)
        all_url_infos.extend(url_infos)
        file_url_counts[md_file.name] = len(url_infos)
        print(f"  Found {len(url_infos)} links")
    
    if not all_url_infos:
        print("No GitHub links found in any files.")
        return
    
    print(f"\n� Testing {len(all_url_infos)} links with multithreading...")
    print("=" * 60)
    
    # Test all URLs concurrently
    all_results = []
    broken_links = []
    working_links = []
    
    # Use ThreadPoolExecutor for concurrent HTTP requests
    max_workers = min(20, len(all_url_infos))  # Limit to 20 concurrent requests
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all URL tests
        future_to_url = {executor.submit(test_url_worker, url_info): url_info for url_info in all_url_infos}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_url):
            result = future.result()
            all_results.append(result)
            completed += 1
            
            if result['is_working']:
                working_links.append(result)
                print(f"  ✅ ({completed}/{len(all_url_infos)}) {Path(result['file']).name}:{result['line']}")
            else:
                broken_links.append(result)
                print(f"  ❌ ({completed}/{len(all_url_infos)}) {Path(result['file']).name}:{result['line']} - {result['status']}")
    
    # Group results by file for display
    print("\n" + "=" * 60)
    print("📄 DETAILED RESULTS BY FILE")
    print("=" * 60)
    
    results_by_file = {}
    for result in all_results:
        file_name = Path(result['file']).name
        if file_name not in results_by_file:
            results_by_file[file_name] = []
        results_by_file[file_name].append(result)
    
    for file_name, results in results_by_file.items():
        working_count = sum(1 for r in results if r['is_working'])
        broken_count = len(results) - working_count
        
        print(f"\n📄 Scanning: {file_name}")
        for result in results:
            if result['is_working']:
                print(f"  ✅ Line {result['line']}: {result['url'][:80]}{'...' if len(result['url']) > 80 else ''}")
            else:
                print(f"  ❌ Line {result['line']}: {result['url'][:60]}{'...' if len(result['url']) > 60 else ''} (Status: {result['status']})")
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 SUMMARY REPORT")
    print("=" * 60)
    print(f"Total links found: {len(all_results)}")
    print(f"Working links: {len(working_links)}")
    print(f"Broken links: {len(broken_links)}")
    
    if broken_links:
        print(f"\n🔧 BROKEN LINKS REQUIRING FIXES:")
        print("-" * 40)
        
        for result in broken_links:
            file_name = Path(result['file']).name
            print(f"\n📁 {file_name}:{result['line']}")
            print(f"🔗 URL: {result['url']}")
            print(f"📄 Context: {result['line_content'][:100]}...")
            print(f"❌ Status: {result['status']}")
            
            if result['suggested_fixes']:
                print(f"💡 Suggested fixes:")
                for fix in result['suggested_fixes']:
                    print(f"   • {fix}")
        
        print(f"\n📝 Create fix script? (y/n)")
        
    else:
        print("\n🎉 All links are working!")

if __name__ == "__main__":
    main()
