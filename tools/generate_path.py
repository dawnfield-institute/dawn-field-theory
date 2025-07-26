import os
import yaml
import re
from collections import defaultdict


# Find the repo root by traversing up from the current file until .git or a known marker is found
def find_repo_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, '.git')) or os.path.exists(os.path.join(current, 'README.md')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError('Could not find repo root')
        current = parent

REPO_ROOT = find_repo_root(os.path.dirname(os.path.abspath(__file__)))
REPO_NAME = os.path.basename(REPO_ROOT)
OUTPUT_FILE = os.path.join(REPO_ROOT, 'map.yaml')

def compress_numbered_files(files):
    """
    Compress files like symbol_distribution_step_0.csv, symbol_distribution_step_1.csv, ... into a pattern.
    """
    pattern = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")
    groups = defaultdict(list)
    for f in files:
        m = pattern.match(f)
        if m:
            prefix, num, suffix = m.groups()
            groups[(prefix, suffix)].append(int(num))
        else:
            groups[(f, None)].append(None)
    result = []
    for (prefix, suffix), nums in groups.items():
        if suffix and len(nums) > 6:
            nums.sort()
            # Show first 2, ellipsis, last 2
            result.append(f"{prefix}{nums[0]}{suffix}")
            result.append(f"{prefix}{nums[1]}{suffix}")
            result.append("...")
            result.append(f"{prefix}{nums[-2]}{suffix}")
            result.append(f"{prefix}{nums[-1]}{suffix}")
        else:
            for n in sorted(nums):
                if suffix:
                    result.append(f"{prefix}{n}{suffix}")
                else:
                    result.append(prefix)
    return result

def build_tree(path):
    tree = {}
    entries = sorted(os.listdir(path))
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    # Compress numbered files
    compressed_files = compress_numbered_files(files)
    for entry in compressed_files:
        tree[entry] = None
    for entry in dirs:
        if entry.startswith('.') and entry != '.gitignore':
            continue  # skip hidden files/folders except .gitignore
        full_path = os.path.join(path, entry)
        subtree = build_tree(full_path)
        tree[entry + '/'] = subtree
    return tree

def format_tree(tree, indent=0):
    lines = []
    for key, value in tree.items():
        lines.append('  ' * indent + key)
        if isinstance(value, dict):
            lines.extend(format_tree(value, indent + 1))
    return lines

def main():
    tree = {REPO_NAME + '/': build_tree(REPO_ROOT)}
    lines = format_tree(tree)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"map.yaml generated at {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
