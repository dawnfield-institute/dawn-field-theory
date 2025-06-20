import os
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_NAME = os.path.basename(REPO_ROOT)
OUTPUT_FILE = os.path.join(REPO_ROOT, 'map.yaml')

def build_tree(path):
    tree = {}
    for entry in sorted(os.listdir(path)):
        if entry.startswith('.') and entry != '.gitignore':
            continue  # skip hidden files/folders except .gitignore
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subtree = build_tree(full_path)
            tree[entry + '/'] = subtree
        else:
            tree[entry] = None
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
