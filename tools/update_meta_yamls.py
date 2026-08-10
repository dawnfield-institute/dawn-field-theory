def fix_orphaned_block_lists(yaml_content):
    lines = yaml_content.splitlines()
    fixed_lines = []
    last_key = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('-') and (not fixed_lines or not fixed_lines[-1].rstrip().endswith(':')):
            # Orphaned block list, attach to last key
            if last_key:
                fixed_lines.append('  ' + stripped)
            else:
                fixed_lines.append(stripped)
        else:
            fixed_lines.append(line)
            if ':' in stripped and not stripped.startswith('-'):
                last_key = stripped.split(':', 1)[0]
    return '\n'.join(fixed_lines)

import os
import subprocess

import yaml

META_FILENAME = 'meta.yaml'

# Tracked paths only. A filesystem walk has no notion of .gitignore, so it wrote
# __pycache__, SESSION_NOTES.md and the private `internal/` tree into meta.yaml files
# that are committed to a PUBLIC repo — and made CI unpassable, since none of that
# content exists on a runner.
_TRACKED = None


def _tracked_index(repo_root):
    """{abs dir -> (files, child_dirs)} built from `git ls-files`."""
    global _TRACKED
    if _TRACKED is not None:
        return _TRACKED
    out = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=repo_root, capture_output=True, check=True,
    ).stdout.decode('utf-8', 'replace')
    index = {}
    for rel in (p for p in out.split('\0') if p):
        parts = rel.split('/')
        for i in range(len(parts)):
            d = os.path.join(repo_root, *parts[:i]) if i else repo_root
            files, dirs = index.setdefault(os.path.abspath(d), (set(), set()))
            if i == len(parts) - 1:
                files.add(parts[i])
            else:
                dirs.add(parts[i])
    _TRACKED = index
    return index


def get_child_dirs_and_files(path, repo_root):
    files, dirs = _tracked_index(repo_root).get(os.path.abspath(path), (set(), set()))
    return (
        sorted(f for f in files if f != META_FILENAME),
        sorted(dirs),
    )

def update_meta_yaml(path, repo_root):
    meta_path = os.path.join(path, META_FILENAME)
    if not os.path.exists(meta_path):
        return  # Skip if meta.yaml does not exist
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            meta = yaml.safe_load(content)
        except yaml.YAMLError as e:
            import re
            # Try to fix inline lists first
            fixed_content = re.sub(r'(files|child_directories):\s*\[(.*?)\]', lambda m: f"{m.group(1)}:\n  - " + '\n  - '.join([x.strip() for x in m.group(2).split(',')]), content)
            # Try to fix orphaned block lists if still broken
            try:
                meta = yaml.safe_load(fixed_content)
            except Exception:
                fixed_content2 = fix_orphaned_block_lists(fixed_content)
                try:
                    meta = yaml.safe_load(fixed_content2)
                    print(f"[FIXED] Orphaned block list YAML in: {meta_path}")
                except Exception as e2:
                    print(f"[WARN] Skipping invalid YAML: {meta_path}\n{e}\nAuto-fix failed: {e2}")
                    return
            else:
                print(f"[FIXED] Auto-corrected YAML in: {meta_path}")
        if meta is None:
            print(f"[WARN] Skipping empty or invalid YAML: {meta_path}")
            return
    except Exception as e:
        print(f"[WARN] Error reading {meta_path}: {e}")
        return
    files, child_dirs = get_child_dirs_and_files(path, repo_root)
    meta['files'] = files
    meta['child_directories'] = child_dirs
    with open(meta_path, 'w', encoding='utf-8', newline='\n') as f:
        yaml.dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"meta.yaml updated at {meta_path}")


def process_directory(path, repo_root):
    meta_path = os.path.join(path, META_FILENAME)
    if os.path.exists(meta_path):
        update_meta_yaml(path, repo_root)
    _, child_dirs = get_child_dirs_and_files(path, repo_root)
    for entry in child_dirs:
        process_directory(os.path.join(path, entry), repo_root)

def main():
    # Find the repo root by traversing up until we find a directory containing .git or a known root marker
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(current, '.git')) or os.path.exists(os.path.join(current, 'map.yaml')):
            repo_root = current
            break
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError('Could not find repo root')
        current = parent
    process_directory(repo_root, repo_root)

if __name__ == '__main__':
    main()
