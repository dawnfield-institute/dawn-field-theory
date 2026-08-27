"""Generate map.yaml — a plain hierarchical listing of the repository.

Enumerates from `git ls-files` rather than walking the filesystem. This is not a
performance choice: a filesystem walk has no notion of .gitignore, so it swept the
gitignored `internal/` tree (600MB, including private material) into a map.yaml that
is committed to a PUBLIC repo. Tracked files are also the correct semantics for a
navigation map of the repository.
"""
import os
import re
import subprocess
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


def repo_name(root):
    """The repository's name, which is NOT the checkout directory's name.

    In a git worktree the checkout is named for the branch's working directory, so
    `basename(root)` yields e.g. 'dft-m17-exp02' and map.yaml's root label silently changes
    on every worktree-based commit. Since the workspace convention routes all repo work
    through dedicated worktrees, that is the normal case rather than the exotic one.

    `.git` is a DIRECTORY in a primary checkout and a FILE in a worktree; in the latter it
    points at the main checkout's git dir, whose parent is the real repository root.
    """
    dot_git = os.path.join(root, '.git')
    if os.path.isfile(dot_git):
        try:
            with open(dot_git, 'r', encoding='utf-8') as fh:
                line = fh.read().strip()
            if line.startswith('gitdir:'):
                gitdir = line.split(':', 1)[1].strip()
                # .../<main-checkout>/.git/worktrees/<name>  ->  .../<main-checkout>
                marker = os.path.join('.git', 'worktrees')
                if marker in gitdir:
                    return os.path.basename(gitdir.split(marker)[0].rstrip(os.sep))
        except OSError:
            pass
    return os.path.basename(root)


REPO_NAME = repo_name(REPO_ROOT)
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

def tracked_paths():
    """Repo-relative paths of every tracked file, gitignored content excluded."""
    out = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode('utf-8', 'replace')
    return [p for p in out.split('\0') if p]


def build_tree(paths):
    """Nest a flat list of repo-relative paths into {name: subtree | None}."""
    root = {}
    dir_files = defaultdict(list)
    for p in paths:
        parts = p.split('/')
        node = root
        for d in parts[:-1]:
            node = node.setdefault(d + '/', {})
        dir_files[id(node)].append(parts[-1])

    def attach(node):
        files = sorted(dir_files.get(id(node), []))
        subdirs = {k: v for k, v in node.items() if k.endswith('/')}
        node.clear()
        for f in compress_numbered_files(files):
            node[f] = None
        for k in sorted(subdirs):
            node[k] = subdirs[k]
            attach(subdirs[k])
        return node

    return attach(root)

def format_tree(tree, indent=0):
    lines = []
    for key, value in tree.items():
        lines.append('  ' * indent + key)
        if isinstance(value, dict):
            lines.extend(format_tree(value, indent + 1))
    return lines

def main():
    tree = {REPO_NAME + '/': build_tree(tracked_paths())}
    lines = format_tree(tree)
    # newline pinned: without it this writes CRLF on Windows and LF on Linux, so the
    # CI freshness check would fail depending on which platform last regenerated.
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"map.yaml generated at {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
