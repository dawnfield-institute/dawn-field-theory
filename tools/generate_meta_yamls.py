import os
import yaml
import fnmatch
import sys

# Set REPO_ROOT to the current working directory so the script can be called from the repo root
REPO_ROOT = os.getcwd()
REPO_NAME = os.path.basename(REPO_ROOT)

META_FILENAME = 'meta.yaml'

def load_gitignore_patterns():
    """Load and parse .gitignore patterns"""
    gitignore_path = os.path.join(REPO_ROOT, '.gitignore')
    patterns = []
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    patterns.append(line)
    
    return patterns

def is_ignored(path, gitignore_patterns):
    """Check if a path should be ignored based on gitignore patterns"""
    # Get relative path from repo root
    rel_path = os.path.relpath(path, REPO_ROOT)
    
    # Check each gitignore pattern
    for pattern in gitignore_patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_pattern = pattern[:-1]
            if fnmatch.fnmatch(rel_path, dir_pattern) or fnmatch.fnmatch(os.path.basename(path), dir_pattern):
                return True
            # Also check if any parent directory matches
            parts = rel_path.split(os.sep)
            for i, part in enumerate(parts):
                if fnmatch.fnmatch(part, dir_pattern):
                    return True
        else:
            # Handle file patterns
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
            # Check if pattern matches any part of the path
            parts = rel_path.split(os.sep)
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
    
    return False

# Example semantic scopes for common directory names
SEMANTIC_SCOPE_MAP = {
    'tools': ['tools', 'utility'],
    'docs': ['documentation'],
    'experiments': ['experiments'],
    'models': ['models'],
    'utils': ['utils', 'tools'],
    'results': ['results', 'analysis'],
    'core': ['core'],
    'agents': ['agents', 'modeling'],
    'compression': ['compression', 'utilities', 'data'],
    'entropy': ['entropy', 'recursion', 'field theory'],
    'learning': ['learning', 'CIMM'],
    'optimization': ['optimization', 'CIMM'],
    'visualization': ['visualization', 'tools'],
    'reference_material': ['reference', 'experiment'],
}

def get_semantic_scope(dirname):
    return SEMANTIC_SCOPE_MAP.get(dirname.lower(), [dirname])

def get_child_dirs_and_files(path, gitignore_patterns):
    files = []
    child_dirs = []
    for entry in sorted(os.listdir(path)):
        if entry.startswith('.') and entry != '.gitignore':
            continue
        full_path = os.path.join(path, entry)
        
        # Skip if ignored by gitignore
        if is_ignored(full_path, gitignore_patterns):
            continue
            
        if os.path.isdir(full_path):
            child_dirs.append(entry)
        elif entry != META_FILENAME:
            files.append(entry)
    return files, child_dirs

def generate_meta_yaml(path, gitignore_patterns):
    dirname = os.path.basename(path)
    files, child_dirs = get_child_dirs_and_files(path, gitignore_patterns)
    meta = {
        'schema_version': '2.0',
        'directory_name': dirname,
        'description': f"Auto-generated metadata for {dirname} directory.",
        'semantic_scope': get_semantic_scope(dirname),
        'files': files,
        'child_directories': child_dirs,
    }
    return meta

def write_meta_yaml(path, meta):
    meta_path = os.path.join(path, META_FILENAME)
    with open(meta_path, 'w', encoding='utf-8') as f:
        yaml.dump(meta, f, sort_keys=False, allow_unicode=True)
    print(f"meta.yaml generated at {meta_path}")

def process_directory(path, gitignore_patterns):
    # Skip if this directory is ignored
    if is_ignored(path, gitignore_patterns):
        return
        
    # Only create meta.yaml if it doesn't exist
    meta_path = os.path.join(path, META_FILENAME)
    if not os.path.exists(meta_path):
        meta = generate_meta_yaml(path, gitignore_patterns)
        write_meta_yaml(path, meta)
    
    # Recurse into subdirectories
    for entry in os.listdir(path):
        if entry.startswith('.') and entry != '.gitignore':
            continue
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path) and not is_ignored(full_path, gitignore_patterns):
            process_directory(full_path, gitignore_patterns)

def main():
    gitignore_patterns = load_gitignore_patterns()
    print(f"Loaded {len(gitignore_patterns)} gitignore patterns")
    process_directory(REPO_ROOT, gitignore_patterns)

if __name__ == '__main__':
    main()
