import os
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_NAME = os.path.basename(REPO_ROOT)

META_FILENAME = 'meta.yaml'

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

def get_child_dirs_and_files(path):
    files = []
    child_dirs = []
    for entry in sorted(os.listdir(path)):
        if entry.startswith('.') and entry != '.gitignore':
            continue
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            child_dirs.append(entry)
        elif entry != META_FILENAME:
            files.append(entry)
    return files, child_dirs

def generate_meta_yaml(path):
    dirname = os.path.basename(path)
    files, child_dirs = get_child_dirs_and_files(path)
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

def process_directory(path):
    # Only create meta.yaml if it doesn't exist
    meta_path = os.path.join(path, META_FILENAME)
    if not os.path.exists(meta_path):
        meta = generate_meta_yaml(path)
        write_meta_yaml(path, meta)
    # Recurse into subdirectories
    for entry in os.listdir(path):
        if entry.startswith('.') and entry != '.gitignore':
            continue
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            process_directory(full_path)

def main():
    process_directory(REPO_ROOT)

if __name__ == '__main__':
    main()
