# server.py  (no "from __future__ import annotations")
from typing import List
from pathlib import Path
import os


import logging
from mcp.server.fastmcp import FastMCP

import yaml


# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)

# --- Settings (hardcoded repo root on Windows)
REPO_ROOT = Path(r"C:\Users\peter\repos\dawn-field-theory").resolve()

# --- Load CIP Resource Guide ---
RESOURCE_GUIDE_PATH = REPO_ROOT / "cognition_index_protocol/gpt/gpt_resource_guide.yaml"
def load_resource_guide():
    try:
        with open(RESOURCE_GUIDE_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Could not load resource guide: {e}")
        return None

RESOURCE_GUIDE = load_resource_guide()

# --- Utility: Map path/query to CIP instruction ---
def cip_instruction_for_path(path: str) -> str:
    if not RESOURCE_GUIDE:
        return "(CIP instruction unavailable)"
    for theory in RESOURCE_GUIDE.get("theories", []):
        for doc in theory.get("documents", []):
            if path.replace("\\", "/") in doc:
                return theory.get("instruction", theory.get("description", ""))
    return "(No direct CIP instruction found for this path)"

def cip_instruction_for_query(query: str) -> str:
    if not RESOURCE_GUIDE:
        return "(CIP instruction unavailable)"
    for theory in RESOURCE_GUIDE.get("theories", []):
        if query.lower() in theory.get("description", "").lower() or query.lower() in theory.get("instruction", "").lower():
            return theory.get("instruction", theory.get("description", ""))
    return "(No direct CIP instruction found for this query)"

def within_root(rel: str) -> Path:
    p = (REPO_ROOT / (rel or ".")).resolve()
    if not str(p).startswith(str(REPO_ROOT)):
        raise ValueError("Path escapes repo root")
    return p

mcp = FastMCP("repo-mcp")

# -------- Resource (browse via repo://{path}) --------
@mcp.resource("repo://{path}")
def repo_resource(path: str = "") -> dict:
    """Browse repo resources. Returns file contents or directory listing, plus CIP instruction."""
    logging.info(f"[RESOURCE] repo://{path}")
    p = within_root(path)
    cip_instr = cip_instruction_for_path(path)
    if p.is_dir():
        items = [f"{'d' if e.is_dir() else 'f'}\t{e.name}" for e in p.iterdir()]
        return {"type": "directory", "items": items, "cip_instruction": cip_instr}
    return {"type": "file", "content": p.read_text(encoding="utf-8", errors="ignore"), "cip_instruction": cip_instr}

# -------- Tools --------
@mcp.tool()
def list_files(path: str = ".") -> dict:
    """List files under a repo-relative path. Returns CIP instruction for context."""
    logging.info(f"[TOOL] list_files path={path}")
    p = within_root(path)
    cip_instr = cip_instruction_for_path(path)
    items = [f"{'d' if e.is_dir() else 'f'}\t{e.name}" for e in p.iterdir()]
    return {"items": items, "cip_instruction": cip_instr}

@mcp.tool()
def read_file(path: str) -> dict:
    """Read a text file by repo-relative path. Returns CIP instruction for context."""
    logging.info(f"[TOOL] read_file path={path}")
    p = within_root(path)
    cip_instr = cip_instruction_for_path(path)
    if p.is_dir():
        return {"type": "directory", "cip_instruction": cip_instr}
    return {"type": "file", "content": p.read_text(encoding="utf-8", errors="ignore"), "cip_instruction": cip_instr}

@mcp.tool()
def search_repo(query: str, path: str = ".") -> dict:
    """Search repo for a literal string. Returns file:line matches and CIP instruction for context."""
    logging.info(f"[TOOL] search_repo query={query} path={path}")
    root = within_root(path)
    results: List[str] = []
    for r, _dirs, files in os.walk(root):
        for name in files:
            fp = Path(r) / name
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = fp.relative_to(REPO_ROOT)
                    results.append(f"{rel}:{i}:{line}")
    cip_instr = cip_instruction_for_query(query)
    return {"results": results if results else ["(no matches)"], "cip_instruction": cip_instr}

# -------- Batch file read tool --------
@mcp.tool()
def read_files(paths: List[str]) -> dict:
    """Read multiple files by repo-relative paths. Returns a dict mapping each path to its content and CIP instruction."""
    logging.info(f"[TOOL] read_files paths={paths}")
    result = {}
    for path in paths:
        try:
            p = within_root(path)
            cip_instr = cip_instruction_for_path(path)
            if p.is_dir():
                result[path] = {"type": "directory", "cip_instruction": cip_instr}
            else:
                result[path] = {"type": "file", "content": p.read_text(encoding="utf-8", errors="ignore"), "cip_instruction": cip_instr}
        except Exception as e:
            result[path] = {"error": str(e)}
    return result

# -------- CIP Enhancement Tools --------
@mcp.tool()
def validate_cip_compliance(path: str) -> dict:
    """Check if a file/directory follows CIP standards (experimental)."""
    logging.info(f"[TOOL] validate_cip_compliance path={path}")
    p = within_root(path)
    issues = []
    suggestions = []
    
    if p.is_file():
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            # Check for YAML frontmatter
            if not content.startswith("---"):
                issues.append("Missing YAML frontmatter")
                suggestions.append("Add CIP-compliant YAML header with metadata")
            
            # Check for CIP naming convention
            if not any(tag in p.name for tag in ["[m]", "[id]", "[cip]"]):
                issues.append("Filename doesn't follow CIP naming convention")
                suggestions.append("Consider adding CIP tags like [m], [id], or [cip] to filename")
        except Exception as e:
            issues.append(f"Could not read file: {e}")
    
    elif p.is_dir():
        # Check for meta.yaml
        meta_file = p / "meta.yaml"
        if not meta_file.exists():
            issues.append("Missing meta.yaml file")
            suggestions.append("Add meta.yaml file with directory metadata")
    
    cip_instr = cip_instruction_for_path(path)
    return {
        "path": path,
        "compliant": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions,
        "cip_instruction": cip_instr
    }

@mcp.tool()
def find_related_content(path: str) -> dict:
    """Find related files based on CIP metadata connections (experimental)."""
    logging.info(f"[TOOL] find_related_content path={path}")
    p = within_root(path)
    related = []
    
    if not RESOURCE_GUIDE:
        return {"path": path, "related": [], "cip_instruction": "(CIP guide unavailable)"}
    
    # Find which theory this path belongs to
    current_theory = None
    for theory in RESOURCE_GUIDE.get("theories", []):
        for doc in theory.get("documents", []):
            if path.replace("\\", "/") in doc:
                current_theory = theory
                break
        if current_theory:
            break
    
    if current_theory:
        # Add other documents from the same theory
        for doc in current_theory.get("documents", []):
            if doc != path.replace("\\", "/"):
                related.append({"path": doc, "relation": "same_theory", "theory": current_theory["name"]})
        
        # Add experiments from the same theory
        for exp in current_theory.get("experiments", []):
            related.append({"path": exp, "relation": "related_experiment", "theory": current_theory["name"]})
    
    cip_instr = cip_instruction_for_path(path)
    return {
        "path": path,
        "current_theory": current_theory["name"] if current_theory else None,
        "related": related[:10],  # Limit to 10 results
        "cip_instruction": cip_instr
    }

@mcp.tool()
def extract_metadata(path: str) -> dict:
    """Extract CIP metadata from files (experimental)."""
    logging.info(f"[TOOL] extract_metadata path={path}")
    p = within_root(path)
    metadata = {}
    
    if not p.is_file():
        return {"path": path, "error": "Not a file", "cip_instruction": cip_instruction_for_path(path)}
    
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        
        # Extract YAML frontmatter
        if content.startswith("---"):
            try:
                end_marker = content.find("---", 3)
                if end_marker != -1:
                    yaml_content = content[3:end_marker]
                    metadata["yaml_frontmatter"] = yaml.safe_load(yaml_content)
            except Exception as e:
                metadata["yaml_parse_error"] = str(e)
        
        # Extract CIP tags from filename
        filename = p.name
        tags = []
        for tag_pattern in ["[m]", "[id]", "[cip]", "[F]", "[D]", "[E]", "[R]"]:
            if tag_pattern in filename:
                tags.append(tag_pattern)
        metadata["cip_tags"] = tags
        
        # Extract version info
        import re
        version_match = re.search(r"\[v([\d.]+)\]", filename)
        if version_match:
            metadata["version"] = version_match.group(1)
        
        # Basic content analysis
        lines = content.split('\n')
        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(content.split())
        
    except Exception as e:
        metadata["error"] = str(e)
    
    cip_instr = cip_instruction_for_path(path)
    return {
        "path": path,
        "metadata": metadata,
        "cip_instruction": cip_instr
    }

if __name__ == "__main__":
    # Uses STDIO by default for local runs
    mcp.run()
