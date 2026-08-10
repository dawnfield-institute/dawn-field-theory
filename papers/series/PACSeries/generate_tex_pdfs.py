#!/usr/bin/env python3
"""
Generate LaTeX and PDF for all PACSeries v0.2 papers.

Uses pandoc for Markdown → LaTeX conversion with a clean academic template,
then pdflatex for LaTeX → PDF compilation.

Usage:
    python generate_tex_pdfs.py          # All papers
    python generate_tex_pdfs.py 2        # Paper 2 only
    python generate_tex_pdfs.py --tex    # TeX only (no PDF)
"""

import subprocess
import sys
import os
from pathlib import Path

V2_DIR = Path(__file__).parent / "v0.2"

PAPERS = [
    ("structure_cost_of_erasure", 1, "The Structure Cost of Erasure"),
    ("balance_constant_decomposition", 2, "The Balance Constant and Its Decomposition"),
    ("feigenbaum_fibonacci_arithmetic", 3, "Feigenbaum Constants from Fibonacci Arithmetic"),
    ("standard_model_fibonacci_arithmetic", 4, "Standard Model Parameters from Fibonacci Arithmetic"),
    ("classical_physics_information_geometry", 5, "Classical Physics from Information Geometry"),
    ("computational_validation_pac_conservation", 6, "Computational Validation of PAC Conservation"),
]

LATEX_HEADER = r"""
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{unicode-math}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,citecolor=green!50!black,urlcolor=blue!60!black}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small PACSeries Paper \thepapernumber}
\fancyhead[R]{\small Dawn Field Institute}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Paper number counter
\newcounter{papernumber}
"""


def generate_header_file(paper_dir: Path, paper_num: int):
    """Write the LaTeX header include file."""
    header_path = paper_dir / "header.tex"
    header_content = LATEX_HEADER + f"\n\\setcounter{{papernumber}}{{{paper_num}}}\n"
    header_path.write_text(header_content, encoding="utf-8")
    return header_path


def md_to_tex(paper_dir: Path, slug: str, paper_num: int, title: str):
    """Convert paper.md to paper.tex using pandoc."""
    md_path = paper_dir / "paper.md"
    tex_path = paper_dir / "paper.tex"
    header_path = generate_header_file(paper_dir, paper_num)

    if not md_path.exists():
        print(f"  SKIP: {md_path} not found")
        return None

    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(tex_path),
        "--standalone",
        "--pdf-engine=xelatex",
        "-V", f"title={title}",
        "-V", "author=Peter Groom",
        "-V", "institute=Dawn Field Institute",
        "-V", "date=February 2026",
        "-V", "documentclass=article",
        "-V", "fontsize=11pt",
        "-V", "classoption=a4paper",
        "-V", "mainfont=Latin Modern Roman",
        "-H", str(header_path),
        "--number-sections",
        "--toc",
        "-V", "toc-title=Contents",
    ]

    result = subprocess.run(cmd, capture_output=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"  ERROR (pandoc): {(result.stderr or '')[:500]}")
        return None

    # Clean up header file
    header_path.unlink(missing_ok=True)

    print(f"  TeX: {tex_path.name}")
    return tex_path


def tex_to_pdf(tex_path: Path):
    """Compile paper.tex to paper.pdf using xelatex (two passes for TOC)."""
    paper_dir = tex_path.parent
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    for pass_num in range(2):
        cmd = [
            "xelatex",
            "-interaction=nonstopmode",
            "-output-directory", str(paper_dir),
            str(tex_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, cwd=str(paper_dir),
            env=env, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0 and pass_num == 1:
            stdout = result.stdout or ''
            errors = [l for l in stdout.split('\n') if l.startswith('!')]
            if errors:
                print(f"  WARNING (xelatex): {errors[0][:200]}")

    pdf_path = tex_path.with_suffix('.pdf')
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  PDF: {pdf_path.name} ({size_mb:.1f} MB)")
        return pdf_path
    else:
        print(f"  WARNING: PDF not generated")
        return None


def cleanup_aux(paper_dir: Path):
    """Remove LaTeX auxiliary files."""
    for ext in ['.aux', '.log', '.out', '.toc', '.fdb_latexmk', '.fls', '.synctex.gz']:
        for f in paper_dir.glob(f'*{ext}'):
            f.unlink(missing_ok=True)


def main():
    tex_only = '--tex' in sys.argv
    target = None

    for arg in sys.argv[1:]:
        if arg.isdigit():
            target = int(arg)
        elif arg == '--tex':
            continue

    papers = PAPERS if target is None else [(s, n, t) for s, n, t in PAPERS if n == target]

    if not papers:
        print(f"No paper #{target} found")
        return

    print(f"{'='*60}")
    print(f"PACSeries v0.2 — {'TeX' if tex_only else 'TeX + PDF'} Generation")
    print(f"{'='*60}")

    success = 0
    for slug, num, title in papers:
        paper_dir = V2_DIR / slug
        print(f"\nPaper {num}: {title}")
        print(f"  Dir: {slug}/")

        tex_path = md_to_tex(paper_dir, slug, num, title)
        if tex_path is None:
            continue

        if not tex_only:
            pdf_path = tex_to_pdf(tex_path)
            cleanup_aux(paper_dir)
            if pdf_path:
                success += 1
        else:
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(papers)} papers {'converted' if tex_only else 'compiled'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
