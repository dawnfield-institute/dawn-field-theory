#!/usr/bin/env python3
"""
md_to_tex.py — Convert a PACSeries paper.md into paper.tex.

Standalone (no pandoc). Handles the constructs the PACSeries papers use:
headings, paragraphs, $...$/$$...$$ math (passthrough), pipe tables (booktabs),
ordered/unordered lists, bold/italic/inline-code, links, and horizontal rules.
Math and inline-code spans are protected before LaTeX escaping so their contents
are never mangled.

Compiles with xelatex or lualatex (unicode passthrough via unicode-math), or
pdflatex (unicode chars in prose may need the fallback below).

Usage:
    python md_to_tex.py <paper_dir> --number N --date "May 2026"
    # writes <paper_dir>/paper.tex from <paper_dir>/paper.md
"""
import re
import sys
import argparse
from pathlib import Path

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{textcomp}
\else
  \usepackage{unicode-math}
  \defaultfontfeatures{Scale=MatchLowercase}
\fi
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,citecolor=green!50!black,urlcolor=blue!60!black}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small PACSeries Paper %(number)s}
\fancyhead[R]{\small Dawn Field Institute}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\title{%(title)s}
\author{Peter Groom \\ Dawn Field Institute}
\date{%(date)s}

\begin{document}
\maketitle
"""

FOOTER = "\n\\end{document}\n"


def escape_tex(s):
    # s has NO math/code spans (already protected). Escape LaTeX specials.
    out = []
    for ch in s:
        if ch == '\\':
            out.append(r'\textbackslash{}')
        elif ch in '&%#_${}':
            out.append('\\' + ch)
        elif ch == '~':
            out.append(r'\textasciitilde{}')
        elif ch == '^':
            out.append(r'\textasciicircum{}')
        else:
            out.append(ch)
    return ''.join(out)


def inline(text, store):
    """Protect math/code, escape, then apply md inline formatting, then restore."""
    # 1. protect $$...$$, $...$, `code`
    def protect(pat, m):
        store.append(m.group(0) if pat != 'code' else '\\texttt{' + escape_tex(m.group(1)) + '}')
        return f'\x00{len(store)-1}\x00'
    text = re.sub(r'\$\$.*?\$\$', lambda m: protect('math', m), text, flags=re.S)
    text = re.sub(r'(?<!\\)\$.+?(?<!\\)\$', lambda m: protect('math', m), text)
    text = re.sub(r'`([^`]+)`', lambda m: protect('code', m), text)
    # 2. escape the remaining prose
    text = escape_tex(text)
    # 3. md inline -> latex (operate on escaped text; ** and * survive escaping)
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\emph{\1}', text)
    # links [t](u)  -> \href{u}{t}  (u was escaped: unescape _ and # inside url)
    def link(m):
        t, u = m.group(1), m.group(2)
        u = u.replace(r'\_', '_').replace(r'\#', '#').replace(r'\%', '%').replace(r'\&', '&')
        return r'\href{' + u + '}{' + t + '}'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    # 4. restore protected spans
    text = re.sub(r'\x00(\d+)\x00', lambda m: store[int(m.group(1))], text)
    return text


def _split_cells(r):
    # protect $...$ math (which may contain '|') and escaped \| before splitting
    spans = []
    def prot(m):
        spans.append(m.group(0)); return f'\x01{len(spans)-1}\x01'
    r2 = re.sub(r'(?<!\\)\$.+?(?<!\\)\$', prot, r)
    r2 = r2.replace(r'\|', '\x02')
    parts = [p.strip() for p in r2.strip().strip('|').split('|')]
    out = []
    for p in parts:
        p = p.replace('\x02', r'\|')
        p = re.sub(r'\x01(\d+)\x01', lambda m: spans[int(m.group(1))], p)
        out.append(p)
    return out


def convert_table(rows):
    # rows: list of raw md table lines (including header + separator)
    cells = [_split_cells(r) for r in rows]
    header = cells[0]
    body = cells[2:]  # skip separator row (cells[1])
    ncol = len(header)
    align = 'l' * ncol
    out = [r'\begin{longtable}[]{@{}' + align + r'@{}}', r'\toprule']
    store = []
    out.append(' & '.join(inline(h, store) for h in header) + r' \\')
    out.append(r'\midrule\endhead')
    for row in body:
        row = (row + [''] * ncol)[:ncol]
        out.append(' & '.join(inline(c, store) for c in row) + r' \\')
    out.append(r'\bottomrule')
    out.append(r'\end{longtable}')
    return '\n'.join(out)


def convert(md, title):
    lines = md.split('\n')
    out = []
    i = 0
    n = len(lines)
    store = []
    seen_title = False
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced blocks not expected; skip if present
        if stripped.startswith('```'):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(r'\begin{verbatim}')
            out.extend(buf)
            out.append(r'\end{verbatim}')
            continue

        # display math block
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 3:
            out.append(r'\[' + stripped[2:-2].strip() + r'\]')
            i += 1; continue
        if stripped == '$$':
            i += 1; buf = []
            while i < n and lines[i].strip() != '$$':
                buf.append(lines[i]); i += 1
            i += 1
            out.append(r'\[' + '\n'.join(buf) + r'\]')
            continue

        # headings
        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if m:
            level = len(m.group(1)); htext = m.group(2)
            if level == 1 and not seen_title:
                seen_title = True  # title handled in preamble
                i += 1; continue
            cmd = {1: 'section*', 2: 'section*', 3: 'subsection*',
                   4: 'subsubsection*', 5: 'paragraph', 6: 'subparagraph'}.get(level, 'subsection*')
            out.append('\\%s{%s}' % (cmd, inline(htext, store)))
            i += 1; continue

        # horizontal rule
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            out.append(r'\begin{center}\rule{0.5\linewidth}{0.4pt}\end{center}')
            i += 1; continue

        # table (a line with | and next line is separator)
        if '|' in line and i + 1 < n and re.match(r'^\s*\|?[\s:\-|]+\|?\s*$', lines[i+1]) and '-' in lines[i+1]:
            tbl = [lines[i], lines[i+1]]
            i += 2
            while i < n and '|' in lines[i] and lines[i].strip():
                tbl.append(lines[i]); i += 1
            out.append(convert_table(tbl))
            continue

        # unordered list
        if re.match(r'^\s*[-*]\s+', line):
            out.append(r'\begin{itemize}\tightlist')
            while i < n and re.match(r'^\s*[-*]\s+', lines[i]):
                item = re.sub(r'^\s*[-*]\s+', '', lines[i])
                out.append(r'\item ' + inline(item, store))
                i += 1
            out.append(r'\end{itemize}')
            continue

        # ordered list
        if re.match(r'^\s*\d+\.\s+', line):
            out.append(r'\begin{enumerate}\tightlist')
            while i < n and re.match(r'^\s*\d+\.\s+', lines[i]):
                item = re.sub(r'^\s*\d+\.\s+', '', lines[i])
                out.append(r'\item ' + inline(item, store))
                i += 1
            out.append(r'\end{enumerate}')
            continue

        # blank line
        if stripped == '':
            out.append('')
            i += 1; continue

        # paragraph (gather until blank/structural)
        para = [line]
        i += 1
        while i < n and lines[i].strip() != '' and not re.match(r'^(#{1,6}\s|\s*[-*]\s|\s*\d+\.\s|\$\$|```|-{3,}$)', lines[i]) and not ('|' in lines[i] and i+1 < n):
            para.append(lines[i]); i += 1
        out.append(inline(' '.join(p.strip() for p in para), store))

    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paper_dir')
    ap.add_argument('--number', required=True)
    ap.add_argument('--date', default='2026')
    args = ap.parse_args()

    pdir = Path(args.paper_dir)
    md = (pdir / 'paper.md').read_text()
    # title = first '# ' heading
    tm = re.search(r'^#\s+(.*)$', md, flags=re.M)
    title = tm.group(1).strip() if tm else pdir.name
    store = []
    title_tex = inline(title, store)

    body = convert(md, title)
    tex = (PREAMBLE % {'number': args.number, 'title': title_tex, 'date': args.date}) + body + FOOTER
    (pdir / 'paper.tex').write_text(tex)
    print(f"wrote {pdir/'paper.tex'} ({len(tex.splitlines())} lines)")


if __name__ == '__main__':
    main()
