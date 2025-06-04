import os
from docx import Document
import re

def docx_to_markdown(docx_path, output_dir="converted_markdown"):
    def sanitize_filename(filename):
        return re.sub(r'[^\w\-_. ]', '_', filename)

    def parse_docx(doc_path):
        doc = Document(doc_path)
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                lines.append("")
                continue

            if text.lower().startswith("abstract"):
                lines.append("## Abstract")
            elif re.match(r'^\d+(\.\d+)*\s+', text):  # Section numbers like 1., 2.1, etc.
                section_title = re.sub(r'^\d+(\.\d+)*\s+', '', text)
                level = text.count('.') + 1
                lines.append(f"{'#' * (level + 1)} {section_title}")
            elif text.startswith("Title:"):
                lines.append(f"# {text[6:].strip()}")
            elif text.startswith("Author:") or text.startswith("Date:") or text.startswith("Framework:"):
                lines.append(f"*{text}*")
            else:
                lines.append(text)
        return "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(docx_path)
    title = os.path.splitext(filename)[0]
    markdown_filename = sanitize_filename(title) + ".md"
    markdown_path = os.path.join(output_dir, markdown_filename)

    markdown_content = parse_docx(docx_path)
    with open(markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Converted: {docx_path} → {markdown_path}")
    return markdown_path

# Example usage:
if __name__ == "__main__":
    import glob
    docx_files = glob.glob("*.docx")  # adjust if needed
    for file in docx_files:
        docx_to_markdown(file)
