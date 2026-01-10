#!/usr/bin/env python3
"""
QENEX LAB Document Generator v2.0
Generates publication-ready PDFs via LaTeX compilation
Full-Stack Transmutation | 2026-01-08
"""

import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentSection:
    title: str
    content: str
    subsections: List['DocumentSection'] = field(default_factory=list)


@dataclass
class ResearchDocument:
    title: str
    authors: List[str]
    abstract: str
    sections: List[DocumentSection]
    qlang_code: str = ""
    validation_results: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class DocumentGenerator:
    """
    Production document generator with LaTeX compilation
    Integrates with Scout CLI report command
    """

    PUBLICATIONS_DIR = "/opt/qenex_lab/publications"
    SCOUT_CLI = "/opt/qenex/scout-cli/target/release/scout"

    LATEX_TEMPLATE = r'''\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{float}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{fancyhdr}
\geometry{margin=1in}

% Q-Lang syntax highlighting
\definecolor{qlang-keyword}{RGB}{0,0,180}
\definecolor{qlang-comment}{RGB}{100,100,100}
\definecolor{qlang-string}{RGB}{180,0,0}

\lstdefinelanguage{qlang}{
    keywords={assert, lattice, conserve_energy, quantum_constraint, truth_engine, enable, check, let, const},
    keywordstyle=\color{qlang-keyword}\bfseries,
    comment=[l]{//},
    commentstyle=\color{qlang-comment},
    stringstyle=\color{qlang-string},
    morestring=[b]",
    sensitive=true,
}

\lstset{
    language=qlang,
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{gray},
}

\pagestyle{fancy}
\fancyhead{}
\fancyhead[L]{\small QENEX LAB}
\fancyhead[R]{\small \today}
\fancyfoot[C]{\thepage}

\title{%(title)s}
\author{%(authors)s}
\date{%(date)s}

\begin{document}
\maketitle

\begin{abstract}
%(abstract)s
\end{abstract}

\tableofcontents
\newpage

%(sections)s

%(appendix)s

\end{document}
'''

    def __init__(self):
        os.makedirs(self.PUBLICATIONS_DIR, exist_ok=True)
        self.latex_available = self._check_latex()

    def _check_latex(self) -> bool:
        """Check if pdflatex is available"""
        try:
            result = subprocess.run(
                ["pdflatex", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        if not text:
            return ""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    def _render_sections(self, sections: List[DocumentSection]) -> str:
        """Render sections as LaTeX"""
        output = []
        for section in sections:
            output.append(f"\\section{{{self._escape_latex(section.title)}}}")
            output.append(self._escape_latex(section.content))
            output.append("")
            for sub in section.subsections:
                output.append(f"\\subsection{{{self._escape_latex(sub.title)}}}")
                output.append(self._escape_latex(sub.content))
                output.append("")
        return "\n".join(output)

    def _render_appendix(self, doc: ResearchDocument) -> str:
        """Render appendix with Q-Lang code and validation results"""
        appendix = []

        if doc.qlang_code:
            appendix.append(r"\appendix")
            appendix.append(r"\section{Q-Lang Source Code}")
            appendix.append(r"\begin{lstlisting}[language=qlang]")
            appendix.append(doc.qlang_code)
            appendix.append(r"\end{lstlisting}")

        if doc.validation_results:
            appendix.append(r"\section{Validation Results}")
            appendix.append(r"\begin{tabular}{ll}")
            appendix.append(r"\toprule")
            appendix.append(r"Property & Value \\")
            appendix.append(r"\midrule")
            for key, value in doc.validation_results.items():
                key_escaped = self._escape_latex(str(key))
                if isinstance(value, bool):
                    val_str = r"$\checkmark$" if value else r"$\times$"
                elif isinstance(value, float):
                    val_str = f"{value:.4f}"
                else:
                    val_str = self._escape_latex(str(value))
                appendix.append(f"{key_escaped} & {val_str} \\\\")
            appendix.append(r"\bottomrule")
            appendix.append(r"\end{tabular}")

        return "\n".join(appendix)

    async def generate_pdf(self, document: ResearchDocument) -> str:
        """Generate PDF from document via LaTeX compilation"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(
            c if c.isalnum() or c in [' ', '_'] else '_'
            for c in document.title
        ).replace(' ', '_')[:50]

        pdf_filename = f"{safe_title}_{timestamp}.pdf"
        pdf_path = os.path.join(self.PUBLICATIONS_DIR, pdf_filename)

        # Try Scout CLI report first (better formatting)
        scout_result = await self._try_scout_report(document, pdf_path)
        if scout_result:
            return scout_result

        # Fallback to direct LaTeX compilation
        if self.latex_available:
            latex_result = await self._compile_latex(document, pdf_path)
            if latex_result:
                return latex_result

        # Final fallback: generate markdown
        md_path = pdf_path.replace('.pdf', '.md')
        await self._generate_markdown(document, md_path)
        return md_path

    async def _try_scout_report(self, document: ResearchDocument, output_path: str) -> Optional[str]:
        """Try to use Scout CLI for document generation"""
        if not os.path.exists(self.SCOUT_CLI):
            return None

        try:
            # Create temp discovery JSON
            import json
            discovery = {
                "id": f"DISC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "title": document.title,
                "abstract": document.abstract[:500],
                "confidence": document.validation_results.get("confidence", 0.0),
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(discovery, f)
                temp_json = f.name

            result = subprocess.run(
                [self.SCOUT_CLI, "report",
                 os.path.basename(temp_json).replace('.json', ''),
                 "--format", "pdf",
                 "--output", output_path],
                capture_output=True,
                timeout=60,
                cwd=os.path.dirname(temp_json)
            )

            os.unlink(temp_json)

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"[PDF Generator] Scout CLI generated: {output_path}")
                return output_path

        except Exception as e:
            print(f"[PDF Generator] Scout CLI failed: {e}")

        return None

    async def _compile_latex(self, document: ResearchDocument, output_path: str) -> Optional[str]:
        """Compile LaTeX to PDF"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "document.tex")

            # Generate LaTeX content
            latex_content = self.LATEX_TEMPLATE % {
                'title': self._escape_latex(document.title),
                'authors': " \\and ".join(self._escape_latex(a) for a in document.authors),
                'date': datetime.now().strftime("%B %d, %Y"),
                'abstract': self._escape_latex(document.abstract),
                'sections': self._render_sections(document.sections),
                'appendix': self._render_appendix(document),
            }

            with open(tex_path, 'w') as f:
                f.write(latex_content)

            # Compile twice for TOC
            for _ in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "document.tex"],
                    capture_output=True,
                    cwd=tmpdir,
                    timeout=60
                )

            pdf_src = os.path.join(tmpdir, "document.pdf")
            if os.path.exists(pdf_src):
                shutil.copy(pdf_src, output_path)
                print(f"[PDF Generator] LaTeX generated: {output_path}")
                return output_path

        return None

    async def _generate_markdown(self, document: ResearchDocument, output_path: str):
        """Generate markdown as fallback"""

        md_content = f"""# {document.title}

**Authors:** {', '.join(document.authors)}

**Date:** {datetime.now().strftime("%B %d, %Y")}

## Abstract

{document.abstract}

---

"""
        for section in document.sections:
            md_content += f"## {section.title}\n\n{section.content}\n\n"

        if document.qlang_code:
            md_content += f"""
---

## Appendix: Q-Lang Source Code

```qlang
{document.qlang_code}
```
"""

        if document.validation_results:
            md_content += "\n## Validation Results\n\n| Property | Value |\n|----------|-------|\n"
            for key, value in document.validation_results.items():
                md_content += f"| {key} | {value} |\n"

        with open(output_path, 'w') as f:
            f.write(md_content)

        print(f"[PDF Generator] Markdown generated: {output_path}")


# Create singleton instance
_generator = None

def get_generator() -> DocumentGenerator:
    global _generator
    if _generator is None:
        _generator = DocumentGenerator()
    return _generator


async def generate_qenex_paper(topic: str) -> str:
    """Generate a QENEX LAB branded PDF paper (legacy API)"""
    print(f"[PDF Generator] Generating paper for: {topic}")

    generator = get_generator()

    document = ResearchDocument(
        title=f"QENEX LAB Research: {topic}",
        authors=["QENEX LAB Autonomous System", "Scout Discovery Engine"],
        abstract=f"This document presents autonomous research findings on: {topic}. "
                 f"Generated using the QENEX LAB multi-agent orchestration system with "
                 f"Trinity Blueprint methodology and 18-expert physics validation.",
        sections=[
            DocumentSection(
                title="Introduction",
                content=f"This research investigates {topic} using the QENEX LAB "
                        f"autonomous scientific computing platform."
            ),
            DocumentSection(
                title="Methodology",
                content="The research follows the Trinity Blueprint methodology:\n"
                        "1. DeepSeek hypothesis generation\n"
                        "2. Q-Lang formal physics specification\n"
                        "3. Scout CLI 18-expert validation"
            ),
            DocumentSection(
                title="Results",
                content="Validation pending. Full results will be generated upon "
                        "completion of the autonomous research loop."
            ),
            DocumentSection(
                title="Conclusion",
                content=f"Further investigation of {topic} is recommended using "
                        f"the QENEX LAB discovery acceleration framework."
            ),
        ],
        qlang_code="// Q-Lang formalization pending\n@truth_engine.enable(\"physics\")\n",
        validation_results={"status": "pending", "confidence": 0.0},
        metadata={"generated": datetime.now().isoformat(), "version": "2.0"}
    )

    return await generator.generate_pdf(document)
