#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Markdown and PDF documentation for the SiteXML standalone bundle.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import shutil
import tempfile

from markdown import markdown
from playwright.sync_api import sync_playwright


DOCS = (
    (
        "standalone-executable-usage-guide.md",
        "SiteXML-Standalone-Usage-Guide.md",
        "SiteXML-Standalone-Usage-Guide.pdf",
    ),
    (
        "tabular-input-reference.md",
        "tabular-input-reference.md",
        "tabular-input-reference.pdf",
    ),
    (
        "quality-indexes-guide.md",
        "quality-indexes-guide.md",
        "quality-indexes-guide.pdf",
    ),
)


CSS = """
@page {
    margin: 18mm 16mm;
}

body {
    color: #1f2933;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11pt;
    line-height: 1.45;
}

h1, h2, h3, h4 {
    color: #102a43;
    line-height: 1.2;
}

h1 {
    font-size: 24pt;
    margin: 0 0 18pt;
}

h2 {
    border-top: 1px solid #d9e2ec;
    font-size: 17pt;
    margin: 30pt 0 10pt;
    padding-top: 14pt;
}

h3 {
    font-size: 13pt;
    margin: 22pt 0 8pt;
}

a {
    color: #1d4ed8;
    text-decoration: none;
}

blockquote {
    border-left: 4px solid #9fb3c8;
    color: #334e68;
    margin: 12pt 0;
    padding: 1pt 0 1pt 10pt;
}

code {
    background: #f0f4f8;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 9.5pt;
    padding: 1px 3px;
}

pre {
    background: #f0f4f8;
    border: 1px solid #d9e2ec;
    border-radius: 4px;
    overflow-wrap: break-word;
    padding: 9pt;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    padding: 0;
}

table {
    border-collapse: collapse;
    font-size: 9.5pt;
    margin: 12pt 0;
    width: 100%;
}

th, td {
    border: 1px solid #bcccdc;
    padding: 5pt 6pt;
    vertical-align: top;
}

th {
    background: #e6f0ff;
    color: #102a43;
    font-weight: 700;
}

tr:nth-child(even) td {
    background: #f8fafc;
}
"""


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _title_from_markdown(markdown_text):
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "SiteXML Documentation"


def _render_html(source_path):
    markdown_text = source_path.read_text(encoding="utf-8")
    body = markdown(
        markdown_text,
        extensions=[
            "extra",
            "fenced_code",
            "sane_lists",
            "smarty",
            "tables",
            "toc",
        ],
        output_format="html5",
    )
    return HTML_TEMPLATE.format(
        title=_title_from_markdown(markdown_text),
        css=CSS,
        body=body,
    )


def build_docs(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir, sync_playwright() as p:
        tmpdir = Path(tmpdir)
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            for source_name, markdown_name, pdf_name in DOCS:
                source_path = source_dir / source_name
                markdown_path = output_dir / markdown_name
                pdf_path = output_dir / pdf_name
                html_path = tmpdir / (pdf_path.stem + ".html")

                shutil.copyfile(source_path, markdown_path)
                html_path.write_text(
                    _render_html(source_path),
                    encoding="utf-8",
                )
                page.goto(html_path.as_uri(), wait_until="networkidle")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                )
        finally:
            browser.close()


def main(argv=None):
    parser = ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--source-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="folder containing the standalone Markdown source files")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="folder where Markdown and PDF files will be written")
    args = parser.parse_args(argv)

    build_docs(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
