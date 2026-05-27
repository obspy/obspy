#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Markdown and PDF documentation for the SiteXML standalone bundle.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import date
from html import escape
from pathlib import Path
import tempfile

from markdown import markdown
from playwright.sync_api import sync_playwright


DOCS = (
    (
        "sitexml-standalone-usage-guide.md",
        "sitexml-standalone-usage-guide.md",
        "sitexml-standalone-usage-guide.pdf",
    ),
    (
        "sitexml-tabular-input-reference.md",
        "sitexml-tabular-input-reference.md",
        "sitexml-tabular-input-reference.pdf",
    ),
    (
        "sitexml-quality-indexes-guide.md",
        "sitexml-quality-indexes-guide.md",
        "sitexml-quality-indexes-guide.pdf",
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

ul, ol {
    margin: 5pt 0 9pt;
    padding-left: 20pt;
}

li {
    margin: 2pt 0;
}

li > ul,
li > ol {
    margin: 2pt 0 4pt;
    padding-left: 18pt;
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


def _render_html(markdown_text):
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


def _normalize_release_version(release_version):
    if not release_version:
        return ""
    for prefix in ("sitexml-scripts-", "sitexml-scripts_"):
        if release_version.startswith(prefix):
            release_version = release_version[len(prefix):]
            break
    if release_version.startswith("v") and len(release_version) > 1:
        return release_version[1:]
    return release_version


def _footer_template(release_version, build_date):
    release_version = escape(_normalize_release_version(release_version))
    build_date = escape(build_date or "")
    return f"""
    <div style="
        color: #52606d;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 11px;
        padding: 0 16mm;
        width: 100%;
    ">
      <div style="
          border-top: 1px solid #d9e2ec;
          display: flex;
          justify-content: space-between;
          padding-top: 4px;
          width: 100%;
      ">
        <span>Release version: {release_version}</span>
        <span>&copy; 2026 ORFEUS and ObsPy contributors</span>
        <span>Build date: {build_date}</span>
      </div>
    </div>
    """


def build_docs(source_dir, output_dir, release_version=None, build_date=None):
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
                markdown_text = source_path.read_text(encoding="utf-8")

                markdown_path.write_text(markdown_text, encoding="utf-8")
                html_path.write_text(
                    _render_html(markdown_text),
                    encoding="utf-8",
                )
                page.goto(html_path.as_uri(), wait_until="networkidle")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    display_header_footer=True,
                    header_template="<span></span>",
                    footer_template=_footer_template(
                        release_version, build_date),
                    margin={
                        "top": "18mm",
                        "right": "16mm",
                        "bottom": "22mm",
                        "left": "16mm",
                    },
                    print_background=True,
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
    parser.add_argument(
        "--release-version",
        default=None,
        help="standalone release version to write into the PDF footer")
    parser.add_argument(
        "--build-date",
        default=date.today().isoformat(),
        help="build date to write into the PDF footer")
    args = parser.parse_args(argv)

    build_docs(
        args.source_dir,
        args.output_dir,
        release_version=args.release_version,
        build_date=args.build_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
