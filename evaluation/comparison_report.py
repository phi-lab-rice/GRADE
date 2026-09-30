#!/usr/bin/env python3
"""Build a side-by-side HTML comparison of reproduced and golden artifacts."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reproduced-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def files_below(root: Path, suffixes: set[str]) -> set[Path]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def href(path: Path) -> str:
    return quote(path.as_posix(), safe="/._-")


def artifact_directory(root: Path, primary: str, legacy: str) -> Path:
    """Prefer the indexed artifact layout while retaining old run support."""

    preferred = root / primary
    return preferred if preferred.is_dir() else root / legacy


def table_card(title: str, reference: Path, reproduced: Path) -> str:
    def text_or_missing(path: Path) -> str:
        if not path.is_file():
            return "<p class=\"missing\">Missing</p>"
        return f"<pre>{html.escape(path.read_text(encoding='utf-8', errors='replace'))}</pre>"

    return "\n".join(
        [
            "<section class=\"comparison\">",
            f"<h3>{html.escape(title)}</h3>",
            "<div class=\"pair\">",
            f"<article><h4>Golden reference</h4>{text_or_missing(reference)}</article>",
            f"<article><h4>Newly reproduced</h4>{text_or_missing(reproduced)}</article>",
            "</div></section>",
        ]
    )


def figure_card(title: str, reference: Path, reproduced: Path, report_root: Path) -> str:
    def image_or_missing(path: Path) -> str:
        if not path.is_file():
            return "<p class=\"missing\">Missing</p>"
        relative = Path(os.path.relpath(path, report_root))
        return f"<a href=\"{href(relative)}\"><img src=\"{href(relative)}\" alt=\"{html.escape(title)}\"></a>"

    return "\n".join(
        [
            "<section class=\"comparison\">",
            f"<h3>{html.escape(title)}</h3>",
            "<div class=\"pair\">",
            f"<article><h4>Golden reference</h4>{image_or_missing(reference)}</article>",
            f"<article><h4>Newly reproduced</h4>{image_or_missing(reproduced)}</article>",
            "</div></section>",
        ]
    )


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root.resolve()
    reproduced_root = args.reproduced_root.resolve()
    output = (args.output or reproduced_root / "comparison.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Golden results use the published ``paper_*`` hierarchy.  A fresh
    # reproduction mirrors indexed figure panels there, while its text tables
    # remain in ``tables/``.  The fallback paths retain compatibility with an
    # older artifact layout.
    reference_tables = artifact_directory(reference_root, "paper_tables", "tables")
    reproduced_tables = artifact_directory(reproduced_root, "paper_tables", "tables")
    reference_figures = artifact_directory(reference_root, "paper_figures", "figures")
    reproduced_figures = artifact_directory(reproduced_root, "paper_figures", "figures")
    table_names = sorted(files_below(reference_tables, {".txt"}) | files_below(reproduced_tables, {".txt"}))
    # The paper bundle also contains introductory and qualitative panels that
    # are deliberately outside this quantitative reproduction workflow.  Show
    # exactly the panels the run generated, paired with their golden versions.
    figure_names = sorted(
        files_below(reference_figures, {".png", ".jpg", ".jpeg", ".svg"})
        & files_below(reproduced_figures, {".png", ".jpg", ".jpeg", ".svg"})
    )

    table_sections = "\n".join(
        table_card(name.as_posix(), reference_tables / name, reproduced_tables / name)
        for name in table_names
    ) or "<p>No table artifacts were found.</p>"
    figure_sections = "\n".join(
        figure_card(name.as_posix(), reference_figures / name, reproduced_figures / name, output.parent)
        for name in figure_names
    ) or "<p>No figure artifacts were found.</p>"

    document = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>GRADE artifact comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; background: #fbfcfc; }}
h1 {{ margin-bottom: .25rem; }} .note {{ color: #566573; }}
.comparison {{ margin: 2rem 0; }} .pair {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
article {{ background: white; border: 1px solid #d5d8dc; border-radius: .5rem; padding: 1rem; overflow: auto; }}
h4 {{ margin-top: 0; }} pre {{ margin: 0; white-space: pre; font-family: ui-monospace, monospace; font-size: .8rem; }}
img {{ max-width: 100%; height: auto; display: block; }} .missing {{ color: #b03a2e; font-weight: 600; }}
@media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>GRADE evaluation: golden vs. reproduced</h1>
<p class=\"note\">Golden inputs: {html.escape(str(reference_root))}<br>New outputs: {html.escape(str(reproduced_root))}</p>
<h2>Tables</h2>{table_sections}
<h2>Figures</h2>{figure_sections}
</body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Saved comparison report: {output}")


if __name__ == "__main__":
    main()
