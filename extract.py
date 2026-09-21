"""
Extraction layer.
Converts xlsx / docx / pdf tables into a common intermediate format:

    ExtractedTable(
        source_file: str,
        location: str,      # sheet name / table index / page number
        headers: list[str],
        rows: list[list[str]],
    )

Downstream code never needs to know which parser produced a table.
"""
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass
class ExtractedTable:
    source_file: str
    location: str
    headers: list
    rows: list
    metadata: dict = field(default_factory=dict)


def extract_xlsx(path: str) -> list[ExtractedTable]:
    tables = []
    sheets = pd.read_excel(path, sheet_name=None, header=0, dtype=str)
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        if df.empty:
            continue
        tables.append(
            ExtractedTable(
                source_file=Path(path).name,
                location=f"sheet:{sheet_name}",
                headers=[str(c) for c in df.columns],
                rows=df.astype(str).values.tolist(),
            )
        )
    return tables


def extract_docx(path: str) -> list[ExtractedTable]:
    from docx import Document

    doc = Document(path)
    tables = []
    for i, table in enumerate(doc.tables):
        grid = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not grid:
            continue
        headers, rows = grid[0], grid[1:]
        tables.append(
            ExtractedTable(
                source_file=Path(path).name,
                location=f"table:{i}",
                headers=headers,
                rows=rows,
            )
        )
    return tables


def extract_pdf(path: str, flavor: str = "lattice") -> list[ExtractedTable]:
    """
    Two-pass strategy:
      1. Try pdfplumber first (fast, no external deps, good for clean grids).
      2. If a page yields nothing, fall back to Camelot with the given flavor.
         Use flavor="stream" for whitespace-separated tables with no ruling lines.
    """
    import pdfplumber

    tables = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for t_idx, raw in enumerate(page.extract_tables() or []):
                if not raw or len(raw) < 2:
                    continue
                headers = [c or "" for c in raw[0]]
                rows = [[c or "" for c in r] for r in raw[1:]]
                tables.append(
                    ExtractedTable(
                        source_file=Path(path).name,
                        location=f"page:{page_num}:table:{t_idx}",
                        headers=headers,
                        rows=rows,
                    )
                )
    if tables:
        return tables

    # Fallback: Camelot (needs Ghostscript installed on the host)
    import camelot

    camelot_tables = camelot.read_pdf(path, pages="all", flavor=flavor)
    for i, t in enumerate(camelot_tables):
        df = t.df
        headers = df.iloc[0].tolist()
        rows = df.iloc[1:].values.tolist()
        tables.append(
            ExtractedTable(
                source_file=Path(path).name,
                location=f"camelot:{i}",
                headers=headers,
                rows=rows,
            )
        )
    return tables


def extract(path: str) -> list[ExtractedTable]:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return extract_xlsx(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")
