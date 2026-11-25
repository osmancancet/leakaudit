"""Command-line interface: ``leakaudit data.csv --text TEXT --label LABEL``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .audit import AuditConfig, Thresholds, audit
from .report import to_markdown


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if suf == ".json":
        return pd.read_json(p)
    if suf in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="leakaudit",
                                 description="Audit a labelled text corpus for leakage and shortcut signals.")
    ap.add_argument("data", help="CSV, TSV, JSON(L) or Parquet file")
    ap.add_argument("--text", required=True, help="text column (use --text a+b to join columns)")
    ap.add_argument("--label", required=True, help="label column")
    ap.add_argument("--metadata", nargs="*", default=[], help="metadata columns to test as predictors")
    ap.add_argument("--split", help="column with train/test assignment for layer 3")
    ap.add_argument("--name", help="corpus name for the report")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-cap", type=int, default=20000)
    ap.add_argument("--near-dup-cap", type=int, default=50000)
    ap.add_argument("--min-template-uniqueness", type=float, default=0.95)
    ap.add_argument("--max-near-dup-pairs", type=int, default=0)
    ap.add_argument("--max-normalized-recovery", type=float, default=0.50)
    ap.add_argument("--min-lookup-margin", type=float, default=0.10)
    ap.add_argument("--out", default="leakaudit_report.json", help="JSON report path")
    ap.add_argument("--md", help="optional Markdown summary path")
    a = ap.parse_args(argv)

    df = _read(a.data)
    text_col = a.text
    if "+" in a.text:
        cols = a.text.split("+")
        df["__text__"] = df[cols].fillna("").astype(str).agg("\n".join, axis=1).str.strip()
        text_col = "__text__"
    cfg = AuditConfig(seed=a.seed, sample_cap=a.sample_cap, near_dup_cap=a.near_dup_cap,
                      thresholds=Thresholds(a.min_template_uniqueness, a.max_near_dup_pairs, 0,
                                            a.max_normalized_recovery, a.min_lookup_margin))
    rep = audit(df, text_col, a.label, a.metadata, a.split, cfg, a.name or Path(a.data).stem)
    rep["text_column"] = a.text
    Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    md = to_markdown(rep)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    return 0 if rep["certificate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
