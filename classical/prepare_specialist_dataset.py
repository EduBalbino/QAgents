#!/usr/bin/env python3
"""Merge trio CSVs and remap attack labels to specialist taxonomy.

Discovers CIC-BCCC-*/*.csv under --root-dir, maps raw attack names to
the specialist label set (NORMAL + 14 attack classes + Others), and writes
a single output CSV with all feature columns + Attack_label + Attack_type.

Output columns:
  Attack_label — multiclass: NORMAL, Backdoor, DDoS_HTTP, ..., Others
  Attack_type — binary: 0 (benign) / 1 (attack)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

# Raw attack name -> specialist label. Unmapped names default to "Others".
LABEL_MAP: dict[str, str] = {
    "Benign Traffic": "NORMAL",
    "Backdoor": "Backdoor",
    "DDoS HTTP Flood": "DDoS_HTTP",
    "DDoS ICMP Fragmentation": "DDoS_ICMP",
    "DDoS TCP SYN Flood": "DDoS_TCP",
    "DDoS ACK Fragmentation": "DDoS_TCP",       # merge
    "ACK Flood": "DDoS_TCP",                    # merge
    "SYN Flood": "DDoS_TCP",                    # merge
    "DDoS PSHACK Flood": "DDoS_TCP",            # merge
    "DDoS RSTFIN Flood": "DDoS_TCP",            # merge
    "Mirai UDP Plain": "DDoS_UDP",
    "OS Fingerprinting": "Fingerprinting",
    "MITM ARP Spoofing": "MITM",
    "Password Attack": "Password",
    "Telnet Brute Force": "Password",            # merge
    "Port Scanning": "Port_Scanning",
    "Recon Port Scan": "Port_Scanning",          # merge
    "Ransomware": "Ransomware",
    "SQL Injection": "SQL_injection",
    "Uploading Attack": "Uploading",
    "Vulnerability Scanner": "Vulnerability_scanner",
    "XSS": "XSS",
}

SRC_COL_CANDIDATES = ("Attack Name", "attack_name", "Attack_type", "attack")
EXCLUDE_LOWER = {"label", "attack_type", "attack_label"}


def find_src_col(columns: list[str]) -> str:
    for c in SRC_COL_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"No attack column found. Tried: {SRC_COL_CANDIDATES}")


def label_expr(src: str) -> pl.Expr:
    """Vectorized when/then chain — no Python UDFs."""
    expr = pl.lit("Others")
    for raw, mapped in LABEL_MAP.items():
        expr = pl.when(pl.col(src) == raw).then(pl.lit(mapped)).otherwise(expr)
    return expr.alias("Attack_label")


def main() -> int:
    p = argparse.ArgumentParser(description="Merge trio CSVs with specialist taxonomy.")
    p.add_argument("--root-dir", default="data")
    p.add_argument("--out-csv", default="data/processed/trio_multiclass_final_single.csv")
    p.add_argument("--batch-size", type=int, default=250_000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.root_dir)
    out = Path(args.out_csv)

    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite.")

    csvs = sorted(root.glob("CIC-BCCC-*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs under {root}/CIC-BCCC-*/")

    # First pass: detect source columns, build union feature schema
    file_src: dict[Path, str] = {}
    all_features: list[str] = []
    for csv_path in csvs:
        cols = pl.read_csv(csv_path, n_rows=0, encoding="utf8-lossy", ignore_errors=True).columns
        src = find_src_col(cols)
        file_src[csv_path] = src
        for c in cols:
            if c != src and c.lower() not in EXCLUDE_LOWER and c not in all_features:
                all_features.append(c)

    out.parent.mkdir(parents=True, exist_ok=True)
    wrote = False
    total = 0
    counts: dict[str, int] = {}

    for csv_path in csvs:
        src_col = file_src[csv_path]
        reader = pl.read_csv_batched(
            csv_path,
            batch_size=args.batch_size,
            infer_schema_length=1000,
            ignore_errors=True,
            encoding="utf8-lossy",
        )
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            df = batches[0]
            df = (
                df.with_columns(pl.col(src_col).cast(pl.Utf8).str.strip_chars().alias("__src"))
                .with_columns(label_expr("__src"))
                .with_columns(
                    pl.when(pl.col("Attack_label") == "NORMAL")
                    .then(0)
                    .otherwise(1)
                    .cast(pl.Int8)
                    .alias("Attack_type"),
                )
            )

            select = [
                pl.col(c) if c in df.columns else pl.lit(None).alias(c)
                for c in all_features
            ] + [pl.col("Attack_label"), pl.col("Attack_type")]
            out_df = df.select(select)

            total += out_df.height
            for lbl, cnt in out_df.group_by("Attack_label").len().iter_rows():
                counts[str(lbl)] = counts.get(str(lbl), 0) + int(cnt)

            if not wrote:
                out_df.write_csv(out, include_header=True)
                wrote = True
            else:
                with out.open("ab") as f:
                    out_df.write_csv(f, include_header=False)

    if total == 0:
        raise RuntimeError("No rows processed.")

    print(f"Rows: {total}")
    for lbl, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {lbl}: {cnt} ({100 * cnt / total:.2f}%)")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
