#!/usr/bin/env python3
"""
Build a trio-only merged binary dataset directly from source folders.

Flow: source CSV files (3 fixed datasets) -> one merged binary CSV.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from prepare_merged_iot_datasets import (
    _discover_csv_files,
    _ensure_unique_columns,
    _inventory_files,
    _load_mapping,
    _prepare_chunk,
)


TRIO_SOURCE_DATASETS = (
    "CIC-BCCC-NRC-Edge-IIoTSet-2022",
    "CIC-BCCC-NRC-IoT-2023-Original Training and Testing",
    "CIC-BCCC-NRC-UQ-IOT-2022",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create trio-only merged binary dataset from source CSVs.")
    p.add_argument("--root-dir", default="data/Datasets", help="Root directory containing source datasets.")
    p.add_argument(
        "--out-path",
        default="data/processed/edge_iot_trio_binary.csv",
        help="Output path for trio merged binary CSV.",
    )
    p.add_argument(
        "--mapping-config",
        default="configs/label_group_map_edge_iot.json",
        help="JSON mapping config for canonical label groups.",
    )
    p.add_argument("--chunksize", type=int, default=50000, help="CSV read chunk size.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_path = Path(args.out_path)
    mapping_path = Path(args.mapping_config)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trio_files: List[Path] = []
    for ds in TRIO_SOURCE_DATASETS:
        ds_dir = root_dir / ds
        if not ds_dir.exists():
            raise FileNotFoundError(f"Trio source dataset folder not found: {ds_dir}")
        trio_files.extend(_discover_csv_files(ds_dir))

    infos, union_columns_raw = _inventory_files(trio_files, root_dir)
    union_columns = _ensure_unique_columns(union_columns_raw)
    mapper = _load_mapping(mapping_path)

    write_header = True
    total_rows = 0
    dataset_rows: Dict[str, int] = defaultdict(int)
    file_rows: Dict[str, int] = defaultdict(int)
    binary_counts: Counter = Counter()
    map_cache: Dict[str, Tuple[str, str]] = {}

    for info in infos:
        chunk_iter = pd.read_csv(
            info.path,
            dtype=str,
            chunksize=int(args.chunksize),
            low_memory=False,
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="ignore",
        )
        for chunk in chunk_iter:
            out_chunk, _gcounts, bcounts, _ocounts, _rcnts = _prepare_chunk(
                chunk=chunk,
                info=info,
                union_columns=union_columns,
                mapper=mapper,
                cache=map_cache,
            )
            if out_chunk.empty:
                continue

            binary_chunk = out_chunk.drop(columns=["label_group"])
            binary_chunk.to_csv(
                out_path,
                mode="w" if write_header else "a",
                index=False,
                header=write_header,
                encoding="utf-8",
            )
            write_header = False

            n = len(binary_chunk)
            total_rows += n
            dataset_rows[info.source_dataset] += n
            file_rows[info.source_file] += n
            binary_counts.update(bcounts)

    if total_rows == 0:
        raise RuntimeError("No rows were merged for trio output.")

    summary = {
        "output": str(out_path),
        "total_rows": total_rows,
        "source_dataset_count": len(dataset_rows),
        "source_file_count": len(file_rows),
        "source_datasets": dict(sorted(dataset_rows.items(), key=lambda kv: kv[1], reverse=True)),
        "binary_counts": dict(sorted(binary_counts.items(), key=lambda kv: kv[0])),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Wrote trio merged binary dataset: {out_path}")
    print(f"Wrote trio summary: {summary_path}")
    print(f"Rows merged: {total_rows}")
    for name, rows in sorted(dataset_rows.items(), key=lambda kv: kv[1], reverse=True):
        print(f" - {name}: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

