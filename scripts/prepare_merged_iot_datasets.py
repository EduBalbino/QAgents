#!/usr/bin/env python3
"""
Merge CIC IoT datasets under data/Datasets into:
1) grouped multiclass dataset (label_group)
2) binary dataset (Attack_label)

Also writes QA artifacts:
- merge_summary.json
- merge_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ATTACK_COL_CANDIDATES = ("Attack Name", "attack_name", "attack", "label_name")
LABEL_COL_CANDIDATES = ("Label", "label", "Attack_label", "attack_label")
DEVICE_COL_CANDIDATES = ("Device", "device")

CANONICAL_EXTRA_COLUMNS = [
    "attack_name_raw",
    "label_group",
    "Attack_label",
    "source_dataset",
    "source_file",
]


def _norm_col(col: str) -> str:
    return str(col or "").strip()


def _norm_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _find_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    cols = [_norm_col(c) for c in columns]
    by_norm = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in by_norm:
            return by_norm[key]
    return None


@dataclass(frozen=True)
class CsvFileInfo:
    path: Path
    source_dataset: str
    source_file: str
    columns: Tuple[str, ...]
    attack_col: str
    label_col: str
    device_col: Optional[str]


class LabelMapper:
    def __init__(self, config: Dict[str, object]) -> None:
        self.others_label = str(config.get("others_label", "Others"))
        self.benign_label = str(config.get("benign_label", "Benign"))
        exact_raw = config.get("exact_map", {})
        regex_raw = config.get("regex_rules", [])

        self.exact_map: Dict[str, str] = {}
        if isinstance(exact_raw, dict):
            for k, v in exact_raw.items():
                nk = _norm_text(str(k))
                if nk:
                    self.exact_map[nk] = str(v)

        self.regex_rules: List[Tuple[re.Pattern[str], str]] = []
        if isinstance(regex_raw, list):
            for item in regex_raw:
                if not isinstance(item, dict):
                    continue
                pattern = str(item.get("pattern", "")).strip()
                group = str(item.get("group", "")).strip()
                if not pattern or not group:
                    continue
                self.regex_rules.append((re.compile(pattern, re.IGNORECASE), group))

    def map_label(self, raw_label: str) -> Tuple[str, str]:
        """Return (group, mapping_rule)."""
        key = _norm_text(raw_label)
        if not key:
            return self.others_label, "empty"

        if key in self.exact_map:
            return self.exact_map[key], "exact"

        for pat, group in self.regex_rules:
            if pat.search(key):
                return group, f"regex:{pat.pattern}"

        return self.others_label, "fallback_others"


def _discover_csv_files(root_dir: Path) -> List[Path]:
    files = sorted(p for p in root_dir.rglob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root_dir}")
    return files


def _read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        try:
            row = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc
    return [_norm_col(c) for c in row]


def _inventory_files(csv_files: List[Path], root_dir: Path) -> Tuple[List[CsvFileInfo], List[str]]:
    infos: List[CsvFileInfo] = []
    union_columns: List[str] = []
    union_seen = set()
    errors: List[str] = []

    for p in csv_files:
        rel = p.relative_to(root_dir)
        source_dataset = rel.parts[0] if rel.parts else "unknown_dataset"
        source_file = rel.as_posix()
        try:
            cols = _read_header(p)
        except Exception as exc:
            errors.append(f"{source_file}: failed reading header ({exc})")
            continue

        attack_col = _find_column(cols, ATTACK_COL_CANDIDATES)
        label_col = _find_column(cols, LABEL_COL_CANDIDATES)
        device_col = _find_column(cols, DEVICE_COL_CANDIDATES)

        if not attack_col or not label_col:
            errors.append(
                f"{source_file}: missing required label columns "
                f"(attack_col={attack_col}, label_col={label_col})"
            )
            continue

        for c in cols:
            if c not in union_seen:
                union_seen.add(c)
                union_columns.append(c)

        infos.append(
            CsvFileInfo(
                path=p,
                source_dataset=source_dataset,
                source_file=source_file,
                columns=tuple(cols),
                attack_col=attack_col,
                label_col=label_col,
                device_col=device_col,
            )
        )

    if errors:
        msg = "\n".join(errors)
        raise ValueError(f"Inventory errors detected:\n{msg}")

    # Ensure optional device column exists in the merged schema.
    if "Device" not in union_seen and "device" not in union_seen:
        union_columns.append("Device")

    return infos, union_columns


def _ensure_unique_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    used = set()
    for c in columns:
        base = _norm_col(c)
        if not base:
            base = "unnamed_col"
        cur = base
        idx = 1
        while cur in used:
            idx += 1
            cur = f"{base}_{idx}"
        used.add(cur)
        out.append(cur)
    return out


def _load_mapping(path: Path) -> LabelMapper:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Mapping config must be a JSON object: {path}")
    return LabelMapper(cfg)


def _prepare_chunk(
    chunk: pd.DataFrame,
    info: CsvFileInfo,
    union_columns: List[str],
    mapper: LabelMapper,
    cache: Dict[str, Tuple[str, str]],
) -> Tuple[pd.DataFrame, Counter, Counter, Counter, Counter]:
    chunk.columns = [_norm_col(c) for c in chunk.columns]

    # Fill missing columns for schema alignment.
    for c in union_columns:
        if c not in chunk.columns:
            chunk[c] = ""

    attack_raw_series = chunk[info.attack_col].fillna("").astype(str).str.strip()

    mapped_groups: List[str] = []
    mapped_rules: List[str] = []
    for raw in attack_raw_series:
        key = _norm_text(raw)
        if key in cache:
            g, rule = cache[key]
        else:
            g, rule = mapper.map_label(raw)
            cache[key] = (g, rule)
        mapped_groups.append(g)
        mapped_rules.append(rule)

    out = chunk[union_columns].copy()
    out["attack_name_raw"] = attack_raw_series
    out["label_group"] = mapped_groups
    out["Attack_label"] = [0 if g == mapper.benign_label else 1 for g in mapped_groups]
    out["source_dataset"] = info.source_dataset
    out["source_file"] = info.source_file

    group_counts = Counter(mapped_groups)
    binary_counts = Counter(out["Attack_label"].tolist())
    rule_counts = Counter(mapped_rules)
    others_counts = Counter()
    for raw, g in zip(attack_raw_series.tolist(), mapped_groups):
        if g == mapper.others_label:
            others_counts[_norm_text(raw) or "<empty>"] += 1

    return out, group_counts, binary_counts, others_counts, rule_counts


def _write_markdown_summary(path: Path, summary: Dict[str, object]) -> None:
    def _table(headers: List[str], rows: List[List[str]]) -> str:
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    total_rows = int(summary["total_rows"])
    group_counts = summary["group_counts"]
    binary_counts = summary["binary_counts"]
    dataset_rows = summary["dataset_rows"]
    file_rows = summary["file_rows"]
    others_top = summary["others_top"]
    rule_counts = summary["rule_counts"]

    lines: List[str] = []
    lines.append("# IoT Merge Summary")
    lines.append("")
    lines.append(f"- total_rows: **{total_rows}**")
    lines.append(f"- source_files: **{len(file_rows)}**")
    lines.append(f"- source_datasets: **{len(dataset_rows)}**")
    lines.append("")

    gc_rows = [[k, str(v), f"{(100.0 * v / max(total_rows, 1)):.3f}%"] for k, v in group_counts.items()]
    lines.append("## label_group distribution")
    lines.append("")
    lines.append(_table(["Group", "Rows", "Share"], gc_rows))
    lines.append("")

    bc_rows = [[str(k), str(v), f"{(100.0 * v / max(total_rows, 1)):.3f}%"] for k, v in binary_counts.items()]
    lines.append("## Attack_label distribution")
    lines.append("")
    lines.append(_table(["Attack_label", "Rows", "Share"], bc_rows))
    lines.append("")

    ds_rows = [[k, str(v), f"{(100.0 * v / max(total_rows, 1)):.3f}%"] for k, v in dataset_rows.items()]
    lines.append("## Rows by source dataset")
    lines.append("")
    lines.append(_table(["SourceDataset", "Rows", "Share"], ds_rows))
    lines.append("")

    top_file_rows = sorted(file_rows.items(), key=lambda kv: kv[1], reverse=True)[:30]
    fr_rows = [[k, str(v)] for k, v in top_file_rows]
    lines.append("## Top source files by rows")
    lines.append("")
    lines.append(_table(["SourceFile", "Rows"], fr_rows))
    lines.append("")

    lines.append("## Mapping rules usage")
    lines.append("")
    mr_rows = [[k, str(v)] for k, v in rule_counts.items()]
    lines.append(_table(["Rule", "Rows"], mr_rows))
    lines.append("")

    lines.append("## Top labels mapped to Others")
    lines.append("")
    if others_top:
        ot_rows = [[k, str(v)] for k, v in others_top]
        lines.append(_table(["RawLabel(normalized)", "Rows"], ot_rows))
    else:
        lines.append("_No labels mapped to Others._")
    lines.append("")

    os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare merged IoT datasets with grouped and binary labels.")
    p.add_argument("--root-dir", default="data/Datasets", help="Root directory containing source CSV datasets.")
    p.add_argument("--out-dir", default="data/processed", help="Output directory for merged artifacts.")
    p.add_argument(
        "--mapping-config",
        default="configs/label_group_map_edge_iot.json",
        help="JSON mapping config for canonical label groups.",
    )
    p.add_argument("--chunksize", type=int, default=50000, help="CSV read chunk size.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--max-files", type=int, default=0, help="Optional cap for source files (0 means all).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    mapping_path = Path(args.mapping_config)

    csv_files = _discover_csv_files(root_dir)
    if int(args.max_files) > 0:
        csv_files = csv_files[: int(args.max_files)]

    infos, union_columns_raw = _inventory_files(csv_files, root_dir)
    union_columns = _ensure_unique_columns(union_columns_raw)
    mapper = _load_mapping(mapping_path)

    out_grouped = out_dir / "edge_iot_merged_grouped.csv"
    out_binary = out_dir / "edge_iot_merged_binary.csv"
    out_reports_dir = out_dir / "reports"
    out_summary_json = out_reports_dir / "merge_summary.json"
    out_summary_md = out_reports_dir / "merge_summary.md"

    for p in (out_grouped, out_binary, out_summary_json, out_summary_md):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {p}. Use --overwrite.")

    os.makedirs(out_reports_dir, exist_ok=True)

    write_header_grouped = True
    write_header_binary = True

    total_rows = 0
    dataset_rows: Dict[str, int] = defaultdict(int)
    file_rows: Dict[str, int] = defaultdict(int)
    group_counts: Counter = Counter()
    binary_counts: Counter = Counter()
    others_counts: Counter = Counter()
    rule_counts: Counter = Counter()

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
            out_chunk, gcounts, bcounts, ocounts, rcnts = _prepare_chunk(
                chunk=chunk,
                info=info,
                union_columns=union_columns,
                mapper=mapper,
                cache=map_cache,
            )
            n = len(out_chunk)
            if n == 0:
                continue

            out_chunk.to_csv(
                out_grouped,
                mode="w" if write_header_grouped else "a",
                index=False,
                header=write_header_grouped,
                encoding="utf-8",
            )
            write_header_grouped = False

            binary_chunk = out_chunk.drop(columns=["label_group"])
            binary_chunk.to_csv(
                out_binary,
                mode="w" if write_header_binary else "a",
                index=False,
                header=write_header_binary,
                encoding="utf-8",
            )
            write_header_binary = False

            total_rows += n
            dataset_rows[info.source_dataset] += n
            file_rows[info.source_file] += n
            group_counts.update(gcounts)
            binary_counts.update(bcounts)
            others_counts.update(ocounts)
            rule_counts.update(rcnts)

    summary = {
        "root_dir": str(root_dir),
        "mapping_config": str(mapping_path),
        "grouped_output": str(out_grouped),
        "binary_output": str(out_binary),
        "total_rows": total_rows,
        "source_file_count": len(file_rows),
        "source_dataset_count": len(dataset_rows),
        "group_counts": dict(sorted(group_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "binary_counts": dict(sorted(binary_counts.items(), key=lambda kv: kv[0])),
        "dataset_rows": dict(sorted(dataset_rows.items(), key=lambda kv: kv[1], reverse=True)),
        "file_rows": dict(sorted(file_rows.items(), key=lambda kv: kv[1], reverse=True)),
        "others_top": others_counts.most_common(100),
        "rule_counts": dict(sorted(rule_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }

    with out_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    _write_markdown_summary(out_summary_md, summary=summary)

    print(f"Wrote grouped dataset: {out_grouped}")
    print(f"Wrote binary dataset:  {out_binary}")
    print(f"Wrote summary json:    {out_summary_json}")
    print(f"Wrote summary md:      {out_summary_md}")
    print(f"Rows merged:           {total_rows}")
    print(f"Source files merged:   {len(file_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
