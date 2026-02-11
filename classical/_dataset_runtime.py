"""Dataset runtime wiring for CIC-BCCC folders under ../data.

Features:
- Redirect legacy dataset paths to local `data/CIC-BCCC-*` sources.
- Normalize raw CIC CSV schema to the project canonical schema:
  - feature names in lowercase.dot convention
  - `Attack_label` (specialist multiclass taxonomy)
  - `Attack_type` (binary 0/1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SPECIALIST_LABEL_MAP: dict[str, str] = {
    "Benign Traffic": "NORMAL",
    "Backdoor": "Backdoor",
    "DDoS HTTP Flood": "DDoS_HTTP",
    "DDoS ICMP Fragmentation": "DDoS_ICMP",
    "DDoS TCP SYN Flood": "DDoS_TCP",
    "DDoS ACK Fragmentation": "DDoS_TCP",
    "Mirai UDP Plain": "DDoS_UDP",
    "OS Fingerprinting": "Fingerprinting",
    "MITM ARP Spoofing": "MITM",
    "Password Attack": "Password",
    "Telnet Brute Force": "Password",
    "Port Scanning": "Port_Scanning",
    "Recon Port Scan": "Port_Scanning",
    "Ransomware": "Ransomware",
    "SQL Injection": "SQL_injection",
    "Uploading Attack": "Uploading",
    "Vulnerability Scanner": "Vulnerability_scanner",
    "XSS": "XSS",
    "ACK Flood": "DDoS_TCP",
    "SYN Flood": "DDoS_TCP",
    "DDoS PSHACK Flood": "DDoS_TCP",
    "DDoS RSTFIN Flood": "DDoS_TCP",
}

SEMANTIC_OVERRIDES: dict[str, str] = {
    "Timestamp": "frame.time",
    "Src IP": "ip.src_host",
    "Dst IP": "ip.dst_host",
    "Src Port": "tcp.srcport",
    "Dst Port": "tcp.dstport",
}


def _normalize_col_name(name: str) -> str:
    col = name.strip()
    if col in SEMANTIC_OVERRIDES:
        return SEMANTIC_OVERRIDES[col]
    return (
        col.lower()
        .replace(" ", ".")
        .replace("/", ".")
    )


def _is_probably_binary_numeric(series: Any) -> bool:
    try:
        vals = set(series.dropna().astype(int).unique().tolist())
    except Exception:
        return False
    return vals.issubset({0, 1}) and len(vals) > 0


def _normalize_attack_columns(df: Any) -> Any:
    """Normalize attack columns to Attack_label/Attack_type when possible."""
    if df is None:
        return df

    has_attack_name = "Attack Name" in df.columns
    has_attack_label = "Attack_label" in df.columns
    has_attack_type = "Attack_type" in df.columns

    if has_attack_name and not has_attack_label:
        mapped = (
            df["Attack Name"]
            .astype(str)
            .str.strip()
            .map(SPECIALIST_LABEL_MAP)
            .fillna("Others")
        )
        df["Attack_label"] = mapped

    if not has_attack_type:
        if "Label" in df.columns and _is_probably_binary_numeric(df["Label"]):
            df["Attack_type"] = df["Label"].astype(int)
        elif "label" in df.columns and _is_probably_binary_numeric(df["label"]):
            df["Attack_type"] = df["label"].astype(int)
        elif "is_attack" in df.columns and _is_probably_binary_numeric(df["is_attack"]):
            df["Attack_type"] = df["is_attack"].astype(int)
        elif "Attack_label" in df.columns:
            attack_label_series = df["Attack_label"]
            if _is_probably_binary_numeric(attack_label_series):
                df["Attack_type"] = attack_label_series.astype(int)
            else:
                df["Attack_type"] = (attack_label_series.astype(str) != "NORMAL").astype(int)

    # Drop legacy target columns that create leakage and naming conflicts.
    for legacy_col in ("Attack Name", "Label", "label", "attack_cat", "attack_category", "is_attack"):
        if legacy_col in df.columns and legacy_col not in {"Attack_label", "Attack_type"}:
            df.drop(columns=[legacy_col], inplace=True)

    return df


def _normalize_feature_columns(df: Any) -> Any:
    """Normalize feature column names to lowercase.dot (except target/split)."""
    if df is None:
        return df

    protected = {"Attack_label", "Attack_type", "split"}
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in protected:
            continue
        normalized = _normalize_col_name(str(col))
        if normalized != col:
            rename_map[col] = normalized

    if rename_map:
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def _normalize_cic_dataframe(df: Any) -> Any:
    df = _normalize_attack_columns(df)
    df = _normalize_feature_columns(df)
    return df


def _project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists():
        return cwd
    if (cwd.parent / "data").exists():
        return cwd.parent
    return cwd


def _largest_csv(folder: Path) -> Path | None:
    csvs = [p for p in folder.glob("*.csv") if p.is_file()]
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_size)


def setup_cic_sources(script_path: str | None = None) -> dict[str, str]:
    root = _project_root()
    data_root = root / "data"

    edge_dir = data_root / "CIC-BCCC-NRC-Edge-IIoTSet-2022"
    iot2023_dir = data_root / "CIC-BCCC-NRC-IoT-2023-Original Training and Testing"
    uq_dir = data_root / "CIC-BCCC-NRC-UQ-IOT-2022"
    cic_dirs = [edge_dir, iot2023_dir, uq_dir]

    edge_default = _largest_csv(edge_dir)
    iot2023_default = _largest_csv(iot2023_dir)
    uq_default = _largest_csv(uq_dir)

    alias_map = {
        "ML-EdgeIIoT-dataset.csv": edge_default,
        "ML-EdgeIIoT-dataset-pp.csv": edge_default,
        "edge_pls8_debug.csv": edge_default,
        "edge_pls8_full.csv": edge_default,
        "CICIDS2017_Combinado.csv": iot2023_default,
        "PCA_CICIDS2017.csv": iot2023_default,
        "CIC-DDoS2019_Combinado.csv": iot2023_default,
        "PCA_CIC-DDoS2019.csv": iot2023_default,
        "UNSW_NB15_Combinado.csv": uq_default,
        "UNSW_NB15_Combinado_preprocessing.csv": uq_default,
        "unified_cicids_unsw_common.csv": uq_default,
    }

    try:
        import pandas as pd  # type: ignore

        original_read_csv = pd.read_csv

        def _resolve_path(path_like: Any) -> Any:
            if not isinstance(path_like, (str, Path)):
                return path_like
            p = Path(path_like)
            if p.exists():
                return str(p)

            name = p.name
            mapped = alias_map.get(name)
            if mapped and mapped.exists():
                return str(mapped)

            for folder in cic_dirs:
                candidate = folder / name
                if candidate.exists():
                    return str(candidate)
            raise FileNotFoundError(
                f"Dataset path not found and no alias mapping is available: {path_like}"
            )

        cic_roots = {str(p.resolve()) for p in cic_dirs if p.exists()}

        def _is_from_raw_cic_csv(resolved_path: Any) -> bool:
            if not isinstance(resolved_path, (str, Path)):
                return False
            try:
                p = Path(resolved_path).resolve()
            except Exception:
                return False
            p_str = str(p)
            return any(p_str.startswith(root) for root in cic_roots)

        def _patched_read_csv(path_like: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            resolved = _resolve_path(path_like)
            loaded = original_read_csv(resolved, *args, **kwargs)

            # Keep behavior untouched for non-DataFrame returns (iterators/chunks).
            if not isinstance(loaded, pd.DataFrame):
                return loaded

            if _is_from_raw_cic_csv(resolved):
                return _normalize_cic_dataframe(loaded)
            return loaded

        pd.read_csv = _patched_read_csv  # type: ignore[assignment]
    except Exception:
        pass

    return {
        "data_root": str(data_root),
        "edge_dir": str(edge_dir),
        "iot2023_dir": str(iot2023_dir),
        "uq_dir": str(uq_dir),
        "edge_default": str(edge_default) if edge_default else "",
        "iot2023_default": str(iot2023_default) if iot2023_default else "",
        "uq_default": str(uq_default) if uq_default else "",
    }
