#!/usr/bin/env python3
"""
Create a derived, leakage-safe dataset for the quantum pipeline:

1) Load `classical/dataset_preprocessado_tratado_winsorizado.csv`
2) Use *all* feature columns (everything except Attack_label + Attack_type)
3) Coerce features using train-fit state (numeric median fill, categorical mapping), apply to test (no leakage)
4) Fit supervised PLSRegression(8) on the full feature matrix (train split only)
5) Quantile-map the PLS outputs to U[0,1] (train split only)
6) Write `data/processed/mergido_preprocessado.csv` with:
   - PC_1..PC_8 (the final quantile-mapped PLS features)
   - Attack_label (0/1)
   - Attack_type (deterministic int code; preserves multiclass type without leakage)
   - split ("train"|"test")

This script intentionally stores a *dataset*, not a pickled model. If you want to reuse
the transformers, extend this script to export *_state in JSON/NPZ (same pattern as training).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# Exact mapping required by the project (do not reorder):
#   0  Backdoor
#   1  DDoS_HTTP
#   2  DDoS_ICMP
#   3  DDoS_TCP
#   4  DDoS_UDP
#   5  Fingerprinting
#   6  MITM
#   7  Password
#   8  Port_Scanning
#   9  Ransomware
#   10 SQL_injection
#   11 Uploading
#   12 Vulnerability_scanner
#   13 XSS
#   14 Others
_ATTACK_TYPE_TO_CODE: Dict[str, int] = {
    "backdoor": 0,
    "ddos_http": 1,
    "ddos_icmp": 2,
    "ddos_tcp": 3,
    "ddos_udp": 4,
    "fingerprinting": 5,
    "mitm": 6,
    "password": 7,
    "port_scanning": 8,
    "ransomware": 9,
    "sql_injection": 10,
    "uploading": 11,
    "vulnerability_scanner": 12,
    "xss": 13,
    "others": 14,
}
_ATTACK_TYPE_CODE_TO_NAME: List[str] = [
    "Backdoor",
    "DDoS_HTTP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_UDP",
    "Fingerprinting",
    "MITM",
    "Password",
    "Port_Scanning",
    "Ransomware",
    "SQL_injection",
    "Uploading",
    "Vulnerability_scanner",
    "XSS",
    "Others",
]


def _attack_type_code_from_attack_label(attack_label: pd.Series) -> np.ndarray:
    """
    Map the *string* Attack_label into a stable Attack_type integer code.

    Required mapping/order (0..13):
      0 Backdoor
      1 DDoS_HTTP
      2 DDoS_ICMP
      3 DDoS_TCP
      4 DDoS_UDP
      5 Fingerprinting
      6 MITM
      7 Password
      8 Port_Scanning
      9 Ransomware
      10 SQL_injection
      11 Uploading
      12 Vulnerability_scanner
      13 XSS

    Benign / non-typed rows ("normal", "others", etc.) are coded as -1.
    """
    s = attack_label.astype(str).str.strip().str.lower()
    out = np.full((s.shape[0],), -1, dtype=np.int32)
    # "normal"/benign stays -1 (not an attack type).
    # Everything else must be explicitly mapped; unknowns are an error to avoid silent corruption.
    for i, v in enumerate(s.tolist()):
        if v in ("normal", "benign"):
            out[i] = -1
            continue
        if v in _ATTACK_TYPE_TO_CODE:
            out[i] = int(_ATTACK_TYPE_TO_CODE[v])
            continue
        raise SystemExit(f"Unknown Attack_label token for Attack_type mapping: {v!r}")
    return out


@dataclass(frozen=True)
class Meta:
    input_csv: str
    output_csv: str
    label_col: str
    type_col: str
    feature_cols: List[str]
    attack_type_mapping: List[str]
    split_seed: int
    test_size: float
    pls_components: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mergido_preprocessado.csv (PCA8 + PLS8 + quantile).")
    p.add_argument(
        "--in",
        dest="in_csv",
        default="classical/dataset_preprocessado_tratado_winsorizado.csv",
        help="Input CSV path",
    )
    p.add_argument(
        "--out",
        dest="out_csv",
        default="data/processed/mergido_preprocessado.csv",
        help="Output CSV path",
    )
    p.add_argument("--label", dest="label_col", default="Attack_label", help="Binary label column (0/1)")
    p.add_argument("--type-col", dest="type_col", default="Attack_type", help="Attack type column to preserve")
    p.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction")
    p.add_argument("--seed", type=int, default=42, help="Split/selection RNG seed")
    p.add_argument(
        "--infer-numeric-rows",
        type=int,
        default=50_000,
        help="Rows used to infer numeric feature columns (fast pre-scan)",
    )
    p.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="If >0, only process the first N rows (debug/dev)",
    )
    return p.parse_args()


def _coerce_fit_apply(
    X_train_df: pd.DataFrame, X_test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Train-fit coercion:
    - numeric-like columns: median impute
    - categorical-like columns: string mapping with unknown=-1.0
    """
    coerce_state: Dict[str, Dict[str, Any]] = {}
    X_train_out: List[np.ndarray] = []
    X_test_out: List[np.ndarray] = []

    for c in X_train_df.columns:
        s_tr_num = pd.to_numeric(X_train_df[c], errors="coerce")
        frac_num = float(np.mean(np.isfinite(s_tr_num.to_numpy(dtype=np.float64, copy=False))))
        if frac_num >= 0.98:
            med = float(np.nanmedian(s_tr_num.to_numpy(dtype=np.float64, copy=False)))
            tr = s_tr_num.fillna(med).to_numpy(dtype=np.float64, copy=False)
            te_num = pd.to_numeric(X_test_df[c], errors="coerce")
            te = te_num.fillna(med).to_numpy(dtype=np.float64, copy=False)
            coerce_state[c] = {"mode": "numeric", "median": med}
        else:
            tr_str = X_train_df[c].astype(str).fillna("").to_list()
            # Deterministic mapping: sort unique tokens.
            uniq = sorted(set(tr_str))
            mapping = {k: float(i) for i, k in enumerate(uniq)}
            unk = -1.0
            tr = np.array([mapping.get(str(v), unk) for v in tr_str], dtype=np.float64)
            te_str = X_test_df[c].astype(str).fillna("").to_list()
            te = np.array([mapping.get(str(v), unk) for v in te_str], dtype=np.float64)
            coerce_state[c] = {"mode": "categorical", "mapping": mapping, "unknown": unk}

        X_train_out.append(tr.reshape(-1, 1))
        X_test_out.append(te.reshape(-1, 1))

    X_train = np.concatenate(X_train_out, axis=1)
    X_test = np.concatenate(X_test_out, axis=1)
    return X_train, X_test, coerce_state


def _train_test_split_idx(y01: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(int(y01.shape[0]), dtype=np.int32)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y01,
    )
    return np.asarray(train_idx, dtype=np.int32), np.asarray(test_idx, dtype=np.int32)


def main() -> None:
    args = _parse_args()
    in_csv = os.path.abspath(args.in_csv)
    out_csv = os.path.abspath(args.out_csv)
    label_col = str(args.label_col)
    type_col = str(args.type_col)
    test_size = float(args.test_size)
    seed = int(args.seed)
    infer_rows = int(args.infer_numeric_rows)
    limit_rows = int(args.limit_rows)

    if not os.path.exists(in_csv):
        raise SystemExit(f"Input CSV not found: {in_csv}")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Pass 1: read header to get feature columns.
    df_head = pd.read_csv(in_csv, nrows=max(1000, infer_rows), low_memory=False)
    if label_col not in df_head.columns:
        raise SystemExit(f"Label column '{label_col}' not found in CSV header.")
    if type_col not in df_head.columns:
        raise SystemExit(f"Type column '{type_col}' not found in CSV header.")
    drop_cols = {label_col, type_col}
    feature_cols = [c for c in df_head.columns if c not in drop_cols]
    if len(feature_cols) < 8:
        raise SystemExit(f"Found only {len(feature_cols)} feature columns after dropping labels; need at least 8.")

    usecols = feature_cols + [label_col, type_col]
    read_kwargs: Dict[str, Any] = {"low_memory": False, "usecols": usecols}
    if limit_rows > 0:
        read_kwargs["nrows"] = int(limit_rows)
    df = pd.read_csv(in_csv, **read_kwargs)

    # Label: coerce to binary 0/1 (0=normal/benign, 1=attack).
    y_ser = df[label_col]
    if pd.api.types.is_numeric_dtype(y_ser):
        y_raw = pd.to_numeric(y_ser, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        y01 = (y_raw > 0.0).astype(np.int32, copy=False)
    else:
        y_str = y_ser.astype(str).str.strip()
        y_low = y_str.str.lower()
        uniq_low = sorted(set(y_low.tolist()))
        neg = None
        for tok in ("normal", "benign"):
            if tok in uniq_low:
                neg = tok
                break
        if neg is None:
            raise SystemExit(
                f"Label '{label_col}' is non-numeric and does not contain an explicit 'normal'/'benign' class. "
                f"Unique(lower) sample={uniq_low[:10]}"
            )
        y01 = (y_low != neg).astype(np.int32).to_numpy()
    uniq = np.unique(y01)
    if uniq.size < 2:
        raise SystemExit(
            f"Label '{label_col}' has only one class in the loaded data (unique={uniq.tolist()}). "
            "This pipeline requires both classes for MI ranking and supervised PLS."
        )

    # Preserve multiclass attack type as a stable integer code (derived from Attack_label).
    # We intentionally do NOT use the source Attack_type column here because in this input CSV
    # it is not reliable (often constant/placeholder).
    attack_type_code = _attack_type_code_from_attack_label(df[label_col])

    # Split and coerce using train-fit state (no leakage).
    train_idx, test_idx = _train_test_split_idx(y01, test_size=test_size, seed=seed)
    if np.unique(y01[train_idx]).size < 2 or np.unique(y01[test_idx]).size < 2:
        raise SystemExit(
            "Train/test split ended up with a single class. "
            "Increase dataset size, adjust test_size, or verify label distribution."
        )

    X_train_df = df.loc[train_idx, feature_cols]
    X_test_df = df.loc[test_idx, feature_cols]
    X_train, X_test, _coerce_state = _coerce_fit_apply(X_train_df, X_test_df)
    type_train = attack_type_code[train_idx]
    type_test = attack_type_code[test_idx]

    from sklearn.preprocessing import QuantileTransformer
    from sklearn.cross_decomposition import PLSRegression

    # Supervised dimensionality reduction. Fit on train only (no leakage).
    pls = PLSRegression(n_components=8, scale=True)
    pls.fit(X_train, y01[train_idx])
    Xpls_train = pls.transform(X_train)
    Xpls_test = pls.transform(X_test)

    qt = QuantileTransformer(
        n_quantiles=int(min(1000, Xpls_train.shape[0])),
        output_distribution="uniform",
        subsample=int(1e9),
        random_state=seed,
    )
    Xq_train = qt.fit_transform(Xpls_train)
    Xq_test = qt.transform(Xpls_test)

    feat_cols = [f"PC_{i+1}" for i in range(8)]
    df_train_out = pd.DataFrame(Xq_train, columns=feat_cols)
    df_test_out = pd.DataFrame(Xq_test, columns=feat_cols)
    df_train_out[label_col] = y01[train_idx]
    df_test_out[label_col] = y01[test_idx]
    df_train_out[type_col] = type_train
    df_test_out[type_col] = type_test
    df_train_out["split"] = "train"
    df_test_out["split"] = "test"
    df_out = pd.concat([df_train_out, df_test_out], ignore_index=True)

    df_out.to_csv(out_csv, index=False)

    meta = Meta(
        input_csv=in_csv,
        output_csv=out_csv,
        label_col=label_col,
        type_col=type_col,
        feature_cols=feature_cols,
        attack_type_mapping=list(_ATTACK_TYPE_CODE_TO_NAME),
        split_seed=seed,
        test_size=test_size,
        pls_components=8,
    )
    meta_path = out_csv + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {meta_path}")
    print(f"[OK] Features used (count={len(feature_cols)}): {feature_cols[:12]}{' ...' if len(feature_cols) > 12 else ''}")


if __name__ == "__main__":
    main()
