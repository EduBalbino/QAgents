import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from scripts.core import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate saved QML models on edge_pls8_full.csv (or another CSV)."
    )
    p.add_argument(
        "--models-glob",
        default="models/*.pt",
        help="Glob for model checkpoints (default: models/*.pt)",
    )
    p.add_argument(
        "--csv",
        default="data/edge_pls8_full.csv",
        help="CSV dataset to evaluate (default: data/edge_pls8_full.csv)",
    )
    p.add_argument(
        "--label",
        default="Attack_label",
        help="Label column name (default: Attack_label)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max rows to evaluate (default: 500; use 0 for all rows)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Random sample size (overrides --limit if > 0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Optional directory to write per-model predictions CSVs",
    )
    p.add_argument(
        "--no-scaler",
        action="store_true",
        help="Disable model scaler when predicting (useful if data already scaled)",
    )
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable quantile/PLS/PCA preprocessing from the saved model",
    )
    return p.parse_args()


def _coerce_binary_label(y: pd.Series) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(y):
        y_str = y.astype(str).str.strip()
        uniq = sorted(set(y_str.tolist()))
        if len(uniq) == 2:
            env_pos = os.environ.get("EDGE_POSITIVE_LABEL")
            if env_pos is not None and str(env_pos).strip() in uniq:
                pos = str(env_pos).strip()
                return np.asarray((y_str == pos).astype(int), dtype=int)
            lower_map = {u.lower(): u for u in uniq}
            pos_tokens = ("attack", "malicious", "anomaly", "intrusion", "true", "yes", "positive")
            neg_tokens = ("benign", "normal", "false", "no", "negative")
            pos = next((lower_map[t] for t in pos_tokens if t in lower_map), None)
            neg = next((lower_map[t] for t in neg_tokens if t in lower_map), None)
            if pos is not None and neg is not None and pos != neg:
                return np.asarray((y_str == pos).astype(int), dtype=int)
            return np.asarray((y_str == uniq[-1]).astype(int), dtype=int)
        lo = uniq[0] if uniq else ""
        return np.asarray((y_str != lo).astype(int), dtype=int)
    return (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int).values


def _select_rows(df: pd.DataFrame, limit: int, sample: int, seed: int) -> pd.DataFrame:
    if sample and sample > 0:
        n = min(sample, len(df))
        return df.sample(n=n, random_state=seed)
    if limit and limit > 0:
        n = min(limit, len(df))
        return df.sample(n=n, random_state=seed)
    return df


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def _dataset_stats(y_true: np.ndarray) -> dict:
    n = len(y_true)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    pos_rate = float(pos / n) if n else 0.0
    maj_rate = float(max(pos, neg) / n) if n else 0.0
    return {
        "n": n,
        "pos": pos,
        "neg": neg,
        "pos_rate": pos_rate,
        "majority_baseline": maj_rate,
    }


def _eval_one(
    model_path: str,
    df: pd.DataFrame,
    label_col: str,
    out_dir: str,
    no_scaler: bool,
    no_preprocess: bool,
) -> Tuple[str, int, dict]:
    clf = load_model(model_path)
    if no_scaler:
        clf.scaler = None
    if no_preprocess:
        clf.quantile = None
        clf.pls = None
        clf.pca = None

    if label_col not in df.columns:
        raise SystemExit(f"Label column not found: {label_col}")

    def _pc_columns() -> List[str]:
        cols = []
        for c in df.columns:
            if c.startswith("PC_"):
                try:
                    idx = int(c.split("_", 1)[1])
                except Exception:
                    continue
                cols.append((idx, c))
        return [c for _, c in sorted(cols, key=lambda t: t[0])]

    def _num_qubits() -> Optional[int]:
        try:
            w = getattr(clf, "weights", None)
            if w is not None and hasattr(w, "shape") and len(w.shape) >= 2:
                return int(w.shape[1])
        except Exception:
            pass
        feats = getattr(clf, "features", None) or []
        return int(len(feats)) if feats else None

    df_in = None
    if getattr(clf, "features", None):
        missing = [c for c in clf.features if c not in df.columns]
        if not missing:
            df_in = df[clf.features]
        else:
            pc_cols = _pc_columns()
            n_q = _num_qubits()
            if pc_cols and n_q is not None and len(pc_cols) >= n_q:
                use_cols = pc_cols[:n_q]
                df_in = df[use_cols]
                # Override features to avoid re-selection inside LoadedQuantumClassifier
                try:
                    clf.features = list(use_cols)
                except Exception:
                    pass
                # If we're already in PLS space, disable train-time preprocessing steps
                try:
                    clf.quantile = None
                    clf.pls = None
                    clf.pca = None
                except Exception:
                    pass
                print(
                    f"[WARN] {os.path.basename(model_path)} missing raw features; "
                    f"falling back to PLS columns {use_cols[0]}..{use_cols[-1]}"
                )
            else:
                raise ValueError(
                    f"Missing required feature columns for model {os.path.basename(model_path)}: "
                    + ", ".join(missing)
                )
    else:
        df_in = df.drop(columns=[label_col])

    y_true = _coerce_binary_label(df[label_col])
    scores = np.asarray(clf.decision_function(df_in)).astype(float)
    threshold = float(getattr(clf, "threshold", 0.0))
    y_pred = (scores >= threshold).astype(int)
    auc = _safe_auc(y_true, scores)
    auc_inv = _safe_auc(y_true, -scores) if auc is not None else None

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "auc_inverted": auc_inv,
        "pos_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "pred_pos_rate": float(np.mean(y_pred)) if len(y_pred) else 0.0,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(model_path))[0] + "_preds.csv"
        out_path = os.path.join(out_dir, out_name)
        out_df = df.copy()
        out_df["prediction"] = y_pred
        out_df["score"] = scores
        out_df.to_csv(out_path, index=False)

    return model_path, len(df_in), metrics


def _print_table(rows: List[Tuple[str, int, dict]]) -> None:
    headers = [
        "model",
        "n",
        "acc",
        "bacc",
        "prec",
        "rec",
        "f1",
        "auc",
        "auc_inv",
        "pos_rate",
        "pred_pos",
    ]
    widths = [max(len(h), 10) for h in headers]
    for path, n, m in rows:
        widths[0] = max(widths[0], len(os.path.basename(path)))
        widths[1] = max(widths[1], len(str(n)))
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for path, n, m in rows:
        def _f(v):
            if v is None:
                return "n/a"
            return f"{v:.4f}"
        print(
            fmt.format(
                os.path.basename(path),
                str(n),
                _f(m["accuracy"]),
                _f(m["balanced_accuracy"]),
                _f(m["precision"]),
                _f(m["recall"]),
                _f(m["f1"]),
                _f(m["auc"]),
                _f(m["auc_inverted"]),
                _f(m["pos_rate"]),
                _f(m["pred_pos_rate"]),
            )
        )


def main() -> None:
    args = parse_args()
    model_paths = sorted(glob.glob(args.models_glob))
    if not model_paths:
        raise SystemExit(f"No models found for glob: {args.models_glob}")

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, low_memory=False)
    df = _select_rows(df, args.limit, args.sample, args.seed)
    y_all = _coerce_binary_label(df[args.label])
    stats = _dataset_stats(y_all)
    print(
        "Dataset stats:",
        f"n={stats['n']}",
        f"pos={stats['pos']}",
        f"neg={stats['neg']}",
        f"pos_rate={stats['pos_rate']:.4f}",
        f"majority_baseline={stats['majority_baseline']:.4f}",
    )

    rows: List[Tuple[str, int, dict]] = []
    errors: List[str] = []
    for mp in model_paths:
        try:
            row = _eval_one(
                mp,
                df=df,
                label_col=args.label,
                out_dir=args.out_dir,
                no_scaler=args.no_scaler,
                no_preprocess=args.no_preprocess,
            )
            rows.append(row)
        except Exception as exc:
            errors.append(f"{os.path.basename(mp)}: {exc}")

    if rows:
        _print_table(rows)
        # Heuristic hints for inverted scores
        for path, _, m in rows:
            auc = m.get("auc")
            auc_inv = m.get("auc_inverted")
            if auc is None or auc_inv is None:
                continue
            if auc < 0.5 and auc_inv > auc:
                print(
                    f"[WARN] {os.path.basename(path)}: AUC<0.5 but inverted AUC={auc_inv:.4f}; "
                    "scores/labels may be flipped."
                )
    if errors:
        print("\nErrors:")
        for e in errors:
            print(" -", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
