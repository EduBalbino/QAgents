import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

from scripts.core import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load a saved QML model and run predictions on CSV or inline points."
    )
    p.add_argument("--model", required=True, help="Path to saved .pt model file")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to CSV with header; model features are auto-selected")
    src.add_argument(
        "--point",
        action="append",
        help="Comma-separated values for a single point; can be repeated",
    )
    p.add_argument(
        "--out",
        help="Optional output path (CSV). Writes the inputs plus prediction column",
    )
    p.add_argument(
        "--decision",
        action="store_true",
        help="Also output continuous decision scores (margin)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max number of rows to predict from CSV (default 500)",
    )
    return p.parse_args()


def parse_points(points_args: List[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for s in points_args:
        parts = [x.strip() for x in s.split(",") if x.strip() != ""]
        if not parts:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            raise SystemExit(f"Invalid numeric value in --point: '{s}'")
    if not rows:
        raise SystemExit("No valid --point rows provided")
    # Validate consistent dimensions
    dims = {len(r) for r in rows}
    if len(dims) != 1:
        raise SystemExit(f"All --point rows must have the same dimension, got: {sorted(dims)}")
    return np.asarray(rows, dtype=float)


def main() -> None:
    args = parse_args()
    clf = load_model(args.model)

    if args.csv:
        df = pd.read_csv(args.csv)
        # If model tracks features, select those
        if getattr(clf, "features", None):
            missing = [c for c in clf.features if c not in df.columns]
            if missing:
                raise SystemExit(
                    "Missing required feature columns from CSV: " + ", ".join(missing)
                )
            df_in = df[clf.features]
        else:
            df_in = df

        if args.limit and args.limit > 0:
            df_in = df_in.head(args.limit)
        preds = clf.predict(df_in)
        preds = np.asarray(preds).astype(int)
        print("Predictions (first 20):", preds[:20].tolist())
        if args.decision:
            scores = clf.decision_function(df_in)
            scores = np.asarray(scores).astype(float)
            print("Scores (first 20):", scores[:20].tolist())

        if args.out:
            out_df = df.head(len(df_in)).copy()
            out_df["prediction"] = preds
            if args.decision:
                out_df["score"] = np.asarray(clf.decision_function(df_in)).astype(float)
            out_df.to_csv(args.out, index=False)
            print(f"Saved predictions to {args.out}")
        return

    # Inline points
    X = parse_points(args.point)
    # If model tracks features, optionally pad/trim to expected size
    exp_dim = len(getattr(clf, "features", [])) or X.shape[1]
    if X.shape[1] != exp_dim:
        raise SystemExit(
            f"Point dimension {X.shape[1]} does not match expected {exp_dim}"
        )

    preds = clf.predict(X)
    preds = np.asarray(preds).astype(int)
    for i, y in enumerate(preds):
        print(f"row={i}\tpred={int(y)}")
    if args.decision:
        scores = np.asarray(clf.decision_function(X)).astype(float)
        for i, s in enumerate(scores):
            print(f"row={i}\tscore={s}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

