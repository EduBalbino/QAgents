import argparse

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a full PLS dataset (all rows, no train/test split)."
    )
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--label", default="Attack_label", help="Label column name")
    p.add_argument("--components", type=int, default=8, help="PLS components")
    p.add_argument(
        "--no-quantile",
        action="store_true",
        help="Skip quantile uniformization",
    )
    return p.parse_args()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s_num = pd.to_numeric(out[c], errors="coerce")
        if s_num.notna().any():
            med = float(s_num.median()) if s_num.notna().any() else 0.0
            out[c] = s_num.fillna(med)
        else:
            cats = pd.Index(out[c].astype(str).unique())
            mapping = {k: float(i) for i, k in enumerate(cats)}
            out[c] = out[c].astype(str).map(mapping).astype("float64")
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.label not in df.columns:
        raise SystemExit(f"Label column not found: {args.label}")

    y = df[args.label]
    X = df.drop(columns=[args.label])

    # Coerce to numeric like training
    X = coerce_numeric(X)

    # Binary label {0,1}
    if not pd.api.types.is_numeric_dtype(y):
        codes, uniques = pd.factorize(y.astype(str))
        if len(uniques) == 2:
            y01 = pd.Series(codes, index=y.index)
        else:
            y0 = codes[0] if len(codes) > 0 else 0
            y01 = pd.Series((codes != y0).astype(int), index=y.index)
    else:
        y01 = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)

    Xn = X.values

    if not args.no_quantile:
        n_q = min(1000, len(Xn))
        qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="uniform",
            subsample=int(1e9),
            random_state=42,
        )
        Xn = qt.fit_transform(Xn)

    target_dim = max(1, min(int(args.components), Xn.shape[1]))
    pls = PLSRegression(n_components=target_dim)
    pls.fit(Xn, y01.values)
    X_pls = pls.transform(Xn)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_pls)

    feat_cols = [f"PC_{i+1}" for i in range(X_scaled.shape[1])]
    out_df = pd.DataFrame(X_scaled, columns=feat_cols)
    out_df[args.label] = y01.values
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out_df)} rows and {len(out_df.columns)} columns.")


if __name__ == "__main__":
    main()
