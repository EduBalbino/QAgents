#!/usr/bin/env python3
"""
End-to-end smoke test:
- synthesize a tiny CSV with mixed numeric/categorical columns
- run one tiny training + save a model
- load the saved model via:
  1) scripts/core/builders.load_model (in-repo loader)
  2) scripts/predict.load_model_pt (standalone loader)
- verify decision scores match closely for a few rows

This is a developer utility, not a unit test framework integration.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    # Allow running as a script without installing the package.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from scripts.core.builders import (
        Recipe,
        ansatz,
        csv,
        device,
        encoder,
        measurement,
        pls_to_pow2,
        quantile_uniform,
        save,
        select,
        train,
        run,
        load_model,
    )
    from scripts.predict import load_model_pt

    rng = np.random.default_rng(0)
    n = 128
    feats = [f"f{i}" for i in range(6)] + ["cat"]
    label = "Label"

    df = pd.DataFrame({c: rng.normal(size=n) for c in feats if c != "cat"})
    df["cat"] = rng.choice(["A", "B", "C"], size=n)
    # Slight imbalance
    df[label] = (rng.random(n) > 0.7).astype(int)

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "tiny.csv")
        df.to_csv(csv_path, index=False)
        model_path = os.path.join(td, "model.pt")

        recipe = (
            Recipe()
            | csv(csv_path, sample_size=None)
            | select(features=feats, label=label)
            | quantile_uniform(n_quantiles=64, output_distribution="uniform")
            | pls_to_pow2(max_qubits=8, components=8)
            | device("lightning.qubit", wires_from_features=True)
            | encoder("angle_embedding_y", hadamard=False, reupload=False, angle_range="0_pi")
            | ansatz("ring_rot_cnot", layers=1)
            | measurement("z_vec", wires=list(range(len(feats))))
            | train(
                lr=0.005,
                batch=16,
                epochs=1,
                seed=0,
                test_size=0.2,
                stratify=True,
                balanced_batches=True,
                balanced_pos_frac=0.5,
                class_weights="none",
                abort_on_degen=False,  # smoke test focuses on portability, not model quality
            )
            | save(model_path)
        )

        summary = run(recipe)
        if not os.path.exists(model_path):
            raise SystemExit(f"FAILED: model not saved at {model_path}")

        clf_repo = load_model(model_path)
        clf_port = load_model_pt(model_path)

        df_in = df[feats].head(8).copy()
        s1 = np.asarray(clf_repo.decision_function(df_in), dtype=np.float64)
        s2 = np.asarray(clf_port.decision_function(df_in), dtype=np.float64)

        if s1.shape != s2.shape:
            raise SystemExit(f"FAILED: shape mismatch {s1.shape} vs {s2.shape}")

        max_abs = float(np.max(np.abs(s1 - s2))) if s1.size else 0.0
        print("roundtrip_ok=1")
        print(f"max_abs_diff={max_abs:.3e}")
        print(f"model_path={model_path}")
        print(f"summary_test_bacc={summary.get('metrics', {}).get('balanced_accuracy')}")

        if not np.isfinite(max_abs) or max_abs > 1e-6:
            raise SystemExit("FAILED: scores differ too much; save/load not portable.")


if __name__ == "__main__":
    main()
