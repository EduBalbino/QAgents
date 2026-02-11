#!/usr/bin/env python3
"""
Standalone inference script for saved EdgeIIoT QML models.

Design constraints:
- This file must be copyable into another repo and work with just a saved `.pt` model file.
- No imports from this codebase (only third-party deps: numpy/pandas/torch/sklearn/pennylane).
- The saved artifact must not rely on pickled sklearn objects; we reconstruct from *_state.

Usage:
  uv run python scripts/predict.py --model models/foo.pt --csv data.csv --out preds.csv --decision
  uv run python scripts/predict.py --model models/foo.pt --point "0.1,0.2,..." --decision
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# CLI
# -----------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a saved QML model and run predictions.")
    p.add_argument("--model", required=True, help="Path to saved .pt model file")
    p.add_argument("--device", default=None, help="Optional PennyLane device override (e.g. lightning.qubit)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to CSV with header; feature columns are auto-selected")
    src.add_argument("--point", action="append", help="Comma-separated values for a single point; can be repeated")
    p.add_argument("--out", help="Optional output path (CSV). Writes inputs plus prediction columns.")
    p.add_argument("--decision", action="store_true", help="Also output continuous decision scores.")
    p.add_argument("--no-preprocess", action="store_true", help="Disable quantile/PLS/PCA/scaler from the saved model.")
    p.add_argument("--limit", type=int, default=500, help="Max rows to predict from CSV (default 500)")
    return p.parse_args()


def _parse_points(points_args: Sequence[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for s in points_args:
        parts = [x.strip() for x in s.split(",") if x.strip() != ""]
        if not parts:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError as e:
            raise SystemExit(f"Invalid numeric value in --point: '{s}'") from e
    if not rows:
        raise SystemExit("No valid --point rows provided")
    dims = {len(r) for r in rows}
    if len(dims) != 1:
        raise SystemExit(f"All --point rows must have the same dimension, got: {sorted(dims)}")
    return np.asarray(rows, dtype=np.float64)


# -----------------------------
# Model (portable)
# -----------------------------


def _try_import_sklearn() -> Tuple[Any, Any, Any, Any]:
    try:
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        MinMaxScaler = None
    try:
        from sklearn.preprocessing import QuantileTransformer
    except Exception:
        QuantileTransformer = None
    try:
        from sklearn.cross_decomposition import PLSRegression
    except Exception:
        PLSRegression = None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        PCA = None
    return MinMaxScaler, QuantileTransformer, PLSRegression, PCA


def _rebuild_transformers_from_state(
    scaler_state: Optional[Dict[str, Any]],
    quantile_state: Optional[Dict[str, Any]],
    pls_state: Optional[Dict[str, Any]],
    pca_state: Optional[Dict[str, Any]],
):
    MinMaxScaler, QuantileTransformer, PLSRegression, PCA = _try_import_sklearn()
    scaler = None
    quantile = None
    pls = None
    pca = None

    if scaler_state and MinMaxScaler is not None:
        sc = MinMaxScaler(feature_range=tuple(scaler_state.get("feature_range", (0, 1))))
        for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
            val = scaler_state.get(attr)
            if val is not None:
                setattr(sc, attr, np.asarray(val))
        scaler = sc

    if quantile_state:
        if QuantileTransformer is None:
            raise ValueError("sklearn is missing QuantileTransformer; cannot load this model.")
        qt = QuantileTransformer(
            n_quantiles=int(quantile_state.get("n_quantiles") or 1000),
            output_distribution=quantile_state.get("output_distribution") or "uniform",
            subsample=int(quantile_state.get("subsample") or 1_000_000_000),
            random_state=quantile_state.get("random_state", None),
        )
        for attr in ["n_quantiles_", "quantiles_", "references_", "n_features_in_"]:
            val = quantile_state.get(attr)
            if val is not None:
                setattr(qt, attr, np.asarray(val) if attr != "n_features_in_" else int(val))
        quantile = qt

    if pls_state and PLSRegression is not None:
        pls_r = PLSRegression(n_components=int(pls_state.get("n_components") or 2))
        for attr in [
            "_x_mean",
            "_x_std",
            "x_mean_",
            "x_std_",
            "x_weights_",
            "x_rotations_",
            "n_features_in_",
        ]:
            val = pls_state.get(attr)
            if val is not None:
                setattr(pls_r, attr, np.asarray(val) if attr != "n_features_in_" else int(val))
        # sklearn>=1.8 uses private attrs; populate from public if needed.
        if getattr(pls_r, "_x_mean", None) is None and getattr(pls_r, "x_mean_", None) is not None:
            pls_r._x_mean = np.asarray(getattr(pls_r, "x_mean_"))
        if getattr(pls_r, "_x_std", None) is None and getattr(pls_r, "x_std_", None) is not None:
            pls_r._x_std = np.asarray(getattr(pls_r, "x_std_"))
        if getattr(pls_r, "_x_mean", None) is None or getattr(pls_r, "_x_std", None) is None:
            raise ValueError("Saved model PLS state is missing required private attrs (_x_mean/_x_std).")
        pls = pls_r

    if pca_state and PCA is not None:
        pca_r = PCA(n_components=int(pca_state.get("n_components") or 2))
        for attr in ["components_", "mean_", "n_features_in_"]:
            val = pca_state.get(attr)
            if val is not None:
                setattr(pca_r, attr, np.asarray(val) if attr != "n_features_in_" else int(val))
        pca = pca_r

    return scaler, quantile, pls, pca


# Encoders/ansatze needed for inference. Keep these aligned with training.
# NOTE: input scaling is handled outside the circuit via compiled_input_{scale,shift}.
def _enc_angle_embedding(x, wires, *, rotation: str, hadamard: bool):
    import pennylane as qml

    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


def _enc_amplitude_embedding(x, wires):
    import pennylane as qml

    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)


def _enc_angle_pattern_xyz(x, wires, *, hadamard: bool):
    import pennylane as qml

    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        if i % 3 == 0:
            qml.RX(x[i], wires=w)
        elif i % 3 == 1:
            qml.RY(x[i], wires=w)
        else:
            qml.RZ(x[i], wires=w)


def _enc_angle_pair_xy(x, wires, *, hadamard: bool):
    import pennylane as qml

    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RX(x[i], wires=w)
        qml.RY(x[i], wires=w)


def _ansatz_ring_rot_cnot(W, wires: List[int]):
    import pennylane as qml

    num_qubits = len(wires)
    layers = [W] if getattr(W, "ndim", 0) == 2 else W
    for layer in layers:
        for i in range(num_qubits):
            qml.Rot(layer[i, 0], layer[i, 1], layer[i, 2], wires=wires[i])
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[-1], wires[0]])


def _ansatz_strongly_entangling(W, wires: List[int]):
    import pennylane as qml

    qml.StronglyEntanglingLayers(W, wires=wires)


def _make_encoder(enc_name: str):
    enc_name = str(enc_name)
    if enc_name == "angle_embedding_x":
        return lambda x, wires, hadamard: _enc_angle_embedding(x, wires, rotation="X", hadamard=hadamard)
    if enc_name == "angle_embedding_y":
        return lambda x, wires, hadamard: _enc_angle_embedding(x, wires, rotation="Y", hadamard=hadamard)
    if enc_name == "angle_embedding_z":
        return lambda x, wires, hadamard: _enc_angle_embedding(x, wires, rotation="Z", hadamard=hadamard)
    if enc_name == "amplitude_embedding":
        return lambda x, wires, hadamard: _enc_amplitude_embedding(x, wires)
    if enc_name == "angle_pattern_xyz":
        return lambda x, wires, hadamard: _enc_angle_pattern_xyz(x, wires, hadamard=hadamard)
    if enc_name == "angle_pair_xy":
        return lambda x, wires, hadamard: _enc_angle_pair_xy(x, wires, hadamard=hadamard)
    raise ValueError(f"Unknown encoder in saved model: {enc_name}")


def _make_ansatz(anz_name: str):
    anz_name = str(anz_name)
    if anz_name == "ring_rot_cnot":
        return _ansatz_ring_rot_cnot
    if anz_name == "strongly_entangling":
        return _ansatz_strongly_entangling
    raise ValueError(f"Unknown ansatz in saved model: {anz_name}")


@dataclass
class LoadedQuantumClassifier:
    version: int
    features: List[str]
    label: str
    coerce_state: Dict[str, Dict[str, Any]]
    scaler: Any
    quantile: Any
    pls: Any
    pca: Any
    compiled_input_scale: float
    compiled_input_shift: float
    weights: Any
    w_ro: Any
    bias: float
    alpha: float
    score_sign: float
    threshold: float
    device_name: str
    num_qubits: int
    encoder_name: str
    encoder_opts: Dict[str, Any]
    ansatz_name: str
    layers: int
    measurement: Dict[str, Any]

    _qnode: Any = None

    def _coerce_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            st = self.coerce_state.get(c)
            if not isinstance(st, dict):
                continue
            mode = st.get("mode")
            if mode == "categorical":
                mapping = st.get("mapping", {})
                unk = float(st.get("unknown", -1.0))
                out[c] = out[c].astype(str).map(mapping).fillna(unk).astype("float64")
            else:
                med = st.get("median", None)
                s_num = pd.to_numeric(out[c], errors="coerce")
                if med is not None:
                    out[c] = s_num.fillna(float(med))
                else:
                    out[c] = s_num
        return out

    def _to_numpy(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Be forgiving: callers often include label columns in the payload.
            drop_lower = {"attack_label", "attack_type", "label"}
            if self.label:
                drop_lower.add(str(self.label).strip().lower())
            cols_to_drop = [c for c in X.columns if str(c).strip().lower() in drop_lower]
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop, errors="ignore")

            if self.features:
                missing = [c for c in self.features if c not in X.columns]
                if missing:
                    raise ValueError("Missing required feature columns: " + ", ".join(missing))
                X = X[self.features]
            Xdf = self._coerce_dataframe(X) if self.coerce_state else X.copy()
            Xn = Xdf.values
        elif isinstance(X, pd.Series):
            Xn = X.values.reshape(1, -1)
        else:
            Xn = np.asarray(X)

        Xn = np.asarray(Xn, dtype=np.float64)
        if self.quantile is not None:
            Xn = self.quantile.transform(Xn)
        if self.pls is not None:
            Xn = self.pls.transform(Xn)
        if self.pca is not None:
            Xn = self.pca.transform(Xn)
        if self.scaler is not None:
            Xn = self.scaler.transform(Xn)
        s = float(self.compiled_input_scale)
        b = float(self.compiled_input_shift)
        if s != 1.0 or b != 0.0:
            Xn = Xn * s + b
        return Xn

    def _build_qnode(self):
        import pennylane as qml

        dev_name = self.device_name
        dev_kwargs: Dict[str, Any] = {}
        if dev_name == "lightning.gpu":
            dev_kwargs["c_dtype"] = np.complex64
        dev = qml.device(dev_name, wires=self.num_qubits, **dev_kwargs)

        wires = list(range(self.num_qubits))
        encoder_fn = _make_encoder(self.encoder_name)
        ansatz_fn = _make_ansatz(self.ansatz_name)

        enc_opts = dict(self.encoder_opts or {})
        hadamard = bool(enc_opts.get("hadamard", False))
        reupload = bool(enc_opts.get("reupload", False))

        if not isinstance(self.measurement, dict) or not self.measurement.get("name") or "wires" not in self.measurement:
            raise ValueError("Loaded model is missing required measurement config.")
        meas_cfg = dict(self.measurement)
        meas_name = str(meas_cfg["name"])
        meas_wires = list(meas_cfg["wires"] or [])
        if not meas_wires:
            raise ValueError("Loaded model has empty measurement.wires.")

        def _circuit(weights, w_ro, x):
            if reupload and self.ansatz_name == "ring_rot_cnot":
                def _reupload_layer(W):
                    encoder_fn(x, wires, hadamard)
                    ansatz_fn(W, wires)
                qml.layer(_reupload_layer, self.layers, weights)
            else:
                encoder_fn(x, wires, hadamard)
                ansatz_fn(weights, wires)

            if meas_name == "mean_z":
                if not meas_wires:
                    raise ValueError("mean_z measurement requires at least one wire")
                coeffs = [1.0 / len(meas_wires)] * len(meas_wires)
                ops = [qml.PauliZ(w) for w in meas_wires]
                return qml.expval(qml.Hamiltonian(coeffs, ops))
            if meas_name == "mean_z_readout":
                # Trainable readout as a gate layer (avoids relying on coefficient gradients).
                if len(w_ro) != 3 * len(meas_wires):
                    raise ValueError(
                        f"mean_z_readout expects w_ro dim=3*len(wires)={3*len(meas_wires)}; got {len(w_ro)}"
                    )
                for i, w in enumerate(meas_wires):
                    j = 3 * i
                    qml.Rot(w_ro[j + 0], w_ro[j + 1], w_ro[j + 2], wires=int(w))
                coeffs = [1.0 / len(meas_wires)] * len(meas_wires)
                ops = [qml.PauliZ(w) for w in meas_wires]
                return qml.expval(qml.Hamiltonian(coeffs, ops))
            if meas_name == "z_vec":
                ops = [qml.PauliZ(int(w)) for w in meas_wires]
                return qml.expval(qml.Hamiltonian(w_ro, ops))
            if meas_name == "z0":
                return qml.expval(qml.PauliZ(0))
            raise ValueError(f"Unknown measurement in saved model: {meas_name}")

        self._qnode = qml.QNode(_circuit, dev, interface="autograd", cache=True)

    def decision_function(self, X: Any) -> np.ndarray:
        import pennylane as qml
        from pennylane import numpy as pnp

        Xn = self._to_numpy(X)
        if self._qnode is None:
            self._build_qnode()

        # Autograd interface: keep weights/bias as pennylane.numpy arrays.
        w = pnp.array(self.weights, requires_grad=False)
        b = float(self.bias)
        a = float(self.alpha)
        s = float(self.score_sign)

        w_ro = pnp.array(np.asarray(self.w_ro, dtype=np.float64).reshape((-1,)), requires_grad=False)

        # Canonical scoring:
        # - v4: logit = score_sign * (expval + bias) with alpha == 1
        # - v5: logit = alpha * (expval + bias) (score_sign ignored if present)
        if int(self.version) == 4:
            if not np.isfinite(a) or abs(a - 1.0) > 1e-6:
                raise ValueError(f"Unsupported alpha={a} for version=4 model. Re-save the model (version=5).")

        if Xn.ndim == 1:
            ev = float(np.asarray(self._qnode(w, w_ro, Xn), dtype=np.float64).reshape(()))
            raw = float(ev + b)
            if int(self.version) == 4:
                return np.asarray([s * raw], dtype=np.float64)
            return np.asarray([a * raw], dtype=np.float64)

        # Batch: Python loop is slow but robust; deployment should batch moderately.
        out = np.empty((Xn.shape[0],), dtype=np.float64)
        for i in range(Xn.shape[0]):
            ev = float(np.asarray(self._qnode(w, w_ro, Xn[i]), dtype=np.float64).reshape(()))
            out[i] = float(ev + b)
        if int(self.version) == 4:
            return s * out
        return float(a) * out

    def predict(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= float(self.threshold), 1.0, -1.0)

    def predict01(self, X: Any) -> np.ndarray:
        """Convenience for deployments that expect 0/1 (0=normal, 1=attack)."""
        pred_signed = self.predict(X)
        return (pred_signed > 0).astype(np.int32)


def load_model_pt(path: str, device_override: Optional[str] = None) -> LoadedQuantumClassifier:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Invalid model file: expected a dict-like state")
    ver = int(state.get("version") or 0)
    if ver not in (4, 5):
        raise ValueError(
            f"Unsupported model version {ver}. Re-save the model with the current training pipeline (version=5)."
        )

    # Hard requirement: sklearn>=1.8 PLS transform needs private mean/std (if PLS is present).
    pls_state = state.get("pls_state")
    if isinstance(pls_state, dict) and (
        ("x_rotations_" in pls_state) and (pls_state.get("_x_mean") is None or pls_state.get("_x_std") is None)
    ):
        raise ValueError(
            "This model file is missing required PLS state (_x_mean/_x_std). "
            "Train and save again with the updated pipeline."
        )

    requires_preprocess = bool(state.get("requires_preprocess", True))
    scaler = quantile = pls = pca = None
    if requires_preprocess:
        scaler, quantile, pls, pca = _rebuild_transformers_from_state(
            state.get("scaler_state"),
            state.get("quantile_state"),
            state.get("pls_state"),
            state.get("pca_state"),
        )

    for k in ["weights", "w_ro", "bias", "alpha", "compiled_input_scale", "compiled_input_shift"]:
        if k not in state:
            raise ValueError(f"Saved model is missing required key '{k}'.")
    if ver == 4 and "score_sign" not in state:
        raise ValueError("Saved model is missing required key 'score_sign' for version=4.")

    weights = state["weights"].detach().cpu().numpy()
    w_ro = state["w_ro"].detach().cpu().numpy().astype(np.float64, copy=False)
    bias = float(state["bias"].detach().cpu().numpy().reshape(()))
    alpha = float(state["alpha"].detach().cpu().numpy().reshape(()))
    score_sign = float(state.get("score_sign", torch.tensor(1.0)).detach().cpu().numpy().reshape(()))

    if "features" not in state or not isinstance(state["features"], (list, tuple)) or not state["features"]:
        raise ValueError("Saved model is missing non-empty 'features' list.")
    features = list(state["features"])
    if "label" not in state:
        raise ValueError("Saved model is missing required 'label'.")
    label = str(state["label"])
    if "coerce_state" not in state or not isinstance(state["coerce_state"], dict):
        raise ValueError("Saved model is missing required 'coerce_state' dict.")
    coerce_state = state["coerce_state"]
    if "threshold" not in state:
        raise ValueError("Saved model is missing required 'threshold'.")
    threshold = float(state["threshold"])
    compiled_input_scale = float(state["compiled_input_scale"])
    compiled_input_shift = float(state["compiled_input_shift"])

    dev_name = device_override or state.get("device", None)
    if not dev_name:
        raise ValueError("Saved model is missing required 'device' and no device_override was provided.")
    if "num_qubits" not in state:
        raise ValueError("Saved model is missing required 'num_qubits'.")
    num_qubits = int(state["num_qubits"])
    meas = dict(state.get("measurement") or {})
    if not meas.get("name") or "wires" not in meas:
        raise ValueError("Saved model is missing required 'measurement' config.")
    mw = list(meas.get("wires") or [])
    if not mw:
        raise ValueError("Saved model has empty measurement.wires.")

    return LoadedQuantumClassifier(
        version=int(ver),
        features=features,
        label=label,
        coerce_state=coerce_state,
        scaler=None if scaler is None else scaler,
        quantile=None if quantile is None else quantile,
        pls=None if pls is None else pls,
        pca=None if pca is None else pca,
        compiled_input_scale=compiled_input_scale,
        compiled_input_shift=compiled_input_shift,
        weights=weights,
        w_ro=w_ro,
        bias=bias,
        alpha=alpha,
        score_sign=score_sign,
        threshold=threshold,
        device_name=dev_name,
        num_qubits=num_qubits,
        encoder_name=str(state.get("encoder")),
        encoder_opts=dict(state.get("encoder_opts") or {}),
        ansatz_name=str(state.get("ansatz")),
        layers=int(state.get("layers") or 1),
        measurement=meas,
    )


def main() -> None:
    args = _parse_args()
    clf = load_model_pt(args.model, device_override=args.device)

    if args.no_preprocess:
        clf.scaler = None
        clf.quantile = None
        clf.pls = None
        clf.pca = None

    if args.csv:
        df = pd.read_csv(args.csv)
        df_in = df
        if clf.features:
            missing = [c for c in clf.features if c not in df.columns]
            if missing:
                raise SystemExit("Missing required feature columns from CSV: " + ", ".join(missing))
            df_in = df[clf.features]
        if args.limit and args.limit > 0:
            df_in = df_in.head(args.limit)
        preds = clf.predict(df_in).astype(int)
        print("Predictions (first 20):", preds[:20].tolist())
        scores = None
        if args.decision:
            scores = clf.decision_function(df_in).astype(float)
            print("Scores (first 20):", scores[:20].tolist())

        if args.out:
            out_df = df.head(len(df_in)).copy()
            out_df["prediction"] = preds
            if scores is not None:
                out_df["score"] = scores
            out_df.to_csv(args.out, index=False)
            print(f"Saved predictions to {args.out}")
        return

    X = _parse_points(args.point or [])
    exp_dim = len(clf.features) if clf.features else X.shape[1]
    if X.shape[1] != exp_dim:
        raise SystemExit(f"Point dimension {X.shape[1]} does not match expected {exp_dim}")
    preds = clf.predict(X).astype(int)
    for i, y in enumerate(preds):
        print(f"row={i}\tpred={int(y)}")
    if args.decision:
        scores = clf.decision_function(X).astype(float)
        for i, s in enumerate(scores):
            print(f"row={i}\tscore={s}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
