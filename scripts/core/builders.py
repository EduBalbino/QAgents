from __future__ import annotations

# Central functional DSL for building and running QML experiments
# Keeps per-experiment shims tiny while concentrating shared logic here.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Dedup guards for log lines within a single process run
_PRINTED_SAVED_PATHS: set = set()


# -----------------------------
# DSL primitives
# -----------------------------


@dataclass
class Step:
    kind: str
    params: Dict[str, Any]


class Recipe:
    def __init__(self, parts: Optional[List[Step]] = None) -> None:
        self.parts: List[Step] = parts or []

    def __or__(self, other: Step) -> "Recipe":
        return Recipe(self.parts + [other])


def csv(path: str, sample_size: Optional[int] = None) -> Step:
    return Step("dataset.csv", {"path": path, "sample_size": sample_size})


def select(features: List[str], label: str) -> Step:
    return Step("dataset.select", {"features": features, "label": label})


def device(name: str = "lightning.qubit", wires_from_features: bool = True) -> Step:
    return Step("device", {"name": name, "wires_from_features": wires_from_features})


def encoder(name: str, **kwargs: Any) -> Step:
    return Step("vqc.encoder", {"name": name, **kwargs})


def ansatz(name: str, layers: int, **kwargs: Any) -> Step:
    return Step("vqc.ansatz", {"name": name, "layers": layers, **kwargs})


def train(
    lr: float = 0.1,
    batch: int = 100,
    epochs: int = 1,
    class_weights: Optional[str] = "balanced",
    seed: int = 42,
    test_size: float = 0.2,
    stratify: bool = True,
) -> Step:
    return Step(
        "train",
        {
            "lr": lr,
            "batch": batch,
            "epochs": epochs,
            "class_weights": class_weights,
            "seed": seed,
            "test_size": test_size,
            "stratify": stratify,
        },
    )


def rf_baseline(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> Step:
    return Step(
        "baseline.rf",
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "random_state": random_state,
        },
    )

def pca_to_pow2(max_qubits: Optional[int] = None) -> Step:
    return Step("dataset.pca_pow2", {"max_qubits": max_qubits})

def quantile_uniform(n_quantiles: int = 1000, output_distribution: str = "uniform") -> Step:
    return Step("dataset.quantile_uniform", {"n_quantiles": n_quantiles, "output_distribution": output_distribution})

def pls_to_pow2(max_qubits: Optional[int] = None, components: Optional[int] = None) -> Step:
    return Step("dataset.pls_pow2", {"max_qubits": max_qubits, "components": components})

# Model persistence
def save(path: str) -> Step:
    return Step("model.save", {"path": path})


# -----------------------------
# Topology registries
# -----------------------------


EncoderFn = Callable[[Any, Any], None]  # Accept extra kwargs at call site
AnsatzFn = Callable[[Any, Any], None]


ENCODERS: Dict[str, EncoderFn] = {}
ANSAETZE: Dict[str, AnsatzFn] = {}


def register_encoder(name: str) -> Callable[[EncoderFn], EncoderFn]:
    def inner(fn: EncoderFn) -> EncoderFn:
        ENCODERS[name] = fn
        return fn

    return inner


def register_ansatz(name: str) -> Callable[[AnsatzFn], AnsatzFn]:
    def inner(fn: AnsatzFn) -> AnsatzFn:
        ANSAETZE[name] = fn
        return fn

    return inner


# Built-in encoders/ansätze


@register_encoder("angle_embedding_y")
def _enc_angle_y(x: Any, wires: Any, hadamard: bool = False, angle_scale: Optional[float] = None, **_: Any) -> None:
    import pennylane as qml  # local import to avoid import cost until run
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    x_scaled = x * angle_scale if angle_scale is not None else x
    qml.AngleEmbedding(x_scaled, wires=wires, rotation="Y")


@register_encoder("angle_embedding_x")
def _enc_angle_x(x: Any, wires: Any, hadamard: bool = False, angle_scale: Optional[float] = None, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    x_scaled = x * angle_scale if angle_scale is not None else x
    qml.AngleEmbedding(x_scaled, wires=wires, rotation="X")


@register_encoder("angle_embedding_z")
def _enc_angle_z(x: Any, wires: Any, hadamard: bool = False, angle_scale: Optional[float] = None, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    x_scaled = x * angle_scale if angle_scale is not None else x
    qml.AngleEmbedding(x_scaled, wires=wires, rotation="Z")


@register_encoder("amplitude_embedding")
def _enc_amplitude(x: Any, wires: Any, **_: Any) -> None:
    import pennylane as qml

    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)


@register_encoder("angle_pattern_xyz")
def _enc_angle_pattern_xyz(x: Any, wires: Any, hadamard: bool = False, angle_scale: Optional[float] = None, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    # Cycle X, Y, Z by wire index for diverse Bloch trajectories
    x_scaled = x * angle_scale if angle_scale is not None else x
    for i, w in enumerate(wires):
        if i % 3 == 0:
            qml.RX(x_scaled[i], wires=w)
        elif i % 3 == 1:
            qml.RY(x_scaled[i], wires=w)
        else:
            qml.RZ(x_scaled[i], wires=w)


@register_encoder("angle_pair_xy")
def _enc_angle_pair_xy(x: Any, wires: Any, hadamard: bool = False, angle_scale: Optional[float] = None, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    x_scaled = x * angle_scale if angle_scale is not None else x
    # Apply RX then RY per wire to enrich expressivity with minimal overhead
    for i, w in enumerate(wires):
        qml.RX(x_scaled[i], wires=w)
        qml.RY(x_scaled[i], wires=w)


@register_ansatz("ring_rot_cnot")
def _ansatz_ring_rot_cnot(W: Any, wires: List[int]) -> None:
    import pennylane as qml

    num_qubits = len(wires)
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=wires[i])
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])


@register_ansatz("strongly_entangling")
def _ansatz_sel(W: Any, wires: List[int]) -> None:
    import pennylane as qml
    # StronglyEntanglingLayers expects shape (layers, n_wires, 3).
    # Our circuit calls ansatz per-layer with W shaped (n_wires, 3), so expand dims.
    try:
        W3 = qml.numpy.expand_dims(W, axis=0)
    except Exception:
        # Fallback: if already correct shape, use as-is
        W3 = W
    qml.templates.StronglyEntanglingLayers(W3, wires=wires)


# -----------------------------
# Runner
# -----------------------------


def _setup_logger(log_filename: str, tee_to_terminal: bool = True):
    import sys

    class Logger(object):
        def __init__(self, filename: str) -> None:
            self.terminal = sys.stdout
            self.log = open(filename, "w")
            self.tee = tee_to_terminal

        def write(self, message: str) -> None:
            if self.tee:
                self.terminal.write(message)
            self.log.write(message)

        def flush(self) -> None:
            if self.tee:
                self.terminal.flush()
            self.log.flush()

        def isatty(self) -> bool:
            # Report non-interactive to libraries that check TTY (e.g., W&B)
            try:
                return bool(getattr(self.terminal, "isatty", lambda: False)())
            except Exception:
                return False

    logger = Logger(log_filename)
    sys.stdout = logger
    sys.stderr = logger


def run(recipe: Recipe) -> Dict[str, Any]:
    import os
    import datetime
    import time

    # Lazy imports for heavy deps
    import pandas as pd
    import pennylane as qml
    from pennylane import numpy as np
    import numpy as _np
    from pennylane.optimize import AdamOptimizer
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    # Collect config from steps
    cfg: Dict[str, Any] = {}
    for step in recipe.parts:
        cfg[step.kind] = {**cfg.get(step.kind, {}), **step.params}

    # Configure logging
    os.makedirs("logs", exist_ok=True)
    # High-resolution timestamp plus pid to avoid filename collisions under parallel execution
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    ds_name = os.path.basename(cfg.get("dataset.csv", {}).get("path", "dataset")).replace(".csv", "")
    log_path = os.path.join("logs", f"DSL_{ds_name}_{ts}.log")
    # In child processes, avoid printing to terminal to prevent interleaved output
    try:
        import multiprocessing as _mp
        is_main = (_mp.current_process().name == "MainProcess")
    except Exception:
        is_main = True
    tee_flag = os.environ.get("QAGENTS_TEE", "1") == "1" and is_main
    _setup_logger(log_path, tee_to_terminal=tee_flag)

    print("--- Starting experiment (DSL) ---")

    # Load dataset (with sampling) or synthesize if missing
    data_cfg = cfg.get("dataset.csv", {})
    path = data_cfg.get("path")
    sample_size = data_cfg.get("sample_size")

    sel_cfg = cfg.get("dataset.select", {})
    features: List[str] = sel_cfg.get("features", [])
    label_col: str = sel_cfg.get("label")

    def _synthesize(n_rows: int) -> "pd.DataFrame":
        rng = np.random.default_rng(42)
        if not features:
            # default to 8 PCs if not given yet
            feats = [f"PC_{i}" for i in range(1, 9)]
        else:
            feats = features
        data = {col: rng.normal(size=n_rows) for col in feats}
        y = (rng.random(n_rows) > 0.5).astype(int)
        data[label_col or "Label"] = y
        return pd.DataFrame(data)

    import random as _random

    if path and os.path.exists(path):
        if sample_size is None:
            df = pd.read_csv(path, low_memory=False)
        else:
            # two-pass memory-efficient sampling
            with open(path, "r") as f:
                num_lines = sum(1 for _ in f) - 1
            k = min(int(sample_size), max(1, num_lines))
            to_skip = sorted(_random.sample(range(1, num_lines + 1), num_lines - k))
            df = pd.read_csv(path, skiprows=to_skip, low_memory=False)
        print(f"Dataset loaded from {path}. Shape: {df.shape}")
    else:
        # Fallback synthetic data for quick smoke test
        n_rows = int(sample_size or 1000)
        df = _synthesize(n_rows)
        print(
            f"Dataset file not found: {path}. Using synthetic data with shape {df.shape} for smoke test."
        )

    # Feature/label selection
    if not features or not label_col:
        raise ValueError("Both features and label must be specified via select(...)")
    X = df[features]
    y = df[label_col]

    # Define a train-fitted coercion that applies consistently to test to avoid leakage
    def _coerce_fit_apply(Xtr: "pd.DataFrame", Xte: "pd.DataFrame") -> Tuple["np.ndarray", "np.ndarray"]:
        Xtr2 = Xtr.copy()
        Xte2 = Xte.copy()
        for c in Xtr2.columns:
            trc = Xtr2[c]
            if pd.api.types.is_numeric_dtype(trc):
                # Already numeric
                continue
            tr_num = pd.to_numeric(trc, errors="coerce")
            te_num = pd.to_numeric(Xte2[c], errors="coerce")
            if tr_num.notna().any():
                med = float(tr_num.median()) if tr_num.notna().any() else 0.0
                Xtr2[c] = tr_num.fillna(med)
                Xte2[c] = te_num.fillna(med)
            else:
                cats = pd.Index(trc.astype(str).unique())
                mapping = {k: float(i) for i, k in enumerate(cats)}
                Xtr2[c] = trc.astype(str).map(mapping).astype("float64")
                Xte2[c] = Xte2[c].astype(str).map(mapping).fillna(-1).astype("float64")
        return Xtr2.values, Xte2.values

    # Make sure label is binary numeric {0,1}
    if not pd.api.types.is_numeric_dtype(y):
        codes, uniques = pd.factorize(y.astype(str))
        if len(uniques) == 2:
            y = pd.Series(codes, index=y.index)
        else:
            # Collapse to binary by treating first observed value as 0, others as 1
            y0 = codes[0] if len(codes) > 0 else 0
            y = pd.Series((codes != y0).astype(int), index=y.index)
    else:
        # Map any non-zero to 1
        y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
    print("Features and labels extracted.")

    # Split first (to avoid leakage), then coerce using train-fit, then optional PCA (fit on train)
    tr_cfg = cfg.get("train", {})
    test_size = float(tr_cfg.get("test_size", 0.2))
    stratify = bool(tr_cfg.get("stratify", True))
    stratify_y = y if stratify else None
    split_seed = int(cfg.get("train", {}).get("seed", 42))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=stratify_y
    )
    print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # Coercion fit on train, apply to test
    X_train, X_test = _coerce_fit_apply(X_train, X_test)

    # Track preprocessing pipeline components for persistence
    qt = None
    pls = None
    pca = None

    # Optional quantile uniformization (fit on train only)
    q_cfg = cfg.get("dataset.quantile_uniform", None)
    if q_cfg is not None:
        from sklearn.preprocessing import QuantileTransformer
        n_q = int(q_cfg.get("n_quantiles", min(1000, len(X_train))))
        out_dist = q_cfg.get("output_distribution", "uniform")
        qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution=out_dist,
            subsample=int(1e9),
            random_state=42,
        )
        X_train = qt.fit_transform(X_train)
        X_test = qt.transform(X_test)

    # Optional supervised dimensionality reduction to nearest power of two using PLS
    pls_cfg = cfg.get("dataset.pls_pow2", None)
    if pls_cfg is not None:
        from sklearn.cross_decomposition import PLSRegression
        import math as _math
        d0 = X_train.shape[1]
        max_power = d0.bit_length() - 1
        max_qubits = pls_cfg.get("max_qubits")
        if max_qubits is not None:
            max_power = min(max_power, int(max_qubits))
        target_dim = int(pls_cfg.get("components") or max(1, 2 ** max_power))
        target_dim = max(1, min(target_dim, d0))
        pls = PLSRegression(n_components=target_dim)
        # Use {0,1} as response for classification
        Y01_pls = (np.array(y_train.values) > 0).astype(int)
        pls.fit(X_train, Y01_pls)
        X_train = pls.transform(X_train)
        X_test = pls.transform(X_test)

    # Optional PCA to a power-of-two feature count (useful for amplitude embedding), fit on train only
    pca_cfg = cfg.get("dataset.pca_pow2", None)
    if pca_cfg is not None:
        max_qubits = pca_cfg.get("max_qubits")
        import math as _math
        d0 = X_train.shape[1]
        max_power = d0.bit_length() - 1
        if max_qubits is not None:
            max_power = min(max_power, int(max_qubits))
        target_dim = max(1, 2 ** max_power)
        if target_dim != d0:
            pca = PCA(n_components=target_dim, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features scaled. X_train_scaled shape={X_train_scaled.shape}, X_test_scaled shape={X_test_scaled.shape}")

    # Labels to {-1, 1} and {0,1}
    Y_train = np.array(y_train.values * 2 - 1, requires_grad=False)
    Y_test = np.array(y_test.values * 2 - 1, requires_grad=False)
    Y_train01 = (Y_train > 0).astype(int)
    Y_test01 = (Y_test > 0).astype(int)

    # Optional export of the PLS-transformed, scaled dataset as seen by QML.
    # This runs after quantile + PLS + MinMax scaling and uses numbered feature
    # columns (PC_1, PC_2, ...) plus a binary label column.
    export_path = os.environ.get("EDGE_EXPORT_PLS_DATASET")
    print(f"[PLS-EXPORT] EDGE_EXPORT_PLS_DATASET={export_path!r}")
    if not export_path:
        print("[PLS-EXPORT] Skipping export: EDGE_EXPORT_PLS_DATASET is not set or empty.")
    else:
        try:
            num_feats = X_train_scaled.shape[1]
            print(f"[PLS-EXPORT] Preparing export with num_feats={num_feats}, rows_train={X_train_scaled.shape[0]}, rows_test={X_test_scaled.shape[0]}")
            feat_cols = [f"PC_{i+1}" for i in range(num_feats)]
            df_train_pls = pd.DataFrame(X_train_scaled, columns=feat_cols)
            df_test_pls = pd.DataFrame(X_test_scaled, columns=feat_cols)
            df_train_pls["Attack_label"] = Y_train01
            df_test_pls["Attack_label"] = Y_test01
            df_train_pls["split"] = "train"
            df_test_pls["split"] = "test"
            df_pls = pd.concat([df_train_pls, df_test_pls], ignore_index=True)
            target_dir = os.path.dirname(export_path) or "."
            print(f"[PLS-EXPORT] Ensuring directory exists: {target_dir}")
            os.makedirs(target_dir, exist_ok=True)
            print(f"[PLS-EXPORT] Writing CSV to {export_path}")
            df_pls.to_csv(export_path, index=False)
            print(f"[PLS-EXPORT] Done. Exported PLS-transformed dataset to {export_path} with {num_feats} features and {len(df_pls)} rows.")
        except Exception as _exc:
            print(f"[PLS-EXPORT] Failed to export PLS-transformed dataset to {export_path}: {_exc}")

    # Optional classical baseline: Random Forest
    if "baseline.rf" in cfg:
        from sklearn.ensemble import RandomForestClassifier
        rf_cfg = cfg.get("baseline.rf", {})
        n_estimators = int(rf_cfg.get("n_estimators", 200))
        max_depth = rf_cfg.get("max_depth", None)
        class_weight = rf_cfg.get("class_weight", "balanced")
        random_state = int(rf_cfg.get("random_state", int(tr_cfg.get("seed", 42))))

        print(f"Training RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth}, class_weight={class_weight})")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth is not None else None,
            class_weight=class_weight if class_weight not in (None, "None") else None,
            n_jobs=-1,
            random_state=random_state,
        )
        import time as _t
        _t0 = _t.time()
        rf.fit(X_train_scaled, Y_train01)
        train_time = _t.time() - _t0

        # Validation threshold via ROC curve maximizing balanced accuracy
        try:
            import numpy as _np
            from sklearn.metrics import roc_curve
            val_size = min(max(1000, int(0.1 * len(X_train_scaled))), len(X_train_scaled))
            val_idx = np.random.randint(0, len(X_train_scaled), val_size)
            X_val = X_train_scaled[val_idx]
            Y_val01 = Y_train01[val_idx]
            # Ensure both classes present; if not, fallback to a larger slice or default threshold
            if len(_np.unique(Y_val01)) < 2:
                val_idx = _np.arange(0, min(len(X_train_scaled), 5000))
                X_val = X_train_scaled[val_idx]
                Y_val01 = Y_train01[val_idx]
            prob_val = rf.predict_proba(X_val)[:, 1]
            fpr, tpr, thr = roc_curve(Y_val01, prob_val)
            # balanced accuracy = (tpr + (1 - fpr)) / 2
            bacc_arr = (tpr + (1.0 - fpr)) / 2.0
            best_i = int(_np.nanargmax(bacc_arr)) if len(bacc_arr) else 0
            best_t = 0.5
            best_bacc = -1.0
            if len(thr):
                best_t = float(thr[best_i])
                best_bacc = float(bacc_arr[best_i])
        except Exception:
            best_t = 0.5
            best_bacc = float('nan')

        prob_test = rf.predict_proba(X_test_scaled)[:, 1]
        predictions_signed = np.where(prob_test >= best_t, 1, -1)
        acc = float(accuracy_score(Y_test, predictions_signed))
        prec = float(precision_score(Y_test, predictions_signed, labels=[-1, 1], average='macro', zero_division=0))
        rec = float(recall_score(Y_test, predictions_signed, labels=[-1, 1], average='macro', zero_division=0))
        f1 = float(f1_score(Y_test, predictions_signed, labels=[-1, 1], average='macro', zero_division=0))
        bacc = float(balanced_accuracy_score(Y_test, predictions_signed))
        try:
            auc = float(roc_auc_score(Y_test01, prob_test))
        except Exception:
            auc = float('nan')
        print("--- RF Test Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:   {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Balanced Acc: {bacc:.4f}")
        print(f"Threshold: {best_t:.6f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"Train Time (s): {train_time:.2f}")
        print("-----------------------")
        # Return summary compatible with aggregator
        return {
            "log_path": log_path,
            "dataset": ds_name,
            "encoder": "none",
            "encoder_opts": {},
            "ansatz": "random_forest",
            "layers": 0,
            "measurement": {"name": "none", "wires": []},
            "train": {"n_estimators": n_estimators, "max_depth": max_depth, "class_weight": class_weight},
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bacc,
                "auc": auc,
                "val_balanced_accuracy": best_bacc,
                "threshold": best_t,
            },
            "train_time_s": train_time,
            "class_distribution": {
                "train_pos": int(Y_train01.sum()),
                "train_neg": int((1 - Y_train01).sum()),
                "test_pos": int(Y_test01.sum()),
                "test_neg": int((1 - Y_test01).sum()),
            },
        }

    # Determine encoder early to size device correctly (amplitude embedding uses log2 dimension)
    enc_cfg = cfg.get("vqc.encoder", {"name": "angle_embedding_y"})
    enc_name_early = enc_cfg.get("name")

    # Device and wires
    dev_cfg = cfg.get("device", {})
    feature_dim = X_train_scaled.shape[1]
    if enc_name_early == "amplitude_embedding":
        import math as _math
        if feature_dim <= 0:
            raise ValueError("Feature dimension must be positive for amplitude embedding")
        q = int(_math.log2(feature_dim))
        if 2 ** q != feature_dim:
            raise ValueError(f"Amplitude embedding requires feature dimension to be a power of two. Got {feature_dim}.")
        num_qubits = q
    else:
        num_qubits = feature_dim
    wires = list(range(num_qubits))
    dev_name = dev_cfg.get("name", "lightning.qubit")
    # Allow environment override for device selection (e.g., QML_DEVICE=lightning.gpu)
    env_device = os.environ.get("QML_DEVICE")
    if env_device:
        dev_name = env_device
    dev = qml.device(dev_name, wires=num_qubits)
    print(f"Quantum device '{dev_name}' initialized with {num_qubits} wires.")

    # Encoder / Ansatz
    enc_cfg = cfg.get("vqc.encoder", {"name": "angle_embedding_y"})
    anz_cfg = cfg.get("vqc.ansatz", {"name": "ring_rot_cnot", "layers": 3})
    enc_name = enc_cfg.get("name")
    anz_name = anz_cfg.get("name")
    num_layers = int(anz_cfg.get("layers", 3))

    if enc_name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {enc_name}")
    if anz_name not in ANSAETZE:
        raise ValueError(f"Unknown ansatz: {anz_name}")

    encoder_fn = ENCODERS[enc_name]
    ansatz_fn = ANSAETZE[anz_name]

    # Measurement configuration
    meas_cfg = cfg.get("measurement", {"name": "z0", "wires": [0]})
    meas_name = meas_cfg.get("name", "z0")
    meas_wires = list(meas_cfg.get("wires", [0]))
    # Ensure measurement wires are within device range; default to all wires if out-of-range
    if any((int(w) >= num_qubits or int(w) < 0) for w in meas_wires):
        meas_wires = list(range(num_qubits))

    # Angle scaling for angle encoders
    angle_scale = None
    if enc_name.startswith("angle_embedding"):
        if enc_cfg.get("angle_range") == "0_pi":
            angle_scale = qml.numpy.pi
        elif enc_cfg.get("angle_scale") is not None:
            angle_scale = float(enc_cfg.get("angle_scale"))

    # Log concise configuration line
    print(
        " | ".join(
            [
                f"Config enc={enc_name}",
                f"hadamard={bool(enc_cfg.get('hadamard', False))}",
                f"reupload={bool(enc_cfg.get('reupload', False))}",
                f"angle={'0..pi' if enc_cfg.get('angle_range')=='0_pi' else enc_cfg.get('angle_scale', '-')}",
                f"ansatz={anz_name}",
                f"layers={num_layers}",
                f"meas={meas_name}:{','.join(map(str, meas_wires))}",
                f"seed={int(tr_cfg.get('seed', 42))}",
            ]
        )
    )

    # QNode (built as a base QNode, then wrapped with batch_input)
    def _circuit(weights, x):
        # Optionally apply encoder before each layer (re-upload)
        reupload = bool(enc_cfg.get("reupload", False))
        if reupload:
            def _reupload_layer(W):
                encoder_fn(x, wires, hadamard=bool(enc_cfg.get("hadamard", False)), angle_scale=angle_scale)
                ansatz_fn(W, wires)
            qml.layer(_reupload_layer, num_layers, weights)
        else:
            encoder_fn(x, wires, hadamard=bool(enc_cfg.get("hadamard", False)), angle_scale=angle_scale)
            qml.layer(ansatz_fn, num_layers, weights, wires=wires)

        # Measurement
        if meas_name == "mean_z":
            if not meas_wires:
                raise ValueError("mean_z measurement requires at least one wire")
            coeffs = [1.0 / len(meas_wires)] * len(meas_wires)
            ops = [qml.PauliZ(w) for w in meas_wires]
            return qml.expval(qml.Hamiltonian(coeffs, ops))
        else:  # default z0
            return qml.expval(qml.PauliZ(0))

    def square_loss(labels, predictions, class_weights=None):
        loss = (labels - predictions) ** 2
        if class_weights is not None:
            loss = class_weights * loss
        return np.mean(loss)

    def accuracy(labels, predictions):
        return np.sum(np.sign(predictions) == labels) / len(labels)

    # Optional Weights & Biases streaming of training and evaluation metrics
    _wandb = None
    _wandb_can_log = False
    # Controlled via env; set EDGE_WANDB_LIVE=0 to disable streaming even if a run is active
    _wandb_live = os.environ.get("EDGE_WANDB_LIVE", "1") != "0"
    if _wandb_live:
        try:
            import wandb as _wandb_mod  # type: ignore[import-not-found]

            # Only enable if there's an active run; builders.run should not call wandb.init()
            if getattr(_wandb_mod, "run", None) is not None:
                _wandb = _wandb_mod
                _wandb_can_log = True
        except Exception:
            _wandb = None
            _wandb_can_log = False

    def _log_train_metrics_to_wandb(
        *,
        epoch: int,
        iter_in_epoch: int,
        total_iters: int,
        batch_size: int,
        loss: float,
        acc: float,
        start_time: float,
        eval_metrics: Dict[str, float] | None = None,
    ) -> None:
        """Best-effort streaming of current training and evaluation metrics to W&B.

        Uses wall-clock seconds since training start (`time_s`) so runs with different
        hyperparameters can be compared on a shared time-based axis.
        """
        if not _wandb_can_log:
            return
        try:
            t_rel = time.time() - start_time
            payload: Dict[str, float] = {
                "train/loss": float(loss),
                "train/accuracy": float(acc),
                "train/epoch": float(epoch),
                "train/iter_in_epoch": float(iter_in_epoch),
                "train/total_iters": float(total_iters),
                "train/batch_size": float(batch_size),
                # Time-based x-axis reference (seconds since training started)
                "time_s": float(t_rel),
            }
            if eval_metrics is not None:
                # Log evolving evaluation metrics under the same keys used for final metrics
                for k, v in eval_metrics.items():
                    payload[f"metrics/{k}"] = float(v)
            _wandb.log(payload)  # type: ignore[union-attr]
        except Exception:
            # Never let logging issues break training
            pass

    # Training setup
    seed = int(tr_cfg.get("seed", 42))
    np.random.seed(seed)
    _np.random.seed(seed)
    try:
        import random as _py_random
        _py_random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if hasattr(_torch, "cuda") and hasattr(_torch.cuda, "manual_seed_all"):
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    alpha_init = np.array(1.0, requires_grad=True)

    def _build_batched_qnode(sample_x):
        base_qnode = qml.QNode(_circuit, dev, interface="autograd", diff_method="adjoint", cache=True)
        tape = base_qnode.construct([weights_init, sample_x], {})
        all_params = tape.get_parameters(trainable_only=False)
        trainable = set(tape.trainable_params)
        argnum = [i for i in range(len(all_params)) if i not in trainable]
        if not argnum:
            return base_qnode
        return qml.batch_input(base_qnode, argnum=argnum)

    sample_x = _np.asarray(X_train_scaled[0], dtype=_np.float64)
    circuit = _build_batched_qnode(sample_x)

    def variational_classifier(weights, bias, alpha, X_np):
        X_np = _np.asarray(X_np, dtype=_np.float64)
        res = circuit(weights, X_np)
        return alpha * res + bias

    def _predict_logits(weights, bias, alpha, X_input):
        return variational_classifier(weights, bias, alpha, X_input)

    lr = float(tr_cfg.get("lr", 0.1))
    # Always use a fixed mini-batch size of 256 for training (clipped by dataset size)
    epochs = int(tr_cfg.get("epochs", 1))
    batch_size = min(256, len(X_train_scaled))
    opt = AdamOptimizer(lr)

    class_weights_mode = tr_cfg.get("class_weights", "balanced")
    class_weights_map = None
    if class_weights_mode == "balanced":
        cls_labels = np.unique(Y_train)
        cls_weights_array = compute_class_weight(
            class_weight="balanced", classes=cls_labels, y=Y_train
        )
        class_weights_map = {label: weight for label, weight in zip(cls_labels, cls_weights_array)}
        print(f"Class weights: {class_weights_map}")
    # Weights for {0,1} labels (for logistic loss)
    class_weights01_map = None
    if class_weights_mode == "balanced":
        cls01 = np.unique(Y_train01)
        w01 = compute_class_weight(class_weight="balanced", classes=cls01, y=Y_train01)
        class_weights01_map = {int(label): float(weight) for label, weight in zip(cls01, w01)}

    def _bce_with_logits(logits, targets01, sample_weights=None):
        # Stable BCEWithLogits: max(0, x) - x*y + log(1+exp(-|x|))
        x = logits
        y01 = targets01
        term = np.maximum(0.0, x) - x * y01 + np.log1p(np.exp(-np.abs(x)))
        if sample_weights is not None:
            term = sample_weights * term
        return np.mean(term)

    def cost(weights, bias, alpha, X_np, Y_np):
        preds = _predict_logits(weights, bias, alpha, X_np)
        # Use logistic loss for classification
        y01 = (Y_np > 0).astype(int)
        wts = None
        if class_weights01_map:
            wts = np.array([class_weights01_map[int(label)] for label in y01])
        return _bce_with_logits(preds, y01, sample_weights=wts)

    # Training loop (epochs)
    weights = weights_init
    bias = bias_init
    alpha = alpha_init
    start_time = time.time()
    total_iters = 0
    _early_stop = False
    for ep in range(epochs):
        num_it = max(1, len(X_train_scaled) // batch_size)
        print(f"Epoch {ep+1}/{epochs} | iters={num_it} | batch_size={batch_size}", flush=True)
        for it in range(num_it):
            batch_index = _np.random.randint(0, len(X_train_scaled), (batch_size,))
            X_batch = X_train_scaled[batch_index]
            Y_batch = Y_train[batch_index]
            print(f"Epoch {ep+1} Iter {it+1}/{num_it} start", flush=True)
            iter_start = time.time()
            weights, bias, alpha = opt.step(
                lambda w, b, a: cost(w, b, a, X_batch, Y_batch), weights, bias, alpha
            )
            iter_s = time.time() - iter_start
            total_iters += 1
            if (it + 1) % 10 == 0 or (it + 1) == num_it:
                preds_b = _predict_logits(weights, bias, alpha, X_batch)
                c_b = cost(weights, bias, alpha, X_batch, Y_batch)
                a_b = accuracy(Y_batch, preds_b)
                print(
                    f"Epoch {ep+1} Iter {it+1}/{num_it} | "
                    f"Batch Loss: {c_b:0.7f} | Batch Acc: {a_b:0.7f} | "
                    f"Iter Time: {iter_s:.2f}s",
                    flush=True,
                )
                # Optionally evaluate on a held-out subset of the test set to track
                # how metrics evolve over wall-clock time. This is intentionally
                # approximate (subset, fixed threshold) but gives a useful trend.
                eval_metrics = None
                if _wandb_can_log:
                    try:
                        eval_size = min(1000, len(X_test_scaled))
                        eval_idx = _np.random.randint(0, len(X_test_scaled), eval_size)
                        X_eval = X_test_scaled[eval_idx]
                        Y_eval = Y_test[eval_idx]
                        Y_eval01 = Y_test01[eval_idx]
                        logits_eval = _predict_logits(weights, bias, alpha, X_eval)
                        preds_eval = np.where(logits_eval >= 0.0, 1, -1)
                        acc_eval = float(accuracy_score(Y_eval, preds_eval))
                        prec_eval = float(precision_score(Y_eval, preds_eval, average="macro", zero_division=0))
                        rec_eval = float(recall_score(Y_eval, preds_eval, average="macro", zero_division=0))
                        f1_eval = float(f1_score(Y_eval, preds_eval, average="macro", zero_division=0))
                        # Avoid sklearn's "y_pred contains classes not in y_true" warning
                        # by skipping balanced_accuracy when predicted classes extend
                        # beyond those present in the evaluation slice.
                        true_classes_eval = np.unique(Y_eval)
                        pred_classes_eval = np.unique(preds_eval)
                        if (
                            len(true_classes_eval) < 2
                            or len(np.setdiff1d(pred_classes_eval, true_classes_eval)) > 0
                        ):
                            bacc_eval = float("nan")
                        else:
                            bacc_eval = float(balanced_accuracy_score(Y_eval, preds_eval))
                        try:
                            auc_eval = float(roc_auc_score(Y_eval01, logits_eval))
                        except Exception:
                            auc_eval = float("nan")
                        eval_metrics = {
                            "accuracy": acc_eval,
                            "precision": prec_eval,
                            "recall": rec_eval,
                            "f1": f1_eval,
                            "balanced_accuracy": bacc_eval,
                            "auc": auc_eval,
                        }
                    except Exception:
                        eval_metrics = None

                # Stream "current" metrics (train + eval) to W&B on a time-based axis
                _log_train_metrics_to_wandb(
                    epoch=ep + 1,
                    iter_in_epoch=it + 1,
                    total_iters=total_iters,
                    batch_size=batch_size,
                    loss=float(c_b),
                    acc=float(a_b),
                    start_time=start_time,
                    eval_metrics=eval_metrics,
                )
                # Early exit on non-finite loss to avoid noisy logs and wasted compute
                if not np.isfinite(c_b):
                    print("Early exit: non-finite batch loss detected; stopping training early.")
                    _early_stop = True
                    break
            elif (it + 1) % 2 == 0:
                # Heartbeat to show forward/backward is still running
                print(
                    f"Epoch {ep+1} Iter {it+1}/{num_it} | Iter Time: {iter_s:.2f}s",
                    flush=True,
                )
        if _early_stop:
            break

    print(f"Training finished in {time.time() - start_time:.2f}s over {epochs} epoch(s), {total_iters} iters.")

    # Validation quick check (use a small random subset if large)
    val_size = min(5 * batch_size, len(X_train_scaled))
    val_idx = _np.random.randint(0, len(X_train_scaled), val_size)
    X_val = X_train_scaled[val_idx]
    Y_val = Y_train[val_idx]
    preds_val = _predict_logits(weights, bias, alpha, X_val)
    # Choose threshold to maximize balanced accuracy on validation.
    # Guard against degenerate validation slices that contain only a single class,
    # which would otherwise trigger sklearn's
    # "y_pred contains classes not in y_true" warning inside balanced_accuracy_score.
    try:
        import numpy as _np

        unique_val_classes = _np.unique(Y_val)
        if len(unique_val_classes) < 2:
            # Cannot compute a meaningful balanced accuracy if only one class is present.
            # Fall back to a default threshold without scanning candidates.
            best_t = 0.0
            best_bacc = float("nan")
            print(
                "Validation subset contained a single class only; "
                "using default threshold 0.0 without balanced accuracy sweep."
            )
        else:
            th_candidates = _np.unique(
                _np.concatenate([[-_np.inf, _np.inf], preds_val])
            )
            best_t = 0.0
            best_bacc = -1.0
            for t in th_candidates:
                preds_lab = _np.where(preds_val >= t, 1, -1)
                b = float(balanced_accuracy_score(Y_val, preds_lab))
                if b > best_bacc:
                    best_bacc = b
                    best_t = float(t)
            print(
                f"Validation Balanced Acc (best): {best_bacc:0.4f} at threshold {best_t:0.6f}"
            )
    except Exception:
        best_t = 0.0
        best_bacc = float("nan")

    # Test evaluation (batched)
    predictions = np.array(_predict_logits(weights, bias, alpha, X_test_scaled))

    # Threshold from validation to maximize balanced accuracy
    predictions_signed = np.where(predictions >= best_t, 1, -1)
    acc = float(accuracy_score(Y_test, predictions_signed))
    prec = float(precision_score(Y_test, predictions_signed, average='macro', zero_division=0))
    rec = float(recall_score(Y_test, predictions_signed, average='macro', zero_division=0))
    f1 = float(f1_score(Y_test, predictions_signed, average='macro', zero_division=0))
    # Guard balanced_accuracy_score against degenerate cases where predictions
    # contain classes not present in Y_test (which would emit sklearn warnings).
    true_classes_test = np.unique(Y_test)
    pred_classes_test = np.unique(predictions_signed)
    if (
        len(true_classes_test) < 2
        or len(np.setdiff1d(pred_classes_test, true_classes_test)) > 0
    ):
        bacc = float("nan")
    else:
        bacc = float(balanced_accuracy_score(Y_test, predictions_signed))
    # AUC on raw logits
    try:
        auc = float(roc_auc_score(Y_test01, predictions))
    except Exception:
        auc = float('nan')
    print("--- Test Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Balanced Acc: {bacc:.4f}")
    print(f"Threshold: {best_t:.6f}")
    print(f"ROC AUC: {auc:.4f}")
    print("--------------------")
    print("Experiment complete.")

    # Optional model save
    save_cfg = cfg.get("model.save")
    if save_cfg is not None:
        save_path = save_cfg.get("path", os.path.join("models", f"{ds_name}_{ts}.pt"))
        _save_model_torch(
            path=save_path,
            created_at=ts,
            dataset=ds_name,
            device_name=dev_name,
            num_qubits=num_qubits,
            encoder_name=enc_name,
            encoder_opts={k: v for k, v in enc_cfg.items() if k != "name"},
            ansatz_name=anz_name,
            layers=num_layers,
            measurement={"name": meas_name, "wires": meas_wires},
            features=features,
            label=label_col,
            scaler=scaler,
            quantile=qt,
            pls=pls,
            pca=pca,
            weights=weights,
            bias=bias,
            alpha=alpha,
            train_cfg={"lr": lr, "batch": batch_size, "epochs": epochs},
            metrics={"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": bacc, "auc": auc, "val_balanced_accuracy": best_bacc, "threshold": best_t},
        )

    # Return a structured summary for A/B aggregators
    return {
        "log_path": log_path,
        "dataset": ds_name,
        "encoder": enc_name,
        "encoder_opts": {k: v for k, v in enc_cfg.items() if k != "name"},
        "ansatz": anz_name,
        "layers": num_layers,
        "measurement": {"name": meas_name, "wires": meas_wires},
        "train": {"lr": lr, "batch": batch_size, "epochs": epochs},
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": bacc, "auc": auc, "val_balanced_accuracy": best_bacc, "threshold": best_t},
    }


def _save_model_torch(
    *,
    path: str,
    created_at: str,
    dataset: str,
    device_name: str,
    num_qubits: int,
    encoder_name: str,
    encoder_opts: Dict[str, Any],
    ansatz_name: str,
    layers: int,
    measurement: Dict[str, Any],
    features: List[str],
    label: str,
    scaler: Any,
    quantile: Any,
    pls: Any,
    pca: Any,
    weights: Any,
    bias: Any,
    alpha: Any,
    train_cfg: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    import os as _os
    import torch as _torch
    from pennylane import numpy as _np

    _dir = _os.path.dirname(path)
    if _dir:
        _os.makedirs(_dir, exist_ok=True)

    # Convert parameters to torch tensors (detach from autograd if present)
    weights_np = _np.array(weights)
    bias_np = _np.array(bias)
    alpha_np = _np.array(alpha)
    # Try to capture scaler/preprocessing state minimally to avoid unsafe pickle on load
    scaler_state = None
    try:
        # Only keep attributes needed to reconstruct MinMaxScaler
        from sklearn.preprocessing import MinMaxScaler as _SkMinMax
        if isinstance(scaler, _SkMinMax):
            scaler_state = {
                "feature_range": getattr(scaler, "feature_range", (0, 1)),
                "min_": getattr(scaler, "min_", None),
                "scale_": getattr(scaler, "scale_", None),
                "data_min_": getattr(scaler, "data_min_", None),
                "data_max_": getattr(scaler, "data_max_", None),
                "data_range_": getattr(scaler, "data_range_", None),
                "n_samples_seen_": getattr(scaler, "n_samples_seen_", None),
            }
    except Exception:
        pass
    quantile_state = None
    pls_state = None
    pca_state = None
    try:
        from sklearn.preprocessing import QuantileTransformer as _SkQuantile
        if isinstance(quantile, _SkQuantile):
            quantile_state = {
                "n_quantiles": getattr(quantile, "n_quantiles", None),
                "subsample": getattr(quantile, "subsample", None),
                "output_distribution": getattr(quantile, "output_distribution", None),
                "random_state": getattr(quantile, "random_state", None),
                "n_quantiles_": getattr(quantile, "n_quantiles_", None),
                "quantiles_": getattr(quantile, "quantiles_", None),
                "references_": getattr(quantile, "references_", None),
                "n_features_in_": getattr(quantile, "n_features_in_", None),
            }
    except Exception:
        pass
    try:
        from sklearn.cross_decomposition import PLSRegression as _SkPLS
        if isinstance(pls, _SkPLS):
            pls_state = {
                "n_components": getattr(pls, "n_components", None),
                "x_mean_": getattr(pls, "x_mean_", None),
                "x_std_": getattr(pls, "x_std_", None),
                "x_weights_": getattr(pls, "x_weights_", None),
                "x_rotations_": getattr(pls, "x_rotations_", None),
                "n_features_in_": getattr(pls, "n_features_in_", None),
            }
    except Exception:
        pass
    try:
        from sklearn.decomposition import PCA as _SkPCA
        if isinstance(pca, _SkPCA):
            pca_state = {
                "n_components": getattr(pca, "n_components", None),
                "components_": getattr(pca, "components_", None),
                "mean_": getattr(pca, "mean_", None),
                "n_features_in_": getattr(pca, "n_features_in_", None),
            }
    except Exception:
        pass

    state = {
        "version": 2,
        "framework": "pennylane",
        "created_at": created_at,
        "dataset": dataset,
        "device": device_name,
        "num_qubits": int(num_qubits),
        "encoder": encoder_name,
        "encoder_opts": encoder_opts,
        "ansatz": ansatz_name,
        "layers": int(layers),
        "measurement": measurement,
        "features": list(features),
        "label": label,
        # Keep original for backward compat but also include a safe state
        "scaler": scaler,
        "scaler_state": scaler_state,
        "quantile": quantile,
        "quantile_state": quantile_state,
        "pls": pls,
        "pls_state": pls_state,
        "pca": pca,
        "pca_state": pca_state,
        "weights": _torch.tensor(weights_np, dtype=_torch.float32),
        "bias": _torch.tensor(bias_np, dtype=_torch.float32),
        "alpha": _torch.tensor(alpha_np, dtype=_torch.float32),
        "train": train_cfg,
        "metrics": metrics,
    }
    _torch.save(state, path)
    # Avoid redundant duplicate log lines for the same model within one process
    global _PRINTED_SAVED_PATHS
    if path not in _PRINTED_SAVED_PATHS:
        print(f"Model saved to {path}")
        _PRINTED_SAVED_PATHS.add(path)


def load_model(path: str, device_override: Optional[str] = None):
    import torch as _torch
    import pennylane as qml
    from pennylane import numpy as _np
    import numpy as _npy
    import pandas as _pd

    # Allowlist sklearn components for safe unpickling of legacy checkpoints
    try:
        from sklearn.preprocessing import MinMaxScaler as _SkMinMax
        from sklearn.preprocessing import QuantileTransformer as _SkQuantile
        from sklearn.cross_decomposition import PLSRegression as _SkPLS
        from sklearn.decomposition import PCA as _SkPCA
        # Both public and private path (varies by sklearn versions)
        import torch.serialization as _ts
        _ts.add_safe_globals([_SkMinMax, _SkQuantile, _SkPLS, _SkPCA])
        # Also attempt to allow the private module path string
        try:
            import sklearn.preprocessing._data as _sk_data
            _ts.add_safe_globals([getattr(_sk_data, "MinMaxScaler", _SkMinMax)])
        except Exception:
            pass
        try:
            import sklearn.preprocessing._data as _sk_qt
            _ts.add_safe_globals([getattr(_sk_qt, "QuantileTransformer", _SkQuantile)])
        except Exception:
            pass
        try:
            import sklearn.cross_decomposition._pls as _sk_pls
            _ts.add_safe_globals([getattr(_sk_pls, "PLSRegression", _SkPLS)])
        except Exception:
            pass
        try:
            import sklearn.decomposition._pca as _sk_pca
            _ts.add_safe_globals([getattr(_sk_pca, "PCA", _SkPCA)])
        except Exception:
            pass
    except Exception:
        _SkMinMax = None
        _SkQuantile = None
        _SkPLS = None
        _SkPCA = None

    # Explicitly set weights_only=False to support object unpickling from our trusted files
    state = _torch.load(path, map_location="cpu", weights_only=False)

    dev_name = device_override or state.get("device", "lightning.qubit")
    num_qubits = int(state["num_qubits"]) if "num_qubits" in state else len(state.get("features", []))
    dev = qml.device(dev_name, wires=num_qubits)

    enc_name = state["encoder"]
    anz_name = state["ansatz"]
    num_layers = int(state.get("layers", len(state.get("weights", []))))
    if enc_name not in ENCODERS:
        raise ValueError(f"Unknown encoder in saved model: {enc_name}")
    if anz_name not in ANSAETZE:
        raise ValueError(f"Unknown ansatz in saved model: {anz_name}")
    encoder_fn = ENCODERS[enc_name]
    ansatz_fn = ANSAETZE[anz_name]

    wires = list(range(num_qubits))
    enc_opts = state.get("encoder_opts", {})
    meas_cfg = state.get("measurement", {"name": "z0", "wires": [0]})
    meas_name = meas_cfg.get("name", "z0")
    meas_wires = meas_cfg.get("wires", [0])

    angle_scale = None
    if enc_name.startswith("angle_embedding"):
        if enc_opts.get("angle_range") == "0_pi":
            angle_scale = qml.numpy.pi
        elif enc_opts.get("angle_scale") is not None:
            angle_scale = float(enc_opts.get("angle_scale"))

    def _circuit(weights, x):
        reupload = bool(enc_opts.get("reupload", False))
        if reupload:
            def _reupload_layer(W):
                encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)), angle_scale=angle_scale)
                ansatz_fn(W, wires)
            qml.layer(_reupload_layer, num_layers, weights)
        else:
            encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)), angle_scale=angle_scale)
            qml.layer(ansatz_fn, num_layers, weights, wires=wires)
        if meas_name == "mean_z":
            if not meas_wires:
                raise ValueError("mean_z measurement requires at least one wire")
            coeffs = [1.0 / len(meas_wires)] * len(meas_wires)
            ops = [qml.PauliZ(w) for w in meas_wires]
            return qml.expval(qml.Hamiltonian(coeffs, ops))
        else:
            return qml.expval(qml.PauliZ(0))

    # Restore parameters and scaler
    weights_t = state["weights"].detach().cpu().numpy()
    bias_t = state["bias"].detach().cpu().numpy().item() if state["bias"].ndim == 0 else state["bias"].detach().cpu().numpy()
    alpha_t = state.get("alpha")
    if alpha_t is None:
        alpha_t = _np.array(1.0)
    else:
        alpha_t = alpha_t.detach().cpu().numpy().item() if alpha_t.ndim == 0 else alpha_t.detach().cpu().numpy()
    weights_np = _np.array(weights_t, requires_grad=False)
    bias_np = _np.array(bias_t, requires_grad=False)
    alpha_np = _np.array(alpha_t, requires_grad=False)
    # Rebuild preprocessing either from embedded object or from safe state
    scaler = state.get("scaler")
    if scaler is None and state.get("scaler_state") is not None:
        st = state["scaler_state"]
        try:
            if _SkMinMax is not None:
                sc = _SkMinMax(feature_range=tuple(st.get("feature_range", (0, 1))))
                # Assign learned attributes if present
                for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
                    val = st.get(attr)
                    if val is not None:
                        setattr(sc, attr, _np.array(val))
                scaler = sc
        except Exception:
            scaler = None
    quantile = state.get("quantile")
    if quantile is None and state.get("quantile_state") is not None:
        st = state["quantile_state"]
        try:
            if _SkQuantile is not None:
                qt = _SkQuantile(
                    n_quantiles=int(st.get("n_quantiles") or 1000),
                    output_distribution=st.get("output_distribution") or "uniform",
                    subsample=int(st.get("subsample") or 1e9),
                    random_state=st.get("random_state", None),
                )
                for attr in ["n_quantiles_", "quantiles_", "references_", "n_features_in_"]:
                    val = st.get(attr)
                    if val is not None:
                        setattr(qt, attr, _np.array(val) if attr != "n_features_in_" else int(val))
                quantile = qt
        except Exception:
            quantile = None
    pls = state.get("pls")
    if pls is None and state.get("pls_state") is not None:
        st = state["pls_state"]
        try:
            if _SkPLS is not None:
                pls_r = _SkPLS(n_components=int(st.get("n_components") or 2))
                for attr in ["x_mean_", "x_std_", "x_weights_", "x_rotations_", "n_features_in_"]:
                    val = st.get(attr)
                    if val is not None:
                        setattr(pls_r, attr, _np.array(val) if attr != "n_features_in_" else int(val))
                pls = pls_r
        except Exception:
            pls = None
    pca = state.get("pca")
    if pca is None and state.get("pca_state") is not None:
        st = state["pca_state"]
        try:
            if _SkPCA is not None:
                pca_r = _SkPCA(n_components=int(st.get("n_components") or 2))
                for attr in ["components_", "mean_", "n_features_in_"]:
                    val = st.get(attr)
                    if val is not None:
                        setattr(pca_r, attr, _np.array(val) if attr != "n_features_in_" else int(val))
                pca = pca_r
        except Exception:
            pca = None
    features = state.get("features", [])

    class LoadedQuantumClassifier:
        def __init__(self):
            self.features = features
            self.scaler = scaler
            self.quantile = quantile
            self.pls = pls
            self.pca = pca
            self.weights = weights_np
            self.bias = bias_np
            self.alpha = alpha_np
            self._circuit = None

        def _to_numpy(self, X):
            if isinstance(X, _pd.DataFrame):
                if self.features:
                    X = X[self.features]
                X = X.values
            elif isinstance(X, _pd.Series):
                X = X.values.reshape(1, -1)
            else:
                X = _npy.asarray(X)
            if self.quantile is not None:
                X = self.quantile.transform(X)
            if self.pls is not None:
                X = self.pls.transform(X)
            if self.pca is not None:
                X = self.pca.transform(X)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return X

        def _get_circuit(self, Xn):
            if self._circuit is not None:
                return self._circuit
            sample_x = Xn[0] if len(Xn.shape) > 1 else Xn
            sample_x = _npy.asarray(sample_x, dtype=_npy.float64)
            base_qnode = qml.QNode(_circuit, dev, interface="autograd", diff_method="adjoint", cache=True)
            tape = base_qnode.construct([self.weights, sample_x], {})
            all_params = tape.get_parameters(trainable_only=False)
            trainable = set(tape.trainable_params)
            argnum = [i for i in range(len(all_params)) if i not in trainable]
            if argnum:
                self._circuit = qml.batch_input(base_qnode, argnum=argnum)
            else:
                self._circuit = base_qnode
            return self._circuit

        def _variational_classifier(self, X_np):
            X_np = _npy.asarray(X_np, dtype=_npy.float64)
            circuit = self._get_circuit(X_np)
            res = circuit(self.weights, X_np)
            return self.alpha * res + self.bias

        def decision_function(self, X):
            Xn = self._to_numpy(X)
            return _np.array(self._variational_classifier(Xn))

        def predict(self, X):
            scores = self.decision_function(X)
            return _npy.sign(_npy.asarray(scores))

    return LoadedQuantumClassifier()


