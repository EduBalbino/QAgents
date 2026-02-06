from __future__ import annotations

# Central functional DSL for building and running QML experiments
# Keeps per-experiment shims tiny while concentrating shared logic here.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Dedup guards for log lines within a single process run
_PRINTED_SAVED_PATHS: set = set()
_BATCH_INDEX_CACHE: dict = {}


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
    try:
        ndim = int(getattr(W, "ndim", 0))
    except Exception:
        ndim = 0
    layers = [W] if ndim == 2 else W
    for layer in layers:
        for i in range(num_qubits):
            qml.Rot(layer[i, 0], layer[i, 1], layer[i, 2], wires=wires[i])
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[-1], wires[0]])


@register_ansatz("strongly_entangling")
def _ansatz_sel(weights: Any, wires: List[int]) -> None:
    import pennylane as qml
    qml.StronglyEntanglingLayers(weights, wires=wires)


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
    import math
    import time

    # Lazy imports for heavy deps
    import pandas as pd
    import pennylane as qml
    import numpy as np
    import jax
    import jax.numpy as jnp
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight
    from scripts.core.compiled_core import Backend, get_compiled_core
    from scripts.core.compiler import assert_jax_array

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

    # Training config/seed is needed early for deterministic dataset sampling.
    tr_cfg = cfg.get("train", {})
    seed = int(tr_cfg.get("seed", 42))
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

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
    def _coerce_fit_apply(
        Xtr: "pd.DataFrame", Xte: "pd.DataFrame"
    ) -> Tuple["np.ndarray", "np.ndarray", Dict[str, Dict[str, Any]]]:
        Xtr2 = Xtr.copy()
        Xte2 = Xte.copy()
        coerce_state: Dict[str, Dict[str, Any]] = {}
        for c in Xtr2.columns:
            trc = Xtr2[c]
            if pd.api.types.is_numeric_dtype(trc):
                # Already numeric
                coerce_state[c] = {"mode": "numeric", "median": None}
                continue
            tr_num = pd.to_numeric(trc, errors="coerce")
            te_num = pd.to_numeric(Xte2[c], errors="coerce")
            if tr_num.notna().any():
                med = float(tr_num.median()) if tr_num.notna().any() else 0.0
                Xtr2[c] = tr_num.fillna(med)
                Xte2[c] = te_num.fillna(med)
                coerce_state[c] = {"mode": "numeric", "median": med}
            else:
                cats = pd.Index(trc.astype(str).unique())
                mapping = {k: float(i) for i, k in enumerate(cats)}
                Xtr2[c] = trc.astype(str).map(mapping).astype("float64")
                Xte2[c] = Xte2[c].astype(str).map(mapping).fillna(-1).astype("float64")
                coerce_state[c] = {"mode": "categorical", "mapping": mapping, "unknown": -1.0}
        return Xtr2.values, Xte2.values, coerce_state

    def _coerce_binary_label(_y: "pd.Series", label_name: str) -> "pd.Series":
        """Deterministically coerce labels to {0,1} without order-dependent factorization."""
        if pd.api.types.is_numeric_dtype(_y):
            return (pd.to_numeric(_y, errors="coerce").fillna(0) > 0).astype(int)

        y_str = _y.astype(str).str.strip()
        uniq = sorted(set(y_str.tolist()))
        if len(uniq) != 2:
            # Collapse to binary using deterministic lexicographic baseline class.
            lo = uniq[0] if uniq else ""
            return (y_str != lo).astype(int)

        # Optional explicit override, e.g. EDGE_POSITIVE_LABEL=Attack.
        env_pos = os.environ.get("EDGE_POSITIVE_LABEL")
        if env_pos is not None:
            pos = str(env_pos).strip()
            if pos in uniq:
                return (y_str == pos).astype(int)

        # Domain-aware mapping for common binary security labels.
        lower_map = {u.lower(): u for u in uniq}
        pos_tokens = ("attack", "malicious", "anomaly", "intrusion", "true", "yes", "positive")
        neg_tokens = ("benign", "normal", "false", "no", "negative")
        pos = next((lower_map[t] for t in pos_tokens if t in lower_map), None)
        neg = next((lower_map[t] for t in neg_tokens if t in lower_map), None)
        if pos is not None and neg is not None and pos != neg:
            print(f"Label mapping ({label_name}): positive='{pos}', negative='{neg}'")
            return (y_str == pos).astype(int)

        # Fallback deterministic mapping independent of first-seen order.
        pos = uniq[-1]
        neg = uniq[0]
        print(f"Label mapping ({label_name}) fallback lexicographic: positive='{pos}', negative='{neg}'")
        return (y_str == pos).astype(int)

    # Make sure label is binary numeric {0,1}
    y = _coerce_binary_label(y, label_col or "label")
    print("Features and labels extracted.")

    # Split first (to avoid leakage), then coerce using train-fit, then optional PCA (fit on train)
    test_size = float(tr_cfg.get("test_size", 0.2))
    stratify = bool(tr_cfg.get("stratify", True))
    stratify_y = y if stratify else None
    split_seed = int(cfg.get("train", {}).get("seed", 42))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=stratify_y
    )
    print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # Coercion fit on train, apply to test
    X_train, X_test, coerce_state = _coerce_fit_apply(X_train, X_test)

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

    # Labels to {-1, 1} and {0,1} (NumPy only)
    Y_train = np.array(y_train.values * 2 - 1)
    Y_test = np.array(y_test.values * 2 - 1)
    Y_train01 = (Y_train > 0).astype(np.int32)
    Y_test01 = (Y_test > 0).astype(np.int32)

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
            import numpy as np
            from sklearn.metrics import roc_curve
            val_size = min(max(1000, int(0.1 * len(X_train_scaled))), len(X_train_scaled))
            val_idx = np.random.randint(0, len(X_train_scaled), val_size)
            X_val = X_train_scaled[val_idx]
            Y_val01 = Y_train01[val_idx]
            # Ensure both classes present; if not, fallback to a larger slice or default threshold
            if len(np.unique(Y_val01)) < 2:
                val_idx = np.arange(0, min(len(X_train_scaled), 5000))
                X_val = X_train_scaled[val_idx]
                Y_val01 = Y_train01[val_idx]
            prob_val = rf.predict_proba(X_val)[:, 1]
            fpr, tpr, thr = roc_curve(Y_val01, prob_val)
            # balanced accuracy = (tpr + (1 - fpr)) / 2
            bacc_arr = (tpr + (1.0 - fpr)) / 2.0
            best_i = int(np.nanargmax(bacc_arr)) if len(bacc_arr) else 0
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
    dev_kwargs: Dict[str, Any] = {}
    if dev_name == "lightning.gpu":
        # Use single precision complex dtype for materially higher GPU throughput.
        dev_kwargs["c_dtype"] = np.complex64
    dev = qml.device(dev_name, wires=num_qubits, **dev_kwargs)
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

    # Measurement configuration
    meas_cfg = cfg.get("measurement", {"name": "z0", "wires": [0]})
    meas_name = meas_cfg.get("name", "z0")
    meas_wires = list(meas_cfg.get("wires", [0]))
    # Ensure measurement wires are within device range; default to all wires if out-of-range
    if any((int(w) >= num_qubits or int(w) < 0) for w in meas_wires):
        meas_wires = list(range(num_qubits))

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

    spec_hash = (
        f"enc={enc_name}|ansatz={anz_name}|layers={num_layers}|meas={meas_name}"
        f"|mw={','.join(map(str, meas_wires))}|had={int(bool(enc_cfg.get('hadamard', False)))}"
        f"|reu={int(bool(enc_cfg.get('reupload', False)))}|qubits={num_qubits}|feat={feature_dim}"
    )
    backend = Backend(
        device_name=dev_name,
        dtype=jnp.float32,
        compile_opts={"autograph": False},
    )
    batch_size = max(1, int(tr_cfg.get("batch", 256)))
    lr = float(tr_cfg.get("lr", 0.1))
    batched_forward = None


    # Optional Weights & Biases streaming of training metrics
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
            _wandb.log(payload)  # type: ignore[union-attr]
        except Exception:
            # Never let logging issues break training
            pass

    # Training setup
    np.random.seed(seed)
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
    weights_init = jnp.array(0.01 * np.random.randn(num_layers, num_qubits, 3), dtype=backend.dtype)
    bias_init = jnp.array(0.0, dtype=backend.dtype)
    assert_jax_array("weights_init", weights_init, backend.dtype)
    assert_jax_array("bias_init", bias_init, backend.dtype)

    epochs = int(tr_cfg.get("epochs", 1))
    cpu_fuse_epochs = (
        dev_name != "lightning.gpu"
        and epochs > 1
        and os.environ.get("EDGE_CPU_FUSE_EPOCHS", "1") != "0"
    )
    np_rng = np.random.default_rng(seed)
    val_frac = float(tr_cfg.get("val_size", 0.1))

    # Hold out a true validation split from training data for calibration/monitoring.
    if (
        0.0 < val_frac < 0.5
        and len(X_train_scaled) >= 20
        and len(np.unique(Y_train01)) >= 2
    ):
        X_fit_scaled, X_val_scaled, Y_fit, Y_val_hold, Y_fit01, Y_val_hold01 = train_test_split(
            X_train_scaled,
            Y_train,
            Y_train01,
            test_size=val_frac,
            random_state=seed,
            stratify=Y_train01,
        )
    else:
        X_fit_scaled = X_train_scaled
        Y_fit = Y_train
        Y_fit01 = Y_train01
        X_val_scaled = X_train_scaled
        Y_val_hold = Y_train
        Y_val_hold01 = Y_train01

    class_weights_mode = tr_cfg.get("class_weights", "balanced")
    class_weights_map = None
    if class_weights_mode == "balanced":
        cls_labels = np.unique(Y_fit)
        cls_weights_array = compute_class_weight(
            class_weight="balanced", classes=cls_labels, y=Y_fit
        )
        class_weights_map = {int(label): float(weight) for label, weight in zip(cls_labels, cls_weights_array)}
        print(f"Class weights: {class_weights_map}")
    # Weights for {0,1} labels (for logistic loss)
    class_weights01_map = None
    if class_weights_mode == "balanced":
        cls01 = np.unique(Y_fit01)
        w01 = compute_class_weight(class_weight="balanced", classes=cls01, y=Y_fit01)
        class_weights01_map = {int(label): float(weight) for label, weight in zip(cls01, w01)}

    # Training pool (pad once so batch size and epoch batch tensor shapes stay constant)
    X_train_pool = X_fit_scaled
    Y_train_pool = Y_fit
    Y_train01_pool = Y_fit01
    if len(X_train_pool) < batch_size:
        pad_count = batch_size - len(X_train_pool)
        pad_idx = np_rng.integers(0, len(X_train_pool), pad_count)
        X_train_pool = np.concatenate([X_train_pool, X_train_pool[pad_idx]], axis=0)
        Y_train_pool = np.concatenate([Y_train_pool, Y_train_pool[pad_idx]], axis=0)
        Y_train01_pool = np.concatenate([Y_train01_pool, Y_train01_pool[pad_idx]], axis=0)
    rem = len(X_train_pool) % batch_size
    if rem != 0:
        pad_count = batch_size - rem
        pad_idx = np_rng.integers(0, len(X_train_pool), pad_count)
        X_train_pool = np.concatenate([X_train_pool, X_train_pool[pad_idx]], axis=0)
        Y_train_pool = np.concatenate([Y_train_pool, Y_train_pool[pad_idx]], axis=0)
        Y_train01_pool = np.concatenate([Y_train01_pool, Y_train01_pool[pad_idx]], axis=0)
    if class_weights01_map:
        w_train_pool = np.array([class_weights01_map[int(label)] for label in Y_train01_pool], dtype=np.float32)
    else:
        w_train_pool = np.ones((len(X_train_pool),), dtype=np.float32)
    # Pre-scale once on host so the compiled quantum graph stays minimal.
    if enc_name in {"angle_embedding_y", "angle_pair_xy"}:
        if enc_cfg.get("angle_range") == "0_pi":
            angle_scale = np.float32(np.pi)
        elif enc_cfg.get("angle_scale") is not None:
            angle_scale = np.float32(float(enc_cfg.get("angle_scale")))
        else:
            angle_scale = np.float32(1.0)
    else:
        angle_scale = np.float32(1.0)
    X_train_pool = np.asarray(X_train_pool, dtype=np.float32) * angle_scale
    X_val_scaled = np.asarray(X_val_scaled, dtype=np.float32) * angle_scale
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32) * angle_scale

    # Compile an epoch-sized training kernel with device-side batch traversal.
    num_it_data = max(1, len(X_train_pool) // batch_size)
    compiled_steps_cfg = tr_cfg.get("compiled_steps")
    if compiled_steps_cfg is None:
        compiled_steps_env = os.environ.get("EDGE_COMPILED_STEPS")
        compiled_steps_cfg = int(compiled_steps_env) if compiled_steps_env else 0
    compiled_steps = int(compiled_steps_cfg) if compiled_steps_cfg else 0
    num_it = compiled_steps if compiled_steps > 0 else num_it_data
    compile_num_batches = num_it * epochs if cpu_fuse_epochs else num_it
    lr_j = jnp.asarray(lr, dtype=backend.dtype)
    compiled = get_compiled_core(
        num_qubits,
        num_layers,
        backend,
        spec_hash,
        shape_key=(batch_size, feature_dim, num_qubits),
        encoder_name=str(enc_name),
        ansatz_name=str(anz_name),
        measurement_name=str(meas_name),
        measurement_wires=tuple(int(w) for w in meas_wires),
        hadamard=bool(enc_cfg.get("hadamard", False)),
        reupload=bool(enc_cfg.get("reupload", False)),
        num_batches=compile_num_batches,
        batch_size=batch_size,
    )
    batched_forward = compiled["batched_forward"]
    train_epoch_compiled = compiled["train_epoch_compiled"]
    init_opt_state = compiled["init_opt_state"]
    assert_no_python_callback_ir = compiled.get("assert_no_python_callback_ir")
    # Force qjit compilation with concrete arrays before entering jitted epoch scans.
    _ = np.asarray(batched_forward(weights_init, jnp.asarray(X_train_pool[:1], dtype=backend.dtype)))

    # Training loop (epochs)
    alpha_init = jnp.array(1.0, dtype=backend.dtype)
    assert_jax_array("alpha_init", alpha_init, backend.dtype)
    params = (weights_init, bias_init, alpha_init)
    train_state = (params, init_opt_state(params))
    rng_key = jax.random.PRNGKey(seed)
    _warm_idx = np.arange(compile_num_batches * batch_size, dtype=np.int32).reshape(
        compile_num_batches, batch_size
    )
    X_steps_warm = jnp.asarray(X_train_pool[_warm_idx % len(X_train_pool)], dtype=backend.dtype)
    Y_steps_warm = jnp.asarray(Y_train01_pool[_warm_idx % len(Y_train01_pool)], dtype=backend.dtype)
    w_steps_warm = jnp.asarray(w_train_pool[_warm_idx % len(w_train_pool)], dtype=backend.dtype)
    if os.environ.get("EDGE_ENFORCE_NO_PY_CALLBACK", "0") != "0" and callable(assert_no_python_callback_ir):
        assert_no_python_callback_ir(train_state, rng_key, X_steps_warm, Y_steps_warm, w_steps_warm, lr_j)
    start_time = time.time()
    total_iters = 0
    _early_stop = False
    X_val_j = jnp.asarray(X_val_scaled, dtype=backend.dtype)
    X_test_j = jnp.asarray(X_test_scaled, dtype=backend.dtype)
    cache_key = (
        int(seed),
        int(len(X_train_pool)),
        int(epochs),
        int(num_it),
        int(batch_size),
        int(compiled_steps),
    )
    idx_steps_all = _BATCH_INDEX_CACHE.get(cache_key)
    if idx_steps_all is None:
        idx_steps_all = np.empty((epochs, num_it, batch_size), dtype=np.int32)
        for ep in range(epochs):
            if compiled_steps > 0:
                idx_steps_all[ep] = np_rng.integers(
                    0, len(X_train_pool), size=(num_it, batch_size), dtype=np.int32
                )
            else:
                perm = np_rng.permutation(len(X_train_pool)).astype(np.int32, copy=False)
                idx_steps_all[ep] = perm[: num_it * batch_size].reshape(num_it, batch_size)
        _BATCH_INDEX_CACHE[cache_key] = idx_steps_all

    if cpu_fuse_epochs:
        print(
            f"CPU fused training | epochs={epochs} | iters/epoch={num_it} | total_iters={compile_num_batches} | "
            f"batch_size={batch_size}",
            flush=True,
        )
        idx_steps_flat = idx_steps_all.reshape(compile_num_batches, batch_size)
        X_steps_j = jnp.asarray(X_train_pool[idx_steps_flat], dtype=backend.dtype)
        Y_steps_j = jnp.asarray(Y_train01_pool[idx_steps_flat], dtype=backend.dtype)
        w_steps_j = jnp.asarray(w_train_pool[idx_steps_flat], dtype=backend.dtype)
        iter_start = time.time()
        train_state, rng_key, loss_stats = train_epoch_compiled(
            train_state,
            rng_key,
            X_steps_j,
            Y_steps_j,
            w_steps_j,
            lr_j,
        )
        loss_stats.block_until_ready()
        iter_s = time.time() - iter_start
        total_iters = compile_num_batches
        params, _ = train_state
        weights, bias, alpha = params
        loss_stats_np = np.asarray(loss_stats)
        c_mean = float(loss_stats_np[0]) if loss_stats_np.size >= 1 else float("nan")
        c_b = float(loss_stats_np[1]) if loss_stats_np.size >= 2 else float("nan")
        print(
            f"CPU fused done | mean_loss={c_mean:0.7f} | last_loss={c_b:0.7f} | Time: {iter_s:.2f}s",
            flush=True,
        )
        _log_train_metrics_to_wandb(
            epoch=epochs,
            iter_in_epoch=num_it,
            total_iters=total_iters,
            batch_size=batch_size,
            loss=float(c_mean),
            acc=float("nan"),
            start_time=start_time,
        )
    else:
        for ep in range(epochs):
            print(f"Epoch {ep+1}/{epochs} | iters={num_it} | batch_size={batch_size}", flush=True)
            idx_steps = idx_steps_all[ep]
            X_steps_j = jnp.asarray(X_train_pool[idx_steps], dtype=backend.dtype)
            Y_steps_j = jnp.asarray(Y_train01_pool[idx_steps], dtype=backend.dtype)
            w_steps_j = jnp.asarray(w_train_pool[idx_steps], dtype=backend.dtype)
            iter_start = time.time()
            train_state, rng_key, loss_stats = train_epoch_compiled(
                train_state,
                rng_key,
                X_steps_j,
                Y_steps_j,
                w_steps_j,
                lr_j,
            )
            # Ensure async dispatches are accounted in timings/log output.
            loss_stats.block_until_ready()
            iter_s = time.time() - iter_start
            total_iters += num_it
            params, _ = train_state
            weights, bias, alpha = params
            loss_stats_np = np.asarray(loss_stats)
            c_mean = float(loss_stats_np[0]) if loss_stats_np.size >= 1 else float("nan")
            c_b = float(loss_stats_np[1]) if loss_stats_np.size >= 2 else float("nan")
            print(
                f"Epoch {ep+1}/{epochs} done | mean_loss={c_mean:0.7f} | "
                f"last_loss={c_b:0.7f} | Epoch Time: {iter_s:.2f}s",
                flush=True,
            )

            _log_train_metrics_to_wandb(
                epoch=ep + 1,
                iter_in_epoch=num_it,
                total_iters=total_iters,
                batch_size=batch_size,
                loss=float(c_mean),
                acc=float("nan"),
                start_time=start_time,
            )
            if not np.isfinite(c_b):
                print("Early exit: non-finite batch loss detected; stopping training early.")
                _early_stop = True
            if _early_stop:
                break

    train_time_s = time.time() - start_time
    iters_per_s = float(total_iters / train_time_s) if train_time_s > 0 else float("nan")
    print(
        f"Training finished in {train_time_s:.2f}s over {epochs} epoch(s), {total_iters} iters. "
        f"Iters/sec: {iters_per_s:0.2f}"
    )

    params, _ = train_state
    weights, bias, alpha = params

    # Validation calibration on held-out validation split.
    X_val = X_val_scaled
    Y_val = Y_val_hold
    Y_val01 = Y_val_hold01
    preds_val = np.array(alpha * batched_forward(weights, X_val_j) + bias)
    val_auc_raw = float("nan")
    val_auc_inv = float("nan")
    try:
        if len(np.unique(Y_val01)) >= 2:
            val_auc_raw = float(roc_auc_score(Y_val01, preds_val))
            val_auc_inv = float(roc_auc_score(Y_val01, -preds_val))
            if np.isfinite(val_auc_raw) and np.isfinite(val_auc_inv) and val_auc_inv > val_auc_raw:
                print(
                    f"[WARN] Validation AUC inversion detected: auc={val_auc_raw:.4f}, "
                    f"auc_inv={val_auc_inv:.4f}. Check label/score polarity."
                )
    except Exception:
        pass
    # Choose threshold to maximize balanced accuracy on validation.
    # Guard against degenerate validation slices that contain only a single class,
    # which would otherwise trigger sklearn's
    # "y_pred contains classes not in y_true" warning inside balanced_accuracy_score.
    try:
        import numpy as np

        unique_val_classes = np.unique(Y_val)
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
            th_candidates = np.unique(
                np.concatenate([[-np.inf, np.inf], preds_val])
            )
            best_t = 0.0
            best_bacc = -1.0
            for t in th_candidates:
                preds_lab = np.where(preds_val >= t, 1, -1)
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
    predictions = np.array(alpha * batched_forward(weights, X_test_j) + bias)

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

    alpha = np.array(alpha)
    score_sign = np.array(1.0)

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
            coerce_state=coerce_state,
            scaler=scaler,
            quantile=qt,
            pls=pls,
            pca=pca,
            weights=weights,
            bias=bias,
            alpha=alpha,
            score_sign=score_sign,
            compiled_input_scale=float(angle_scale),
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
    coerce_state: Dict[str, Dict[str, Any]],
    scaler: Any,
    quantile: Any,
    pls: Any,
    pca: Any,
    weights: Any,
    bias: Any,
    alpha: Any,
    score_sign: Any,
    compiled_input_scale: float,
    train_cfg: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    import os as _os
    import torch as _torch
    from pennylane import numpy as np

    _dir = _os.path.dirname(path)
    if _dir:
        _os.makedirs(_dir, exist_ok=True)

    # Convert parameters to torch tensors (detach from autograd if present)
    weightsnp = np.array(weights)
    biasnp = np.array(bias)
    alphanp = np.array(alpha)
    score_sign_np = np.array(score_sign)
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
        "coerce_state": coerce_state,
        # Keep original for backward compat but also include a safe state
        "scaler": scaler,
        "scaler_state": scaler_state,
        "quantile": quantile,
        "quantile_state": quantile_state,
        "pls": pls,
        "pls_state": pls_state,
        "pca": pca,
        "pca_state": pca_state,
        "weights": _torch.tensor(weightsnp, dtype=_torch.float32),
        "bias": _torch.tensor(biasnp, dtype=_torch.float32),
        "alpha": _torch.tensor(alphanp, dtype=_torch.float32),
        "score_sign": _torch.tensor(score_sign_np, dtype=_torch.float32),
        "compiled_input_scale": float(compiled_input_scale),
        "train": train_cfg,
        "metrics": metrics,
        "threshold": metrics.get("threshold", 0.0),
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
    from pennylane import numpy as np
    import numpy as npy
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
    dev_kwargs: Dict[str, Any] = {}
    if dev_name == "lightning.gpu":
        dev_kwargs["c_dtype"] = npy.complex64
    dev = qml.device(dev_name, wires=num_qubits, **dev_kwargs)

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
        if reupload and anz_name == "ring_rot_cnot":
            def _reupload_layer(W):
                encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)), angle_scale=angle_scale)
                ansatz_fn(W, wires)
            qml.layer(_reupload_layer, num_layers, weights)
        else:
            encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)), angle_scale=angle_scale)
            ansatz_fn(weights, wires)
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
        alpha_t = np.array(1.0)
    else:
        alpha_t = alpha_t.detach().cpu().numpy().item() if alpha_t.ndim == 0 else alpha_t.detach().cpu().numpy()
    weightsnp = np.array(weights_t, requires_grad=False)
    biasnp = np.array(bias_t, requires_grad=False)
    alphanp = np.array(alpha_t, requires_grad=False)
    score_sign_t = state.get("score_sign")
    if score_sign_t is None:
        score_sign_t = np.array(1.0)
    elif hasattr(score_sign_t, "detach"):
        score_sign_t = (
            score_sign_t.detach().cpu().numpy().item()
            if getattr(score_sign_t, "ndim", 0) == 0
            else score_sign_t.detach().cpu().numpy()
        )
    else:
        score_sign_t = npy.asarray(score_sign_t)
    score_sign_np = np.array(score_sign_t, requires_grad=False)
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
                        setattr(sc, attr, np.array(val))
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
                        setattr(qt, attr, np.array(val) if attr != "n_features_in_" else int(val))
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
                        setattr(pls_r, attr, np.array(val) if attr != "n_features_in_" else int(val))
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
                        setattr(pca_r, attr, np.array(val) if attr != "n_features_in_" else int(val))
                pca = pca_r
        except Exception:
            pca = None
    features = state.get("features", [])
    coerce_state = state.get("coerce_state") or {}
    threshold = float(state.get("threshold", state.get("metrics", {}).get("threshold", 0.0)))
    compiled_input_scale = float(state.get("compiled_input_scale", 1.0))

    class LoadedQuantumClassifier:
        def __init__(self):
            self.features = features
            self.scaler = scaler
            self.quantile = quantile
            self.pls = pls
            self.pca = pca
            self.coerce_state = coerce_state
            self.weights = weightsnp
            self.bias = biasnp
            self.alpha = alphanp
            self.score_sign = score_sign_np
            self.threshold = threshold
            self.compiled_input_scale = compiled_input_scale
            self._circuit = None

        def _to_numpy(self, X):
            if isinstance(X, _pd.DataFrame):
                if self.features:
                    X = X[self.features]
                X_df = X.copy()
                # Re-apply training-time coercion for raw tabular features.
                if self.coerce_state:
                    for c in X_df.columns:
                        st = self.coerce_state.get(c)
                        if not isinstance(st, dict):
                            continue
                        mode = st.get("mode")
                        if mode == "categorical":
                            mapping = st.get("mapping", {})
                            unk = float(st.get("unknown", -1.0))
                            X_df[c] = X_df[c].astype(str).map(mapping).fillna(unk).astype("float64")
                        else:
                            med = st.get("median", None)
                            s_num = _pd.to_numeric(X_df[c], errors="coerce")
                            if med is not None:
                                X_df[c] = s_num.fillna(float(med))
                            else:
                                X_df[c] = s_num
                else:
                    # Backward compatibility: best-effort coercion for older checkpoints.
                    for c in X_df.columns:
                        s_num = _pd.to_numeric(X_df[c], errors="coerce")
                        if s_num.notna().any():
                            med = float(s_num.median()) if s_num.notna().any() else 0.0
                            X_df[c] = s_num.fillna(med)
                        else:
                            cats = _pd.Index(X_df[c].astype(str).unique())
                            mapping = {k: float(i) for i, k in enumerate(cats)}
                            X_df[c] = X_df[c].astype(str).map(mapping).astype("float64")
                X = X_df.values
            elif isinstance(X, _pd.Series):
                X = X.values.reshape(1, -1)
            else:
                X = npy.asarray(X)
            if self.quantile is not None:
                X = self.quantile.transform(X)
            if self.pls is not None:
                X = self.pls.transform(X)
            if self.pca is not None:
                X = self.pca.transform(X)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            if self.compiled_input_scale != 1.0:
                X = X * self.compiled_input_scale
            return X

        def _get_circuit(self, Xn):
            if self._circuit is not None:
                return self._circuit
            self._circuit = qml.QNode(_circuit, dev, interface="autograd", cache=True)
            return self._circuit

        def _variational_classifier(self, Xnp):
            Xnp = npy.asarray(Xnp, dtype=npy.float64)
            circuit = self._get_circuit(Xnp)
            if Xnp.ndim <= 1:
                res = circuit(self.weights, Xnp)
            else:
                res = npy.array([circuit(self.weights, row) for row in Xnp], dtype=npy.float64)
            return self.score_sign * (self.alpha * res + self.bias)

        def decision_function(self, X):
            Xn = self._to_numpy(X)
            return np.array(self._variational_classifier(Xn))

        def predict(self, X):
            scores = self.decision_function(X)
            scores_np = npy.asarray(scores, dtype=float)
            return npy.where(scores_np >= float(self.threshold), 1.0, -1.0)

    return LoadedQuantumClassifier()
