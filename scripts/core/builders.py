from __future__ import annotations

# Central functional DSL for building and running QML experiments
# Keeps per-experiment shims tiny while concentrating shared logic here.

from dataclasses import dataclass
import os
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

def measurement(name: str, wires: List[int]) -> Step:
    return Step("measurement", {"name": name, "wires": wires})


def train(
    lr: float = 0.1,
    batch: int = 100,
    epochs: int = 1,
    class_weights: Optional[str] = "balanced",
    seed: int = 42,
    test_size: float = 0.2,
    stratify: bool = True,
    balanced_batches: bool = True,
    balanced_pos_frac: float = 0.5,
    **kwargs: Any,
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
            "balanced_batches": balanced_batches,
            "balanced_pos_frac": balanced_pos_frac,
            **kwargs,
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
def _enc_angle_y(x: Any, wires: Any, hadamard: bool = False, **_: Any) -> None:
    import pennylane as qml  # local import to avoid import cost until run
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


@register_encoder("angle_embedding_x")
def _enc_angle_x(x: Any, wires: Any, hadamard: bool = False, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    qml.AngleEmbedding(x, wires=wires, rotation="X")


@register_encoder("angle_embedding_z")
def _enc_angle_z(x: Any, wires: Any, hadamard: bool = False, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    qml.AngleEmbedding(x, wires=wires, rotation="Z")


@register_encoder("amplitude_embedding")
def _enc_amplitude(x: Any, wires: Any, **_: Any) -> None:
    import pennylane as qml

    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)


@register_encoder("angle_pattern_xyz")
def _enc_angle_pattern_xyz(x: Any, wires: Any, hadamard: bool = False, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    # Cycle X, Y, Z by wire index for diverse Bloch trajectories
    for i, w in enumerate(wires):
        if i % 3 == 0:
            qml.RX(x[i], wires=w)
        elif i % 3 == 1:
            qml.RY(x[i], wires=w)
        else:
            qml.RZ(x[i], wires=w)


@register_encoder("angle_pair_xy")
def _enc_angle_pair_xy(x: Any, wires: Any, hadamard: bool = False, **_: Any) -> None:
    import pennylane as qml
    if hadamard:
        for w in wires:
            qml.Hadamard(wires=w)
    # Apply RX then RY per wire to enrich expressivity with minimal overhead
    for i, w in enumerate(wires):
        qml.RX(x[i], wires=w)
        qml.RY(x[i], wires=w)


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


#
# Preprocess artifact caching removed:
# - It caused confusing behavior ("cache hit/miss") and background I/O.
# - The pipeline is fast enough without it for current workflows.
# If you need caching later, reintroduce it with a proper CLI flag and tests.


def _rebuild_preprocess_objects(
    *,
    scaler_state: Optional[Dict[str, Any]],
    quantile_state: Optional[Dict[str, Any]],
    pls_state: Optional[Dict[str, Any]],
    pca_state: Optional[Dict[str, Any]],
):
    import numpy as _np

    scaler = None
    quantile = None
    pls = None
    pca = None
    try:
        from sklearn.preprocessing import MinMaxScaler as _SkMinMax
    except Exception:
        _SkMinMax = None
    try:
        from sklearn.preprocessing import QuantileTransformer as _SkQuantile
    except Exception:
        _SkQuantile = None
    try:
        from sklearn.cross_decomposition import PLSRegression as _SkPLS
    except Exception:
        _SkPLS = None
    try:
        from sklearn.decomposition import PCA as _SkPCA
    except Exception:
        _SkPCA = None

    if scaler_state and _SkMinMax is not None:
        try:
            sc = _SkMinMax(feature_range=tuple(scaler_state.get("feature_range", (0, 1))))
            for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
                val = scaler_state.get(attr)
                if val is not None:
                    setattr(sc, attr, _np.array(val))
            scaler = sc
        except Exception:
            scaler = None
    if quantile_state and _SkQuantile is not None:
        try:
            qt = _SkQuantile(
                n_quantiles=int(quantile_state.get("n_quantiles") or 1000),
                output_distribution=quantile_state.get("output_distribution") or "uniform",
                subsample=int(quantile_state.get("subsample") or 1e9),
                random_state=quantile_state.get("random_state", None),
            )
            for attr in ["n_quantiles_", "quantiles_", "references_", "n_features_in_"]:
                val = quantile_state.get(attr)
                if val is not None:
                    setattr(qt, attr, _np.array(val) if attr != "n_features_in_" else int(val))
            quantile = qt
        except Exception:
            quantile = None
    if pls_state and _SkPLS is not None:
        try:
            pls_r = _SkPLS(n_components=int(pls_state.get("n_components") or 2))
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
                    setattr(pls_r, attr, _np.array(val) if attr != "n_features_in_" else int(val))
            # If only public attrs were saved, populate sklearn>=1.8 private attrs.
            if getattr(pls_r, "_x_mean", None) is None and getattr(pls_r, "x_mean_", None) is not None:
                pls_r._x_mean = _np.array(getattr(pls_r, "x_mean_"))
            if getattr(pls_r, "_x_std", None) is None and getattr(pls_r, "x_std_", None) is not None:
                pls_r._x_std = _np.array(getattr(pls_r, "x_std_"))
            pls = pls_r
        except Exception:
            pls = None
    if pca_state and _SkPCA is not None:
        try:
            pca_r = _SkPCA(n_components=int(pca_state.get("n_components") or 2))
            for attr in ["components_", "mean_", "n_features_in_"]:
                val = pca_state.get(attr)
                if val is not None:
                    setattr(pca_r, attr, _np.array(val) if attr != "n_features_in_" else int(val))
            pca = pca_r
        except Exception:
            pca = None

    return scaler, quantile, pls, pca


def run(recipe: Recipe) -> Dict[str, Any]:
    import os
    import datetime
    import math
    import time

    # Lazy imports for heavy deps
    import pandas as pd
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
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
    q_cfg = cfg.get("dataset.quantile_uniform", None)
    pls_cfg = cfg.get("dataset.pls_pow2", None)
    pca_cfg = cfg.get("dataset.pca_pow2", None)

    # Encoder name is needed during preprocessing decisions.
    # IMPORTANT: define it before any later assignments in this function to avoid UnboundLocalError.
    enc_cfg = cfg.get("vqc.encoder", {"name": "angle_embedding_y"})
    enc_name = enc_cfg.get("name")
    batch_size_cfg = max(1, int(tr_cfg.get("batch", 256)))
    val_frac_cfg = float(tr_cfg.get("val_size", 0.1))
    test_size = float(tr_cfg.get("test_size", 0.2))
    stratify = bool(tr_cfg.get("stratify", True))
    split_seed = seed
    env_preprocess_seed = os.environ.get("EDGE_PREPROCESS_SPLIT_SEED")
    if env_preprocess_seed not in (None, "", "None"):
        try:
            split_seed = int(env_preprocess_seed)
        except Exception:
            pass

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

    def _read_csv_exact_sample(_path: str, _k: int, _rng: "Any") -> "pd.DataFrame":
        """
        Deterministic exact-size CSV sampling without allocating O(N) skip lists.

        Pandas supports `skiprows` as a callable. We use reservoir sampling
        to select the *rows to keep* (size k), then skip everything else.
        """
        keep: List[int] = []
        # `skiprows` line numbers are 0-indexed; header is 0, data starts at 1.
        with open(_path, "r", encoding="utf-8", errors="ignore") as _f:
            hdr = _f.readline()
            if not hdr:
                return pd.read_csv(_path, low_memory=False)
            seen = 0
            for line_no, _ in enumerate(_f, start=1):
                if seen < _k:
                    keep.append(line_no)
                else:
                    j = _rng.randrange(seen + 1)
                    if j < _k:
                        keep[j] = line_no
                seen += 1
        if not keep:
            return pd.read_csv(_path, low_memory=False)
        keep_set = set(keep)
        return pd.read_csv(
            _path,
            skiprows=lambda i: (i != 0 and i not in keep_set),
            low_memory=False,
        )

    # Preprocess artifact caching removed: always load the dataframe directly.
    df = None
    if path and os.path.exists(path):
        if sample_size is None:
            df = pd.read_csv(path, low_memory=False)
        else:
            # Deterministic sampling using reservoir sampling; avoids O(N) skip lists.
            k = max(1, int(sample_size))
            rng = _random.Random(seed)
            df = _read_csv_exact_sample(path, k, rng)
        print(f"Dataset loaded from {path}. Shape: {df.shape}")
    else:
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

    # If the dataset is already preprocessed (PC_* columns), do not fit/serialize sklearn transforms.
    # This keeps the saved model artifact minimal and makes inference independent of those objects.
    input_is_preprocessed = bool(features) and all(str(f).startswith("PC_") for f in features)

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

    # Split first (to avoid leakage), then coerce using train-fit.
    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=stratify_y
    )
    print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # Coercion fit on train, apply to test
    X_train, X_test, coerce_state = _coerce_fit_apply(X_train, X_test)

    # Track preprocessing pipeline components for persistence (model saving).
    scaler = None
    qt = None
    pls = None
    pca = None

    # Optional supervised dimensionality reduction to nearest power of two using PLS
    if pls_cfg is not None:
        from sklearn.cross_decomposition import PLSRegression
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
    if pca_cfg is not None:
        d0 = X_train.shape[1]
        max_power = d0.bit_length() - 1
        max_qubits = pca_cfg.get("max_qubits")
        if max_qubits is not None:
            max_power = min(max_power, int(max_qubits))
        target_dim = max(1, 2 ** max_power)
        if target_dim != d0:
            pca = PCA(n_components=target_dim, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    # Optional boundedization via quantile map. For preprocessed PC_* inputs, this is off.
    apply_quantile = (q_cfg is not None) and (not input_is_preprocessed)
    if apply_quantile:
        from sklearn.preprocessing import QuantileTransformer
        cfg_q = q_cfg or {}
        n_q = int(cfg_q.get("n_quantiles", min(1000, len(X_train))))
        qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="uniform",
            subsample=int(1e9),
            random_state=42,
        )
        X_train_scaled = qt.fit_transform(X_train)
        X_test_scaled = qt.transform(X_test)
        print(
            "Features quantile-uniformized to [0,1]. "
            f"X_train_scaled shape={X_train_scaled.shape}, X_test_scaled shape={X_test_scaled.shape}"
        )
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        print(
            "Features left as-is (no quantile mapping configured). "
            f"X_train_scaled shape={X_train_scaled.shape}, X_test_scaled shape={X_test_scaled.shape}"
        )

    # Labels to {-1, 1} and {0,1} (NumPy only)
    Y_train = np.array(y_train.values * 2 - 1)
    Y_test = np.array(y_test.values * 2 - 1)
    Y_train01 = (Y_train > 0).astype(np.int32)
    Y_test01 = (Y_test > 0).astype(np.int32)

    # Optional export of the PLS-transformed dataset as seen by QML.
    # This runs after quantile + PLS (+ optional PCA) and uses numbered feature
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
    enc_name_early = enc_name

    # Device and wires (device is created inside compiled_core; no qml.device here)
    dev_cfg = cfg.get("device")
    if not isinstance(dev_cfg, dict) or not dev_cfg.get("name"):
        raise ValueError("Config must include device.name (e.g. {'device': {'name': 'lightning.qubit'}})")
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
    dev_name = str(dev_cfg["name"])

    # Encoder / Ansatz
    anz_cfg = cfg.get("vqc.ansatz", {"name": "ring_rot_cnot", "layers": 3})
    anz_name = anz_cfg.get("name")
    num_layers = int(anz_cfg.get("layers", 3))

    if enc_name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {enc_name}")
    if anz_name not in ANSAETZE:
        raise ValueError(f"Unknown ansatz: {anz_name}")

    # Measurement configuration (required; no silent defaults)
    meas_cfg = cfg.get("measurement")
    if not isinstance(meas_cfg, dict) or not meas_cfg.get("name") or "wires" not in meas_cfg:
        raise ValueError("Config must include measurement.{name,wires} (e.g. {'measurement': {'name': 'z_vec', 'wires': [0,1]}})")
    meas_name = str(meas_cfg["name"])
    meas_wires = list(meas_cfg["wires"] or [])
    if not meas_wires:
        raise ValueError("measurement.wires must be non-empty")
    bad_wires = [int(w) for w in meas_wires if int(w) < 0 or int(w) >= int(num_qubits)]
    if bad_wires:
        raise ValueError(f"measurement.wires contains out-of-range wires for num_qubits={num_qubits}: {bad_wires}")
    if meas_name == "mean_z":
        print("[WARN] measurement=mean_z uses a single bounded expval; prefer measurement=z_vec for a trainable readout.", flush=True)
    if meas_name == "mean_z_readout":
        print("[INFO] measurement=mean_z_readout uses a trainable post-ansatz gate readout (w_ro as gate params).", flush=True)
    if meas_name == "z0":
        print("[WARN] measurement=z0 is a single-qubit readout; prefer measurement=z_vec unless you intend this bottleneck.", flush=True)

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

    backend = Backend(
        device_name=dev_name,
        dtype=jnp.float32,
        compile_opts={"autograph": False},
    )
    batch_size = max(1, int(tr_cfg.get("batch", 256)))
    lr = float(tr_cfg.get("lr", 0.1))
    epochs = int(tr_cfg.get("epochs", 1))

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
                try:
                    _wandb.define_metric("epoch")  # type: ignore[union-attr]
                    _wandb.define_metric("train/*", step_metric="epoch")  # type: ignore[union-attr]
                    _wandb.define_metric("val/*", step_metric="epoch")  # type: ignore[union-attr]
                    _wandb.define_metric("time/*", step_metric="epoch")  # type: ignore[union-attr]
                    _wandb.define_metric("polarity/*", step_metric="epoch")  # type: ignore[union-attr]
                except Exception:
                    # Best-effort; never fail training for W&B API differences.
                    pass
        except Exception:
            _wandb = None
            _wandb_can_log = False

    def _wandb_log(payload: Dict[str, Any]) -> None:
        if not _wandb_can_log:
            return
        try:
            _wandb.log(payload)  # type: ignore[union-attr]
        except Exception:
            pass

    # Training setup
    np.random.seed(seed)
    try:
        import random as _py_random
        _py_random.seed(seed)
    except Exception:
        pass
    weights_init = jnp.array(0.01 * np.random.randn(num_layers, num_qubits, 3), dtype=backend.dtype)
    bias_init = jnp.array(0.0, dtype=backend.dtype)
    assert_jax_array("weights_init", weights_init, backend.dtype)
    assert_jax_array("bias_init", bias_init, backend.dtype)
    np_rng = np.random.default_rng(seed)
    val_frac = val_frac_cfg

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
        w_train_pool *= len(w_train_pool) / w_train_pool.sum()  # normalize so mean == 1.0
    else:
        w_train_pool = np.ones((len(X_train_pool),), dtype=np.float32)
    # Pre-map once on host so the compiled quantum graph stays minimal.
    #
    # Convention: after preprocessing we aim to have u in [0,1]. For angle encoders we apply an
    # affine map u -> theta on-host:
    # - angle_range="pm_pi"   : theta = 2π(u - 0.5)  ∈ [-π, π]
    # - angle_range="pm_pi_2" : theta =  π(u - 0.5)  ∈ [-π/2, π/2]
    # - angle_range="0_pi"    : theta =  πu          ∈ [0, π] (legacy)
    #
    is_angle_encoder = str(enc_name).startswith("angle_") or str(enc_name).startswith("angle")
    compiled_input_scale = np.float32(1.0)
    compiled_input_shift = np.float32(0.0)
    if is_angle_encoder:
        ar = enc_cfg.get("angle_range")
        if ar is None and enc_cfg.get("angle_scale") is None:
            raise ValueError(
                "Angle encoders require an explicit input map. Set encoder.angle_range "
                "('pm_pi'|'pm_pi_2'|'0_pi') or encoder.angle_scale."
            )
        if ar == "pm_pi":
            compiled_input_scale = np.float32(2.0 * np.pi)
            compiled_input_shift = np.float32(-1.0 * np.pi)
        elif ar == "pm_pi_2":
            compiled_input_scale = np.float32(1.0 * np.pi)
            compiled_input_shift = np.float32(-0.5 * np.pi)
        elif ar == "0_pi":
            compiled_input_scale = np.float32(1.0 * np.pi)
            compiled_input_shift = np.float32(0.0)
        elif enc_cfg.get("angle_scale") is not None:
            compiled_input_scale = np.float32(float(enc_cfg.get("angle_scale")))
            compiled_input_shift = np.float32(0.0)

    X_train_pool = np.asarray(X_train_pool, dtype=np.float32) * compiled_input_scale + compiled_input_shift
    X_val_scaled = np.asarray(X_val_scaled, dtype=np.float32) * compiled_input_scale + compiled_input_shift
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32) * compiled_input_scale + compiled_input_shift

    # Per-batch training only (no epoch fusion, no epoch-compiled loop).
    # Optional fixed batch count keeps iters/epoch stable across sweep runs.
    num_it_data = max(1, len(X_train_pool) // batch_size)
    fixed_num_batches = int(tr_cfg.get("fixed_num_batches", 0) or 0)
    num_it = fixed_num_batches if fixed_num_batches > 0 else num_it_data
    if fixed_num_batches > 0 and fixed_num_batches != num_it_data:
        print(
            f"[INFO] Using fixed_num_batches={fixed_num_batches} "
            f"(data-derived batches/epoch={num_it_data}).",
            flush=True,
        )
    # Optax schedule (per-step). Default: cosine one-cycle.
    lr_schedule = str(tr_cfg.get("lr_schedule", "onecycle_cosine")).strip().lower()
    onecycle_pct_start = float(tr_cfg.get("onecycle_pct_start", 0.3))
    onecycle_div_factor = float(tr_cfg.get("onecycle_div_factor", 25.0))
    onecycle_final_div_factor = float(tr_cfg.get("onecycle_final_div_factor", 10000.0))
    total_steps = int(max(1, epochs) * max(1, num_it))

    if lr_schedule in ("onecycle_cosine", "cosine_onecycle", "onecycle"):
        schedule_fn = optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=float(lr),
            pct_start=float(onecycle_pct_start),
            div_factor=float(onecycle_div_factor),
            final_div_factor=float(onecycle_final_div_factor),
        )
    elif lr_schedule in ("onecycle_linear", "linear_onecycle"):
        schedule_fn = optax.linear_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=float(lr),
            pct_start=float(onecycle_pct_start),
            div_factor=float(onecycle_div_factor),
            final_div_factor=float(onecycle_final_div_factor),
        )
    elif lr_schedule in ("cosine_decay", "cosine"):
        # Standard cosine decay from lr -> lr*alpha over total_steps.
        try:
            cosine_alpha = float(tr_cfg.get("cosine_alpha", 0.01) or 0.01)
        except Exception:
            cosine_alpha = 0.01
        cosine_alpha = max(0.0, min(1.0, cosine_alpha))
        schedule_fn = optax.cosine_decay_schedule(
            init_value=float(lr),
            decay_steps=total_steps,
            alpha=float(cosine_alpha),
        )
    else:
        schedule_fn = lambda step: jnp.asarray(float(lr), dtype=backend.dtype)  # type: ignore[no-any-return]

    print(
        f"LR schedule: {lr_schedule} (lr={float(lr):.6f}, total_steps={total_steps}, pct_start={onecycle_pct_start})",
        flush=True,
    )
    compile_t0 = time.time()

    # If you're already enforcing 50/50 batches, focal loss tends to collapse to its
    # logit≈0 fixed point. Default to plain BCE unless explicitly running imbalanced batches.
    focal_gamma_cfg = float(tr_cfg.get("focal_gamma", 0.0) or 0.0)
    if bool(tr_cfg.get("balanced_batches", True)) and abs(float(tr_cfg.get("balanced_pos_frac", 0.5) or 0.5) - 0.5) < 1e-9:
        if focal_gamma_cfg > 0.0:
            print("[INFO] balanced_batches enabled with pos_frac=0.5; overriding focal_gamma -> 0.0", flush=True)
        focal_gamma_cfg = 0.0

    compiled = get_compiled_core(
        num_qubits,
        num_layers,
        backend,
        batch_size=batch_size,
        feature_dim=feature_dim,
        encoder_name=str(enc_name),
        ansatz_name=str(anz_name),
        measurement_name=str(meas_name),
        measurement_wires=tuple(int(w) for w in meas_wires),
        hadamard=bool(enc_cfg.get("hadamard", False)),
        reupload=bool(enc_cfg.get("reupload", False)),
        focal_gamma=float(focal_gamma_cfg),
        alpha_mode=str(tr_cfg.get("alpha_mode", "softplus")),
    )
    batched_forward = compiled["batched_forward"]
    batch_loss_and_grad = compiled["batch_loss_and_grad"]
    readout_dim = int(compiled.get("readout_dim", 1))
    readout_dim = max(1, readout_dim)
    weights = weights_init
    bias = bias_init
    alpha_mode = str(tr_cfg.get("alpha_mode", "softplus") or "softplus").strip().lower()
    if alpha_mode not in ("direct", "softplus"):
        alpha_mode = "softplus"
    # Alpha parameter:
    # - direct: alpha = alpha_param
    # - softplus: alpha = softplus(alpha_param) + 1e-3
    if alpha_mode == "direct":
        alpha_param = jnp.asarray(1.0, dtype=backend.dtype)
    else:
        # softplus(0.5413...) ≈ 1.0
        alpha_param = jnp.asarray(0.54132485, dtype=backend.dtype)
    # Readout parameters:
    # - z_vec: Hamiltonian coefficients (NOT reliably differentiable under some Catalyst stacks),
    #          so initialize non-zero to avoid constant logits if gradients are dead.
    # - mean_z_readout: gate parameters (Rot per wire), so initialize to 0 for identity readout.
    if str(meas_name) == "mean_z_readout":
        w_ro = jnp.zeros((readout_dim,), dtype=backend.dtype)
    else:
        w_ro = jnp.asarray(np.ones((readout_dim,), dtype=np.float32) / float(readout_dim), dtype=backend.dtype)
    assert_jax_array("w_ro_init", w_ro, backend.dtype)
    assert_jax_array("alpha_param_init", alpha_param, backend.dtype)

    # Force qjit compilation with concrete arrays before entering any timed region.
    # This compiles the forward qjit path (qnode_compiled) plus any surrounding JAX tracing.
    _ = np.asarray(batched_forward(weights, w_ro, jnp.asarray(X_train_pool[:1], dtype=backend.dtype)))

    # Optax optimizer stays in Python (AdamW-style with per-step schedule).
    wd = float(tr_cfg.get("weight_decay", 0.001) or 0.001)
    wd_ro = float(tr_cfg.get("weight_decay_ro", 0.0) or 0.0)
    # Optional param-group LR multipliers (applied to updates after the base optimizer step).
    lr_mult_w_ro = float(tr_cfg.get("lr_mult_w_ro", 1.0) or 1.0)
    lr_mult_alpha = float(tr_cfg.get("lr_mult_alpha", 1.0) or 1.0)
    alpha_train = bool(tr_cfg.get("alpha_train", True))
    if not alpha_train:
        lr_mult_alpha = 0.0

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        # Decoupled weight decay on quantum weights only (not readout vector, not bias).
        optax.add_decayed_weights(wd, mask=(True, False, False, False)) if wd > 0.0 else optax.identity(),
        optax.add_decayed_weights(wd_ro, mask=(False, True, False, False)) if wd_ro > 0.0 else optax.identity(),
        optax.scale_by_schedule(schedule_fn),
        optax.scale(-1.0),
    )
    opt_state = tx.init((weights, w_ro, bias, alpha_param))

    # Preflight compile the per-batch loss/grad kernel so compile time is not charged to epoch 1.
    preflight_compile = bool(tr_cfg.get("preflight_compile", True))
    if preflight_compile:
        try:
            _warm_idx = np.arange(batch_size, dtype=np.int32) % max(1, len(X_train_pool))
            Xb_w = jnp.asarray(X_train_pool[_warm_idx], dtype=backend.dtype)
            yb_w = jnp.asarray(Y_train01_pool[_warm_idx], dtype=backend.dtype)
            wb_w = jnp.asarray(w_train_pool[_warm_idx], dtype=backend.dtype)
            _ls, _gwq, _gwro, _gb, _ga = batch_loss_and_grad(weights, w_ro, bias, alpha_param, Xb_w, yb_w, wb_w)
            _ls.block_until_ready()
        except Exception as _exc:
            print(f"[WARN] Preflight compile failed; continuing without it: {_exc}")

    compile_time_s = time.time() - compile_t0
    print(
        " | ".join(
            [
                f"Compile done in {compile_time_s:.2f}s",
                f"iters/epoch={num_it}",
                f"data_batches/epoch={num_it_data}",
                f"batch_size={batch_size}",
            ]
        ),
        flush=True,
    )
    start_time = time.time()
    total_iters = 0
    X_val_j = jnp.asarray(X_val_scaled, dtype=backend.dtype)
    best_epoch = 0
    val_objective = str(tr_cfg.get("val_objective", "bacc_then_prec")).strip().lower()
    if val_objective not in ("auc", "sep", "bacc", "f1", "bacc_then_prec", "prec_at_bacc"):
        val_objective = "bacc_then_prec"
    try:
        min_bacc = float(tr_cfg.get("min_bacc", 0.8) or 0.8)
    except Exception:
        min_bacc = 0.8
    min_bacc = max(0.0, min(1.0, min_bacc))

    best_val_obj = -1.0
    best_threshold = 0.0
    best_params = (weights, w_ro, bias, alpha_param)
    best_vm: Optional[Dict[str, Any]] = None
    best_any_bacc = -1.0
    best_any_threshold = 0.0
    best_any_epoch = 0
    best_any_params = (weights, w_ro, bias, alpha_param)
    best_any_vm: Optional[Dict[str, Any]] = None
    epoch_eval_history: List[Dict[str, Any]] = []  # val-only
    epoch_train_history: List[Dict[str, Any]] = []  # train-only

    # Early stopping on the validation objective (threshold-free if val_objective is 'auc'/'sep').
    try:
        early_stop_patience = int(tr_cfg.get("early_stop_patience", 0) or 0)
    except Exception:
        early_stop_patience = 0
    try:
        early_stop_min_delta = float(tr_cfg.get("early_stop_min_delta", 0.0) or 0.0)
    except Exception:
        early_stop_min_delta = 0.0
    early_stop_patience = max(0, early_stop_patience)
    early_stop_min_delta = max(0.0, early_stop_min_delta)
    no_improve = 0

    eval_every_epochs = 1
    try:
        eval_every_epochs = int(tr_cfg.get("eval_every_epochs", 1) or 1)
    except Exception:
        eval_every_epochs = 1
    eval_every_epochs = max(1, eval_every_epochs)

    try:
        min_pred_pos_rate = float(tr_cfg.get("min_pred_pos_rate", 0.01) or 0.01)
    except Exception:
        min_pred_pos_rate = 0.01
    try:
        max_pred_pos_rate = float(tr_cfg.get("max_pred_pos_rate", 0.99) or 0.99)
    except Exception:
        max_pred_pos_rate = 0.99
    min_pred_pos_rate = max(0.0, min(0.49, min_pred_pos_rate))
    max_pred_pos_rate = max(0.51, min(1.0, max_pred_pos_rate))
    abort_on_degen = bool(tr_cfg.get("abort_on_degen", True))

    def _best_threshold_by_metric(
        y_signed: np.ndarray, scores: np.ndarray, metric: str
    ) -> Tuple[float, float]:
        """Pick threshold via bounded ROC candidates, with small-grid fallback."""
        y_signed = np.asarray(y_signed)
        scores = np.asarray(scores, dtype=np.float64)
        if len(np.unique(y_signed)) < 2:
            return 0.0, float("nan")

        y01 = (y_signed > 0).astype(np.int32)
        try:
            _, _, thr = roc_curve(y01, scores)
            thr = np.asarray(thr, dtype=np.float64)
            thr = np.unique(thr[np.isfinite(thr)])
            # Keep threshold search bounded for stable eval-time overhead.
            if thr.size > 129:
                q = np.linspace(0.0, 1.0, num=129, dtype=np.float64)
                thr = np.unique(np.quantile(thr, q))
            if thr.size == 0:
                raise ValueError("Empty threshold set from ROC.")
        except Exception:
            lo = float(np.nanmin(scores))
            hi = float(np.nanmax(scores))
            if not np.isfinite(lo) or not np.isfinite(hi):
                return 0.0, float("nan")
            if hi <= lo:
                pred_fixed = np.where(scores >= lo, 1, -1)
                return lo, float(balanced_accuracy_score(y_signed, pred_fixed))
            thr = np.linspace(lo, hi, num=65, dtype=np.float64)

        pos_rate = float(np.mean(y01)) if y01.size else float("nan")
        found_valid = False
        best_t = 0.0
        best_v = -1.0
        best_prec = -1.0
        best_bacc = -1.0
        constrained_found = False
        for t in thr:
            preds_lab = np.where(scores >= t, 1, -1)
            pred01 = (preds_lab > 0).astype(np.int32)
            pred_pos_rate = float(np.mean(pred01)) if pred01.size else float("nan")
            if not np.isfinite(pred_pos_rate):
                continue
            if pred_pos_rate < min_pred_pos_rate or pred_pos_rate > max_pred_pos_rate:
                continue
            found_valid = True
            bacc_t = float(balanced_accuracy_score(y_signed, preds_lab))
            prec_t = float(precision_score(y01, pred01, zero_division=0))

            if metric == "f1":
                v = float(f1_score(y01, pred01, zero_division=0))
                if v > best_v:
                    best_v = v
                    best_t = float(t)
            elif metric == "prec_at_bacc":
                if bacc_t >= min_bacc:
                    constrained_found = True
                    # Primary: maximize precision; tie-break by bacc.
                    if prec_t > best_prec or (prec_t == best_prec and bacc_t > best_bacc):
                        best_prec = prec_t
                        best_bacc = bacc_t
                        best_t = float(t)
            elif metric == "bacc_then_prec":
                # Lexicographic: maximize bacc, tie-break by precision.
                if bacc_t > best_bacc or (bacc_t == best_bacc and prec_t > best_prec):
                    best_bacc = bacc_t
                    best_prec = prec_t
                    best_t = float(t)
                    best_v = bacc_t
            else:
                # "bacc"
                v = bacc_t
                if v > best_v:
                    best_v = v
                    best_t = float(t)
        if not found_valid:
            # deterministic fallback: choose threshold whose pred_pos_rate is closest to true pos_rate
            best_t = float(thr[0]) if len(thr) else 0.0
            best_gap = float("inf")
            for t in thr:
                pred01 = (scores >= t).astype(np.int32)
                ppr = float(np.mean(pred01)) if pred01.size else float("nan")
                if not np.isfinite(ppr) or not np.isfinite(pos_rate):
                    continue
                gap = abs(ppr - pos_rate)
                if gap < best_gap:
                    best_gap = gap
                    best_t = float(t)
            return best_t, float("nan")

        if metric == "prec_at_bacc":
            if constrained_found:
                return best_t, best_prec
            # Fallback: maximize bacc, tie-break by precision.
            best_t = 0.0
            best_bacc = -1.0
            best_prec = -1.0
            for t in thr:
                preds_lab = np.where(scores >= t, 1, -1)
                pred01 = (preds_lab > 0).astype(np.int32)
                bacc_t = float(balanced_accuracy_score(y_signed, preds_lab))
                prec_t = float(precision_score(y01, pred01, zero_division=0))
                if bacc_t > best_bacc or (bacc_t == best_bacc and prec_t > best_prec):
                    best_bacc = bacc_t
                    best_prec = prec_t
                    best_t = float(t)
            return best_t, best_bacc
        return best_t, best_v

    def _alpha_from_param(alpha_param_cur: Any) -> float:
        if alpha_mode == "direct":
            return float(np.asarray(alpha_param_cur, dtype=np.float64))
        # Keep eval in sync with compiled loss: alpha = softplus(alpha_param) + eps.
        a = float(np.asarray(jax.nn.softplus(np.asarray(alpha_param_cur, dtype=np.float64)) + 1e-3))
        return a

    def _signed_scores_from_ev(ev: np.ndarray, bias_cur: Any, alpha_param_cur: Any) -> np.ndarray:
        # Canonical: higher => more positive (attack).
        raw = np.asarray(ev, dtype=np.float64) + float(np.asarray(bias_cur))
        return float(_alpha_from_param(alpha_param_cur)) * raw

    def _scores_range_ok(scores: np.ndarray, eps: float = 1e-12) -> bool:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            return False
        return float(np.nanmax(scores) - np.nanmin(scores)) > float(eps)

    def _eval_scores(weights_cur: Any, w_ro_cur: Any, bias_cur: Any, alpha_param_cur: Any, X_j: Any) -> np.ndarray:
        """Canonical signed scores: higher => y=1."""
        ev = np.array(batched_forward(weights_cur, w_ro_cur, X_j))  # (N,)
        return _signed_scores_from_ev(ev, bias_cur, alpha_param_cur)

    def _val_metrics(weights_cur: Any, w_ro_cur: Any, bias_cur: Any, alpha_param_cur: Any) -> Dict[str, Any]:
        scores_val = _eval_scores(weights_cur, w_ro_cur, bias_cur, alpha_param_cur, X_val_j)
        y01 = np.asarray(Y_val_hold01).astype(np.int32, copy=False)
        ysgn = np.asarray(Y_val_hold)

        thr, objv = _best_threshold_by_metric(ysgn, scores_val, val_objective)
        pos_rate = float(np.mean(y01)) if y01.size else float("nan")
        degen = 0
        degen_reason = ""
        if not _scores_range_ok(scores_val):
            degen = 1
            degen_reason = "scores_constant"
            # Fallback threshold in logit space: 0.0 => sigmoid(logit)=0.5
            thr = 0.0
            objv = float("nan")
        pred_signed = np.where(scores_val >= thr, 1, -1)
        pred01 = (pred_signed > 0).astype(np.int32)
        pred_pos_rate = float(np.mean(pred01)) if pred01.size else float("nan")
        if np.isfinite(pred_pos_rate) and (pred_pos_rate < min_pred_pos_rate or pred_pos_rate > max_pred_pos_rate):
            degen = 1
            degen_reason = degen_reason or "pred_pos_rate_extreme"
            thr = 0.0
            pred_signed = np.where(scores_val >= thr, 1, -1)
            pred01 = (pred_signed > 0).astype(np.int32)
            pred_pos_rate = float(np.mean(pred01)) if pred01.size else float("nan")
            objv = float("nan")
        try:
            # TPR/TNR under the chosen threshold (helps debug bacc=0.5 + high F1).
            tp = float(np.sum((y01 == 1) & (pred01 == 1)))
            fn = float(np.sum((y01 == 1) & (pred01 == 0)))
            tn = float(np.sum((y01 == 0) & (pred01 == 0)))
            fp = float(np.sum((y01 == 0) & (pred01 == 1)))
            tpr = tp / max(tp + fn, 1.0)
            tnr = tn / max(tn + fp, 1.0)
        except Exception:
            tpr = float("nan")
            tnr = float("nan")
        try:
            prec = float(precision_score(y01, pred01, zero_division=0))
            rec = float(recall_score(y01, pred01, zero_division=0))
        except Exception:
            prec = float("nan")
            rec = float("nan")

        try:
            auc = float(roc_auc_score(y01, scores_val)) if len(np.unique(y01)) >= 2 else float("nan")
        except Exception:
            auc = float("nan")
        if abort_on_degen and not np.isfinite(auc):
            raise RuntimeError("[DEGEN] validation AUC is non-finite (check score computation for NaN/inf).")
        try:
            f1v = float(f1_score(y01, pred01, zero_division=0))
        except Exception:
            f1v = float("nan")
        try:
            bacc = float(balanced_accuracy_score(ysgn, pred_signed)) if len(np.unique(ysgn)) >= 2 else float("nan")
        except Exception:
            bacc = float("nan")
        constraint_met = bool(np.isfinite(bacc) and bacc >= min_bacc)

        mean_pos = float(np.mean(scores_val[y01 == 1])) if np.any(y01 == 1) else float("nan")
        mean_neg = float(np.mean(scores_val[y01 == 0])) if np.any(y01 == 0) else float("nan")
        sep = float(mean_pos - mean_neg) if np.isfinite(mean_pos) and np.isfinite(mean_neg) else float("nan")

        # Threshold-free validation loss (BCE-with-logits).
        try:
            y01f = y01.astype(np.float64, copy=False)
            s = np.asarray(scores_val, dtype=np.float64)
            ce = np.logaddexp(0.0, s) - y01f * s
            val_loss = float(np.mean(ce))
        except Exception:
            val_loss = float("nan")

        # If selecting by a threshold-free metric, override objective accordingly.
        if val_objective == "auc":
            objv = float(auc)
        elif val_objective == "sep":
            objv = float(sep)

        return {
            "scores": scores_val,
            "threshold": float(thr),
            "bacc": float(bacc),
            "auc": float(auc),
            "f1": float(f1v),
            "val_loss": float(val_loss),
            "objective": float(objv),
            "objective_name": val_objective,
            "constraint_met": int(constraint_met),
            "pos_rate": pos_rate,
            "pred_pos_rate": pred_pos_rate,
            "precision": float(prec),
            "recall": float(rec),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "score_mean_pos": mean_pos,
            "score_mean_neg": mean_neg,
            "sep": sep,
            "degen": int(degen),
            "degen_reason": str(degen_reason),
        }

    # NOTE: score polarity is learned continuously via the trainable scalar `alpha`.
    cache_key = (
        int(seed),
        int(len(X_train_pool)),
        int(epochs),
        int(num_it),
        int(batch_size),
        int(bool(tr_cfg.get("balanced_batches", True))),
        float(tr_cfg.get("balanced_pos_frac", 0.5)),
    )
    idx_steps_all = _BATCH_INDEX_CACHE.get(cache_key)
    if idx_steps_all is None:
        idx_steps_all = np.empty((epochs, num_it, batch_size), dtype=np.int32)
        balanced_batches = bool(tr_cfg.get("balanced_batches", True))
        try:
            pos_frac = float(tr_cfg.get("balanced_pos_frac", 0.5))
        except Exception:
            pos_frac = 0.5
        pos_frac = max(0.0, min(1.0, pos_frac))

        y_pool01 = np.asarray(Y_train01_pool).astype(np.int32, copy=False)
        pos_idx_all = np.flatnonzero(y_pool01 == 1).astype(np.int32, copy=False)
        neg_idx_all = np.flatnonzero(y_pool01 == 0).astype(np.int32, copy=False)

        if balanced_batches and pos_idx_all.size > 0 and neg_idx_all.size > 0:
            pos_n = int(round(float(batch_size) * float(pos_frac)))
            pos_n = max(1, min(batch_size - 1, pos_n))
            neg_n = int(batch_size - pos_n)

            for ep in range(epochs):
                rng_ep = np.random.default_rng(int(seed) + 10007 * int(ep + 1))
                pos_perm = rng_ep.permutation(pos_idx_all)
                neg_perm = rng_ep.permutation(neg_idx_all)
                ppos = 0
                pneg = 0
                for it in range(num_it):
                    if ppos + pos_n > pos_perm.size:
                        pos_perm = rng_ep.permutation(pos_idx_all)
                        ppos = 0
                    if pneg + neg_n > neg_perm.size:
                        neg_perm = rng_ep.permutation(neg_idx_all)
                        pneg = 0
                    bpos = pos_perm[ppos : ppos + pos_n]
                    bneg = neg_perm[pneg : pneg + neg_n]
                    ppos += pos_n
                    pneg += neg_n
                    b = np.concatenate([bpos, bneg]).astype(np.int32, copy=False)
                    rng_ep.shuffle(b)
                    idx_steps_all[ep, it, :] = b
            print(
                f"[BATCH] balanced sampling enabled | pos_frac={pos_frac:.2f} "
                f"(pos={pos_n}, neg={neg_n}) | pool_pos={int(pos_idx_all.size)} pool_neg={int(neg_idx_all.size)}",
                flush=True,
            )
        else:
            # Fallback: plain shuffled sampling (handles degenerate single-class pools).
            for ep in range(epochs):
                perm = np_rng.permutation(len(X_train_pool)).astype(np.int32, copy=False)
                needed = int(num_it * batch_size)
                if needed <= len(perm):
                    ep_idx = perm[:needed]
                else:
                    # Pad with repeated shuffled indices to preserve static tensor shapes.
                    reps = int(np.ceil(float(needed) / float(len(perm))))
                    ep_idx = np.tile(perm, reps)[:needed]
                idx_steps_all[ep] = ep_idx.reshape(num_it, batch_size)
        _BATCH_INDEX_CACHE[cache_key] = idx_steps_all

    print(
        f"Per-batch training | epochs={epochs} | iters/epoch={num_it} | batch_size={batch_size} | "
        f"balanced_batches={int(bool(tr_cfg.get('balanced_batches', True)))}",
        flush=True,
    )

    for ep in range(epochs):
        ep_start = time.time()
        progress_every = max(1, int(num_it // 5))  # ~5 progress lines/epoch, even for long runs
        print(f"[TRAIN] epoch={ep+1}/{epochs} iters={num_it} starting...", flush=True)
        idx_steps = idx_steps_all[ep]  # (num_it, B)
        X_epoch = jnp.asarray(X_train_pool[idx_steps], dtype=backend.dtype)  # (num_it, B, D)
        Y_epoch = jnp.asarray(Y_train01_pool[idx_steps], dtype=backend.dtype)  # (num_it, B)
        W_epoch = jnp.asarray(w_train_pool[idx_steps], dtype=backend.dtype)  # (num_it, B)

        loss_sum = jnp.asarray(0.0, dtype=backend.dtype)
        loss_sq_sum = jnp.asarray(0.0, dtype=backend.dtype)
        last_gw = None
        last_gwro = None
        last_gb = None
        last_ga = None
        last_lr = float("nan")
        w_ro_start = w_ro

        for it in range(num_it):
            # Train on the SAME score used in eval: alpha * (evs@w_ro + bias).
            loss, gwq, gwro, gb, ga = batch_loss_and_grad(
                weights,
                w_ro,
                bias,
                alpha_param,
                X_epoch[it],
                Y_epoch[it],
                W_epoch[it],
            )

            grads = (gwq, gwro, gb, ga)
            params = (weights, w_ro, bias, alpha_param)
            updates, opt_state = tx.update(grads, opt_state, params)
            # Apply optional param-group LR multipliers without breaking qjit constraints.
            up_w, up_ro, up_b, up_a = updates
            if lr_mult_w_ro != 1.0:
                up_ro = jnp.asarray(lr_mult_w_ro, dtype=backend.dtype) * up_ro
            if lr_mult_alpha != 1.0:
                up_a = jnp.asarray(lr_mult_alpha, dtype=backend.dtype) * up_a
            updates = (up_w, up_ro, up_b, up_a)
            weights, w_ro, bias, alpha_param = optax.apply_updates(params, updates)
            try:
                last_lr = float(np.asarray(schedule_fn(total_iters)))
            except Exception:
                last_lr = float("nan")

            loss_sum = loss_sum + loss
            loss_sq_sum = loss_sq_sum + loss * loss
            last_gw, last_gwro, last_gb, last_ga = gwq, gwro, gb, ga
            total_iters += 1

            # Progress logging without spamming (and without syncing every batch).
            if it == 0 or (it + 1) == num_it or ((it + 1) % progress_every) == 0:
                loss.block_until_ready()
                it_s = time.time() - ep_start
                iters_per_s_live = float((it + 1) / it_s) if it_s > 0 else float("nan")
                try:
                    loss_live = float(np.asarray(loss))
                except Exception:
                    loss_live = float("nan")
                print(
                    f"[TRAIN] epoch={ep+1}/{epochs} it={it+1}/{num_it} "
                    f"loss={loss_live:0.6f} ({iters_per_s_live:0.2f} it/s)",
                    flush=True,
                )

        mean_loss = float((loss_sum / float(max(1, num_it))).block_until_ready())
        mean_sq = float((loss_sq_sum / float(max(1, num_it))).block_until_ready())
        var = max(0.0, mean_sq - mean_loss * mean_loss)
        std_loss = float(math.sqrt(var))
        ep_s = time.time() - ep_start
        iters_per_s_ep = float(num_it / ep_s) if ep_s > 0 else float("nan")

        try:
            param_norm_w = float(jnp.linalg.norm(weights).block_until_ready())
        except Exception:
            param_norm_w = float("nan")
        try:
            param_norm_w_ro = float(jnp.linalg.norm(w_ro).block_until_ready())
        except Exception:
            param_norm_w_ro = float("nan")
        try:
            grad_norm_w = float(jnp.linalg.norm(last_gw).block_until_ready()) if last_gw is not None else float("nan")
        except Exception:
            grad_norm_w = float("nan")
        try:
            grad_norm_head = float(
                jnp.abs(last_gb).block_until_ready()
            ) if (last_gb is not None) else float("nan")
        except Exception:
            grad_norm_head = float("nan")
        try:
            grad_norm_alpha = float(jnp.abs(last_ga).block_until_ready()) if (last_ga is not None) else float("nan")
        except Exception:
            grad_norm_alpha = float("nan")
        try:
            grad_norm_wro = float(jnp.linalg.norm(last_gwro).block_until_ready()) if (last_gwro is not None) else float("nan")
        except Exception:
            grad_norm_wro = float("nan")

        alpha_live = _alpha_from_param(alpha_param)
        try:
            delta_w_ro = float(jnp.linalg.norm(w_ro - w_ro_start).block_until_ready())
        except Exception:
            delta_w_ro = float("nan")

        print(
            f"Epoch {ep+1}/{epochs} done | lr={last_lr:.6f} | loss_mean={mean_loss:0.7f} | "
            f"loss_std={std_loss:0.7f} | bias={float(np.asarray(bias)):.4f} | alpha={alpha_live:.4f} | "
            f"Time: {ep_s:.2f}s ({iters_per_s_ep:0.2f} it/s)",
            flush=True,
        )

        _wandb_log(
            {
                "epoch": int(ep + 1),
                "train/weight_decay": float(wd),
                "train/weight_decay_ro": float(wd_ro),
                "train/lr_mult_w_ro": float(lr_mult_w_ro),
                "train/lr_mult_alpha": float(lr_mult_alpha),
                "train/alpha_train": int(bool(alpha_train)),
                "train/loss_mean": float(mean_loss),
                "train/loss_std": float(std_loss),
                "train/bias": float(np.asarray(bias)),
                "train/alpha": float(alpha_live),
                "train/param_norm_w": float(param_norm_w),
                "train/param_norm_w_ro": float(param_norm_w_ro),
                "train/delta_w_ro": float(delta_w_ro),
                "train/grad_norm_w": float(grad_norm_w),
                "train/grad_norm_w_ro": float(grad_norm_wro),
                "train/grad_norm_head": float(grad_norm_head),
                "train/grad_norm_alpha": float(grad_norm_alpha),
                "time/epoch_s": float(ep_s),
                "time/iters_per_s": float(iters_per_s_ep),
            }
        )
        epoch_train_history.append(
            {
                "epoch": int(ep + 1),
                "train/loss_mean": float(mean_loss),
                "train/loss_std": float(std_loss),
                "train/bias": float(np.asarray(bias)),
                "train/alpha": float(alpha_live),
                "train/param_norm_w": float(param_norm_w),
                "train/param_norm_w_ro": float(param_norm_w_ro),
                "train/delta_w_ro": float(delta_w_ro),
                "train/grad_norm_w": float(grad_norm_w),
                "train/grad_norm_w_ro": float(grad_norm_wro),
                "train/grad_norm_head": float(grad_norm_head),
                "train/grad_norm_alpha": float(grad_norm_alpha),
            }
        )

        if ((ep + 1) % eval_every_epochs) == 0 or (ep + 1) == epochs:
            vm = _val_metrics(weights, w_ro, bias, alpha_param)
            print(
                f"[VAL] epoch={ep+1}/{epochs} "
                f"auc={float(vm['auc']):.4f} "
                f"bacc={float(vm['bacc']):.4f} "
                f"prec={float(vm['precision']):.4f} "
                f"f1={float(vm['f1']):.4f} "
                f"thr={float(vm['threshold']):.6f} "
                f"obj({str(vm['objective_name'])})={float(vm['objective']):.4f} "
                f"pos_rate={float(vm['pos_rate']):.4f} "
                f"pred_pos_rate={float(vm['pred_pos_rate']):.4f} "
                f"degen={int(vm.get('degen', 0))} "
                f"tpr={float(vm['tpr']):.4f} "
                f"tnr={float(vm['tnr']):.4f} "
                f"sep={float(vm['sep']):.6f} "
                f"mean_pos={float(vm['score_mean_pos']):.6f} "
                f"mean_neg={float(vm['score_mean_neg']):.6f}",
                flush=True,
            )
            epoch_eval_history.append(
                {
                    "epoch": int(ep + 1),
                    "val/auc": float(vm["auc"]),
                    "val/bacc": float(vm["bacc"]),
                    "val/precision": float(vm["precision"]),
                    "val/recall": float(vm["recall"]),
                    "val/f1": float(vm["f1"]),
                    "val/loss": float(vm.get("val_loss", float("nan"))),
                    "val/threshold": float(vm["threshold"]),
                    "val/objective": float(vm["objective"]),
                    "val/objective_name": str(vm["objective_name"]),
                    "val/score_mean_pos": float(vm["score_mean_pos"]),
                    "val/score_mean_neg": float(vm["score_mean_neg"]),
                    "val/sep": float(vm["sep"]),
                }
            )

            payload = {
                "epoch": int(ep + 1),
                "val/auc": float(vm["auc"]),
                "val/bacc": float(vm["bacc"]),
                "val/precision": float(vm["precision"]),
                "val/recall": float(vm["recall"]),
                "val/f1": float(vm["f1"]),
                "val/threshold": float(vm["threshold"]),
                "val/objective": float(vm["objective"]),
                "val/constraint_met": int(vm["constraint_met"]),
                "val/pos_rate": float(vm["pos_rate"]),
                "val/pred_pos_rate": float(vm["pred_pos_rate"]),
                "val/tpr": float(vm["tpr"]),
                "val/tnr": float(vm["tnr"]),
                "val/score_mean_pos": float(vm["score_mean_pos"]),
                "val/score_mean_neg": float(vm["score_mean_neg"]),
                "val/sep": float(vm["sep"]),
            }
            if _wandb_can_log:
                try:
                    payload["val/score_hist"] = _wandb.Histogram(vm["scores"])  # type: ignore[union-attr]
                except Exception:
                    pass
            _wandb_log(payload)

            # Track best unconstrained epoch by bacc_then_prec, always.
            cur_bacc = float(vm["bacc"])
            cur_prec = float(vm["precision"])
            if np.isfinite(cur_bacc):
                if (cur_bacc > best_any_bacc) or (cur_bacc == best_any_bacc and cur_prec > float(best_any_vm.get("precision", -1.0)) if best_any_vm else True):
                    best_any_bacc = cur_bacc
                    best_any_epoch = int(ep + 1)
                    best_any_threshold = float(vm["threshold"])
                    best_any_params = (weights, w_ro, bias, alpha_param)
                    best_any_vm = dict(vm)

            # Track best constrained epoch under requested objective.
            cur_obj = float(vm["objective"])
            if val_objective == "prec_at_bacc":
                if int(vm.get("constraint_met", 0)) != 1:
                    cur_obj = float("nan")
                else:
                    cur_obj = float(vm["precision"])

            improved = bool(np.isfinite(cur_obj) and (cur_obj > (best_val_obj + early_stop_min_delta)))
            if improved:
                best_val_obj = cur_obj
                best_epoch = int(ep + 1)
                best_threshold = float(vm["threshold"])
                best_params = (weights, w_ro, bias, alpha_param)
                best_vm = dict(vm)
                no_improve = 0
            else:
                no_improve += 1

            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(
                    f"[EARLY-STOP] epoch={ep+1} no_improve={no_improve} patience={early_stop_patience} "
                    f"(best_epoch={best_epoch}, best_{val_objective}={best_val_obj:.4f})",
                    flush=True,
                )
                break

    train_time_s = time.time() - start_time
    iters_per_s = float(total_iters / train_time_s) if train_time_s > 0 else float("nan")
    print(
        f"Training finished in {train_time_s:.2f}s over {epochs} epoch(s), {total_iters} iters. "
        f"Iters/sec: {iters_per_s:0.2f}"
    )

    # Persist a full val history artifact to W&B (curves already exist via val/* scalars,
    # but a table is convenient for inspection/export).
    if _wandb_can_log and epoch_eval_history:
        try:
            cols = [
                "epoch",
                "val/auc",
                "val/bacc",
                "val/f1",
                "val/threshold",
                "val/sep",
                "val/score_mean_pos",
                "val/score_mean_neg",
            ]
            tbl = _wandb.Table(columns=cols)  # type: ignore[union-attr]
            for r in epoch_eval_history:
                tbl.add_data(
                    int(r.get("epoch", 0)),
                    float(r.get("val/auc", float("nan"))),
                    float(r.get("val/bacc", float("nan"))),
                    float(r.get("val/f1", float("nan"))),
                    float(r.get("val/threshold", float("nan"))),
                    float(r.get("val/sep", float("nan"))),
                    float(r.get("val/score_mean_pos", float("nan"))),
                    float(r.get("val/score_mean_neg", float("nan"))),
                )
            _wandb_log({"epoch": int(epochs), "val/epoch_eval_history": tbl})
            try:
                # Keep a JSON copy in summary for quick grep (may be truncated by W&B UI).
                _wandb.run.summary["val/epoch_eval_history_json"] = epoch_eval_history  # type: ignore[union-attr]
            except Exception:
                pass
        except Exception:
            pass

    # Restore best params (val-selected), evaluate on test exactly once.
    used_constraint = True
    if best_epoch <= 0:
        # No epoch satisfied the constraint (e.g. EDGE_MIN_BACC) or objective was non-finite.
        used_constraint = False
        best_epoch = int(best_any_epoch)
        best_threshold = float(best_any_threshold)
        best_params = best_any_params
        best_vm = best_any_vm
    weights, w_ro, bias, alpha_param = best_params

    if used_constraint:
        print(
            f"[SELECT] best_epoch={best_epoch} val_{val_objective}={best_val_obj:.4f} val_thr={best_threshold:.6f}",
            flush=True,
        )
    else:
        print(
            f"[SELECT] best_epoch={best_epoch} (no epoch met min_bacc={min_bacc:.2f}; fell back to best bacc) "
            f"val_bacc={float(best_any_bacc):.4f} val_thr={best_threshold:.6f}",
            flush=True,
        )

    X_test_j = jnp.asarray(X_test_scaled, dtype=backend.dtype)
    scores_test = _eval_scores(weights, w_ro, bias, alpha_param, X_test_j)
    pred_signed = np.where(scores_test >= best_threshold, 1, -1)
    pred01 = (pred_signed > 0).astype(np.int32)
    y_test01 = np.asarray(Y_test01).astype(np.int32, copy=False)

    acc = float(accuracy_score(Y_test, pred_signed))
    try:
        prec = float(precision_score(y_test01, pred01, zero_division=0))
        rec = float(recall_score(y_test01, pred01, zero_division=0))
        f1 = float(f1_score(y_test01, pred01, zero_division=0))
    except Exception:
        prec = float("nan")
        rec = float("nan")
        f1 = float("nan")
    try:
        bacc = float(balanced_accuracy_score(Y_test, pred_signed)) if len(np.unique(Y_test)) >= 2 else float("nan")
    except Exception:
        bacc = float("nan")
    try:
        auc = float(roc_auc_score(y_test01, scores_test)) if len(np.unique(y_test01)) >= 2 else float("nan")
    except Exception:
        auc = float("nan")
    print("--- Test Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Balanced Acc: {bacc:.4f}")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"ROC AUC: {auc:.4f}")
    print("--------------------")
    print("Experiment complete.")

    payload = {
        "epoch": int(best_epoch if best_epoch > 0 else epochs),
        "test/accuracy": float(acc),
        "test/precision": float(prec),
        "test/recall": float(rec),
        "test/f1": float(f1),
        "test/bacc": float(bacc),
        "test/auc": float(auc),
        "test/threshold": float(best_threshold),
    }
    if _wandb_can_log:
        try:
            payload["test/score_hist"] = _wandb.Histogram(scores_test)  # type: ignore[union-attr]
        except Exception:
            pass
    _wandb_log(payload)

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
            requires_preprocess=bool(not input_is_preprocessed),
            scaler=scaler,
            quantile=qt,
            pls=pls,
            pca=pca,
            weights=weights,
            w_ro=w_ro,
            bias=bias,
            alpha=np.array(_alpha_from_param(alpha_param), dtype=np.float32),
            # Keep the field for backward compatibility with older model loaders.
            score_sign=np.array(1.0),
            compiled_input_scale=float(compiled_input_scale),
            compiled_input_shift=float(compiled_input_shift),
            train_cfg={"lr": lr, "batch": batch_size, "epochs": epochs},
            metrics={
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "balanced_accuracy": bacc,
                "auc": auc,
                "val_balanced_accuracy": float(best_vm.get("bacc", float("nan"))) if best_vm else float("nan"),
                "val_f1": float(best_vm.get("f1", float("nan"))) if best_vm else float("nan"),
                "val_auc": float(best_vm.get("auc", float("nan"))) if best_vm else float("nan"),
                "val_objective_name": str(val_objective),
                f"val_{val_objective}": best_val_obj,
                "threshold": best_threshold,
                "best_epoch": int(best_epoch),
            },
        )
        # Ensure the saved artifact is actually deployable via scripts/predict.py.
        # Keep it cheap: validate on a few validation rows only.
        try:
            verify_n = int(save_cfg.get("verify_rows", 8) or 8)
        except Exception:
            verify_n = 8
        verify_n = max(0, min(int(verify_n), int(len(X_val_scaled))))
        if verify_n > 0:
            raw_df = None
            try:
                if path and os.path.exists(path):
                    import pandas as _pd
                    raw_df = _pd.read_csv(path, low_memory=False).head(int(verify_n))
            except Exception:
                raw_df = None
            _verify_saved_model_portable(
                save_path,
                X_ref=np.asarray(X_val_scaled[:verify_n], dtype=np.float64),
                features=features,
                raw_df=raw_df,
                atol=float(save_cfg.get("verify_atol", 1e-6) or 1e-6),
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
        "compile_time_s": float(compile_time_s),
        "train_time_s": float(train_time_s),
        "num_batches_per_epoch": int(num_it),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_accuracy": bacc,
            "auc": auc,
            "val_balanced_accuracy": float(best_vm.get("bacc", float("nan"))) if best_vm else float("nan"),
            "val_f1": float(best_vm.get("f1", float("nan"))) if best_vm else float("nan"),
            "val_auc": float(best_vm.get("auc", float("nan"))) if best_vm else float("nan"),
            "val_objective_name": str(val_objective),
            f"val_{val_objective}": best_val_obj,
            "threshold": best_threshold,
            "best_epoch": int(best_epoch),
            "alpha": float(_alpha_from_param(alpha_param)),
        },
        "epoch_eval_history": epoch_eval_history,
        "epoch_train_history": epoch_train_history,
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
    requires_preprocess: bool,
    scaler: Any,
    quantile: Any,
    pls: Any,
    pca: Any,
    weights: Any,
    w_ro: Any,
    bias: Any,
    alpha: Any,
    score_sign: Any = 1.0,
    compiled_input_scale: float,
    compiled_input_shift: float,
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
    w_ro_np = np.array(w_ro)
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
                # sklearn>=1.8 PLSRegression.transform() uses private _x_mean/_x_std.
                "_x_mean": getattr(pls, "_x_mean", None),
                "_x_std": getattr(pls, "_x_std", None),
                # Keep public names too for forward/backward compatibility with older checkpoints.
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
        # v5: trainable scalar logit temperature/sign `alpha` (removes fixed score_sign hack).
        # `score_sign` is kept as a deprecated field for backward compatibility.
        "version": 5,
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
        # If false, inference expects already-preprocessed features (e.g. PC_1..PC_8) and
        # will not require quantile/pls/pca state to be present.
        "requires_preprocess": bool(requires_preprocess),
        # Only store safe extracted state for sklearn objects (avoid brittle pickles).
        "scaler_state": scaler_state,
        "quantile_state": quantile_state,
        "pls_state": pls_state,
        "pca_state": pca_state,
        "weights": _torch.tensor(weightsnp, dtype=_torch.float32),
        "w_ro": _torch.tensor(w_ro_np, dtype=_torch.float32),
        "bias": _torch.tensor(biasnp, dtype=_torch.float32),
        "alpha": _torch.tensor(alphanp, dtype=_torch.float32),
        "score_sign": _torch.tensor(score_sign_np, dtype=_torch.float32),
        "compiled_input_scale": float(compiled_input_scale),
        "compiled_input_shift": float(compiled_input_shift),
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

    ver = int(state.get("version") or 0)
    if ver not in (4, 5):
        raise ValueError(
            f"Unsupported model version {ver}. Re-save the model with the current pipeline (version=5)."
        )

    dev_name = device_override or state.get("device")
    if not dev_name:
        raise ValueError("Saved model is missing 'device' and no device_override was provided.")

    if "num_qubits" not in state:
        raise ValueError("Saved model is missing 'num_qubits'.")
    num_qubits = int(state["num_qubits"])
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
    meas_cfg = state.get("measurement")
    if not isinstance(meas_cfg, dict) or not meas_cfg.get("name") or "wires" not in meas_cfg:
        raise ValueError("Saved model is missing required 'measurement' config.")
    meas_name = str(meas_cfg["name"])
    meas_wires = list(meas_cfg["wires"] or [])
    if not meas_wires:
        raise ValueError("Saved model has empty measurement.wires.")

    def _circuit(weights, w_ro, x):
        reupload = bool(enc_opts.get("reupload", False))
        if reupload and anz_name == "ring_rot_cnot":
            def _reupload_layer(W):
                encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)))
                ansatz_fn(W, wires)
            qml.layer(_reupload_layer, num_layers, weights)
        else:
            encoder_fn(x, wires, hadamard=bool(enc_opts.get("hadamard", False)))
            ansatz_fn(weights, wires)
        if meas_name == "mean_z":
            if not meas_wires:
                raise ValueError("mean_z measurement requires at least one wire")
            coeffs = [1.0 / len(meas_wires)] * len(meas_wires)
            ops = [qml.PauliZ(w) for w in meas_wires]
            return qml.expval(qml.Hamiltonian(coeffs, ops))
        if meas_name == "mean_z_readout":
            # Trainable readout as a gate layer (Rot per wire), then fixed mean-Z.
            mw = [int(w) for w in meas_wires]
            bad = [w for w in mw if w < 0 or w >= num_qubits]
            if bad:
                raise ValueError(f"Saved model measurement.wires out of range for num_qubits={num_qubits}: {bad}")
            if len(w_ro) != 3 * len(mw):
                raise ValueError(
                    f"mean_z_readout expects w_ro dim=3*len(wires)={3*len(mw)}; got {len(w_ro)}"
                )
            for i, w in enumerate(mw):
                j = 3 * i
                qml.Rot(w_ro[j + 0], w_ro[j + 1], w_ro[j + 2], wires=w)
            coeffs = [1.0 / len(mw)] * len(mw)
            ops = [qml.PauliZ(w) for w in mw]
            return qml.expval(qml.Hamiltonian(coeffs, ops))
        if meas_name == "z_vec":
            mw = [int(w) for w in meas_wires]
            bad = [w for w in mw if w < 0 or w >= num_qubits]
            if bad:
                raise ValueError(f"Saved model measurement.wires out of range for num_qubits={num_qubits}: {bad}")
            ops = [qml.PauliZ(w) for w in mw]
            return qml.expval(qml.Hamiltonian(w_ro, ops))
        if meas_name == "z0":
            return qml.expval(qml.PauliZ(0))
        raise ValueError(f"Unknown measurement in saved model: {meas_name}")

    # Restore parameters and scaler
    weights_t = state["weights"].detach().cpu().numpy()
    if "w_ro" not in state:
        raise ValueError("Saved model is missing required 'w_ro' vector (version=4 models must include it).")
    w_ro_t = state["w_ro"]
    if hasattr(w_ro_t, "detach"):
        w_ro_t = w_ro_t.detach().cpu().numpy()
    else:
        w_ro_t = npy.asarray(w_ro_t)
    bias_t = state["bias"].detach().cpu().numpy().item() if state["bias"].ndim == 0 else state["bias"].detach().cpu().numpy()
    if "alpha" not in state:
        raise ValueError("Saved model is missing required 'alpha'.")
    alpha_t = state["alpha"].detach().cpu().numpy().item() if state["alpha"].ndim == 0 else state["alpha"].detach().cpu().numpy()
    weightsnp = np.array(weights_t, requires_grad=False)
    w_ro_np = np.array(w_ro_t, requires_grad=False)
    biasnp = np.array(bias_t, requires_grad=False)
    alphanp = np.array(alpha_t, requires_grad=False)
    score_sign_np = np.array(1.0, requires_grad=False)
    if "score_sign" in state:
        score_sign_t = state["score_sign"]
        if hasattr(score_sign_t, "detach"):
            score_sign_t = (
                score_sign_t.detach().cpu().numpy().item()
                if getattr(score_sign_t, "ndim", 0) == 0
                else score_sign_t.detach().cpu().numpy()
            )
        else:
            score_sign_t = npy.asarray(score_sign_t)
        score_sign_np = np.array(score_sign_t, requires_grad=False)
    elif ver == 4:
        raise ValueError("Saved model is missing required 'score_sign' (required for version=4 models).")
    # Rebuild preprocessing either from embedded object or from safe state
    # New checkpoints store only *_state; older checkpoints may have embedded objects.
    scaler = None
    if state.get("scaler_state") is not None:
        st = state["scaler_state"]
        if _SkMinMax is None:
            raise ValueError("Cannot rebuild scaler: sklearn MinMaxScaler unavailable in this environment.")
        sc = _SkMinMax(feature_range=tuple(st.get("feature_range", (0, 1))))
        for attr in ["min_", "scale_", "data_min_", "data_max_", "data_range_", "n_samples_seen_"]:
            val = st.get(attr)
            if val is not None:
                setattr(sc, attr, np.array(val))
        scaler = sc

    requires_preprocess = bool(state.get("requires_preprocess", True))
    quantile = None
    if requires_preprocess:
        if state.get("quantile_state") is None:
            raise ValueError("Saved model is missing required 'quantile_state'.")
        st = state["quantile_state"]
        if _SkQuantile is None:
            raise ValueError("Cannot rebuild quantile transformer: sklearn QuantileTransformer unavailable.")
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

    pls = None
    if state.get("pls_state") is not None:
        st = state["pls_state"]
        if _SkPLS is None:
            raise ValueError("Cannot rebuild PLS: sklearn PLSRegression unavailable.")
        pls_r = _SkPLS(n_components=int(st.get("n_components") or 2))
        for attr in [
            "_x_mean",
            "_x_std",
            "x_mean_",
            "x_std_",
            "x_weights_",
            "x_rotations_",
            "n_features_in_",
        ]:
            val = st.get(attr)
            if val is not None:
                setattr(pls_r, attr, np.array(val) if attr != "n_features_in_" else int(val))
        if getattr(pls_r, "_x_mean", None) is None or getattr(pls_r, "_x_std", None) is None:
            raise ValueError("Saved model PLS state is missing required private attrs (_x_mean/_x_std).")
        pls = pls_r

    pca = None
    if state.get("pca_state") is not None:
        st = state["pca_state"]
        if _SkPCA is None:
            raise ValueError("Cannot rebuild PCA: sklearn PCA unavailable.")
        pca_r = _SkPCA(n_components=int(st.get("n_components") or 2))
        for attr in ["components_", "mean_", "n_features_in_"]:
            val = st.get(attr)
            if val is not None:
                setattr(pca_r, attr, np.array(val) if attr != "n_features_in_" else int(val))
        pca = pca_r

    if "features" not in state:
        raise ValueError("Saved model is missing required 'features' list.")
    features = state["features"]
    if "coerce_state" not in state or not isinstance(state["coerce_state"], dict):
        raise ValueError("Saved model is missing required 'coerce_state' dict.")
    coerce_state = state["coerce_state"]
    if "threshold" not in state:
        raise ValueError("Saved model is missing required 'threshold'.")
    threshold = float(state["threshold"])
    if "compiled_input_scale" not in state or "compiled_input_shift" not in state:
        raise ValueError("Saved model is missing required compiled_input_{scale,shift}.")
    compiled_input_scale = float(state["compiled_input_scale"])
    compiled_input_shift = float(state["compiled_input_shift"])

    class LoadedQuantumClassifier:
        def __init__(self):
            self.version = int(ver)
            self.features = features
            self.scaler = scaler
            self.quantile = quantile
            self.pls = pls
            self.pca = pca
            self.coerce_state = coerce_state
            self.weights = weightsnp
            self.w_ro = w_ro_np
            self.bias = biasnp
            self.alpha = alphanp
            self.score_sign = score_sign_np
            self.threshold = threshold
            self.compiled_input_scale = compiled_input_scale
            self.compiled_input_shift = compiled_input_shift
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
            s = float(self.compiled_input_scale)
            b = float(self.compiled_input_shift)
            if s != 1.0 or b != 0.0:
                X = X * s + b
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
                res = circuit(self.weights, self.w_ro, Xnp)
            else:
                res = npy.array([circuit(self.weights, self.w_ro, row) for row in Xnp], dtype=npy.float64)
            ev = npy.asarray(res, dtype=npy.float64)
            a = float(npy.asarray(self.alpha))
            raw = ev + float(npy.asarray(self.bias))
            if self.version == 4:
                if not npy.isfinite(a) or abs(a - 1.0) > 1e-6:
                    raise ValueError(f"Unsupported alpha={a} for version=4 model. Re-save the model (version=5).")
                return self.score_sign * raw
            return float(a) * raw

        def decision_function(self, X):
            Xn = self._to_numpy(X)
            return np.array(self._variational_classifier(Xn))

        def predict(self, X):
            scores = self.decision_function(X)
            scores_np = npy.asarray(scores, dtype=float)
            return npy.where(scores_np >= float(self.threshold), 1.0, -1.0)

    return LoadedQuantumClassifier()


def _verify_saved_model_portable(
    model_path: str,
    *,
    X_ref: "np.ndarray",
    features: List[str],
    raw_df: Any = None,
    atol: float = 1e-6,
) -> None:
    """
    Post-save validation: ensure the `.pt` is loadable by the standalone loader and
    produces identical decision scores on a small reference set.

    This is intentionally strict: if this fails, the artifact is not deployable.
    """
    import numpy as _np
    import pandas as _pd

    try:
        # Standalone loader (no imports from this codebase inside the file).
        from scripts.predict import load_model_pt as _load_model_portable
    except Exception as _exc:
        raise RuntimeError(f"Portable loader import failed: {_exc}") from _exc

    clf_repo = load_model(model_path)
    clf_port = _load_model_portable(model_path)

    # Prefer verifying on *raw* DataFrame inputs (true deploy path). If not provided, fall back
    # to the numeric X_ref, but disable preprocessing to avoid double-transform.
    if raw_df is not None:
        s_repo = _np.asarray(clf_repo.decision_function(raw_df), dtype=_np.float64)
        s_port = _np.asarray(clf_port.decision_function(raw_df), dtype=_np.float64)
    else:
        # Numeric fallback (already-preprocessed arrays). Ensure both sides skip preprocessing.
        try:
            clf_repo.quantile = None
            clf_repo.pls = None
            clf_repo.pca = None
            clf_repo.scaler = None
        except Exception:
            pass
        try:
            clf_port.quantile = None
            clf_port.pls = None
            clf_port.pca = None
            clf_port.scaler = None
        except Exception:
            pass
        X_ref = _np.asarray(X_ref, dtype=_np.float64)
        if X_ref.ndim != 2:
            raise RuntimeError(f"Portable verify expected 2D X_ref, got shape {X_ref.shape}")
        s_repo = _np.asarray(clf_repo.decision_function(X_ref), dtype=_np.float64)
        s_port = _np.asarray(clf_port.decision_function(X_ref), dtype=_np.float64)

    if s_repo.shape != s_port.shape:
        raise RuntimeError(f"Portable verify shape mismatch: {s_repo.shape} vs {s_port.shape}")
    if not _np.all(_np.isfinite(s_repo)) or not _np.all(_np.isfinite(s_port)):
        raise RuntimeError("Portable verify produced non-finite scores")

    max_abs = float(_np.max(_np.abs(s_repo - s_port))) if s_repo.size else 0.0
    if max_abs > float(atol):
        raise RuntimeError(
            f"Portable verify failed: max_abs_diff={max_abs:.3e} > atol={float(atol):.3e}"
        )

    # Sanity: predictions should match as well.
    p_repo = _np.asarray(clf_repo.predict(raw_df if raw_df is not None else X_ref), dtype=_np.float64)
    p_port = _np.asarray(clf_port.predict(raw_df if raw_df is not None else X_ref), dtype=_np.float64)
    if p_repo.shape != p_port.shape or not _np.all(p_repo == p_port):
        raise RuntimeError("Portable verify failed: predictions differ")

    print(f"[SAVE-VERIFY] portable_ok=1 max_abs_diff={max_abs:.3e}", flush=True)
