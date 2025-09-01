from __future__ import annotations

# Central functional DSL for building and running QML experiments
# Keeps per-experiment shims tiny while concentrating shared logic here.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


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


def pca_to_pow2(max_qubits: Optional[int] = None) -> Step:
    return Step("dataset.pca_pow2", {"max_qubits": max_qubits})


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

    qml.templates.StronglyEntanglingLayers(W, wires=wires)


# -----------------------------
# Runner
# -----------------------------


def _setup_logger(log_filename: str):
    import sys

    class Logger(object):
        def __init__(self, filename: str) -> None:
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message: str) -> None:
            self.terminal.write(message)
            self.log.write(message)

        def flush(self) -> None:
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_filename)


def run(recipe: Recipe) -> None:
    import os
    import datetime
    import time

    # Lazy imports for heavy deps
    import pandas as pd
    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.optimize import AdamOptimizer
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    # Collect config from steps
    cfg: Dict[str, Any] = {}
    for step in recipe.parts:
        cfg[step.kind] = {**cfg.get(step.kind, {}), **step.params}

    # Configure logging
    os.makedirs("logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ds_name = os.path.basename(cfg.get("dataset.csv", {}).get("path", "dataset")).replace(".csv", "")
    log_path = os.path.join("logs", f"DSL_{ds_name}_{ts}.log")
    _setup_logger(log_path)

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
            df = pd.read_csv(path)
        else:
            # two-pass memory-efficient sampling
            with open(path, "r") as f:
                num_lines = sum(1 for _ in f) - 1
            k = min(int(sample_size), max(1, num_lines))
            to_skip = sorted(_random.sample(range(1, num_lines + 1), num_lines - k))
            df = pd.read_csv(path, skiprows=to_skip)
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
    # Optional PCA to a power-of-two feature count (useful for amplitude embedding)
    pca_cfg = cfg.get("dataset.pca_pow2", None)
    if pca_cfg is not None:
        max_qubits = pca_cfg.get("max_qubits")
        # nearest power-of-two <= current features
        import math as _math

        d0 = X.shape[1]
        max_power = d0.bit_length() - 1
        if max_qubits is not None:
            max_power = min(max_power, int(max_qubits))
        target_dim = max(1, 2 ** max_power)
        if target_dim != d0:
            pca = PCA(n_components=target_dim, random_state=42)
            X = pca.fit_transform(X.values)
        else:
            X = X.values
    else:
        # keep as ndarray for later scaling
        X = X.values if hasattr(X, "values") else X
    print("Features and labels extracted.")

    # Split
    tr_cfg = cfg.get("train", {})
    test_size = float(tr_cfg.get("test_size", 0.2))
    stratify = bool(tr_cfg.get("stratify", True))
    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_y
    )
    print(f"Data split: X_train={X_train.shape}, X_test={X_test.shape}")

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    # Labels to {-1, 1}
    Y_train = np.array(y_train.values * 2 - 1, requires_grad=False)
    Y_test = np.array(y_test.values * 2 - 1, requires_grad=False)

    # Device and wires
    dev_cfg = cfg.get("device", {})
    num_qubits = X_train_scaled.shape[1]
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
    meas_wires = meas_cfg.get("wires", [0])

    # Angle scaling for angle encoders
    angle_scale = None
    if enc_name.startswith("angle_embedding"):
        if enc_cfg.get("angle_range") == "0_pi":
            angle_scale = qml.numpy.pi
        elif enc_cfg.get("angle_scale") is not None:
            angle_scale = float(enc_cfg.get("angle_scale"))

    # QNode
    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        # Optionally apply encoder before each layer (re-upload)
        reupload = bool(enc_cfg.get("reupload", False))
        if reupload:
            for W in weights:
                encoder_fn(x, wires, hadamard=bool(enc_cfg.get("hadamard", False)), angle_scale=angle_scale)
                ansatz_fn(W, wires)
        else:
            encoder_fn(x, wires, hadamard=bool(enc_cfg.get("hadamard", False)), angle_scale=angle_scale)
            for W in weights:
                ansatz_fn(W, wires)

        # Measurement
        if meas_name == "mean_z":
            obs = [qml.expval(qml.PauliZ(w)) for w in meas_wires]
            return obs
        else:  # default z0
            return qml.expval(qml.PauliZ(0))

    def variational_classifier(weights, bias, X_np):
        res = circuit(weights, X_np)
        # If multiple expvals were returned, average them
        try:
            res = qml.numpy.mean(res)
        except Exception:
            pass
        return res + bias

    def square_loss(labels, predictions, class_weights=None):
        loss = (labels - predictions) ** 2
        if class_weights is not None:
            loss = class_weights * loss
        return np.mean(loss)

    def accuracy(labels, predictions):
        return np.sum(np.sign(predictions) == labels) / len(labels)

    # Training setup
    seed = int(tr_cfg.get("seed", 42))
    np.random.seed(seed)
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    lr = float(tr_cfg.get("lr", 0.1))
    batch_size = int(tr_cfg.get("batch", 100))
    epochs = int(tr_cfg.get("epochs", 1))
    batch_size = min(batch_size, len(X_train_scaled))
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

    def cost(weights, bias, X_np, Y_np):
        preds = variational_classifier(weights, bias, X_np)
        weights_tensor = None
        if class_weights_map:
            weights_tensor = np.array([class_weights_map[label] for label in Y_np])
        return square_loss(Y_np, preds, class_weights=weights_tensor)

    # Training loop (epochs)
    weights = weights_init
    bias = bias_init
    start_time = time.time()
    total_iters = 0
    for ep in range(epochs):
        num_it = max(1, len(X_train_scaled) // batch_size)
        for it in range(num_it):
            batch_index = np.random.randint(0, len(X_train_scaled), (batch_size,))
            X_batch = X_train_scaled[batch_index]
            Y_batch = Y_train[batch_index]
            weights, bias = opt.step(lambda w, b: cost(w, b, X_batch, Y_batch), weights, bias)
            total_iters += 1
            if (it + 1) % 10 == 0 or (it + 1) == num_it:
                preds_b = variational_classifier(weights, bias, X_batch)
                c_b = square_loss(Y_batch, preds_b)
                a_b = accuracy(Y_batch, np.sign(preds_b))
                print(f"Epoch {ep+1} Iter {it+1}/{num_it} | Batch Cost: {c_b:0.7f} | Batch Acc: {a_b:0.7f}")

    print(f"Training finished in {time.time() - start_time:.2f}s over {epochs} epoch(s), {total_iters} iters.")

    # Validation quick check (use a small random subset if large)
    val_size = min(5 * batch_size, len(X_train_scaled))
    val_idx = np.random.randint(0, len(X_train_scaled), val_size)
    X_val = X_train_scaled[val_idx]
    Y_val = Y_train[val_idx]
    preds_val = variational_classifier(weights, bias, X_val)
    print(
        f"Validation Cost: {square_loss(Y_val, preds_val):0.7f} | Validation Accuracy: {accuracy(Y_val, np.sign(preds_val)):0.7f}"
    )

    # Test evaluation
    # Compute predictions per-sample to ensure a 1D predictions array
    if len(X_test_scaled) < 1000:
        predictions = np.array([variational_classifier(weights, bias, x) for x in X_test_scaled])
    else:
        preds_list = []
        for i in range(0, len(X_test_scaled), batch_size):
            X_b = X_test_scaled[i : i + batch_size]
            preds_list.extend([variational_classifier(weights, bias, x) for x in X_b])
        predictions = np.array(preds_list)

    predictions_signed = np.sign(predictions)
    acc = float(accuracy_score(Y_test, predictions_signed))
    prec = float(precision_score(Y_test, predictions_signed, average='macro', zero_division=0))
    rec = float(recall_score(Y_test, predictions_signed, average='macro', zero_division=0))
    f1 = float(f1_score(Y_test, predictions_signed, average='macro', zero_division=0))
    print("--- Test Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("--------------------")
    print("Experiment complete.")

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
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
    }


