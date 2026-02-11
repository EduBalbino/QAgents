import os
import json
import time
import sys
import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# ---------------------------
# Embedded spec helpers
# ---------------------------
# Folded from the former `scripts/specs.py` to reduce module sprawl and keep
# the benchmark script self-contained.
from dataclasses import asdict, dataclass
import hashlib
import platform
import subprocess

from pydantic import BaseModel, Field, ValidationError, field_validator

# Keep this aligned with scripts/core/compiled_core.py supported encoders.
ALLOWED_ENCODERS = {
    "angle_embedding_y",
    "angle_pair_xy",
    "amplitude_embedding",
}
ALLOWED_ANZ = {"ring_rot_cnot", "strongly_entangling"}
ALLOWED_MEASUREMENTS = {"mean_z", "mean_z_readout", "z0", "z_vec"}


@dataclass(frozen=True)
class EncoderCfg:
    name: str
    hadamard: bool = False
    reupload: bool = False
    angle_range: Optional[str] = None
    angle_scale: Optional[float] = None


@dataclass(frozen=True)
class AnsatzCfg:
    name: str
    layers: int


@dataclass(frozen=True)
class MeasurementCfg:
    name: str
    wires: List[int]


@dataclass(frozen=True)
class DataCfg:
    path: str
    features: List[str]
    sample: int


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    batch: int
    epochs: int
    seed: int
    class_weights: str


@dataclass(frozen=True)
class ExperimentSpec:
    schema_version: int
    encoder: EncoderCfg
    ansatz: AnsatzCfg
    measurement: MeasurementCfg
    data: DataCfg
    train: TrainCfg
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


class SpecValidator(BaseModel):
    schema_version: int = Field(ge=1)
    encoder: Dict[str, Any]
    ansatz: Dict[str, Any]
    measurement: Dict[str, Any]
    data: Dict[str, Any]
    train: Dict[str, Any]
    meta: Dict[str, Any]

    @field_validator("encoder")
    @classmethod
    def _encoder_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v.get("name") not in ALLOWED_ENCODERS:
            raise ValueError(f"encoder.name must be one of {sorted(ALLOWED_ENCODERS)}")
        return v

    @field_validator("ansatz")
    @classmethod
    def _ansatz_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v.get("name") not in ALLOWED_ANZ:
            raise ValueError(f"ansatz.name must be one of {sorted(ALLOWED_ANZ)}")
        layers = v.get("layers")
        if layers is None or layers < 1:
            raise ValueError("ansatz.layers must be >= 1")
        return v

    @field_validator("measurement")
    @classmethod
    def _measurement_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        name = v.get("name")
        wires = v.get("wires")
        if name not in ALLOWED_MEASUREMENTS:
            raise ValueError(f"measurement.name must be one of {sorted(ALLOWED_MEASUREMENTS)}")
        if not isinstance(wires, list) or not all(isinstance(w, int) for w in wires):
            raise ValueError("measurement.wires must be a list of integers")
        return v

    @field_validator("data")
    @classmethod
    def _data_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        sample = v.get("sample")
        features = v.get("features", []) or []
        # sample == 0 means "full dataset" (no sampling); >=1 means "exact sample size".
        if sample is None or int(sample) < 0:
            raise ValueError("data.sample must be >= 0 (0 means full dataset)")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise ValueError("data.features must be a list of feature names")
        if len(features) == 0:
            raise ValueError("data.features must not be empty")
        return v

    @field_validator("train")
    @classmethod
    def _train_ok(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        lr = v.get("lr")
        batch = v.get("batch")
        epochs = v.get("epochs")
        if lr is None or float(lr) <= 0:
            raise ValueError("train.lr must be > 0")
        if batch is None or int(batch) < 1:
            raise ValueError("train.batch must be >= 1")
        if epochs is None or int(epochs) < 1:
            raise ValueError("train.epochs must be >= 1")
        return v


def build_and_validate_spec(**kwargs: Any) -> tuple[ExperimentSpec, str]:
    spec = ExperimentSpec(**kwargs)
    try:
        SpecValidator.model_validate(spec.to_dict())
    except ValidationError as exc:
        raise ValueError(f"Spec invalid: {exc}") from exc
    return spec, spec.hash()


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        compound_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, compound_key))
        else:
            out[compound_key] = value
            if isinstance(value, list):
                out[f"{compound_key}__len"] = len(value)
    return out


def _md5(path: str, block_size: int = 1 << 20) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def provenance(spec: ExperimentSpec) -> Dict[str, Any]:
    data_path = spec.data.path
    data_info: Dict[str, Any] = {
        "path": data_path,
        "exists": os.path.exists(data_path),
        "sample": spec.data.sample,
        "features": spec.data.features,
    }
    data_info["md5"] = _md5(data_path)
    try:
        data_info["size_bytes"] = os.path.getsize(data_path)
    except OSError:
        data_info["size_bytes"] = None

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_rev = None
    try:
        dirty = subprocess.call(["git", "diff", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
    except Exception:
        dirty = None

    return {
        "data": data_info,
        "code": {
            "git_commit": git_rev,
            "git_dirty": dirty,
        },
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def compliance_from_summary(spec: ExperimentSpec, summary: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    expected_qubits = len(spec.data.features)
    actual_qubits = summary.get("circuit_qubits") or summary.get("num_qubits") or expected_qubits
    if actual_qubits != expected_qubits:
        diff["qubits_mismatch"] = {"expected": expected_qubits, "actual": actual_qubits}
    ok = len(diff) == 0
    return {"ok": ok, "diff": diff}


def build_feature_manifest(spec: ExperimentSpec, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = spec.data.features
    used_features = summary.get("used_features")
    if isinstance(used_features, dict):
        used_set = {str(k) for k in used_features.keys()}
    elif isinstance(used_features, list):
        used_set = {str(f) for f in used_features}
    else:
        used_set = {feature for feature in features}
    pca_map = summary.get("pca_index_map", {})
    manifest: List[Dict[str, Any]] = []
    for idx, feature in enumerate(features):
        manifest.append(
            {
                "feature": feature,
                "original_index": idx,
                "selected": int(feature in used_set or str(idx) in used_set),
                "post_pca_index": pca_map.get(idx),
            }
        )
    return manifest

# By default, do not force BLAS/OpenMP thread caps. This prevents stale
# shell-exported limits (e.g., OMP_NUM_THREADS=2) from throttling sweeps.
if os.environ.get("EDGE_USE_ALL_THREADS", "1") != "0":
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.pop(_k, None)

try:
    import wandb
except Exception:
    wandb = None  # type: ignore[assignment]

from scripts.core.builders import (
    Recipe,
    csv,
    select,
    device,
    encoder,
    ansatz,
    measurement,
    train,
    save,
    run,
    rf_baseline,
    quantile_uniform,
    pls_to_pow2,
)


# --- objective metric used by grid and sweeps ---
OBJECTIVE_WEIGHTS = {"f1": 0.5, "balanced_accuracy": 0.3, "auc": 0.2}

def objective(m: Dict[str, float]) -> float:
    # Treat non-finite metrics as 0.0 so sweeps don't get poisoned by NaNs.
    out = 0.0
    for k, w in OBJECTIVE_WEIGHTS.items():
        try:
            v = float(m.get(k, 0.0) or 0.0)
        except Exception:
            v = 0.0
        if v != v or v in (float("inf"), float("-inf")):
            v = 0.0
        out += float(w) * v
    return float(out)


EDGE_DATASET = os.environ.get("EDGE_DATASET", "data/processed/mergido_preprocessado.csv")
# Derived dataset already contains 8 leakage-safe PLS components (PC_1..PC_8).
EDGE_FEATURES = [f"PC_{i}" for i in range(1, 9)]
EDGE_LABEL = os.environ.get("EDGE_LABEL", "Attack_label")

WANDB_BASE_URL = "https://wandb.balbino.io"
os.environ.setdefault("WANDB_BASE_URL", WANDB_BASE_URL)
os.environ.setdefault("WANDB_HOST", WANDB_BASE_URL)
os.environ.setdefault("WANDB_API_HOST", WANDB_BASE_URL)
os.environ.setdefault("EDGE_PREFLIGHT_COMPILE", "1")
# Default to CPU-backed Lightning simulator, but allow user/sweep overrides.
os.environ.setdefault("QML_DEVICE", "lightning.qubit")
_WANDB_SESSION_GROUP = f"edgeiiot-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
_WANDB_LOGIN_OK: Optional[bool] = None

# Default training / sweep hyperparameters (overridable via env or W&B sweeps)
EDGE_FIXED_EPOCHS = 4
EDGE_DEFAULT_SAMPLE = 0
EDGE_DEFAULT_LR = 0.01
EDGE_DEFAULT_BATCH = 64
EDGE_DEFAULT_EPOCHS = EDGE_FIXED_EPOCHS
EDGE_DEFAULT_CLASS_WEIGHTS = "none"
EDGE_DEFAULT_SEED = 42
# Keep preprocessing split stable across sweep seeds.
os.environ.setdefault("EDGE_PREPROCESS_SPLIT_SEED", str(EDGE_DEFAULT_SEED))

# ---------------------------
# Phase A (fixed forever)
# ---------------------------
# This sweep is intentionally hard-coded (no env overrides, no conditional logic) so results
# stay comparable across time and compilation caching stays stable.

PHASE_A_SAMPLE = 120000
PHASE_A_EPOCHS = 20
PHASE_A_SEEDS = [42, 1337, 2024]
PHASE_A_LR_VALUES = [0.01]

PHASE_A_ENC_NAMES = [
    "angle_embedding_y",
]
# "angle_mode" is interpreted by _enc_opts_from_cfg:
# - range_0_pi -> angle_range=0_pi
# - range_pm_pi -> angle_range=pm_pi        (theta = 2*pi*(u-0.5))
# - range_pm_pi_2 -> angle_range=pm_pi_2    (theta = pi*(u-0.5))
# - scale_X    -> angle_scale=float(X)
PHASE_A_ANGLE_MODES = [
    "range_pm_pi",
]

PHASE_A_MEASUREMENTS = ["z0"]
PHASE_A_LAYERS_RING = [3]
PHASE_A_LAYERS_STRONGLY_ENTANGLING = [3]

PHASE_A_RING_COUNT_DEFAULT = 80
PHASE_A_STRONG_COUNT_DEFAULT = 60

EXPLORE_COUNT_DEFAULT = 8
EXPAND_COUNT_DEFAULT = 16

# Default benchmark run should use the full dataset (no sampling).
# Use <=0 to mean "no sampling" and pass sample_size=None into the DSL.
BENCHMARK_DEFAULT_SAMPLE = 0

# Limit number of features to a feasible qubit count for simulators
# Configurable via EDGE_NUM_FEATURES env (default: all EDGE_FEATURES).
def _active_features() -> List[str]:
    try:
        n = int(os.environ.get("EDGE_NUM_FEATURES", str(len(EDGE_FEATURES))))
    except Exception:
        n = len(EDGE_FEATURES)
    n = max(1, min(n, len(EDGE_FEATURES)))
    return EDGE_FEATURES[:n]


def _fmt_bool(v: Any) -> str:
    return "True" if bool(v) else "False"


def _pretty_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join((str(c)).ljust(w) for c, w in zip(cols, widths))


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v not in (None, "", "None") else default
    except Exception:
        return default


def _env_list_int(name: str, default: List[int]) -> List[int]:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        out = [int(x.strip()) for x in v.split(",") if x.strip() != ""]
        return out if out else default
    except Exception:
        return default


def _env_list_str(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name)
    if not v:
        return default
    out = [x.strip() for x in v.split(",") if x.strip() != ""]
    return out if out else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "", "None") else default
    except Exception:
        return default


def _default_train_params(seed: int) -> Dict[str, Any]:
    """
    Hardcoded training hyperparameters for benchmark runs.
    """
    return {
        "lr": EDGE_DEFAULT_LR,
        "batch": EDGE_DEFAULT_BATCH,
        "epochs": EDGE_DEFAULT_EPOCHS,
        "class_weights": EDGE_DEFAULT_CLASS_WEIGHTS,
        "seed": seed,
        "test_size": 0.2,
        "stratify": True,
        # Keep train batches roughly class-balanced; val/test remain untouched.
        "balanced_batches": True,
        "balanced_pos_frac": 0.5,
        # Sane defaults for stability/precision.
        "lr_schedule": "onecycle_cosine",
        "onecycle_pct_start": 0.3,
        "onecycle_div_factor": 25.0,
        "onecycle_final_div_factor": 10000.0,
        "weight_decay": 0.001,
        "focal_gamma": 2.0,
        # Selection policy: maximize precision subject to bacc>=0.8.
        "val_objective": "prec_at_bacc",
        "min_bacc": 0.8,
        # Degeneracy guardrails for threshold search/selection.
        "min_pred_pos_rate": 0.01,
        "max_pred_pos_rate": 0.99,
        "abort_on_degen": True,
    }


def build_recipe(
    sample: Optional[int],
    enc_name: str,
    enc_opts: Dict[str, Any],
    layers: int,
    meas: Dict[str, Any],
    anz_name: str,
    seed: int,
    *,
    features: Optional[List[str]] = None,
    train_params: Optional[Dict[str, Any]] = None,
) -> Recipe:
    # Always drive the recipe feature selection from the same "active features"
    # that get embedded into the spec, so qubit count and compliance stay consistent.
    feats = list(features) if features else _active_features()
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        tp |= {
            k: train_params[k]
            for k in (
                "lr",
                "batch",
                "epochs",
                "class_weights",
                "seed",
                "test_size",
                "stratify",
                "balanced_batches",
                "balanced_pos_frac",
                "lr_schedule",
                "onecycle_pct_start",
                "onecycle_div_factor",
                "onecycle_final_div_factor",
                "weight_decay",
                "focal_gamma",
                "val_objective",
                "min_bacc",
                "min_pred_pos_rate",
                "max_pred_pos_rate",
                "abort_on_degen",
                # Fast-run / micro-ablation knobs (consumed by scripts/core/builders.py)
                "fixed_num_batches",
                "preflight_compile",
                "eval_every_epochs",
                "weight_decay_ro",
                "lr_mult_w_ro",
                "lr_mult_alpha",
                "alpha_train",
            )
            if k in train_params
        }
    r = Recipe() | csv(EDGE_DATASET, sample_size=sample) | select(feats, label=EDGE_LABEL)
    # mergido_preprocessado.csv already contains quantile-mapped PLS components (PC_1..PC_8).
    # Don't apply quantile/PLS again.
    if not (os.path.basename(str(EDGE_DATASET)) == "mergido_preprocessado.csv" or all(str(f).startswith("PC_") for f in feats)):
        r = r | quantile_uniform()
        r = r | pls_to_pow2(components=max(1, len(feats)))
    r = (
        r
        | device("lightning.qubit", wires_from_features=True)
        | encoder(enc_name, **enc_opts)
        | ansatz(anz_name, layers=layers)
        | measurement(str(meas.get("name")), list(meas.get("wires") or []))
        | train(**tp)
    )
    # Embedding receives exactly len(feats) features.
    return r


def _wandb_group() -> str:
    return os.environ.get("WANDB_GROUP") or _WANDB_SESSION_GROUP


def _wandb_project() -> str:
    return os.environ.get("WANDB_PROJECT", "qml-edgeiiot")


def _wandb_entity() -> Optional[str]:
    return os.environ.get("WANDB_ENTITY", "edubalbino")


def _wandb_disabled() -> bool:
    return os.environ.get("WANDB_DISABLED", "0").lower() in ("1", "true", "yes", "on")


def _wandb_base_kwargs(name: str,
                       job_type: str,
                       tags: Optional[List[str]] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "project": _wandb_project(),
        "entity": _wandb_entity(),
        "group": _wandb_group(),
        "job_type": job_type,
        "name": name,
    }
    if tags:
        kwargs["tags"] = tags
    return {k: v for k, v in kwargs.items() if v is not None}


def _resolved_envs() -> Dict[str, str]:
    keys = [
        "EDGE_NUM_FEATURES",
        "EDGE_SAMPLE",
        "EDGE_SEED",
        "EDGE_ANZ",
        "EDGE_ANZ_LIST",
        "EDGE_LAYERS_LIST",
        "WANDB_GROUP",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
    ]
    resolved = {k: os.environ.get(k) for k in keys}
    return {k: v for k, v in resolved.items() if v is not None}


def _wandb_ensure_login(force: bool = False) -> None:
    if wandb is None:
        raise RuntimeError("wandb is not installed in this environment.")
    global _WANDB_LOGIN_OK
    if not force and _WANDB_LOGIN_OK:
        return
    api_key = os.environ.get("WANDB_API_KEY")
    relogin = force or bool(os.environ.get("WANDB_FORCE_RELOGIN"))
    try:
        current_key = None
        try:
            current_key = wandb.Api().api_key
        except Exception:
            current_key = None

        if current_key and not relogin and not api_key:
            _WANDB_LOGIN_OK = True
            return

        if api_key:
            result = wandb.login(key=api_key, host=WANDB_BASE_URL, relogin=relogin)
        else:
            result = wandb.login(host=WANDB_BASE_URL, relogin=relogin)
    except Exception as exc:
        _WANDB_LOGIN_OK = False
        raise RuntimeError(f"Failed to authenticate with Weights & Biases at {WANDB_BASE_URL}: {exc}") from exc
    _WANDB_LOGIN_OK = bool(result or getattr(wandb.Api(), "api_key", None))
    if not _WANDB_LOGIN_OK:
        raise RuntimeError(
            f"Weights & Biases login was not detected. Run `wandb login --relogin --host={WANDB_BASE_URL}` once or set WANDB_API_KEY."
        )


def run_one(sample: Optional[int],
            enc_name: str,
            enc_opts: Dict[str, Any],
            layers: int,
            meas: Dict[str, Any],
            anz_name: str,
            seed: int,
            train_params: Optional[Dict[str, Any]] = None,
            use_current_wandb_run: bool = False) -> Dict[str, Any]:
    measurement_cfg = MeasurementCfg(
        name=meas.get("name", "z0"),
        wires=[int(w) for w in meas.get("wires", [])],
    )
    active_features = _active_features()
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        # Allow overriding the full training surface area so A/B grids are real.
        tp |= {
            k: train_params[k]
            for k in (
                "lr",
                "batch",
                "epochs",
                "class_weights",
                "seed",
                "test_size",
                "stratify",
                "balanced_batches",
                "balanced_pos_frac",
                "lr_schedule",
                "onecycle_pct_start",
                "onecycle_div_factor",
                "onecycle_final_div_factor",
                "weight_decay",
                "focal_gamma",
                "val_objective",
                "min_bacc",
                "min_pred_pos_rate",
                "max_pred_pos_rate",
                "abort_on_degen",
                # QML-specific knobs (consumed by scripts/core/builders.py)
                "alpha_mode",
                "fixed_num_batches",
                "preflight_compile",
                "eval_every_epochs",
                "weight_decay_ro",
                "lr_mult_w_ro",
                "lr_mult_alpha",
                "alpha_train",
            )
            if k in train_params
        }
    spec, spec_hash = build_and_validate_spec(
        schema_version=1,
        encoder=EncoderCfg(
            name=enc_name,
            hadamard=bool(enc_opts.get("hadamard", False)),
            reupload=bool(enc_opts.get("reupload", False)),
            angle_range=enc_opts.get("angle_range"),
            angle_scale=enc_opts.get("angle_scale"),
        ),
        ansatz=AnsatzCfg(name=anz_name, layers=layers),
        measurement=measurement_cfg,
        # Spec carries a stable integer; use 0 to represent "full dataset" (no sampling).
        data=DataCfg(path=EDGE_DATASET, features=active_features, sample=(0 if sample is None else int(sample))),
        train=TrainCfg(lr=float(tp["lr"]), batch=int(tp["batch"]), epochs=int(tp["epochs"]), seed=int(tp["seed"]), class_weights=str(tp["class_weights"])),
        meta={"device": os.environ.get("QML_DEVICE", "lightning.qubit"), "env": _resolved_envs()},
    )
    spec_dict = spec.to_dict()
    phase = os.environ.get("EDGE_PHASE", "").strip() or "unspecified"
    short_hash = spec_hash[:6]
    # Make reupload explicit in run naming for easier tracking
    reupload_flag = bool(enc_opts.get("reupload", False))
    hadamard_flag = bool(enc_opts.get("hadamard", False))
    reup_suffix = "R" if reupload_flag else "NR"
    run_name = f"{phase}-{enc_name}-{anz_name}-L{layers}-{reup_suffix}-s{seed}-{short_hash}"
    tags = [
        f"enc:{enc_name}",
        f"anz:{anz_name}",
        f"L:{layers}",
        f"meas:{measurement_cfg.name}",
        f"wires:{len(measurement_cfg.wires)}",
        f"seed:{seed}",
        f"reupload:{int(reupload_flag)}",
        f"hadamard:{int(hadamard_flag)}",
        f"spec:{spec_hash}",
        f"phase:{phase}",
    ]
    wandb_kwargs = _wandb_base_kwargs(run_name, job_type="benchmark-run", tags=tags)
    wandb_run = None
    artifact_paths: List[str] = []
    if _wandb_disabled():
        wandb_run = None
    elif use_current_wandb_run:
        wandb_run = wandb.run
    else:
        try:
            _wandb_ensure_login()
        except RuntimeError as auth_err:
            print(f"W&B authentication unavailable for run '{run_name}': {auth_err}")
        else:
            try:
                wandb_run = wandb.init(**wandb_kwargs)
            except Exception as init_err:
                print(f"Failed to initialize W&B run '{run_name}': {init_err}")

    if wandb_run is not None:
        try:
            cfg_payload = spec_dict | {"provenance": provenance(spec)}
            wandb_run.config.update(cfg_payload, allow_val_change=True)
            wandb_run.config.update({"spec_flat": flatten(cfg_payload)}, allow_val_change=True)
            wandb_run.config.update({"spec_hash": spec_hash}, allow_val_change=True)
            # Also expose encoder flags in top-level config for easier querying
            wandb_run.config.update(
                {
                    "encoder_name": enc_name,
                    "encoder_hadamard": hadamard_flag,
                    "encoder_reupload": reupload_flag,
                },
                allow_val_change=True,
            )
            spec_dir = "wandb_specs"
            os.makedirs(spec_dir, exist_ok=True)
            spec_artifact_path = os.path.join(spec_dir, f"spec-{spec_hash}.json")
            with open(spec_artifact_path, "w", encoding="utf-8") as f:
                json.dump(cfg_payload, f, indent=2)
            artifact = wandb.Artifact(name=f"spec-{spec_hash}", type="experiment-spec")
            artifact.add_file(spec_artifact_path)
            wandb_run.log_artifact(artifact)
            artifact_paths.append(spec_artifact_path)
        except Exception as exc:
            print(f"Failed to register spec for run '{run_name}': {exc}")

    # Build base recipe (data → quantile/PLS → device/encoder/ansatz/train)
    recipe = build_recipe(
        sample,
        enc_name,
        enc_opts,
        layers,
        meas,
        anz_name,
        seed,
        features=active_features,
        train_params=tp,
    )
    # Always persist trained models for this run; use spec hash for stable, unique names.
    model_dir = os.environ.get("EDGE_MODEL_DIR", "models")
    model_name = f"edgeiiot_{enc_name}_{anz_name}_L{layers}_s{seed}_{spec_hash}.pt"
    model_path = os.path.join(model_dir, model_name)
    recipe = recipe | save(model_path)
    if wandb_run is not None:
        try:
            # Log the recipe structure directly to W&B without writing a local file
            recipe_payload = [{"kind": part.kind, "params": part.params} for part in recipe.parts]
            wandb_run.summary.update({"recipe": recipe_payload})
        except Exception as exc:
            print(f"Failed to persist recipe for run '{run_name}': {exc}")

    start_time = time.time()
    summary = run(recipe)
    wall_time_s = time.time() - start_time
    compile_time_s = float(summary.get("compile_time_s", float("nan")))
    core_train_time_s = float(summary.get("train_time_s", float("nan")))
    try:
        metrics = summary.get("metrics", {})
        if wandb_run is not None and metrics:
            wandb_run.log({f"metrics/{k}": v for k, v in metrics.items()})
            wandb_run.summary.update({k: v for k, v in metrics.items()})
            try:
                obj = objective(metrics)
                wandb_run.log({"objective": obj})
                wandb_run.summary.update({"objective": obj})
                denom = core_train_time_s if core_train_time_s == core_train_time_s else wall_time_s
                wandb_run.log(
                    {
                        "objective_per_s": obj / max(float(denom), 1e-9),
                        "time/core_train_s": core_train_time_s,
                        "time/compile_s": compile_time_s,
                        "time/wall_s": wall_time_s,
                    }
                )
            except Exception:
                pass
        if wandb_run is not None:
            compliance = compliance_from_summary(spec, summary)
            wandb_run.summary.update({
                "dataset": summary.get("dataset"),
                "log_path": summary.get("log_path"),
                "seed": seed,
                "spec_hash": spec_hash,
                "time/core_train_s": core_train_time_s,
                "time/compile_s": compile_time_s,
                "time/wall_s": wall_time_s,
                "compliance/ok": compliance["ok"],
                "compliance/diff": json.dumps(compliance["diff"]),
            })
            manifest_rows = build_feature_manifest(spec, summary)
            if manifest_rows:
                manifest_table = wandb.Table(columns=["feature", "original_index", "selected", "post_pca_index"])
                for row in manifest_rows:
                    manifest_table.add_data(
                        row["feature"],
                        row["original_index"],
                        row["selected"],
                        row["post_pca_index"],
                    )
                wandb_run.log({"manifest/features": manifest_table})
            log_path = summary.get("log_path")
            if log_path and os.path.exists(log_path):
                artifact = wandb.Artifact(name=f"logs-{spec_hash}", type="edge-logs")
                artifact.add_file(log_path)
                wandb_run.log_artifact(artifact)
                artifact_paths.append(log_path)
                try:
                    os.remove(log_path)
                except OSError:
                    pass

    except Exception as exc:
        if wandb_run is not None:
            wandb_run.summary.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        if wandb_run is not None and not use_current_wandb_run:
            wandb_run.finish()
        for path in artifact_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    out: Dict[str, Any] = {
        **summary,
        "enc_name": enc_name,
        "enc_opts": enc_opts,
        "layers": layers,
        "measurement": meas,
        "ansatz": summary.get("ansatz", anz_name),
        "seed": seed,
        "wandb_run": wandb_run.name if wandb_run is not None else run_name,
        "spec": spec_dict,
        "spec_hash": spec_hash,
        "spec_flat": flatten(spec_dict),
        "core_train_time_s": core_train_time_s,
        "compile_time_s": compile_time_s,
        "wall_time_s": wall_time_s,
    }
    return out


def _benchmark_worker(payload: Tuple[Optional[int], int, str, Dict[str, Any], str, Dict[str, Any], int]):
    sample, seed, enc_name, enc_opts, anz_name, meas, layers = payload
    try:
        return run_one(sample, enc_name, enc_opts, layers, meas, anz_name, seed)
    except Exception as e:
        print(f"Run failed for enc={enc_name} anz={anz_name} opts={enc_opts} meas={meas} L={layers}: {e}")
        return None


def main() -> None:
    # Compiled-safe benchmark defaults.
    encoders: List[Tuple[str, Dict[str, Any]]] = [
        ("angle_embedding_y", {"angle_scale": 0.5, "reupload": True}),
    ]

    feats = _active_features()
    # Default readout: mean-Z across all active wires (stable, fast baseline).
    measurement: Dict[str, Any] = {"name": "mean_z", "wires": list(range(len(feats)))}

    # Grid of ansatz/layers, overridable via env lists
    anz_list = _env_list_str("EDGE_ANZ_LIST", ["ring_rot_cnot"]) or [
        os.environ.get("EDGE_ANZ", "ring_rot_cnot")
    ]
    layers_list = _env_list_int("EDGE_LAYERS_LIST", [4])

    sample_raw = _env_int("EDGE_SAMPLE", BENCHMARK_DEFAULT_SAMPLE)
    sample = None if int(sample_raw) <= 0 else int(sample_raw)
    seed = _env_int("EDGE_SEED", 42)

    jobs: List[Tuple[Optional[int], int, str, Dict[str, Any], str, Dict[str, Any], int]] = []
    for (enc_name, enc_opts) in encoders:
        for anz_name in anz_list:
            for layers in layers_list:
                jobs.append((sample, seed, enc_name, enc_opts, anz_name, measurement, int(layers)))

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_headers = [
        "SpecHash", "Encoder", "Hadamard", "Reupload", "AngleScale", "Meas", "Layers", "Ansatz",
        "Acc", "Prec", "Rec", "F1", "BAcc", "AUC", "ValBAcc", "Thresh", "CompileS", "TrainS", "WallS", "Log"
    ]

    aggregate_name = os.environ.get("WANDB_AGGREGATE_RUN") or f"edgeiiot-benchmark-{ts}"
    aggregate_tags = ["aggregate", "edgeiiot"]
    aggregate_kwargs = _wandb_base_kwargs(aggregate_name, job_type="benchmark-aggregate", tags=aggregate_tags)
    aggregate_run = None
    table = None
    if not _wandb_disabled():
        try:
            _wandb_ensure_login()
            aggregate_run = wandb.init(**aggregate_kwargs)
            table = wandb.Table(columns=run_headers)
        except Exception as exc:
            print(f"Failed to initialize W&B aggregate run '{aggregate_name}': {exc}")

    if aggregate_run is not None:
        aggregate_run.config.update({
            "sample": sample,
            "seed": seed,
            "encoders": encoders,
            "ansatz_list": anz_list,
            "layers_list": layers_list,
            "measurement": measurement,
            "dataset": EDGE_DATASET,
            "features": feats,
        }, allow_val_change=True)

    # Sequential execution (no multiprocessing)
    results: List[Dict[str, Any]] = []
    try:
        for payload in jobs:
            res = _benchmark_worker(payload)
            if res is not None:
                results.append(res)
                enc_opts = res.get("encoder_opts", res.get("enc_opts", {}))
                meas = res.get("measurement", {})
                row = [
                    res.get("spec_hash", ""),
                    res.get("encoder", res.get("enc_name", "")),
                    _fmt_bool(enc_opts.get("hadamard", False)),
                    _fmt_bool(enc_opts.get("reupload", False)),
                    str(enc_opts.get("angle_range", enc_opts.get("angle_scale", "-"))),
                    f"{meas.get('name')}:{','.join(map(str, meas.get('wires', [])))}",
                    str(res.get("layers", "")),
                    res.get("ansatz", ""),
                    f"{res['metrics']['accuracy']:.4f}",
                    f"{res['metrics']['precision']:.4f}",
                    f"{res['metrics']['recall']:.4f}",
                    f"{res['metrics']['f1']:.4f}",
                    f"{res['metrics'].get('balanced_accuracy', float('nan')):.4f}",
                    f"{res['metrics'].get('auc', float('nan')):.4f}",
                    f"{res['metrics'].get('val_balanced_accuracy', float('nan')):.4f}",
                    f"{res['metrics'].get('threshold', float('nan')):.6f}",
                    f"{float(res.get('compile_time_s', float('nan'))):.2f}",
                    f"{float(res.get('core_train_time_s', float('nan'))):.2f}",
                    f"{float(res.get('wall_time_s', float('nan'))):.2f}",
                    os.path.basename(res.get("log_path", "")),
                ]
                if table is not None:
                    table.add_data(*row)
    finally:
        if aggregate_run is not None:
            if table is not None:
                aggregate_run.log({"benchmark_runs": table})
            if results:
                best = max(results, key=lambda r: objective(r.get("metrics", {})))
                aggregate_run.summary.update({
                    "best_spec_hash": best.get("spec_hash"),
                    "best_objective": objective(best.get("metrics", {})),
                    "best_encoder": best.get("encoder", best.get("enc_name")),
                    "best_ansatz": best.get("ansatz"),
                    "best_layers": best.get("layers"),
                    "best_measurement": best.get("measurement"),
                })
            aggregate_run.finish()

    # Prepare human-readable table
    headers = run_headers
    rows: List[List[str]] = []
    for r in results:
        enc_opts = r.get("encoder_opts", r.get("enc_opts", {}))
        meas = r.get("measurement", {})
        rows.append([
            r.get("spec_hash", ""),
            r.get("encoder", r.get("enc_name", "")),
            _fmt_bool(enc_opts.get("hadamard", False)),
            _fmt_bool(enc_opts.get("reupload", False)),
            str(enc_opts.get("angle_range", enc_opts.get("angle_scale", "-"))),
            f"{meas.get('name')}:{','.join(map(str, meas.get('wires', [])))}",
            str(r.get("layers", "")),
            r.get("ansatz", ""),
            f"{r['metrics']['accuracy']:.4f}",
            f"{r['metrics']['precision']:.4f}",
            f"{r['metrics']['recall']:.4f}",
            f"{r['metrics']['f1']:.4f}",
            f"{r['metrics'].get('balanced_accuracy', float('nan')):.4f}",
            f"{r['metrics'].get('auc', float('nan')):.4f}",
            f"{r['metrics'].get('val_balanced_accuracy', float('nan')):.4f}",
            f"{r['metrics'].get('threshold', float('nan')):.6f}",
            f"{float(r.get('compile_time_s', float('nan'))):.2f}",
            f"{float(r.get('core_train_time_s', float('nan'))):.2f}",
            f"{float(r.get('wall_time_s', float('nan'))):.2f}",
            os.path.basename(r.get("log_path", "")),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                w = len(cell)
                if w > widths[i]:
                    widths[i] = w

    sep = "-+-".join("-" * w for w in widths)
    print("\n===== EdgeIIoT Binary QML Benchmark =====")
    print(_pretty_row(headers, widths))
    print(sep)
    for row in rows:
        print(_pretty_row(row, widths))
    print("=======================================\n")


def _enc_opts_from_cfg(cfg) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "hadamard": bool(getattr(cfg, "hadamard", False)),
        "reupload": bool(getattr(cfg, "reupload", False)),
    }
    mode = str(getattr(cfg, "angle_mode", "none"))
    if mode == "range_0_pi":
        out["angle_range"] = "0_pi"
    elif mode == "range_pm_pi":
        out["angle_range"] = "pm_pi"
    elif mode == "range_pm_pi_2":
        out["angle_range"] = "pm_pi_2"
    elif mode.startswith("scale_"):
        try:
            out["angle_scale"] = float(mode.split("_", 1)[1])
        except Exception:
            pass
    return {k: v for k, v in out.items() if v is not None}


def _sweep_main() -> None:
    # Deprecated: unified into _sweep_train
    _sweep_train()


def _build_phase_a_sweep_config(*, ansatz_name: str) -> Dict[str, Any]:
    if ansatz_name != "ring_rot_cnot":
        raise ValueError(f"Unsupported Phase A ansatz: {ansatz_name}")
    return {
        "name": f"edgeiiot-phase-a-{ansatz_name}-sample{PHASE_A_SAMPLE}-b{EDGE_DEFAULT_BATCH}-e{PHASE_A_EPOCHS}",
        "method": "random",
        "metric": {"name": "objective", "goal": "maximize"},
        "parameters": {
            "phase": {"value": "phase_a"},
            "sample": {"value": PHASE_A_SAMPLE},
            "enc_name": {"values": PHASE_A_ENC_NAMES},
            "angle_mode": {"value": "range_pm_pi"},
            "hadamard": {"value": True},
            "reupload": {"value": True},
            "anz_name": {"value": ansatz_name},
            "measurement": {"values": PHASE_A_MEASUREMENTS},
            "layers": {"value": 3},
            "lr": {"values": PHASE_A_LR_VALUES},
            "epochs": {"value": PHASE_A_EPOCHS},
            "seed": {"values": PHASE_A_SEEDS},
        },
        # Keep early termination conservative; Phase A runs are already short.
        "early_terminate": {"type": "hyperband", "min_iter": 2, "s": 2},
    }

def _sweep_train() -> None:
    if _wandb_disabled():
        print("[ERROR] Sweep mode requires W&B. Unset WANDB_DISABLED or use EDGE_MODE=grid.")
        return
    cfg_kwargs = _wandb_base_kwargs(name=None, job_type="sweep-run")
    # Add phase tag if present in config later
    run = wandb.init(**cfg_kwargs)
    cfg = wandb.config
    phase_tag = f"phase:{getattr(cfg, 'phase', 'unknown')}"
    try:
        if run is not None:
            tags = list(getattr(run, "tags", []) or [])
            if phase_tag not in tags:
                run.tags = tags + [phase_tag]
    except Exception:
        pass
    feats = _active_features()
    meas_name = str(getattr(cfg, "measurement", "z0"))
    meas = (
        {"name": "mean_z", "wires": list(range(len(feats)))}
        if (meas_name == "mean_z" and feats)
        else {"name": "z0", "wires": [0]}
    )
    enc_opts = _enc_opts_from_cfg(cfg)
    # Allow env overrides for test_size/stratify to control exact train/test split
    env_test_size = os.environ.get("EDGE_TEST_SIZE")
    env_strat = os.environ.get("EDGE_STRATIFY", "1")
    train_params = {
        "lr": float(getattr(cfg, "lr", EDGE_DEFAULT_LR)),
        # Respect fixed batch size used in builders.run; do not sweep batch.
        "batch": EDGE_DEFAULT_BATCH,
        "epochs": int(getattr(cfg, "epochs", EDGE_DEFAULT_EPOCHS)),
        "class_weights": getattr(cfg, "class_weights", EDGE_DEFAULT_CLASS_WEIGHTS),
        "seed": int(getattr(cfg, "seed", EDGE_DEFAULT_SEED)),
    }
    if env_test_size is not None:
        try:
            train_params["test_size"] = float(env_test_size)
        except Exception:
            pass
    train_params["stratify"] = env_strat not in ("0", "false", "False")
    # Force compiled-safe encoder regardless of legacy sweep configs.
    forced_enc = "angle_embedding_y"
    if str(getattr(cfg, "enc_name", forced_enc)) != forced_enc:
        print(f"[SWEEP] Overriding enc_name={getattr(cfg, 'enc_name', None)} -> {forced_enc}")
    _ = run_one(
        sample=int(getattr(cfg, "sample", 120000)),
        enc_name=forced_enc,
        enc_opts=enc_opts,
        layers=int(getattr(cfg, "layers")),
        meas=meas,
        anz_name=str(getattr(cfg, "anz_name")),
        seed=int(getattr(cfg, "seed", 42)),
        train_params=train_params,
        use_current_wandb_run=True,
    )


def _run_phase_a_local(ansatz_name: str) -> None:
    if ansatz_name != "ring_rot_cnot":
        raise ValueError(f"Unsupported Phase A ansatz: {ansatz_name}")
    feats = _active_features()
    results: List[Dict[str, Any]] = []
    total = (
        len(PHASE_A_ENC_NAMES)
        * len(PHASE_A_ANGLE_MODES)
        * len(PHASE_A_MEASUREMENTS)
        * len(PHASE_A_LR_VALUES)
        * len(PHASE_A_SEEDS)
    )
    i = 0
    for enc_name in PHASE_A_ENC_NAMES:
        for angle_mode in PHASE_A_ANGLE_MODES:
            class _Cfg:
                pass
            cfg = _Cfg()
            cfg.hadamard = True
            cfg.reupload = True
            cfg.angle_mode = angle_mode
            enc_opts = _enc_opts_from_cfg(cfg)
            for meas_name in PHASE_A_MEASUREMENTS:
                meas = (
                    {"name": "mean_z", "wires": list(range(len(feats)))}
                    if (meas_name == "mean_z" and feats)
                    else {"name": "z0", "wires": [0]}
                )
                for lr in PHASE_A_LR_VALUES:
                    for seed in PHASE_A_SEEDS:
                        i += 1
                        print(
                            f"[PHASE-A] Run {i}/{total} | enc={enc_name} angle={angle_mode} "
                            f"meas={meas_name} lr={lr} seed={seed}",
                            flush=True,
                        )
                        res = run_one(
                            sample=PHASE_A_SAMPLE,
                            enc_name=enc_name,
                            enc_opts=enc_opts,
                            layers=3,
                            meas=meas,
                            anz_name=ansatz_name,
                            seed=int(seed),
                            train_params={
                                "lr": float(lr),
                                "batch": EDGE_DEFAULT_BATCH,
                                "epochs": int(PHASE_A_EPOCHS),
                                "class_weights": EDGE_DEFAULT_CLASS_WEIGHTS,
                                "seed": int(seed),
                            },
                            use_current_wandb_run=False,
                        )
                        results.append(res)
    if not results:
        print("[PHASE-A] No runs executed.")
        return
    best = max(results, key=lambda r: objective(r.get("metrics", {})))
    print(
        "[PHASE-A] Best run | "
        f"obj={objective(best.get('metrics', {})):.4f} "
        f"enc={best.get('encoder', best.get('enc_name'))} "
        f"ansatz={best.get('ansatz')} L={best.get('layers')} "
        f"seed={best.get('seed')} spec={best.get('spec_hash')}"
    )


def _create_sweep(ansatz_name: str, project: str, entity: str) -> str:
    cfg = _build_phase_a_sweep_config(ansatz_name=ansatz_name)
    sid = wandb.sweep(cfg, project=project, entity=entity)
    print(f"[W&B] Created new sweep {sid}")
    return sid


def _run_sweeps_autorun() -> None:
    if _wandb_disabled():
        print("[INFO] W&B disabled; sweeps require W&B. Use EDGE_MODE=grid instead.")
        return
    # Legacy entrypoint. Kept minimal: defaults to Phase A sweeps.
    explore_count = _env_int("EDGE_EXPLORE_COUNT", PHASE_A_RING_COUNT_DEFAULT)
    project = _wandb_project()
    entity = _wandb_entity()
    try:
        _wandb_ensure_login()
    except Exception as exc:
        print(f"Cannot login to W&B: {exc}")
        return
    explore_id = _create_sweep("ring_rot_cnot", project, entity)
    wandb.agent(explore_id, function=_sweep_train, count=explore_count)


def run_rf_baseline(sample: int = 60000, seed: int = 42, n_estimators: int = 200, class_weight: str = "balanced", stratify: bool = True, test_size: float = 0.2) -> Dict[str, Any]:
    feats = _active_features()
    measurement = {"name": "none", "wires": []}
    r = Recipe() | csv(EDGE_DATASET, sample_size=sample) | select(feats, label=EDGE_LABEL)
    r = r | quantile_uniform()
    r = r | pls_to_pow2(components=max(1, len(feats)))
    r = r | train(lr=0.0, batch=0, epochs=0, class_weights=None, seed=seed, test_size=test_size, stratify=stratify)
    r = r | rf_baseline(n_estimators=n_estimators, class_weight=class_weight, random_state=seed)
    recipe = r
    summary = run(recipe)
    try:
        cd = summary.get("class_distribution", {})
        print(f"Class distribution | train(+/−): {cd.get('train_pos')}/{cd.get('train_neg')} | test(+/−): {cd.get('test_pos')}/{cd.get('test_neg')}")
    except Exception:
        pass
    # Optionally log to W&B in a small run for traceability
    if not _wandb_disabled():
        try:
            _wandb_ensure_login()
            run_name = f"rf-baseline-s{seed}-{summary.get('dataset','edge')}"
            wkwargs = _wandb_base_kwargs(run_name, job_type="rf-baseline", tags=["baseline","rf"])
            w = wandb.init(**wkwargs)
            if "metrics" in summary:
                w.log({f"metrics/{k}": v for k, v in summary["metrics"].items()})
                w.summary.update(summary["metrics"])
            if "class_distribution" in summary:
                w.log({"class_distribution": summary["class_distribution"]})
            w.finish()
        except Exception:
            pass
    return summary

if __name__ == "__main__":
    # Minimal CLI to avoid accidental sweep launches (e.g. `--help` previously triggered sweeps).
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(
            "\n".join(
                [
                    "Usage:",
                    "  uv run python scripts/QML_ML-EdgeIIoT-benchmark.py            # grid (default)",
                    "  uv run python scripts/QML_ML-EdgeIIoT-benchmark.py --sweep    # one W&B sweep step (uses wandb.agent(function=...))",
                    "  uv run python scripts/QML_ML-EdgeIIoT-benchmark.py --rf       # random-forest baseline",
                    "  uv run python scripts/QML_ML-EdgeIIoT-benchmark.py --phase-a-ring",
                ]
            )
        )
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1].startswith("-") and sys.argv[1] not in (
        "--sweep",
        "--rf",
        "--phase-a-ring",
    ):
        print(f"Unknown option: {sys.argv[1]}. Use --help.")
        raise SystemExit(2)

    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        _sweep_train()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase-a-ring":
        _run_phase_a_local("ring_rot_cnot")
    elif len(sys.argv) > 1 and sys.argv[1] == "--rf":
        run_rf_baseline(
            sample=_env_int("EDGE_SAMPLE", BENCHMARK_DEFAULT_SAMPLE),
            seed=_env_int("EDGE_SEED", EDGE_DEFAULT_SEED),
            n_estimators=_env_int("RF_N_ESTIMATORS", 200),
            class_weight=os.environ.get("RF_CLASS_WEIGHT", "balanced"),
            stratify=os.environ.get("EDGE_STRATIFY", "1") not in ("0", "false", "False"),
            test_size=float(os.environ.get("EDGE_TEST_SIZE", "0.2")),
        )
    else:
        # Default behavior: grid. Sweeps must be explicitly requested.
        main()
