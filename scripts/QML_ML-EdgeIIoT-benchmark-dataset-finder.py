import os
import json
import time
import sys
import datetime as _dt
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import itertools
import random

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

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
    train,
    save,
    run,
    rf_baseline,
    quantile_uniform,
    pls_to_pow2,
)
from scripts.specs import (
    AnsatzCfg,
    DataCfg,
    EncoderCfg,
    MeasurementCfg,
    TrainCfg,
    build_and_validate_spec,
    build_feature_manifest,
    compliance_from_summary,
    flatten,
    provenance,
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


EDGE_DATASET = os.environ.get("EDGE_DATASET", "data/ML-EdgeIIoT-dataset-binario.csv")
# Columns come from QML_ML-EdgeIIoT-Binario.py
EDGE_FEATURES = [
    'ip.src_host',
    'ip.dst_host',
    'arp.dst.proto_ipv4',
    'arp.opcode',
    'arp.hw.size',
    'arp.src.proto_ipv4',
    'icmp.checksum',
    'icmp.seq_le',
    'icmp.transmit_timestamp',
    'icmp.unused',
    'http.file_data',
    'http.content_length',
    'http.request.uri.query',
    'http.request.method',
    'http.referer',
    'http.request.full_uri',
    'http.request.version',
    'http.response',
    'http.tls_port',
    'tcp.ack',
    'tcp.ack_raw',
    'tcp.checksum',
    'tcp.connection.fin',
    'tcp.connection.rst',
    'tcp.connection.syn',
    'tcp.connection.synack',
    'tcp.dstport',
    'tcp.flags',
    'tcp.flags.ack',
    'tcp.len',
    'tcp.options',
    'tcp.payload',
    'tcp.seq',
    'tcp.srcport',
    'udp.port',
    'udp.stream',
    'udp.time_delta',
    'dns.qry.name',
    'dns.qry.name.len',
    'dns.qry.qu',
    'dns.qry.type',
    'dns.retransmission',
    'dns.retransmit_request',
    'dns.retransmit_request_in',
    'mqtt.conack.flags',
    'mqtt.conflag.cleansess',
    'mqtt.conflags',
    'mqtt.hdrflags',
    'mqtt.len',
    'mqtt.msg_decoded_as',
    'mqtt.msg',
    'mqtt.msgtype',
    'mqtt.proto_len',
    'mqtt.protoname',
    'mqtt.topic',
    'mqtt.topic_len',
    'mqtt.ver',
    'mbtcp.len',
    'mbtcp.trans_id',
    'mbtcp.unit_id',
]
EDGE_LABEL = os.environ.get("EDGE_LABEL", "Attack_label")

WANDB_BASE_URL = "https://wandb.balbino.io"
os.environ.setdefault("WANDB_BASE_URL", WANDB_BASE_URL)
os.environ.setdefault("WANDB_HOST", WANDB_BASE_URL)
os.environ.setdefault("WANDB_API_HOST", WANDB_BASE_URL)
os.environ.setdefault("EDGE_CPU_FUSE_EPOCHS", "1")
os.environ.setdefault("EDGE_ENFORCE_NO_PY_CALLBACK", "0")
os.environ.setdefault("EDGE_PREFLIGHT_COMPILE", "1")
os.environ.setdefault("EDGE_CPU_FUSE_EPOCHS_CHUNK", "5")
# Reuse preprocessing artifact across sweep runs by default.
os.environ.setdefault("EDGE_PREPROCESS_ARTIFACT", os.path.join(_ROOT, "cache", "edgeiiot-preprocess"))
# CPU only: force a CPU-backed Lightning simulator.
os.environ["QML_DEVICE"] = "lightning.qubit"
_WANDB_SESSION_GROUP = f"edgeiiot-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
_WANDB_LOGIN_OK: Optional[bool] = None

# Default training / sweep hyperparameters (overridable via env or W&B sweeps)
EDGE_FIXED_EPOCHS = 50
EDGE_DEFAULT_SAMPLE = 120000
EDGE_DEFAULT_LR = 0.1
EDGE_DEFAULT_BATCH = 64
EDGE_DEFAULT_EPOCHS = EDGE_FIXED_EPOCHS
EDGE_DEFAULT_CLASS_WEIGHTS = "balanced"
EDGE_DEFAULT_SEED = 42
# Keep preprocessing split stable across sweep seeds.
os.environ.setdefault("EDGE_PREPROCESS_SPLIT_SEED", str(EDGE_DEFAULT_SEED))

# ---------------------------
# Phase A (fixed forever)
# ---------------------------
# This sweep is intentionally hard-coded (no env overrides, no conditional logic) so results
# stay comparable across time and compilation caching stays stable.

PHASE_A_SAMPLE = 80000
PHASE_A_EPOCHS = 50
PHASE_A_SEEDS = [42, 1337]
PHASE_A_LR_VALUES = [0.01, 0.005]

PHASE_A_ENC_NAMES = [
    "angle_embedding_y",
    "angle_pair_xy",
    "angle_pattern_xyz",
]
# "angle_mode" is interpreted by _enc_opts_from_cfg:
# - range_0_pi -> angle_range=0_pi
# - scale_X    -> angle_scale=float(X)
PHASE_A_ANGLE_MODES = [
    "scale_0.5",
]

PHASE_A_MEASUREMENTS = ["z0", "mean_z"]
PHASE_A_LAYERS_RING = [3]
PHASE_A_LAYERS_STRONGLY_ENTANGLING = [3]

PHASE_A_RING_COUNT_DEFAULT = 80
PHASE_A_STRONG_COUNT_DEFAULT = 60

EXPLORE_COUNT_DEFAULT = 8
EXPAND_COUNT_DEFAULT = 16

BENCHMARK_DEFAULT_SAMPLE = 60000

# ---------------------------
# Trio dataset search (PoC)
# ---------------------------
# Compact, fixed config for quickly ranking source datasets.
TRIO_SAMPLE = 60000
TRIO_SEEDS = [42, 1337, 2024]
TRIO_LR_VALUES = [0.01, 0.005]
TRIO_MEASUREMENTS = ["z0"]
TRIO_ENCODER = "angle_embedding_y"
TRIO_ANGLE_MODE = "scale_0.5"
TRIO_LAYERS = 3
TRIO_FIXED_DATASETS = [
    "CIC-BCCC-NRC-Edge-IIoTSet-2022",
    "CIC-BCCC-NRC-IoT-2023-Original Training and Testing",
    "CIC-BCCC-NRC-UQ-IOT-2022",
]

TRIO_NON_FEATURE_COLUMNS = {
    "Attack_label",
    "Label",
    "label_group",
    "attack_name_raw",
    "source_dataset",
    "source_file",
    "Attack Name",
    "Device",
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
}

# Limit number of features to a feasible qubit count for simulators
# Configurable via EDGE_NUM_FEATURES env (default 8)
def _active_features() -> List[str]:
    try:
        n = int(os.environ.get("EDGE_NUM_FEATURES", "8"))
    except Exception:
        n = 8
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


def _slug(s: str) -> str:
    import re as _re

    x = str(s or "").strip()
    if not x:
        return "unknown"
    x = _re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x[:120]


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "", "None") else default
    except Exception:
        return default


def _infer_features_for_dataset(dataset_path: str) -> List[str]:
    import pandas as pd

    feature_cap = _env_int("EDGE_TRIO_FEATURE_CAP", 60)
    numeric_ratio_min = _env_float("EDGE_TRIO_NUMERIC_RATIO_MIN", 0.8)

    # Read small sample for schema + numeric profiling.
    sample_df = pd.read_csv(
        dataset_path,
        nrows=3000,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    )
    cols = [str(c) for c in sample_df.columns]

    # Prefer classic feature list intersection when available.
    classic = [c for c in EDGE_FEATURES if c in cols]
    if classic:
        return classic[:feature_cap] if feature_cap > 0 else classic

    candidates = [c for c in cols if c not in TRIO_NON_FEATURE_COLUMNS]
    scored: List[Tuple[str, float]] = []
    for c in candidates:
        s = pd.to_numeric(sample_df[c], errors="coerce")
        ratio = float(s.notna().mean()) if len(s) else 0.0
        if ratio >= float(numeric_ratio_min):
            scored.append((c, ratio))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    chosen = [c for c, _ in scored]

    if not chosen:
        # Last-resort fallback: keep all non-label candidates.
        chosen = candidates

    if feature_cap > 0:
        chosen = chosen[:feature_cap]
    return chosen


def _default_train_params(seed: int) -> Dict[str, Any]:
    """
    Base training hyperparameters, with env overrides for easy tuning:
    - EDGE_LR
    - EDGE_CLASS_WEIGHTS
    """
    return {
        "lr": _env_float("EDGE_LR", EDGE_DEFAULT_LR),
        "batch": EDGE_DEFAULT_BATCH,
        "epochs": EDGE_DEFAULT_EPOCHS,
        "class_weights": os.environ.get("EDGE_CLASS_WEIGHTS", EDGE_DEFAULT_CLASS_WEIGHTS),
        "seed": seed,
    }


def build_recipe(sample: int,
                 enc_name: str,
                 enc_opts: Dict[str, Any],
                 layers: int,
                 meas: Dict[str, Any],
                 anz_name: str,
                 seed: int,
                 train_params: Optional[Dict[str, Any]] = None,
                 dataset_path: Optional[str] = None,
                 features_override: Optional[List[str]] = None) -> Recipe:
    # Use all raw features; supervised PLS will reduce to 8 components
    feats = features_override if features_override else EDGE_FEATURES
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        tp |= {
            k: train_params[k]
            for k in ("lr", "batch", "epochs", "class_weights", "seed", "test_size", "stratify")
            if k in train_params
        }
    # Use train-fitted quantile mapping and supervised PLS to 8 components for QML
    dataset = str(dataset_path or EDGE_DATASET)
    r = Recipe() | csv(dataset, sample_size=sample) | select(feats, label=EDGE_LABEL)
    r = r | quantile_uniform()
    r = r | pls_to_pow2(components=8)
    dev_name = os.environ.get("QML_DEVICE", "lightning.qubit")
    r = (
        r
        | device(dev_name, wires_from_features=True)
        | encoder(enc_name, **enc_opts)
        | ansatz(anz_name, layers=layers)
        | train(**tp)
    )
    # Amplitude embedding receives 8 PLS components (pow2), no PCA insertion needed
    # Insert measurement as a trailing step for builders.run to pick up
    r.parts.append(type(r.parts[0])(kind="measurement", params=meas))
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


def run_one(sample: int,
            enc_name: str,
            enc_opts: Dict[str, Any],
            layers: int,
            meas: Dict[str, Any],
            anz_name: str,
            seed: int,
            train_params: Optional[Dict[str, Any]] = None,
            use_current_wandb_run: bool = False,
            dataset_path: Optional[str] = None,
            features_override: Optional[List[str]] = None) -> Dict[str, Any]:
    measurement_cfg = MeasurementCfg(
        name=meas.get("name", "z0"),
        wires=[int(w) for w in meas.get("wires", [])],
    )
    active_features = features_override if features_override else _active_features()
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        tp |= {
            k: train_params[k]
            for k in ("lr", "batch", "epochs", "class_weights", "seed", "test_size", "stratify")
            if k in train_params
        }
    dataset = str(dataset_path or EDGE_DATASET)
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
        data=DataCfg(path=dataset, features=active_features, sample=sample),
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
    ds_slug = _slug(os.path.basename(dataset))
    run_name = f"{phase}-{ds_slug}-{enc_name}-{anz_name}-L{layers}-{reup_suffix}-s{seed}-{short_hash}"
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
        f"dataset:{ds_slug}",
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
        train_params=tp,
        dataset_path=dataset,
        features_override=active_features,
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
        "dataset_path": dataset,
        "features_used": active_features,
        "wandb_run": wandb_run.name if wandb_run is not None else run_name,
        "spec": spec_dict,
        "spec_hash": spec_hash,
        "spec_flat": flatten(spec_dict),
        "core_train_time_s": core_train_time_s,
        "compile_time_s": compile_time_s,
        "wall_time_s": wall_time_s,
    }
    return out


def _benchmark_worker(payload: Tuple[int, int, str, Dict[str, Any], str, Dict[str, Any], int]):
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
        ("angle_pair_xy", {"angle_scale": 0.5, "reupload": True}),
    ]

    # Mean-Z readout across active wires.
    feats = _active_features()
    measurement: Dict[str, Any] = {"name": "z0", "wires": [0]}

    # Grid of ansatz/layers, overridable via env lists
    anz_list = _env_list_str("EDGE_ANZ_LIST", ["ring_rot_cnot"]) or [
        os.environ.get("EDGE_ANZ", "ring_rot_cnot")
    ]
    layers_list = _env_list_int("EDGE_LAYERS_LIST", [3])

    sample = _env_int("EDGE_SAMPLE", 60000)
    seed = _env_int("EDGE_SEED", 42)

    jobs: List[Tuple[int, int, str, Dict[str, Any], str, Dict[str, Any], int]] = []
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
        "program": "scripts/QML_ML-EdgeIIoT-benchmark.py",
        "command": ["${env}", "python", "${program}", "--sweep"],
        "parameters": {
            "phase": {"value": "phase_a"},
            "sample": {"value": PHASE_A_SAMPLE},
            "enc_name": {"values": PHASE_A_ENC_NAMES},
            "angle_mode": {"value": "scale_0.5"},
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
    _ = run_one(
        sample=int(getattr(cfg, "sample", 120000)),
        enc_name=str(getattr(cfg, "enc_name")),
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
                            dataset_path=EDGE_DATASET,
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


def _collect_top_source_datasets(merged_csv_path: str, topn: int) -> List[str]:
    import pandas as pd
    from collections import Counter

    c = Counter()
    chunks = 0
    for chunk in pd.read_csv(
        merged_csv_path,
        dtype=str,
        usecols=["source_dataset"],
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        vals = chunk["source_dataset"].fillna("").astype(str).tolist()
        c.update([v for v in vals if v])
        if chunks % 20 == 0:
            print(f"[TRIO] Scanning source datasets... chunks={chunks}", flush=True)
    return [k for k, _ in c.most_common(max(int(topn), 1))]


def _collect_source_dataset_counts(merged_csv_path: str) -> Dict[str, int]:
    import pandas as pd
    from collections import Counter

    c = Counter()
    chunks = 0
    for chunk in pd.read_csv(
        merged_csv_path,
        dtype=str,
        usecols=["source_dataset"],
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        vals = chunk["source_dataset"].fillna("").astype(str).tolist()
        c.update([v for v in vals if v])
        if chunks % 20 == 0:
            print(f"[TRIO] Counting source datasets... chunks={chunks}", flush=True)
    return dict(c)


def _materialize_subsets_from_merged(merged_csv_path: str, dataset_names: List[str], out_dir: str) -> Dict[str, str]:
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    wanted = {d: os.path.join(out_dir, f"{_slug(d)}.csv") for d in dataset_names}
    header_written = {d: False for d in dataset_names}

    for p in wanted.values():
        if os.path.exists(p):
            os.remove(p)

    chunks = 0
    for chunk in pd.read_csv(
        merged_csv_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        if "source_dataset" not in chunk.columns:
            raise ValueError(
                f"Dataset '{merged_csv_path}' does not contain 'source_dataset'. "
                "Set EDGE_TRIO_DATASETS with explicit CSV paths instead."
            )
        for d in dataset_names:
            sub = chunk[chunk["source_dataset"] == d]
            if sub.empty:
                continue
            sub.to_csv(
                wanted[d],
                mode="a" if header_written[d] else "w",
                index=False,
                header=not header_written[d],
                encoding="utf-8",
            )
            header_written[d] = True
        if chunks % 20 == 0:
            ready = sum(1 for v in header_written.values() if v)
            print(
                f"[TRIO] Materializing subsets... chunks={chunks} datasets_with_rows={ready}/{len(dataset_names)}",
                flush=True,
            )

    return {d: p for d, p in wanted.items() if os.path.exists(p)}


def _materialize_sampled_subsets_from_merged(
    merged_csv_path: str,
    dataset_names: List[str],
    dataset_counts: Dict[str, int],
    out_dir: str,
    sample_rows_per_dataset: int,
    random_seed: int,
) -> Dict[str, str]:
    import pandas as pd
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    wanted = {d: os.path.join(out_dir, f"{_slug(d)}_sampled.csv") for d in dataset_names}
    header_written = {d: False for d in dataset_names}
    kept = {d: 0 for d in dataset_names}
    rng = np.random.default_rng(int(random_seed))
    target = set(dataset_names)

    for p in wanted.values():
        if os.path.exists(p):
            os.remove(p)

    probs: Dict[str, float] = {}
    for d in dataset_names:
        denom = max(int(dataset_counts.get(d, 0)), 1)
        probs[d] = min(1.0, float(sample_rows_per_dataset) / float(denom))

    chunks = 0
    for chunk in pd.read_csv(
        merged_csv_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        if "source_dataset" not in chunk.columns:
            raise ValueError(f"Dataset '{merged_csv_path}' does not contain 'source_dataset'.")
        sub = chunk[chunk["source_dataset"].fillna("").astype(str).isin(target)]
        if sub.empty:
            if chunks % 20 == 0:
                print(f"[TRIO] Sampling subsets... chunks={chunks}", flush=True)
            continue

        for d in dataset_names:
            remaining = int(sample_rows_per_dataset) - kept[d]
            if remaining <= 0:
                continue
            ds_sub = sub[sub["source_dataset"] == d]
            if ds_sub.empty:
                continue
            p = probs[d]
            if p >= 1.0:
                take = ds_sub
            else:
                mask = rng.random(len(ds_sub)) < p
                take = ds_sub[mask]
            if take.empty:
                continue
            if len(take) > remaining:
                take = take.sample(n=remaining, random_state=int(random_seed))
            take.to_csv(
                wanted[d],
                mode="a" if header_written[d] else "w",
                index=False,
                header=not header_written[d],
                encoding="utf-8",
            )
            header_written[d] = True
            kept[d] += len(take)

        if chunks % 20 == 0:
            ready = sum(1 for d in dataset_names if kept[d] > 0)
            print(
                f"[TRIO] Sampling subsets... chunks={chunks} nonempty={ready}/{len(dataset_names)} "
                f"mean_rows={int(sum(kept.values())/max(len(dataset_names),1))}",
                flush=True,
            )
        if all(kept[d] >= int(sample_rows_per_dataset) for d in dataset_names):
            break

    out = {d: p for d, p in wanted.items() if os.path.exists(p)}
    print("[TRIO] Sampled subset sizes:")
    for d in dataset_names:
        print(f"  - {d}: {kept[d]} rows")
    return out


def _concat_csv_files(paths: List[str], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total_lines = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, p in enumerate(paths):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for j, line in enumerate(f):
                    if i > 0 and j == 0:
                        continue
                    out.write(line)
                    total_lines += 1
    # subtract header row
    return max(total_lines - 1, 0)


def _profile_binary_dataset(csv_path: str) -> Dict[str, float]:
    import pandas as pd

    rows = 0
    pos = 0
    groups = set()
    usecols = None
    for chunk in pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
        usecols=usecols,
    ):
        rows += len(chunk)
        if "Attack_label" in chunk.columns:
            y = pd.to_numeric(chunk["Attack_label"], errors="coerce").fillna(0).astype(int)
            pos += int(y.sum())
        if "label_group" in chunk.columns:
            for g in chunk["label_group"].fillna("").astype(str).tolist():
                if g:
                    groups.add(g)
    benign_share = 1.0 - (float(pos) / float(rows)) if rows > 0 else float("nan")
    return {
        "rows": float(rows),
        "attack_share": float(pos) / float(rows) if rows > 0 else float("nan"),
        "benign_share": benign_share,
        "group_diversity": float(len(groups)),
    }


def _profile_datasets_from_merged(merged_csv_path: str, dataset_names: List[str]) -> Dict[str, Dict[str, float]]:
    import pandas as pd
    from collections import defaultdict

    target = set(dataset_names)
    rows = defaultdict(int)
    pos = defaultdict(int)
    groups = defaultdict(set)
    chunks = 0

    for chunk in pd.read_csv(
        merged_csv_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        if "source_dataset" not in chunk.columns:
            raise ValueError(f"Dataset '{merged_csv_path}' does not contain 'source_dataset'.")
        sub = chunk[chunk["source_dataset"].fillna("").astype(str).isin(target)]
        if not sub.empty:
            ds_vals = sub["source_dataset"].fillna("").astype(str)
            y_vals = (
                pd.to_numeric(sub["Attack_label"], errors="coerce").fillna(0).astype(int)
                if "Attack_label" in sub.columns
                else None
            )
            g_vals = sub["label_group"].fillna("").astype(str) if "label_group" in sub.columns else None
            for idx, ds in enumerate(ds_vals.tolist()):
                rows[ds] += 1
                if y_vals is not None:
                    pos[ds] += int(y_vals.iloc[idx])
                if g_vals is not None:
                    gv = g_vals.iloc[idx]
                    if gv:
                        groups[ds].add(gv)
        if chunks % 20 == 0:
            print(f"[TRIO] Profiling from merged... chunks={chunks}", flush=True)

    out: Dict[str, Dict[str, float]] = {}
    for ds in dataset_names:
        r = rows[ds]
        p = pos[ds]
        benign_share = 1.0 - (float(p) / float(r)) if r > 0 else float("nan")
        out[ds] = {
            "rows": float(r),
            "attack_share": float(p) / float(r) if r > 0 else float("nan"),
            "benign_share": benign_share,
            "group_diversity": float(len(groups[ds])),
        }
    return out


def _heuristic_dataset_score(profile: Dict[str, float]) -> float:
    rows = float(profile.get("rows", 0.0))
    benign = float(profile.get("benign_share", float("nan")))
    diversity = float(profile.get("group_diversity", 0.0))
    # Prefer medium-large datasets, some benign support, and diverse classes.
    size_term = min(rows / 1_500_000.0, 1.0) * 0.45
    if benign == benign:
        # Ideal benign share for quick binary PoC: ~10-35%
        if 0.10 <= benign <= 0.35:
            benign_term = 0.40
        elif 0.05 <= benign <= 0.50:
            benign_term = 0.28
        else:
            benign_term = 0.10
    else:
        benign_term = 0.15
    diversity_term = min(diversity / 6.0, 1.0) * 0.15
    return float(size_term + benign_term + diversity_term)


def _extract_from_merged() -> None:
    import pandas as pd

    merged_path = os.environ.get("EDGE_EXTRACT_MERGED_DATASET", EDGE_DATASET)
    source_file = os.environ.get("EDGE_EXTRACT_SOURCE_FILE", "").strip()
    source_dataset = os.environ.get("EDGE_EXTRACT_SOURCE_DATASET", "").strip()
    out_path = os.environ.get("EDGE_EXTRACT_OUT", os.path.join(_ROOT, "cache", "extracted_source.csv"))

    if not source_file and not source_dataset:
        print(
            "[EXTRACT] Set EDGE_EXTRACT_SOURCE_FILE (exact source_file) "
            "or EDGE_EXTRACT_SOURCE_DATASET (dataset folder name)."
        )
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    wrote = False
    rows = 0
    for chunk in pd.read_csv(
        merged_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        if "source_file" not in chunk.columns and "source_dataset" not in chunk.columns:
            raise ValueError(
                f"'{merged_path}' does not contain source provenance columns "
                "(source_file/source_dataset)."
            )
        mask = None
        if source_file:
            if "source_file" not in chunk.columns:
                raise ValueError("source_file column not found in merged dataset.")
            mask = (chunk["source_file"].fillna("").astype(str) == source_file)
        else:
            if "source_dataset" not in chunk.columns:
                raise ValueError("source_dataset column not found in merged dataset.")
            mask = (chunk["source_dataset"].fillna("").astype(str) == source_dataset)
        sub = chunk[mask]
        if sub.empty:
            continue
        sub.to_csv(out_path, mode="a" if wrote else "w", index=False, header=not wrote, encoding="utf-8")
        wrote = True
        rows += len(sub)

    if not wrote:
        print("[EXTRACT] No rows matched your selector.")
        return
    print(f"[EXTRACT] Wrote {rows} rows to: {out_path}")


def _prepare_fixed_trio_dataset() -> None:
    import pandas as pd

    merged_path = os.environ.get("EDGE_TRIO_MERGED_DATASET", EDGE_DATASET)
    out_path = os.environ.get(
        "EDGE_TRIO_OUT",
        os.path.join(_ROOT, "data", "processed", "edge_iot_trio_fixed_binary.csv"),
    )
    wanted = set(TRIO_FIXED_DATASETS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    wrote = False
    rows = 0
    chunks = 0
    for chunk in pd.read_csv(
        merged_path,
        dtype=str,
        chunksize=200000,
        low_memory=False,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    ):
        chunks += 1
        if "source_dataset" not in chunk.columns:
            raise ValueError(f"Dataset '{merged_path}' does not contain 'source_dataset'.")
        sub = chunk[chunk["source_dataset"].fillna("").astype(str).isin(wanted)]
        if sub.empty:
            if chunks % 20 == 0:
                print(f"[TRIO-PREP] chunks={chunks}", flush=True)
            continue
        sub.to_csv(out_path, mode="a" if wrote else "w", index=False, header=not wrote, encoding="utf-8")
        wrote = True
        rows += len(sub)
        if chunks % 20 == 0:
            print(f"[TRIO-PREP] chunks={chunks} rows_written={rows}", flush=True)

    if not wrote:
        raise RuntimeError("No rows written for fixed trio. Check merged dataset/source_dataset values.")
    print(f"[TRIO-PREP] Wrote fixed trio dataset: {out_path}")
    print(f"[TRIO-PREP] Rows: {rows}")
    print(f"[TRIO-PREP] Trio: {TRIO_FIXED_DATASETS}")


def _run_phase_a_trio_search(ansatz_name: str) -> None:
    if ansatz_name != "ring_rot_cnot":
        raise ValueError(f"Unsupported trio-search ansatz: {ansatz_name}")

    merged_path = os.environ.get("EDGE_TRIO_MERGED_DATASET", EDGE_DATASET)
    explicit_paths = _env_list_str("EDGE_TRIO_DATASETS", [])
    sample = _env_int("EDGE_TRIO_SAMPLE", TRIO_SAMPLE)
    random_sets = _env_int("EDGE_TRIO_RANDOM_SETS", 8)
    random_seed = _env_int("EDGE_TRIO_RANDOM_SEED", 1337)
    profile_only = os.environ.get("EDGE_TRIO_PROFILE_ONLY", "0").lower() in ("1", "true", "yes", "on")
    verbose = os.environ.get("EDGE_TRIO_VERBOSE", "1").lower() not in ("0", "false", "no", "off")
    smart_mode = os.environ.get("EDGE_TRIO_SMART_MODE", "1").lower() not in ("0", "false", "no", "off")
    sample_rows_per_dataset = _env_int("EDGE_TRIO_DATASET_SAMPLE_ROWS", 120000)

    if verbose:
        print("[TRIO] Configuration")
        print(f"  merged_path: {merged_path}")
        print(f"  profile_only: {profile_only}")
        print(f"  fixed_trio: {TRIO_FIXED_DATASETS}")
        print(f"  sample: {sample}")
        print(f"  random_sets: {random_sets}")
        print(f"  random_seed: {random_seed}")
        print(f"  smart_mode: {smart_mode}")
        print(f"  sample_rows_per_dataset: {sample_rows_per_dataset}")
        print(f"  measurements: {TRIO_MEASUREMENTS}")
        print(f"  lrs: {TRIO_LR_VALUES}")
        print(f"  seeds: {TRIO_SEEDS}")
        print("")

    dataset_map: Dict[str, str] = {}
    if explicit_paths:
        for p in explicit_paths:
            dataset_map[os.path.basename(p)] = p
        if verbose:
            print(f"[TRIO] Using {len(dataset_map)} explicit dataset paths from EDGE_TRIO_DATASETS")
        dataset_counts = {}
    else:
        dataset_counts = _collect_source_dataset_counts(merged_path)
        explicit_names = [d for d in TRIO_FIXED_DATASETS if d in dataset_counts]
        if verbose:
            print(f"[TRIO] Using fixed trio datasets: {explicit_names}")
        if profile_only:
            dataset_map = {n: merged_path for n in explicit_names}
            if verbose:
                print("[TRIO] Profile-only mode: skipping subset materialization.")
                print("")
        else:
            subset_dir = os.path.join(_ROOT, "cache", "trio_subsets")
            if smart_mode:
                dataset_map = _materialize_sampled_subsets_from_merged(
                    merged_csv_path=merged_path,
                    dataset_names=explicit_names,
                    dataset_counts=dataset_counts,
                    out_dir=subset_dir,
                    sample_rows_per_dataset=int(sample_rows_per_dataset),
                    random_seed=int(random_seed),
                )
            else:
                dataset_map = _materialize_subsets_from_merged(merged_path, explicit_names, subset_dir)
            if verbose:
                print(f"[TRIO] Prepared {len(dataset_map)} dataset subsets under: {subset_dir}")
                for ds_name, ds_path in dataset_map.items():
                    print(f"  - {ds_name} -> {ds_path}")
                print("")

    if len(dataset_map) < 3:
        print(
            f"[TRIO] Need fixed trio datasets present in merged data, got {len(dataset_map)}.",
            flush=True,
        )
        return

    feats = _active_features()
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.hadamard = True
    cfg.reupload = True
    cfg.angle_mode = TRIO_ANGLE_MODE
    enc_opts = _enc_opts_from_cfg(cfg)

    dataset_profiles: Dict[str, Dict[str, float]] = {}
    if profile_only and not explicit_paths:
        dataset_profiles = _profile_datasets_from_merged(merged_path, list(dataset_map.keys()))
    else:
        for ds_name, ds_path in dataset_map.items():
            dataset_profiles[ds_name] = _profile_binary_dataset(ds_path)
    if verbose:
        print("[TRIO] Dataset profiles")
        for ds_name, prof in sorted(dataset_profiles.items(), key=lambda kv: kv[1].get("rows", 0.0), reverse=True):
            print(
                f"  - {ds_name}: rows={int(prof.get('rows', 0.0))} "
                f"attack_share={float(prof.get('attack_share', float('nan'))):.4f} "
                f"benign_share={float(prof.get('benign_share', float('nan'))):.4f} "
                f"group_diversity={int(prof.get('group_diversity', 0.0))}"
            )
        print("")

    all_results: List[Dict[str, Any]] = []
    trio_benchmark_rows: List[Dict[str, Any]] = []
    if not profile_only:
        names_for_combos = sorted(dataset_map.keys())
        combos = list(itertools.combinations(names_for_combos, 3))
        rng = random.Random(int(random_seed))
        sampled_trios = rng.sample(combos, min(int(random_sets), len(combos))) if combos else []
        if verbose:
            print(
                f"[TRIO] Benchmarking randomized trios: total_combos={len(combos)} sampled={len(sampled_trios)}",
                flush=True,
            )
        combo_dir = os.path.join(_ROOT, "cache", "trio_combos")
        os.makedirs(combo_dir, exist_ok=True)
        total_runs = len(sampled_trios) * len(TRIO_LR_VALUES) * len(TRIO_SEEDS)
        run_i = 0
        for trio_idx, trio in enumerate(sampled_trios, start=1):
            trio_slug = _slug("__".join(trio))
            trio_csv = os.path.join(combo_dir, f"{trio_slug}.csv")
            trio_rows = _concat_csv_files([dataset_map[x] for x in trio], trio_csv)
            if verbose:
                print(
                    f"[TRIO] Trio {trio_idx}/{len(sampled_trios)} prepared | rows={trio_rows} | trio={trio}",
                    flush=True,
                )
            trio_features = _infer_features_for_dataset(trio_csv)
            if verbose:
                print(
                    f"[TRIO] Trio {trio_idx} inferred features: n={len(trio_features)} "
                    f"(first={trio_features[:6]})",
                    flush=True,
                )
            meas = {"name": "z0", "wires": [0]}
            trio_objs: List[float] = []
            for lr in TRIO_LR_VALUES:
                for seed in TRIO_SEEDS:
                    run_i += 1
                    print(
                        f"[TRIO] Run {run_i}/{total_runs} | trio={trio_idx} lr={lr} seed={seed}",
                        flush=True,
                    )
                    res = run_one(
                        sample=int(sample),
                        enc_name=TRIO_ENCODER,
                        enc_opts=enc_opts,
                        layers=int(TRIO_LAYERS),
                        meas=meas,
                        anz_name=ansatz_name,
                        seed=int(seed),
                        train_params={
                            "lr": float(lr),
                            "batch": EDGE_DEFAULT_BATCH,
                            "epochs": int(PHASE_A_EPOCHS),
                            "class_weights": EDGE_DEFAULT_CLASS_WEIGHTS,
                            "seed": int(seed),
                            "stratify": True,
                        },
                        use_current_wandb_run=False,
                        dataset_path=str(trio_csv),
                        features_override=trio_features,
                    )
                    obj = objective(res.get("metrics", {}))
                    trio_objs.append(obj)
                    res["trio_dataset_name"] = " | ".join(trio)
                    all_results.append(res)
            if trio_objs:
                mean_obj = sum(trio_objs) / len(trio_objs)
                # simple std without numpy dependency
                var = sum((x - mean_obj) ** 2 for x in trio_objs) / max(len(trio_objs) - 1, 1)
                trio_benchmark_rows.append(
                    {
                        "dataset": " | ".join(trio),
                        "runs": len(trio_objs),
                        "obj_mean": mean_obj,
                        "obj_best": max(trio_objs),
                        "obj_std": var ** 0.5,
                        "mode": "benchmark_trio",
                        "benign_share": float("nan"),
                        "rows_total": int(trio_rows),
                    }
                )

    rows: List[Dict[str, Any]] = []
    if profile_only:
        for ds, prof in dataset_profiles.items():
            proxy = _heuristic_dataset_score(prof)
            rows.append(
                {
                    "dataset": ds,
                    "runs": 0,
                    "obj_mean": proxy,
                    "obj_best": proxy,
                    "mode": "heuristic",
                    "benign_share": prof.get("benign_share", float("nan")),
                    "rows_total": int(prof.get("rows", 0.0)),
                }
            )
    else:
        rows = trio_benchmark_rows[:]
    rows.sort(key=lambda d: (d["obj_mean"], d["obj_best"]), reverse=True)

    top3 = rows[:3]
    print("\n[TRIO] Top dataset candidates:")
    for idx, r in enumerate(top3, start=1):
        print(
            f"  {idx}. {r['dataset']} | mean_obj={r['obj_mean']:.4f} | "
            f"best_obj={r['obj_best']:.4f} | runs={r['runs']} | "
            f"benign_share={float(r.get('benign_share', float('nan'))):.3f} | "
            f"rows={int(r.get('rows_total', 0))}"
        )

    # Randomized trio proposals from available candidates.
    trio_rows: List[Dict[str, Any]] = []
    if profile_only:
        names = [str(r["dataset"]) for r in rows]
        combos = list(itertools.combinations(names, 3))
        rng = random.Random(int(random_seed))
        if combos:
            if len(combos) > int(random_sets):
                sampled = rng.sample(combos, int(random_sets))
            else:
                sampled = combos
        else:
            sampled = []
        if verbose:
            print(
                f"[TRIO] Trio combinations: total={len(combos)} "
                f"sampled={len(sampled)}"
            )
        row_by_name = {str(r["dataset"]): r for r in rows}
        for c in sampled:
            members = [row_by_name[x] for x in c]
            mean_obj = sum(float(m["obj_mean"]) for m in members) / 3.0
            benign_vals = [float(m.get("benign_share", float("nan"))) for m in members if float(m.get("benign_share", float("nan"))) == float(m.get("benign_share", float("nan")))]
            benign_mean = (sum(benign_vals) / len(benign_vals)) if benign_vals else float("nan")
            balance_bonus = 0.03 if (benign_mean == benign_mean and 0.08 <= benign_mean <= 0.35) else 0.0
            trio_score = mean_obj + balance_bonus
            trio_rows.append(
                {
                    "datasets": c,
                    "score": trio_score,
                    "mean_obj": mean_obj,
                    "benign_mean": benign_mean,
                }
            )
        trio_rows.sort(key=lambda d: d["score"], reverse=True)
    else:
        for r in rows:
            trio_rows.append(
                {
                    "datasets": tuple(str(r["dataset"]).split(" | ")),
                    "score": float(r["obj_mean"]),
                    "mean_obj": float(r["obj_mean"]),
                    "benign_mean": float("nan"),
                    "obj_std": float(r.get("obj_std", float("nan"))),
                }
            )

    print("\n[TRIO] Trio candidates (top 5):")
    for idx, t in enumerate(trio_rows[:5], start=1):
        ds = " | ".join(t["datasets"])
        if profile_only:
            print(
                f"  {idx}. score={t['score']:.4f} obj_mean={t['mean_obj']:.4f} "
                f"benign_mean={t['benign_mean']:.3f} :: {ds}"
            )
        else:
            print(
                f"  {idx}. score={t['score']:.4f} obj_mean={t['mean_obj']:.4f} "
                f"obj_std={float(t.get('obj_std', float('nan'))):.4f} :: {ds}"
            )
    if verbose and trio_rows:
        chosen = trio_rows[0]
        if profile_only:
            print(
                "\n[TRIO] Selected best randomized trio rationale:\n"
                f"  score={chosen['score']:.4f} (mean_obj={chosen['mean_obj']:.4f}, "
                f"benign_mean={chosen['benign_mean']:.4f})\n"
                f"  datasets={chosen['datasets']}"
            )
        else:
            print(
                "\n[TRIO] Selected best randomized trio rationale:\n"
                f"  score={chosen['score']:.4f} (mean_obj={chosen['mean_obj']:.4f}, "
                f"obj_std={float(chosen.get('obj_std', float('nan'))):.4f})\n"
                f"  datasets={chosen['datasets']}"
            )

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_md = os.path.join(out_dir, f"edgeiiot_trio_search_{ts}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# EdgeIIoT Trio Dataset Search\n\n")
        f.write(f"- sample: {sample}\n")
        f.write(f"- seeds: {TRIO_SEEDS}\n")
        f.write(f"- lrs: {TRIO_LR_VALUES}\n")
        f.write(f"- measurements: {TRIO_MEASUREMENTS}\n")
        f.write(f"- encoder: {TRIO_ENCODER}\n")
        f.write(f"- angle_mode: {TRIO_ANGLE_MODE}\n")
        f.write(f"- layers: {TRIO_LAYERS}\n\n")
        f.write(f"- mode: {'heuristic' if profile_only else 'benchmark_trio'}\n\n")
        f.write("| Rank | Dataset | Runs | ObjMean | ObjBest | ObjStd | BenignShare | Rows |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for idx, r in enumerate(rows, start=1):
            f.write(
                f"| {idx} | {r['dataset']} | {r['runs']} | {r['obj_mean']:.4f} | "
                f"{r['obj_best']:.4f} | {float(r.get('obj_std', float('nan'))):.4f} | "
                f"{float(r.get('benign_share', float('nan'))):.4f} | "
                f"{int(r.get('rows_total', 0))} |\n"
            )
        f.write("\n")
        if top3:
            f.write("## Recommended Trio\n\n")
            for idx, r in enumerate(top3, start=1):
                f.write(f"{idx}. `{r['dataset']}`\n")
        if trio_rows:
            f.write("\n## Randomized Trio Candidates\n\n")
            f.write("| Rank | TrioScore | MeanObj | BenignMean | Datasets |\n")
            f.write("| --- | ---: | ---: | ---: | --- |\n")
            for idx, t in enumerate(trio_rows[:10], start=1):
                ds = " ; ".join(t["datasets"])
                f.write(
                    f"| {idx} | {t['score']:.4f} | {t['mean_obj']:.4f} | "
                    f"{t['benign_mean']:.4f} | {ds} |\n"
                )
    print(f"[TRIO] Wrote report: {out_md}")


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
    # Always derive 8 supervised components from all features for RF
    feats = EDGE_FEATURES
    measurement = {"name": "none", "wires": []}
    r = Recipe() | csv(EDGE_DATASET, sample_size=None) | select(feats, label=EDGE_LABEL)
    r = r | quantile_uniform()
    r = r | pls_to_pow2(components=8)
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
    # Default: run sweeps if W&B enabled, else grid. Set EDGE_MODE to override.
    mode = os.environ.get("EDGE_MODE", "").lower()
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        _run_phase_a_local("ring_rot_cnot")
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase-a-ring":
        _run_phase_a_local("ring_rot_cnot")
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase-a-trio-search":
        _run_phase_a_trio_search("ring_rot_cnot")
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-source":
        _extract_from_merged()
    elif len(sys.argv) > 1 and sys.argv[1] == "--prepare-fixed-trio":
        _prepare_fixed_trio_dataset()
    elif mode == "grid":
        main()
    elif mode == "rf":
        run_rf_baseline(
            sample=_env_int("EDGE_SAMPLE", BENCHMARK_DEFAULT_SAMPLE),
            seed=_env_int("EDGE_SEED", EDGE_DEFAULT_SEED),
            n_estimators=_env_int("RF_N_ESTIMATORS", 200),
            class_weight=os.environ.get("RF_CLASS_WEIGHT", "balanced"),
            stratify=os.environ.get("EDGE_STRATIFY", "1") not in ("0", "false", "False"),
            test_size=float(os.environ.get("EDGE_TEST_SIZE", "0.2")),
        )
    elif _wandb_disabled():
        # Sweeps require W&B; fall back to grid
        main()
    else:
        _run_sweeps_autorun()
