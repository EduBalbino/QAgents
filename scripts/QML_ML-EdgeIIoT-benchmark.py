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
EDGE_FIXED_EPOCHS = 4
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

PHASE_A_SAMPLE = 120000
PHASE_A_EPOCHS = 100
PHASE_A_SEEDS = [42, 1337, 2024]
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

PHASE_A_MEASUREMENTS = ["z0"]
PHASE_A_LAYERS_RING = [3]
PHASE_A_LAYERS_STRONGLY_ENTANGLING = [3]

PHASE_A_RING_COUNT_DEFAULT = 80
PHASE_A_STRONG_COUNT_DEFAULT = 60

EXPLORE_COUNT_DEFAULT = 8
EXPAND_COUNT_DEFAULT = 16

BENCHMARK_DEFAULT_SAMPLE = 60000

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


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "", "None") else default
    except Exception:
        return default


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
                 train_params: Optional[Dict[str, Any]] = None) -> Recipe:
    # Use all raw features; supervised PLS will reduce to 8 components
    feats = EDGE_FEATURES
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        tp |= {
            k: train_params[k]
            for k in ("lr", "batch", "epochs", "class_weights", "seed", "test_size", "stratify")
            if k in train_params
        }
    # Use train-fitted quantile mapping and supervised PLS to 8 components for QML
    r = Recipe() | csv(EDGE_DATASET, sample_size=sample) | select(feats, label=EDGE_LABEL)
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
            use_current_wandb_run: bool = False) -> Dict[str, Any]:
    measurement_cfg = MeasurementCfg(
        name=meas.get("name", "z0"),
        wires=[int(w) for w in meas.get("wires", [])],
    )
    active_features = _active_features()
    tp: Dict[str, Any] = _default_train_params(seed)
    if train_params:
        tp |= {
            k: train_params[k]
            for k in ("lr", "batch", "epochs", "class_weights", "seed", "test_size", "stratify")
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
        data=DataCfg(path=EDGE_DATASET, features=active_features, sample=sample),
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
    recipe = build_recipe(sample, enc_name, enc_opts, layers, meas, anz_name, seed, train_params=tp)
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
