from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json
import os
import platform
import subprocess

from pydantic import BaseModel, Field, ValidationError, field_validator

ALLOWED_ENCODERS = {"angle_embedding_y", "angle_pair_xy", "amplitude_embedding"}
ALLOWED_ANZ = {"ring_rot_cnot", "strongly_entangling"}
ALLOWED_MEASUREMENTS = {"mean_z", "z0"}


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
        if sample is None or sample < 1:
            raise ValueError("data.sample must be >= 1")
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
    if spec.encoder.name == "amplitude_embedding":
        expected_qubits = summary.get("pca_qubits") or summary.get("pca_dim") or expected_qubits
        if summary.get("pca_qubits") is None and summary.get("pca_dim") is None:
            diff["pca_expected_but_missing"] = True
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
        manifest.append({
            "feature": feature,
            "original_index": idx,
            "selected": int(feature in used_set or str(idx) in used_set),
            "post_pca_index": pca_map.get(idx),
        })
    return manifest


__all__ = [
    "ALLOWED_ENCODERS",
    "ALLOWED_ANZ",
    "ALLOWED_MEASUREMENTS",
    "EncoderCfg",
    "AnsatzCfg",
    "MeasurementCfg",
    "DataCfg",
    "TrainCfg",
    "ExperimentSpec",
    "SpecValidator",
    "build_and_validate_spec",
    "flatten",
    "provenance",
    "compliance_from_summary",
    "build_feature_manifest",
]
