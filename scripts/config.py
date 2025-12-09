from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import hashlib
import json


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
    encoder: EncoderCfg
    ansatz: AnsatzCfg
    measurement: MeasurementCfg
    data: DataCfg
    train: TrainCfg
    meta: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload | {"schema_version": 1}

    def hash(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode()).hexdigest()[:10]
