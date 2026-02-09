#!/usr/bin/env python3
"""
Programmatic W&B reporting for EdgeIIoT sweeps.

This script writes local artifacts (CSV + Markdown) to make sweep comparisons easy.
It does not require the W&B Reports API.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


OBJECTIVE_WEIGHTS = {"f1": 0.5, "balanced_accuracy": 0.3, "auc": 0.2}


def objective(m: Dict[str, Any]) -> float:
    return sum(OBJECTIVE_WEIGHTS[k] * float(m.get(k, 0.0) or 0.0) for k in OBJECTIVE_WEIGHTS)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "all"
    # Keep filenames safe and short.
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:120]


@dataclass(frozen=True)
class RunRow:
    run_id: str
    name: str
    group: str
    state: str
    created_at: str

    # config-like
    encoder: str
    hadamard: int
    reupload: int
    angle_mode: str
    ansatz: str
    layers: int
    measurement: str
    lr: float
    seed: int
    batch_forward_impl: str
    compiled_steps: int

    # metrics
    acc: float
    prec: float
    rec: float
    f1: float
    bacc: float
    auc: float
    val_bacc: float
    threshold: float

    # timings
    compile_s: float
    train_s: float
    wall_s: float

    # derived
    objective: float
    objective_per_s: float
    spec_hash: str


def _iter_wandb_runs(api, entity: Optional[str], project: str, filters: Dict[str, Any]) -> Iterable[Any]:
    # wandb.Api().runs supports path "entity/project" or just "project" (if default entity is set).
    path = f"{entity}/{project}" if entity else project
    return api.runs(path, filters=filters)


def _row_from_run(r: Any) -> Optional[RunRow]:
    try:
        cfg = dict(getattr(r, "config", {}) or {})
        summ = dict(getattr(r, "summary", {}) or {})
        name = str(getattr(r, "name", "") or "")
        group = str(getattr(r, "group", "") or "")
        state = str(getattr(r, "state", "") or "")
        rid = str(getattr(r, "id", "") or "")
        created = str(getattr(r, "created_at", "") or "")

        encoder = str(cfg.get("encoder_name") or cfg.get("enc_name") or _get(cfg, "encoder", "name", default="") or "")
        had = int(bool(cfg.get("encoder_hadamard") if "encoder_hadamard" in cfg else cfg.get("hadamard", False)))
        reup = int(bool(cfg.get("encoder_reupload") if "encoder_reupload" in cfg else cfg.get("reupload", False)))
        # Keep a simple angle mode label for grouping.
        angle_mode = str(cfg.get("angle_mode") or ("range_0_pi" if _get(cfg, "encoder", "angle_range") == "0_pi" else "scale_1.0"))
        ansatz = str(cfg.get("anz_name") or _get(cfg, "ansatz", "name", default="") or "")
        layers = _safe_int(cfg.get("layers") or _get(cfg, "ansatz", "layers", default=0) or 0) or 0
        measurement = str(cfg.get("measurement") or _get(cfg, "measurement", "name", default="") or "")
        lr = _safe_float(cfg.get("lr") or _get(cfg, "train", "lr", default=float("nan")))
        seed = _safe_int(cfg.get("seed") or _get(cfg, "train", "seed", default=0) or 0) or 0
        batch_forward_impl = str(cfg.get("batch_forward_impl") or cfg.get("EDGE_BATCH_FORWARD_IMPL") or "")
        compiled_steps = _safe_int(cfg.get("compiled_steps") or cfg.get("EDGE_COMPILED_STEPS") or 0) or 0

        # metrics (prefer summary.* stored by QML_ML-EdgeIIoT-benchmark.py)
        metrics = _get(summ, "metrics", default={}) if isinstance(_get(summ, "metrics", default={}), dict) else {}
        acc = _safe_float(metrics.get("accuracy", summ.get("accuracy")))
        prec = _safe_float(metrics.get("precision", summ.get("precision")))
        rec = _safe_float(metrics.get("recall", summ.get("recall")))
        f1 = _safe_float(metrics.get("f1", summ.get("f1")))
        bacc = _safe_float(metrics.get("balanced_accuracy", summ.get("balanced_accuracy")))
        auc = _safe_float(metrics.get("auc", summ.get("auc")))
        val_bacc = _safe_float(metrics.get("val_balanced_accuracy", summ.get("val_balanced_accuracy")))
        threshold = _safe_float(metrics.get("threshold", summ.get("threshold")))

        # timings (populated by updated benchmark script)
        compile_s = _safe_float(summ.get("time/compile_s", summ.get("compile_time_s", summ.get("compile_time"))))
        train_s = _safe_float(summ.get("time/core_train_s", summ.get("core_train_time_s", summ.get("train_time_s"))))
        wall_s = _safe_float(summ.get("time/wall_s", summ.get("wall_time_s")))

        spec_hash = str(cfg.get("spec_hash") or summ.get("spec_hash") or "")

        obj = objective({"f1": f1, "balanced_accuracy": bacc, "auc": auc})
        denom = train_s if train_s == train_s and train_s > 0 else wall_s
        obj_per_s = obj / max(float(denom) if denom == denom and denom > 0 else 1e-9, 1e-9)

        return RunRow(
            run_id=rid,
            name=name,
            group=group,
            state=state,
            created_at=created,
            encoder=encoder,
            hadamard=had,
            reupload=reup,
            angle_mode=angle_mode,
            ansatz=ansatz,
            layers=int(layers),
            measurement=measurement,
            lr=float(lr),
            seed=int(seed),
            batch_forward_impl=batch_forward_impl,
            compiled_steps=int(compiled_steps),
            acc=acc,
            prec=prec,
            rec=rec,
            f1=f1,
            bacc=bacc,
            auc=auc,
            val_bacc=val_bacc,
            threshold=threshold,
            compile_s=compile_s,
            train_s=train_s,
            wall_s=wall_s,
            objective=obj,
            objective_per_s=obj_per_s,
            spec_hash=spec_hash,
        )
    except Exception:
        return None


def _write_csv(path: str, rows: List[RunRow]) -> None:
    import csv

    fields = list(RunRow.__dataclass_fields__.keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k) for k in fields])


def _group_key(r: RunRow) -> Tuple[Any, ...]:
    return (
        r.encoder,
        r.hadamard,
        r.reupload,
        r.angle_mode,
        r.ansatz,
        r.layers,
        r.measurement,
        r.lr,
        r.batch_forward_impl,
        r.compiled_steps,
    )


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    return float(statistics.fmean(xs)) if xs else float("nan")


def _stdev(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return 0.0
    try:
        return float(statistics.stdev(xs))
    except Exception:
        return 0.0


def _write_markdown(path: str, title: str, rows: List[RunRow], topn: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Aggregate across seeds for each config key.
    buckets: Dict[Tuple[Any, ...], List[RunRow]] = {}
    for r in rows:
        buckets.setdefault(_group_key(r), []).append(r)

    agg = []
    for k, rs in buckets.items():
        rs_sorted = sorted(rs, key=lambda x: (x.objective, x.objective_per_s), reverse=True)
        best = rs_sorted[0]
        objs = [x.objective for x in rs]
        train_s = [x.train_s for x in rs]
        comp_s = [x.compile_s for x in rs]
        agg.append(
            {
                "key": k,
                "n": len(rs),
                "obj_mean": _mean(objs),
                "obj_std": _stdev(objs),
                "train_mean": _mean(train_s),
                "compile_mean": _mean(comp_s),
                "best_seed": best.seed,
                "best_obj": best.objective,
                "best_obj_per_s": best.objective_per_s,
            }
        )
    agg_sorted = sorted(agg, key=lambda d: (d["obj_mean"], d["best_obj"]), reverse=True)

    headers = [
        "Rank",
        "Enc",
        "H",
        "R",
        "Angle",
        "Ansatz",
        "L",
        "Meas",
        "LR",
        "Impl",
        "CSteps",
        "N",
        "Obj(mean±std)",
        "TrainS(mean)",
        "CompileS(mean)",
        "Best(seed,obj,obj/s)",
    ]
    table_rows = []
    for i, d in enumerate(agg_sorted[:topn], start=1):
        enc, had, reup, am, anz, layers, meas, lr, impl, csteps = d["key"]
        table_rows.append(
            [
                str(i),
                str(enc),
                str(had),
                str(reup),
                str(am),
                str(anz),
                str(layers),
                str(meas),
                f"{float(lr):.4g}" if lr == lr else "nan",
                str(impl),
                str(csteps),
                str(d["n"]),
                f"{d['obj_mean']:.4f}±{d['obj_std']:.4f}",
                f"{d['train_mean']:.2f}",
                f"{d['compile_mean']:.2f}",
                f"s{d['best_seed']},{d['best_obj']:.4f},{d['best_obj_per_s']:.4f}",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("Objective: 0.5*f1 + 0.3*balanced_accuracy + 0.2*auc\n\n")
        f.write(f"Runs: {len(rows)}\n\n")
        f.write(_md_table(headers, table_rows))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate programmatic W&B sweep reports for EdgeIIoT.")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "qml-edgeiiot"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--group", default=os.environ.get("WANDB_GROUP", ""), help="Run group to report on (recommended).")
    p.add_argument("--tag", default="", help="Optional tag filter (exact match).")
    p.add_argument("--name-re", default="", help="Optional regex filter on run name.")
    p.add_argument("--state", default="finished", help="Run state filter (default: finished).")
    p.add_argument("--out-dir", default="reports", help="Local output directory.")
    p.add_argument("--topn", type=int, default=30, help="Number of aggregated configs to include in markdown.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise SystemExit(f"wandb is required for this script: {exc}")

    # Respect custom host if configured in env.
    host = os.environ.get("WANDB_BASE_URL") or os.environ.get("WANDB_HOST") or None
    try:
        wandb.login(host=host, relogin=bool(os.environ.get("WANDB_FORCE_RELOGIN")))  # type: ignore[arg-type]
    except Exception:
        # Allow offline/no-login if the machine already has creds configured.
        pass

    api = wandb.Api()  # type: ignore

    filters: Dict[str, Any] = {}
    entity = args.entity.strip() or None
    if args.group:
        filters["group"] = args.group
    if args.state:
        filters["state"] = args.state
    if args.tag:
        filters["tags"] = {"$in": [args.tag]}

    name_re = re.compile(args.name_re) if args.name_re else None

    runs = list(_iter_wandb_runs(api, entity, args.project, filters))
    rows: List[RunRow] = []
    for r in runs:
        rr = _row_from_run(r)
        if rr is None:
            continue
        if name_re and not name_re.search(rr.name):
            continue
        # Only keep runs that have core metrics; sweep runs without metrics are not useful for comparisons.
        if rr.f1 != rr.f1 and rr.acc != rr.acc:
            continue
        rows.append(rr)

    # Sort by objective (best first)
    rows.sort(key=lambda x: (x.objective, x.objective_per_s), reverse=True)

    group = args.group or "all"
    tag = _now_tag()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    base = f"edgeiiot_{_slug(group)}_{tag}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    md_path = os.path.join(out_dir, f"{base}.md")

    _write_csv(csv_path, rows)
    _write_markdown(md_path, title=f"EdgeIIoT Sweep Summary ({group})", rows=rows, topn=int(args.topn))

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote MD:  {md_path}")
    print(f"Runs loaded: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
