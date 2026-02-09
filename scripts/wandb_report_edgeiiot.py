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
    sweep_id: str
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
    angle_effective_scale: float
    ansatz: str
    layers: int
    measurement: str
    measurement_wires_count: int
    lr: float
    seed: int

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


def _run_sweep_id(r: Any) -> str:
    # W&B SDK/Api has historically exposed sweep info in multiple shapes.
    try:
        sid = getattr(r, "sweep_id", None)
        if sid:
            return str(sid)
    except Exception:
        pass
    try:
        sweep = getattr(r, "sweep", None)
        if sweep is None:
            return ""
        sid = getattr(sweep, "id", None) or getattr(sweep, "_id", None)  # type: ignore[attr-defined]
        return str(sid or "")
    except Exception:
        return ""


def _as_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("name") or "")
    return str(x or "")


def _row_from_run(r: Any) -> Optional[RunRow]:
    try:
        cfg = dict(getattr(r, "config", {}) or {})
        summ = dict(getattr(r, "summary", {}) or {})
        name = str(getattr(r, "name", "") or "")
        group = str(getattr(r, "group", "") or "")
        state = str(getattr(r, "state", "") or "")
        rid = str(getattr(r, "id", "") or "")
        created = str(getattr(r, "created_at", "") or "")
        sweep_id = _run_sweep_id(r)

        enc_cfg = cfg.get("encoder") if isinstance(cfg.get("encoder"), dict) else {}
        train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
        ans_cfg = cfg.get("ansatz") if isinstance(cfg.get("ansatz"), dict) else {}
        meas_cfg = cfg.get("measurement") if isinstance(cfg.get("measurement"), dict) else {}

        encoder = str(cfg.get("encoder_name") or enc_cfg.get("name") or cfg.get("enc_name") or "")
        had = int(bool(cfg.get("encoder_hadamard") if "encoder_hadamard" in cfg else enc_cfg.get("hadamard", cfg.get("hadamard", False))))
        reup = int(bool(cfg.get("encoder_reupload") if "encoder_reupload" in cfg else enc_cfg.get("reupload", cfg.get("reupload", False))))

        angle_range = str(enc_cfg.get("angle_range") or "")
        angle_scale = _safe_float(enc_cfg.get("angle_scale"))
        if angle_range == "0_pi":
            angle_mode = "range_0_pi"
            angle_eff = 3.141592653589793
        elif angle_scale == angle_scale and angle_scale > 0:
            angle_mode = f"scale_{angle_scale:g}"
            angle_eff = float(angle_scale)
        else:
            angle_mode = "scale_1.0"
            angle_eff = 1.0

        ansatz = str(cfg.get("anz_name") or ans_cfg.get("name") or "")
        layers = _safe_int(cfg.get("layers") or ans_cfg.get("layers") or 0) or 0

        measurement = str(cfg.get("measurement_name") or _as_name(cfg.get("measurement")) or meas_cfg.get("name") or cfg.get("measurement") or "")
        meas_wires = meas_cfg.get("wires") if isinstance(meas_cfg.get("wires"), list) else []
        meas_wires_count = _safe_int(cfg.get("measurement_wires_count") or (len(meas_wires) if meas_wires else 0)) or 0

        lr = _safe_float(cfg.get("train_lr") or cfg.get("lr") or train_cfg.get("lr"))
        seed = _safe_int(cfg.get("seed") or train_cfg.get("seed") or 0) or 0

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
            sweep_id=str(sweep_id or ""),
            run_id=rid,
            name=name,
            group=group,
            state=state,
            created_at=created,
            encoder=encoder,
            hadamard=had,
            reupload=reup,
            angle_mode=angle_mode,
            angle_effective_scale=float(angle_eff),
            ansatz=ansatz,
            layers=int(layers),
            measurement=measurement,
            measurement_wires_count=int(meas_wires_count),
            lr=float(lr),
            seed=int(seed),
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


def _group_key_with_lr(r: RunRow) -> Tuple[Any, ...]:
    return (
        r.encoder,
        r.hadamard,
        r.reupload,
        r.angle_mode,
        f"{r.angle_effective_scale:.6g}",
        r.ansatz,
        r.layers,
        r.measurement,
        r.measurement_wires_count,
        r.lr,
    )

def _group_key_arch_only(r: RunRow) -> Tuple[Any, ...]:
    return (
        r.encoder,
        r.hadamard,
        r.reupload,
        r.angle_mode,
        f"{r.angle_effective_scale:.6g}",
        r.ansatz,
        r.layers,
        r.measurement,
        r.measurement_wires_count,
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


def _seed_reduced(rs: List[RunRow]) -> Tuple[List[float], Dict[int, RunRow]]:
    """Reduce multiple runs per (config, seed) to a single representative (best objective)."""
    best_by_seed: Dict[int, RunRow] = {}
    for r in rs:
        cur = best_by_seed.get(r.seed)
        if cur is None or (r.objective, r.objective_per_s) > (cur.objective, cur.objective_per_s):
            best_by_seed[r.seed] = r
    objs = [rr.objective for rr in best_by_seed.values()]
    return objs, best_by_seed


def _write_markdown(path: str, title: str, rows: List[RunRow], topn: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Aggregate across seeds for each full config key (includes lr).
    buckets_full: Dict[Tuple[Any, ...], List[RunRow]] = {}
    buckets_arch: Dict[Tuple[Any, ...], List[RunRow]] = {}
    for r in rows:
        buckets_full.setdefault(_group_key_with_lr(r), []).append(r)
        buckets_arch.setdefault(_group_key_arch_only(r), []).append(r)

    full_agg = []
    for k, rs in buckets_full.items():
        objs_seed, best_by_seed = _seed_reduced(rs)
        best = max(best_by_seed.values(), key=lambda x: (x.objective, x.objective_per_s))
        full_agg.append(
            {
                "key": k,
                "runs": len(rs),
                "seeds": len(best_by_seed),
                "obj_mean": _mean(objs_seed),
                "obj_std": _stdev(objs_seed),
                "train_mean": _mean([x.train_s for x in best_by_seed.values()]),
                "compile_mean": _mean([x.compile_s for x in best_by_seed.values()]),
                "best_seed": best.seed,
                "best_obj": best.objective,
                "best_run_id": best.run_id,
            }
        )
    full_sorted = sorted(full_agg, key=lambda d: (d["obj_mean"], d["best_obj"]), reverse=True)

    # Architecture-only: for each arch key, pick the LR that maximizes seed-reduced mean objective.
    arch_best = []
    for ak, rs in buckets_arch.items():
        # Split by lr inside an architecture.
        by_lr: Dict[float, List[RunRow]] = {}
        for r in rs:
            by_lr.setdefault(float(r.lr), []).append(r)
        best_lr = None
        best_stat = None
        best_example = None
        for lr, lrrs in by_lr.items():
            objs_seed, best_by_seed = _seed_reduced(lrrs)
            stat = (_mean(objs_seed), -_stdev(objs_seed), len(best_by_seed))
            if best_stat is None or stat > best_stat:
                best_stat = stat
                best_lr = lr
                best_example = max(best_by_seed.values(), key=lambda x: (x.objective, x.objective_per_s)) if best_by_seed else None
        if best_lr is None or best_stat is None or best_example is None:
            continue
        arch_best.append(
            {
                "arch": ak,
                "best_lr": float(best_lr),
                "seeds": int(best_stat[2]),
                "obj_mean": float(best_stat[0]),
                "obj_std": _stdev([r.objective for r in by_lr[best_lr]]),
                "best_run_id": best_example.run_id,
                "best_seed": best_example.seed,
                "best_obj": best_example.objective,
            }
        )
    arch_sorted = sorted(arch_best, key=lambda d: (d["obj_mean"], d["best_obj"]), reverse=True)

    headers = [
        "Rank",
        "Enc",
        "H",
        "R",
        "Angle",
        "Scale",
        "Ansatz",
        "L",
        "Meas",
        "MW",
        "LR",
        "Seeds",
        "Runs",
        "Obj(mean±std)",
        "TrainS(mean)",
        "CompileS(mean)",
        "Best(run,seed,obj)",
    ]
    table_rows = []
    for i, d in enumerate(full_sorted[:topn], start=1):
        enc, had, reup, am, scale_s, anz, layers, meas, mw, lr = d["key"]
        table_rows.append(
            [
                str(i),
                str(enc),
                str(had),
                str(reup),
                str(am),
                str(scale_s),
                str(anz),
                str(layers),
                str(meas),
                str(mw),
                f"{float(lr):.4g}" if lr == lr else "nan",
                str(d["seeds"]),
                str(d["runs"]),
                f"{d['obj_mean']:.4f}±{d['obj_std']:.4f}",
                f"{d['train_mean']:.2f}",
                f"{d['compile_mean']:.2f}",
                f"{d['best_run_id']},s{d['best_seed']},{d['best_obj']:.4f}",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("Objective: 0.5*f1 + 0.3*balanced_accuracy + 0.2*auc\n\n")
        f.write(f"Runs: {len(rows)}\n\n")
        if arch_sorted:
            best_arch = arch_sorted[0]
            enc, had, reup, am, scale_s, anz, layers, meas, mw = best_arch["arch"]
            f.write("## Best Architecture (LR-Optimized)\n\n")
            f.write(
                f"- winner: enc={enc} had={had} reup={reup} angle={am} scale={scale_s} "
                f"ansatz={anz} L={layers} meas={meas} mw={mw}\n"
            )
            f.write(f"- best_lr: {best_arch['best_lr']:.4g}\n")
            f.write(f"- objective_mean: {best_arch['obj_mean']:.4f}\n")
            f.write(f"- best_run: {best_arch['best_run_id']} (seed {best_arch['best_seed']}, obj {best_arch['best_obj']:.4f})\n\n")
            f.write("## Best Full Configs (Seed-Reduced)\n\n")
        else:
            f.write("## Best Full Configs (Seed-Reduced)\n\n")
        f.write(_md_table(headers, table_rows))
        f.write("\n")

        if arch_sorted:
            f.write("\n## Top Architectures (LR-Optimized)\n\n")
            a_headers = ["Rank", "Enc", "H", "R", "Angle", "Scale", "Ansatz", "L", "Meas", "MW", "BestLR", "Seeds", "ObjMean", "Best(run,seed,obj)"]
            a_rows: List[List[str]] = []
            for i, d in enumerate(arch_sorted[: min(topn, 30)], start=1):
                enc, had, reup, am, scale_s, anz, layers, meas, mw = d["arch"]
                a_rows.append(
                    [
                        str(i),
                        str(enc),
                        str(had),
                        str(reup),
                        str(am),
                        str(scale_s),
                        str(anz),
                        str(layers),
                        str(meas),
                        str(mw),
                        f"{d['best_lr']:.4g}",
                        str(d["seeds"]),
                        f"{d['obj_mean']:.4f}",
                        f"{d['best_run_id']},s{d['best_seed']},{d['best_obj']:.4f}",
                    ]
                )
            f.write(_md_table(a_headers, a_rows))
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
    p.add_argument("--by-sweep", action="store_true", help="Write one report per sweep id (recommended).")
    p.add_argument("--sweep-id", default="", help="Restrict report to a single sweep id.")
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
    skipped = 0
    for r in runs:
        rr = _row_from_run(r)
        if rr is None:
            skipped += 1
            continue
        if name_re and not name_re.search(rr.name):
            continue
        if args.sweep_id and rr.sweep_id != args.sweep_id:
            continue
        # Only keep "real" experiment runs (ignore aggregate/meta runs).
        if not rr.spec_hash or not rr.encoder or not rr.ansatz or rr.layers <= 0:
            continue
        # Only keep runs that have core metrics; runs without metrics are not useful for comparisons.
        if (rr.f1 != rr.f1) and (rr.acc != rr.acc):
            continue
        rows.append(rr)

    tag = _now_tag()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.by_sweep:
        by_sweep: Dict[str, List[RunRow]] = {}
        for r in rows:
            by_sweep.setdefault(r.sweep_id or "no_sweep", []).append(r)
        index_lines: List[str] = []
        index_lines.append("# EdgeIIoT Sweep Index\n")
        index_lines.append("Objective: 0.5*f1 + 0.3*balanced_accuracy + 0.2*auc\n")
        index_lines.append(f"Total runs loaded: {len(rows)} (skipped parse: {skipped})\n")
        index_lines.append("| SweepID | Runs | BestArch(obj_mean) | BestArch(best_lr) | BestRunID |")
        index_lines.append("| --- | --- | --- | --- | --- |")
        for sid, srows in sorted(by_sweep.items(), key=lambda kv: len(kv[1]), reverse=True):
            # Sort by objective for stable "best run" pick.
            srows.sort(key=lambda x: (x.objective, x.objective_per_s), reverse=True)
            base = f"edgeiiot_sweep_{_slug(sid)}_{tag}"
            csv_path = os.path.join(out_dir, f"{base}.csv")
            md_path = os.path.join(out_dir, f"{base}.md")
            _write_csv(csv_path, srows)
            _write_markdown(md_path, title=f"EdgeIIoT Sweep Summary (sweep {sid})", rows=srows, topn=int(args.topn))

            # Pull a light "best arch" line by reusing the markdown reducer logic.
            # Cheaply recompute from the same functions here.
            buckets_arch: Dict[Tuple[Any, ...], List[RunRow]] = {}
            for rr in srows:
                buckets_arch.setdefault(_group_key_arch_only(rr), []).append(rr)
            best_arch_mean = float("nan")
            best_arch_lr = float("nan")
            for ak, rs in buckets_arch.items():
                by_lr: Dict[float, List[RunRow]] = {}
                for rr in rs:
                    by_lr.setdefault(float(rr.lr), []).append(rr)
                for lr, lrrs in by_lr.items():
                    objs_seed, best_by_seed = _seed_reduced(lrrs)
                    m = _mean(objs_seed)
                    if best_arch_mean != best_arch_mean or m > best_arch_mean:
                        best_arch_mean = m
                        best_arch_lr = float(lr)

            best_run = srows[0] if srows else None
            best_run_id = best_run.run_id if best_run else ""
            index_lines.append(f"| `{sid}` | {len(srows)} | {best_arch_mean:.4f} | {best_arch_lr:.4g} | `{best_run_id}` |")
        index_path = os.path.join(out_dir, f"edgeiiot_index_{tag}.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(index_lines) + "\n")
        print(f"Wrote index: {index_path}")
        print(f"Sweeps reported: {len(by_sweep)}")
    else:
        # Single report (legacy behavior)
        rows.sort(key=lambda x: (x.objective, x.objective_per_s), reverse=True)
        group = args.group or "all"
        base = f"edgeiiot_{_slug(group)}_{tag}"
        csv_path = os.path.join(out_dir, f"{base}.csv")
        md_path = os.path.join(out_dir, f"{base}.md")
        _write_csv(csv_path, rows)
        _write_markdown(md_path, title=f"EdgeIIoT Sweep Summary ({group})", rows=rows, topn=int(args.topn))
        print(f"Wrote CSV: {csv_path}")
        print(f"Wrote MD:  {md_path}")
        print(f"Runs loaded: {len(rows)} (skipped parse: {skipped})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
