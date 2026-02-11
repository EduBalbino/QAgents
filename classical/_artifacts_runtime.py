"""Runtime bootstrap to persist script outputs under ./artifacts.

This module is intentionally lightweight and non-invasive:
- mirrors stdout/stderr to log files;
- configures file logging under artifacts;
- auto-saves matplotlib and plotly figures to artifacts.
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_STATE: dict[str, Any] = {}


class _TeeStream(io.TextIOBase):
    def __init__(self, original_stream: io.TextIOBase, mirror_file: Path) -> None:
        self._original_stream = original_stream
        self._mirror_handle = mirror_file.open("a", encoding="utf-8", buffering=1)

    @property
    def encoding(self) -> str:
        return getattr(self._original_stream, "encoding", "utf-8")

    def write(self, data: str) -> int:
        if data is None:
            return 0
        text = data if isinstance(data, str) else str(data)
        self._original_stream.write(text)
        self._mirror_handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._original_stream.flush()
        if not self._mirror_handle.closed:
            self._mirror_handle.flush()

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
        if not self._mirror_handle.closed:
            self._mirror_handle.close()

    def isatty(self) -> bool:
        return bool(getattr(self._original_stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original_stream.fileno()

    def writable(self) -> bool:
        return True


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "script"


def _next_path(figures_dir: Path, prefix: str, ext: str) -> Path:
    _STATE["figure_counter"] += 1
    return figures_dir / f"{_STATE['figure_counter']:05d}_{_slugify(prefix)}.{ext}"


def _patch_matplotlib(figures_dir: Path) -> None:
    try:
        import matplotlib.figure as mpl_figure
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("matplotlib patch skipped: %s", exc)
        return

    if getattr(plt, "_artifacts_runtime_patched", False):
        return

    original_show = plt.show
    original_plt_savefig = plt.savefig
    original_fig_savefig = mpl_figure.Figure.savefig

    def _mirror_figure(fig: Any, prefix: str) -> None:
        try:
            mirror_path = _next_path(figures_dir, prefix, "png")
            original_fig_savefig(fig, mirror_path, dpi=150, bbox_inches="tight")
            logging.getLogger(__name__).info("Saved matplotlib figure -> %s", mirror_path)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("Failed to mirror matplotlib figure: %s", exc)

    def patched_show(*args: Any, **kwargs: Any) -> Any:
        for fig_number in plt.get_fignums():
            fig = plt.figure(fig_number)
            _mirror_figure(fig, f"show_fig_{fig_number}")
        return original_show(*args, **kwargs)

    def patched_plt_savefig(*args: Any, **kwargs: Any) -> Any:
        result = original_plt_savefig(*args, **kwargs)
        _mirror_figure(plt.gcf(), "savefig")
        return result

    def patched_fig_savefig(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fig_savefig(self, *args, **kwargs)
        _mirror_figure(self, "figure_savefig")
        return result

    plt.show = patched_show
    plt.savefig = patched_plt_savefig
    mpl_figure.Figure.savefig = patched_fig_savefig
    plt._artifacts_runtime_patched = True


def _patch_plotly(figures_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("plotly patch skipped: %s", exc)
        return

    if getattr(go.Figure, "_artifacts_runtime_patched", False):
        return

    original_show = go.Figure.show
    original_write_html = go.Figure.write_html
    original_write_image = getattr(go.Figure, "write_image", None)

    def _mirror_plotly_html(fig: Any, prefix: str) -> None:
        try:
            mirror_path = _next_path(figures_dir, prefix, "html")
            original_write_html(fig, str(mirror_path), include_plotlyjs="cdn")
            logging.getLogger(__name__).info("Saved plotly figure -> %s", mirror_path)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("Failed to mirror plotly figure: %s", exc)

    def patched_show(self: Any, *args: Any, **kwargs: Any) -> Any:
        _mirror_plotly_html(self, "plotly_show")
        return original_show(self, *args, **kwargs)

    def patched_write_html(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_write_html(self, *args, **kwargs)
        _mirror_plotly_html(self, "plotly_write_html")
        return result

    go.Figure.show = patched_show
    go.Figure.write_html = patched_write_html

    if original_write_image is not None:
        def patched_write_image(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_write_image(self, *args, **kwargs)
            _mirror_plotly_html(self, "plotly_write_image")
            return result

        go.Figure.write_image = patched_write_image

    go.Figure._artifacts_runtime_patched = True


def setup_artifacts(script_path: str | None = None) -> dict[str, str]:
    """Initialize artifacts capture once per process and return run paths."""
    if _STATE.get("initialized"):
        return _STATE["context"]

    cwd = Path.cwd()
    # Force artifacts under the classical workspace, independent of caller cwd.
    artifacts_root = Path(__file__).resolve().parent / "artifacts"
    script_name = Path(script_path or (sys.argv[0] if sys.argv else "script")).stem
    script_slug = _slugify(script_name)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = artifacts_root / script_slug / run_id
    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"

    for directory in (artifacts_root, run_dir, logs_dir, figures_dir, tables_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stdout_log = logs_dir / "stdout.log"
    stderr_log = logs_dir / "stderr.log"

    tee_stdout = _TeeStream(sys.stdout, stdout_log)
    tee_stderr = _TeeStream(sys.stderr, stderr_log)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    runtime_log = logs_dir / "runtime.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(runtime_log, encoding="utf-8"),
        ],
        force=True,
    )
    logging.captureWarnings(True)

    _STATE["figure_counter"] = 0
    _patch_matplotlib(figures_dir)
    _patch_plotly(figures_dir)

    context = {
        "artifacts_root": str(artifacts_root),
        "run_dir": str(run_dir),
        "logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "reports_dir": str(reports_dir),
    }

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "script": script_path or (sys.argv[0] if sys.argv else ""),
                "cwd": str(cwd),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "context": context,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.getLogger(__name__).info("Artifacts capture initialized at: %s", run_dir)

    def _close_streams() -> None:
        try:
            tee_stdout.flush()
            tee_stderr.flush()
        except Exception:
            pass
        try:
            tee_stdout.close()
            tee_stderr.close()
        except Exception:
            pass

    atexit.register(_close_streams)

    _STATE["initialized"] = True
    _STATE["context"] = context
    return context
