"""Live terminal dashboard for TeCH soccer Stage2 training."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from humanoidverse.monitor import (
    as_float,
    format_count,
    format_duration,
    format_metric,
    metric_values,
    query_gpus,
    sparkline,
    trend_text,
)

DEFAULT_TOTAL_ITERATIONS = 30_000
METRIC_LOG_NAME = "localhost[0].log"
CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)\.pt$")

KICK_METRICS = (
    "reward_mean",
    "reward/kick/approach_progress",
    "reward/kick/ball_velocity",
    "reward/kick/correct_foot",
    "reward/kick/post_kick_stabilize",
    "reward/kick/wrong_foot",
)
CONTACT_METRICS = (
    "contact_rate",
    "correct_touch_rate",
    "has_valid_kick_rate",
    "ball_velocity_error_rate",
    "wrong_touch_rate",
    "termination_rate",
    "truncation_rate",
)
PPO_METRICS = ("policy_loss", "value_loss", "approx_kl", "entropy")


@dataclass(frozen=True)
class KickRunPaths:
    run_dir: Path
    metric_log: Path
    config: Path


def _metric_logs(root: Path) -> list[Path]:
    logs = []
    for path in root.rglob("localhost*.log"):
        if path.name == METRIC_LOG_NAME:
            logs.append(path)
    return logs


def resolve_kick_run(path: str | Path | None, search_root: str | Path = "runs") -> KickRunPaths:
    """Resolve a Stage2 run directory or the newest run under ``search_root``."""
    if path is None:
        logs = _metric_logs(Path(search_root).expanduser())
        if not logs:
            raise FileNotFoundError(f"No {METRIC_LOG_NAME} found under {Path(search_root).resolve()}")
        metric_log = max(logs, key=lambda item: item.stat().st_mtime)
        run_dir = metric_log.parents[2]
    else:
        supplied = Path(path).expanduser()
        if supplied.is_file():
            metric_log = supplied
            run_dir = supplied.parents[2]
        else:
            run_dir = supplied
            logs = _metric_logs(run_dir)
            metric_log = max(logs, key=lambda item: item.stat().st_mtime) if logs else run_dir / "torchrunx" / METRIC_LOG_NAME
    return KickRunPaths(run_dir.resolve(), metric_log.resolve(), (run_dir / "config.json").resolve())


def read_metric_rows(path: Path, history: int = 40) -> list[dict[str, str]]:
    """Read complete JSON metric objects emitted by rank zero."""
    if not path.exists():
        return []
    rows: deque[dict[str, str]] = deque(maxlen=history)
    decoder = json.JSONDecoder()
    with path.open(errors="replace") as file:
        for line in file:
            start = line.find("{")
            if start < 0:
                continue
            try:
                value, _ = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "iteration" in value:
                rows.append({str(key): str(item) for key, item in value.items()})
    return list(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def latest_checkpoint(run_dir: Path) -> tuple[Path | None, int | None]:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None, None
    iteration, path = max(candidates)
    return path, iteration


def _metric_table(rows: list[dict[str, str]], keys: tuple[str, ...], title: str) -> Table:
    table = Table(title=title, expand=True, box=None, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Recent", justify="right", style="dim")
    populated = False
    latest = rows[-1] if rows else {}
    for key in keys:
        values = metric_values(rows, key)
        if not values:
            continue
        populated = True
        table.add_row(key, format_metric(as_float(latest.get(key))), trend_text(values), sparkline(values))
    if not populated:
        table.add_row("No matching metrics yet", "-", "-", "-")
    return table


def _gpu_table(gpus: list[dict[str, str]]) -> Table:
    table = Table(title="GPUs", expand=True, box=None, padding=(0, 1))
    table.add_column("GPU", style="bold")
    table.add_column("Util", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Temp", justify="right")
    for gpu in gpus:
        table.add_row(
            f"{gpu['index']} {gpu['name']}",
            f"{gpu['utilization.gpu']}%",
            f"{gpu['memory.used']}/{gpu['memory.total']} MiB",
            f"{gpu['temperature.gpu']} C",
        )
    return table


def build_dashboard(
    paths: KickRunPaths,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    total_iterations: int,
    gpus: list[dict[str, str]],
) -> Group:
    checkpoint_path, checkpoint_iteration = latest_checkpoint(paths.run_dir)
    latest_iteration = as_float(rows[-1].get("iteration")) if rows else None
    log_age = max(0.0, time.time() - paths.metric_log.stat().st_mtime) if paths.metric_log.exists() else None
    if latest_iteration is not None and latest_iteration >= total_iterations:
        status, status_style = "COMPLETE", "green"
    elif log_age is None:
        status, status_style = "WAITING FOR LOG", "yellow"
    elif log_age <= 120:
        status, status_style = "ACTIVE", "green"
    elif log_age <= 600:
        status, status_style = "BETWEEN UPDATES", "yellow"
    else:
        status, status_style = "STALE", "red"

    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(justify="right")
    header.add_row(Text(paths.run_dir.name, style="bold bright_white"), Text(status, style=f"bold {status_style}"))
    header.add_row(str(paths.run_dir), "")
    header.add_row(
        f"iteration {format_count(latest_iteration)} / {format_count(total_iterations)}",
        f"updated {format_duration(log_age)} ago" if log_age is not None else "",
    )

    progress = Progress(
        TextColumn("[bold]Progress"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
        expand=True,
    )
    progress.add_task("training", total=total_iterations, completed=latest_iteration or 0)

    overview = Table.grid(expand=True, padding=(0, 2))
    for _ in range(4):
        overview.add_column(justify="center", ratio=1)
    overview.add_row("Latest iteration", "Checkpoint", "Envs / GPU", "World size")
    overview.add_row(
        Text(format_count(latest_iteration), style="bold"),
        Text(format_count(checkpoint_iteration), style="bold"),
        Text(str(config.get("num_envs", "1024")), style="bold"),
        Text(str(config.get("world_size", "8")), style="bold"),
    )
    if checkpoint_path is not None:
        overview.add_row("", str(checkpoint_path.name), "", "")

    parts: list[Any] = [Panel(Group(header, progress, overview), title="TeCH Kick Stage2 Monitor", border_style=status_style)]
    parts.append(
        Columns(
            [
                Panel(_metric_table(rows, KICK_METRICS, "Kick rewards"), border_style="cyan"),
                Panel(_metric_table(rows, CONTACT_METRICS, "Contact and safety"), border_style="yellow"),
            ],
            equal=True,
            expand=True,
        )
    )
    parts.append(
        Columns(
            [Panel(_metric_table(rows, PPO_METRICS, "PPO"), border_style="magenta"), Panel(_gpu_table(gpus), border_style="blue")],
            equal=True,
            expand=True,
        )
        if gpus
        else Panel(_metric_table(rows, PPO_METRICS, "PPO"), border_style="magenta")
    )
    parts.append(Align.center(Text(f"Rows: {len(rows)} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Ctrl-C to exit", style="dim")))
    return Group(*parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor TeCH soccer Stage2 training in the terminal.")
    parser.add_argument("path", nargs="?", help="Stage2 run directory or rank-zero metric log.")
    parser.add_argument("--search-root", default="runs", help="Directory searched for the newest Stage2 run.")
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh interval in seconds.")
    parser.add_argument("--history", type=int, default=40, help="Recent metric rows used for trends.")
    parser.add_argument("--total-iterations", type=int, default=DEFAULT_TOTAL_ITERATIONS)
    parser.add_argument("--no-gpu", action="store_true", help="Do not query nvidia-smi.")
    parser.add_argument("--once", action="store_true", help="Render once and exit.")
    args = parser.parse_args(argv)
    if args.refresh <= 0 or args.history < 2 or args.total_iterations <= 0:
        parser.error("refresh, history, and total-iterations must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = resolve_kick_run(args.path, args.search_root)
    except FileNotFoundError as exc:
        Console(stderr=True).print(f"[bold red]Error:[/] {exc}")
        return 2
    config = read_json(paths.config)
    total_iterations = int(config.get("iterations", args.total_iterations))
    console = Console()

    def render() -> Group:
        return build_dashboard(paths, read_metric_rows(paths.metric_log, args.history), config, total_iterations, [] if args.no_gpu else query_gpus())

    if args.once:
        console.print(render())
        return 0
    try:
        with Live(render(), console=console, refresh_per_second=4, screen=True) as live:
            while True:
                time.sleep(args.refresh)
                live.update(render(), refresh=True)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
