"""Live terminal dashboard for an active UFO training run."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import statistics
import subprocess
import time
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

TRAIN_LOG_FILENAME = "train_log.txt"
CONFIG_FILENAME = "config.json"
CHECKPOINT_STATUS_PATH = Path("checkpoint/train_status.json")

CORE_METRICS = (
    "actor_loss",
    "critic_loss",
    "fb_loss",
    "disc_loss",
    "disc_train_loss",
    "mean_disc_reward",
    "mean_aux_reward",
    "mean_next_Q",
    "mean_next_auxQ",
)

ROBOT_METRICS = (
    "torso_z_mean",
    "torso_z_min",
    "torso_z_p05",
    "torso_contact_force_max",
    "torso_contact_force_contact_frac",
)

AUX_REWARD_METRICS = (
    "aux_rew/limits_dof_pos",
    "aux_rew/limits_torque",
    "aux_rew/penalty_action_rate",
    "aux_rew/penalty_ankle_roll",
    "aux_rew/penalty_feet_ori",
    "aux_rew/penalty_slippage",
    "aux_rew/penalty_torques",
    "aux_rew/penalty_undesired_contact",
)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    train_log: Path
    config: Path
    checkpoint_status: Path


def resolve_run_paths(path: str | Path | None, search_root: str | Path = "runs") -> RunPaths:
    if path is None:
        candidates = list(Path(search_root).glob(f"**/{TRAIN_LOG_FILENAME}"))
        if not candidates:
            raise FileNotFoundError(
                f"No {TRAIN_LOG_FILENAME} found under {Path(search_root).resolve()}. Pass a run directory or log file explicitly."
            )
        train_log = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
        run_dir = train_log.parent
    else:
        supplied = Path(path).expanduser()
        if supplied.is_file() or supplied.name == TRAIN_LOG_FILENAME:
            train_log = supplied
            run_dir = supplied.parent
        else:
            run_dir = supplied
            train_log = run_dir / TRAIN_LOG_FILENAME

    return RunPaths(
        run_dir=run_dir.resolve(),
        train_log=train_log.resolve(),
        config=(run_dir / CONFIG_FILENAME).resolve(),
        checkpoint_status=(run_dir / CHECKPOINT_STATUS_PATH).resolve(),
    )


def read_csv_rows(path: Path, history: int = 40) -> list[dict[str, str]]:
    """Read complete CSV rows while tolerating a concurrent append."""
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as file:
        reader = csv.reader(file)
        try:
            fieldnames = next(reader)
        except StopIteration:
            return []
        field_count = len(fieldnames)
        rows: list[dict[str, str]] = []
        schema_mismatch = False
        for values in reader:
            if not values:
                continue
            if len(values) != field_count:
                schema_mismatch = True
                continue
            rows.append(dict(zip(fieldnames, values, strict=True)))

    # A resumed run can append rows from a newer metric schema without
    # rewriting the existing CSV header. Use the structured train.log in that
    # case so the dashboard does not silently show stale pre-resume rows.
    if schema_mismatch and path.name == TRAIN_LOG_FILENAME:
        fallback = read_train_log_rows(path.with_name("train.log"), history)
        if fallback:
            return fallback
    return rows[-history:]


def read_train_log_rows(path: Path, history: int = 40) -> list[dict[str, str]]:
    """Read metric dictionaries emitted by the human-readable training log."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(errors="replace") as file:
        for line in file:
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                value = ast.literal_eval(line[start : end + 1])
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, dict):
                rows.append({str(key): str(item) for key, item in value.items()})
    return rows[-history:]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as file:
            value = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def latest_number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def format_count(value: float | int | None) -> str:
    if value is None:
        return "-"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:,.0f}"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "-"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    absolute = abs(value)
    if absolute >= 10000 or (absolute != 0 and absolute < 0.001):
        return f"{value:.3e}"
    return f"{value:.4f}"


def sparkline(values: list[float], width: int = 12) -> str:
    values = [value for value in values if math.isfinite(value)][-width:]
    if not values:
        return "-"
    if len(values) == 1:
        return "."
    low, high = min(values), max(values)
    levels = "._-~=+*#"
    if math.isclose(low, high):
        return "-" * len(values)
    return "".join(levels[min(len(levels) - 1, int((value - low) / (high - low) * len(levels)))] for value in values)


def metric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def trend_text(values: list[float]) -> Text:
    if len(values) < 2:
        return Text("-")
    previous = values[-2]
    current = values[-1]
    difference = current - previous
    scale = max(abs(previous), 1e-12)
    percent = difference / scale * 100
    if math.isclose(difference, 0.0, abs_tol=1e-12):
        return Text("0.0%", style="dim")
    style = "cyan" if difference > 0 else "magenta"
    return Text(f"{percent:+.1f}%", style=style)


def query_gpus() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) == 7:
            rows.append(dict(zip(fields.split(","), columns, strict=True)))
    return rows


def status_for_log(
    path: Path,
    total_steps: float | None,
    current_steps: float | None,
    expected_update_seconds: float | None = None,
) -> tuple[str, str, float | None]:
    if not path.exists():
        return "WAITING FOR LOG", "yellow", None
    age = max(0.0, time.time() - path.stat().st_mtime)
    if total_steps is not None and current_steps is not None and current_steps >= total_steps:
        return "COMPLETE", "green", age
    active_threshold = max(180.0, (expected_update_seconds or 0.0) * 1.5)
    stale_threshold = max(600.0, (expected_update_seconds or 0.0) * 3.0)
    if age <= active_threshold:
        return "ACTIVE", "green", age
    if age <= stale_threshold:
        return "BETWEEN UPDATES", "yellow", age
    return "STALE", "red", age


def build_metric_table(rows: list[dict[str, str]], keys: tuple[str, ...], title: str) -> Table:
    latest = rows[-1]
    table = Table(title=title, expand=True, box=None, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Recent", justify="right", style="dim")
    populated = False
    for key in keys:
        values = metric_values(rows, key)
        if not values:
            continue
        populated = True
        table.add_row(key, format_metric(as_float(latest.get(key))), trend_text(values), sparkline(values))
    if not populated:
        table.add_row("No matching metrics yet", "-", "-", "-")
    return table


def build_gpu_table(gpus: list[dict[str, str]]) -> Table:
    table = Table(title="GPUs", expand=True, box=None, padding=(0, 1))
    table.add_column("GPU", style="bold")
    table.add_column("Util", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("Power", justify="right")
    for gpu in gpus:
        memory_used = as_float(gpu["memory.used"])
        memory_total = as_float(gpu["memory.total"])
        memory = f"{memory_used:.0f}/{memory_total:.0f} MiB" if memory_used is not None and memory_total is not None else "-"
        table.add_row(
            f"{gpu['index']} {gpu['name']}",
            f"{gpu['utilization.gpu']}%",
            memory,
            f"{gpu['temperature.gpu']} C",
            f"{gpu['power.draw']} W",
        )
    return table


def build_dashboard(
    paths: RunPaths,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    gpus: list[dict[str, str]],
) -> Group:
    if not rows:
        message = Text()
        message.append(f"Run: {paths.run_dir}\n", style="bold")
        message.append(f"Waiting for {paths.train_log}\n", style="yellow")
        message.append("The dashboard will update when the first metrics row is written.", style="dim")
        parts: list[Any] = [Panel(message, title="UFO Training Monitor", border_style="yellow")]
        if gpus:
            parts.append(Panel(build_gpu_table(gpus), border_style="blue"))
        return Group(*parts)

    latest = rows[-1]
    current_steps = latest_number(latest, "distributed/global_env_steps", "timestep")
    total_steps = as_float(config.get("num_env_steps"))
    optimizer_steps = latest_number(latest, "distributed/optimizer_steps")
    fps_values = metric_values(rows, "FPS")[-5:]
    fps = statistics.median(fps_values) if fps_values else None
    log_interval_steps = as_float(config.get("log_every_updates"))
    expected_update_seconds = log_interval_steps / fps if log_interval_steps is not None and fps and fps > 0 else None
    eta = (total_steps - current_steps) / fps if total_steps is not None and current_steps is not None and fps and fps > 0 else None
    elapsed_minutes = latest_number(latest, "duration [minutes]")
    status, status_style, age = status_for_log(
        paths.train_log,
        total_steps,
        current_steps,
        expected_update_seconds,
    )
    checkpoint_steps = as_float(checkpoint.get("global_time", checkpoint.get("time")))
    checkpoint_age = max(0.0, time.time() - paths.checkpoint_status.stat().st_mtime) if paths.checkpoint_status.exists() else None

    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(justify="right")
    header.add_row(Text(paths.run_dir.name, style="bold bright_white"), Text(status, style=f"bold {status_style}"))
    header.add_row(str(paths.run_dir), "")
    header.add_row(
        f"session {format_duration(elapsed_minutes * 60)}" if elapsed_minutes is not None else "",
        f"updated {format_duration(age)} ago" if age is not None else "",
    )

    progress = Progress(
        TextColumn("[bold]Progress"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
        expand=True,
    )
    progress.add_task("training", total=total_steps or max(current_steps or 1, 1), completed=current_steps or 0)

    overview = Table.grid(expand=True, padding=(0, 2))
    for _ in range(5):
        overview.add_column(justify="center", ratio=1)
    world_size = latest_number(latest, "distributed/world_size")
    batch_size = latest_number(latest, "distributed/effective_batch_size")
    overview.add_row("Global steps", "FPS", "ETA", "Optimizer", "World / batch")
    overview.add_row(
        Text(format_count(current_steps), style="bold"),
        Text(format_count(fps), style="bold"),
        Text(format_duration(eta), style="bold"),
        Text(format_count(optimizer_steps), style="bold"),
        Text(f"{format_count(world_size)} / {format_count(batch_size)}", style="bold"),
    )
    if checkpoint_steps is not None:
        overview.add_row(
            "",
            "",
            f"checkpoint {format_count(checkpoint_steps)} ({format_duration(checkpoint_age)} ago)",
            "",
            "",
        )

    parts = [
        Panel(Group(header, progress, overview), title="UFO Training Monitor", border_style=status_style),
        Columns(
            [
                Panel(build_metric_table(rows, CORE_METRICS, "Learning"), border_style="cyan"),
                Panel(build_metric_table(rows, AUX_REWARD_METRICS, "Auxiliary penalties"), border_style="yellow"),
            ],
            equal=True,
            expand=True,
        ),
    ]
    if gpus:
        parts.append(
            Columns(
                [
                    Panel(build_metric_table(rows, ROBOT_METRICS, "Robot health"), border_style="magenta"),
                    Panel(build_gpu_table(gpus), border_style="blue"),
                ],
                equal=True,
                expand=True,
            )
        )
    else:
        parts.append(Panel(build_metric_table(rows, ROBOT_METRICS, "Robot health"), border_style="magenta"))
    footer = Align.center(
        Text(f"Rows: {len(rows)} shown | Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Ctrl-C to exit", style="dim")
    )
    parts.append(footer)
    return Group(*parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor an active UFO training run in the terminal.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Run directory or train_log.txt path. Defaults to the newest run under --search-root.",
    )
    parser.add_argument("--search-root", default="runs", help="Directory searched when path is omitted (default: runs).")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds (default: 2).")
    parser.add_argument("--history", type=int, default=40, help="Recent log rows used for trends (default: 40).")
    parser.add_argument("--no-gpu", action="store_true", help="Do not query nvidia-smi.")
    parser.add_argument("--once", action="store_true", help="Render once and exit; useful for scripts and SSH checks.")
    args = parser.parse_args(argv)
    if args.refresh <= 0:
        parser.error("--refresh must be positive")
    if args.history < 2:
        parser.error("--history must be at least 2")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = resolve_run_paths(args.path, args.search_root)
    except FileNotFoundError as exc:
        Console(stderr=True).print(f"[bold red]Error:[/] {exc}")
        return 2

    if args.path is not None and not paths.run_dir.exists():
        Console(stderr=True).print(f"[bold red]Error:[/] Run directory does not exist: {paths.run_dir}")
        return 2

    console = Console()

    def render() -> Group:
        rows = read_csv_rows(paths.train_log, args.history)
        config = read_json(paths.config)
        checkpoint = read_json(paths.checkpoint_status)
        gpus = [] if args.no_gpu else query_gpus()
        return build_dashboard(paths, rows, config, checkpoint, gpus)

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
