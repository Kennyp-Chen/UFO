from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, Final

COMMAND_LOW: Final = (-0.8, -0.5, -0.8)
COMMAND_HIGH: Final = (0.8, 0.5, 0.8)
DEFAULT_LEGACY_REPO: Final = Path("/root/autodl-tmp/chenyupeng/HT_BFM")
DEFAULT_LEGACY_PYTHON: Final = Path("/root/autodl-tmp/chenyupeng/.conda/envs/HT_BFM/bin/python")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _resolved_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _resolved_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _validate_command(command: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in command)
    if len(values) != 3 or any(value < low or value > high for value, low, high in zip(values, COMMAND_LOW, COMMAND_HIGH)):
        raise ValueError(f"fixed command must be within {COMMAND_LOW} and {COMMAND_HIGH}")
    return values[0], values[1], values[2]


def build_command(
    legacy_repo: Path,
    *,
    checkpoint: Path,
    decoder_path: Path,
    robot_config: Path,
    expert_dataset: Path,
    python_executable: Path | None = None,
    model_folder: Path | None = None,
    device: str = "cpu",
    fixed_command: Sequence[float] = (0.4, 0.0, 0.0),
    command_smoothing: float | None = None,
    max_steps: int | None = 250,
    max_episode_length_s: float | None = None,
    headless: bool = True,
    save_mp4: bool = False,
    output: Path | None = None,
    render_size: int | None = None,
    fps: float | None = None,
    realtime: bool | None = None,
) -> tuple[str, ...]:
    if python_executable is None:
        python = Path(sys.executable).resolve()
    else:
        python = _resolved_file(python_executable, "legacy Python executable")
    checkpoint_path = _resolved_file(checkpoint, "checkpoint")
    decoder = _resolved_file(decoder_path, "decoder")
    robot = _resolved_file(robot_config, "robot config")
    dataset = _resolved_file(expert_dataset, "expert dataset")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_episode_length_s is not None and max_episode_length_s <= 0.0:
        raise ValueError("max_episode_length_s must be positive")
    if command_smoothing is not None and not 0.0 < command_smoothing <= 1.0:
        raise ValueError("command_smoothing must be in (0, 1]")
    if render_size is not None and render_size <= 0:
        raise ValueError("render_size must be positive")
    if fps is not None and fps <= 0.0:
        raise ValueError("fps must be positive")
    if output is not None and not save_mp4:
        raise ValueError("output requires save_mp4")
    command_values = _validate_command(fixed_command)
    command: list[str] = [
        str(python),
        "-m",
        "humanoidverse.speed_stage2_play",
    ]
    if model_folder is not None:
        command.extend(("--model-folder", str(_resolved_directory(model_folder, "model folder"))))
    command.extend(
        (
            "--checkpoint",
            str(checkpoint_path),
            "--decoder-path",
            str(decoder),
            "--robot-config",
            str(robot),
            "--expert-dataset",
            str(dataset),
            "--device",
            device,
        )
    )
    if command_values != (0.4, 0.0, 0.0):
        command.extend(("--fixed-command", *(str(value) for value in command_values)))
    if command_smoothing is not None:
        command.extend(("--command-smoothing", str(command_smoothing)))
    if max_steps is not None:
        command.extend(("--max-steps", str(max_steps)))
    if max_episode_length_s is not None:
        command.extend(("--max-episode-length-s", str(max_episode_length_s)))
    if headless:
        command.append("--headless")
    if save_mp4:
        command.append("--save-mp4")
        if output is not None:
            command.extend(("--output", str(output.expanduser().resolve())))
        if render_size is not None:
            command.extend(("--render-size", str(render_size)))
        if fps is not None:
            command.extend(("--fps", str(fps)))
    if realtime is False:
        command.append("--no-realtime")
    return tuple(command)


def build_environment(legacy_repo: Path, environment: Mapping[str, str] | None = None) -> dict[str, str]:
    repo = _resolved_directory(legacy_repo, "legacy repository")
    result = dict(os.environ)
    if environment is not None:
        result.update(environment)
    inherited_pythonpath = result.get("PYTHONPATH", "")
    result["PYTHONPATH"] = os.pathsep.join(value for value in (str(repo), inherited_pythonpath) if value)
    return result


def _run_process(argv: Sequence[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, check=False, text=True)


def launch(
    legacy_repo: Path,
    *,
    checkpoint: Path,
    decoder_path: Path,
    robot_config: Path,
    expert_dataset: Path,
    python_executable: Path | None = None,
    model_folder: Path | None = None,
    device: str = "cpu",
    fixed_command: Sequence[float] = (0.4, 0.0, 0.0),
    command_smoothing: float | None = None,
    max_steps: int | None = 250,
    max_episode_length_s: float | None = None,
    headless: bool = True,
    save_mp4: bool = False,
    output: Path | None = None,
    render_size: int | None = None,
    fps: float | None = None,
    realtime: bool | None = None,
    runner: Runner = _run_process,
    environment: Mapping[str, str] | None = None,
) -> int:
    command = build_command(
        legacy_repo,
        checkpoint=checkpoint,
        decoder_path=decoder_path,
        robot_config=robot_config,
        expert_dataset=expert_dataset,
        python_executable=python_executable,
        model_folder=model_folder,
        device=device,
        fixed_command=fixed_command,
        command_smoothing=command_smoothing,
        max_steps=max_steps,
        max_episode_length_s=max_episode_length_s,
        headless=headless,
        save_mp4=save_mp4,
        output=output,
        render_size=render_size,
        fps=fps,
        realtime=realtime,
    )
    repo = _resolved_directory(legacy_repo, "legacy repository")
    result = runner(command, cwd=repo, env=build_environment(repo, environment))
    return int(result.returncode)


def _default_python() -> Path:
    if DEFAULT_LEGACY_PYTHON.is_file():
        return DEFAULT_LEGACY_PYTHON
    return Path(sys.executable)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play speed-stage2 checkpoints through the HT_BFM legacy backend.")
    parser.add_argument("--legacy-repo", type=Path, default=DEFAULT_LEGACY_REPO)
    parser.add_argument("--legacy-python", type=Path, default=_default_python())
    parser.add_argument("--model-folder", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-path", type=Path, required=True)
    parser.add_argument("--expert-dataset", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed-command", type=float, nargs=3, default=(0.4, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command-smoothing", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-episode-length-s", type=float, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--render-size", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return launch(
        args.legacy_repo,
        checkpoint=args.checkpoint,
        decoder_path=args.decoder_path,
        robot_config=args.robot_config,
        expert_dataset=args.expert_dataset,
        python_executable=args.legacy_python,
        model_folder=args.model_folder,
        device=args.device,
        fixed_command=args.fixed_command,
        command_smoothing=args.command_smoothing,
        max_steps=args.max_steps,
        max_episode_length_s=args.max_episode_length_s,
        headless=args.headless,
        save_mp4=args.save_mp4,
        output=args.output,
        render_size=args.render_size,
        fps=args.fps,
        realtime=False if args.no_realtime else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
