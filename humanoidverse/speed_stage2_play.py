"""Run a bounded, headless PiPlus H0W speed-PPO playback."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from humanoidverse.piplus_h0w_stage2_play import (
    load_playback_checkpoint,
    playback,
    print_playback_result,
    resolve_playback_asset,
    resolve_required_playback_asset,
)
from humanoidverse.speed_stage2 import (
    DEFAULT_DECODER_PATH,
    PROJECT_ROOT,
    SPEED_STAGE2_TASK,
)

DEFAULT_WORK_DIR = PROJECT_ROOT / "runs/speed_stage2_piplus_h0w"


def checkpoint_task_is_compatible(metadata: dict[str, object]) -> bool:
    return metadata.get("task") in {None, SPEED_STAGE2_TASK}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bfm-model", type=Path, default=None)
    parser.add_argument("--decoder-path", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--motion-dataset", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-command", type=float, nargs=3, default=(0.4, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command-smoothing", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--max-episode-length-s", type=float, default=20.0)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--render-size", type=int, default=720)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main(parsed_args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if parsed_args is None else parsed_args
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_path, metadata, policy_state = load_playback_checkpoint(
        model_folder=args.model_folder.expanduser().resolve(),
        checkpoint=args.checkpoint,
        expected_task=SPEED_STAGE2_TASK,
        device=device,
    )
    bfm_model = resolve_playback_asset(args.bfm_model, metadata, "bfm_model")
    decoder_path = resolve_required_playback_asset(args.decoder_path, metadata, "decoder_path", fallback=DEFAULT_DECODER_PATH)
    robot_config = resolve_required_playback_asset(args.robot_config, metadata, "robot_config")
    motion_dataset = resolve_required_playback_asset(args.motion_dataset, metadata, "motion_dataset")
    print_playback_result(
        playback(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            policy_state=policy_state,
            bfm_model=bfm_model,
            decoder_path=decoder_path,
            robot_config=robot_config,
            motion_dataset=motion_dataset,
            device=device,
            fixed_command=tuple(args.fixed_command),
            command_smoothing=args.command_smoothing,
            max_steps=args.max_steps,
            max_episode_length_s=args.max_episode_length_s,
            seed=args.seed,
            video_path=args.video_path,
            render_every=args.render_every,
            render_size=args.render_size,
        )
    )


if __name__ == "__main__":
    main()
