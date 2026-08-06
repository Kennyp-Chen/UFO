from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from humanoidverse.speed_stage2_legacy_play import (
    DEFAULT_LEGACY_PYTHON,
    DEFAULT_LEGACY_REPO,
    PROJECT_ROOT,
    build_command,
    build_environment,
    launch,
)


class FakeRunner:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.argv: Sequence[str] | None = None
        self.cwd: Path | None = None
        self.env: dict[str, str] | None = None

    def __call__(self, argv: Sequence[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.argv = argv
        self.cwd = cwd
        self.env = env
        return subprocess.CompletedProcess(argv, self.returncode)


class SpeedStage2LegacyPlayTest(unittest.TestCase):
    def test_defaults_do_not_use_user_specific_absolute_paths(self) -> None:
        self.assertEqual(DEFAULT_LEGACY_REPO, PROJECT_ROOT.parent / "HT_BFM")
        self.assertEqual(DEFAULT_LEGACY_PYTHON, Path(sys.executable))

    def _paths(self, directory: str) -> dict[str, Path]:
        root = Path(directory)
        legacy_repo = root / "legacy repo"
        legacy_repo.mkdir()
        paths = {
            "legacy_repo": legacy_repo,
            "checkpoint": root / "checkpoint override.pt",
            "decoder_path": root / "decoder override.onnx",
            "robot_config": root / "robot override.yaml",
            "expert_dataset": root / "expert override.pkl",
        }
        for path in paths.values():
            if path != legacy_repo:
                path.touch()
        return paths

    def test_build_command_uses_legacy_speed_play_arguments(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._paths(directory)
            command = build_command(
                paths["legacy_repo"],
                checkpoint=paths["checkpoint"],
                decoder_path=paths["decoder_path"],
                robot_config=paths["robot_config"],
                expert_dataset=paths["expert_dataset"],
                device="cpu",
                max_steps=7,
            )

        self.assertEqual(
            command,
            (
                os.fspath(Path(sys.executable).resolve()),
                "-m",
                "humanoidverse.speed_stage2_play",
                "--checkpoint",
                os.fspath(paths["checkpoint"].resolve()),
                "--decoder-path",
                os.fspath(paths["decoder_path"].resolve()),
                "--robot-config",
                os.fspath(paths["robot_config"].resolve()),
                "--expert-dataset",
                os.fspath(paths["expert_dataset"].resolve()),
                "--device",
                "cpu",
                "--max-steps",
                "7",
                "--headless",
            ),
        )

    def test_build_environment_prefixes_pythonpath_and_preserves_override(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._paths(directory)
            inherited = os.fspath(Path(directory) / "caller pythonpath")
            environment = build_environment(paths["legacy_repo"], {"PYTHONPATH": inherited, "UFO_TEST": "distinct"})

        self.assertEqual(
            environment["PYTHONPATH"],
            os.pathsep.join((os.fspath(paths["legacy_repo"].resolve()), inherited)),
        )
        self.assertEqual(environment["UFO_TEST"], "distinct")

    def test_launch_uses_absolute_cwd_and_environment_and_returns_runner_code(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._paths(directory)
            runner = FakeRunner(returncode=23)
            result = launch(
                paths["legacy_repo"],
                checkpoint=paths["checkpoint"],
                decoder_path=paths["decoder_path"],
                robot_config=paths["robot_config"],
                expert_dataset=paths["expert_dataset"],
                runner=runner,
                environment={"PYTHONPATH": "caller-only", "UFO_TEST": "override"},
            )

        self.assertEqual(result, 23)
        self.assertEqual(runner.cwd, paths["legacy_repo"].resolve())
        self.assertEqual(runner.env["UFO_TEST"], "override")
        self.assertEqual(runner.env["PYTHONPATH"].split(os.pathsep)[0], os.fspath(paths["legacy_repo"].resolve()))
        self.assertEqual(runner.argv[0], os.fspath(Path(sys.executable).resolve()))
        self.assertEqual(runner.argv[2], "humanoidverse.speed_stage2_play")

    def test_build_command_rejects_missing_paths(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._paths(directory)
            paths["decoder_path"] = Path(directory) / "missing decoder.onnx"

            with self.assertRaises(FileNotFoundError):
                build_command(
                    paths["legacy_repo"],
                    checkpoint=paths["checkpoint"],
                    decoder_path=paths["decoder_path"],
                    robot_config=paths["robot_config"],
                    expert_dataset=paths["expert_dataset"],
                )

    def test_build_command_rejects_nonpositive_max_steps(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._paths(directory)

            with self.assertRaises(ValueError):
                build_command(
                    paths["legacy_repo"],
                    checkpoint=paths["checkpoint"],
                    decoder_path=paths["decoder_path"],
                    robot_config=paths["robot_config"],
                    expert_dataset=paths["expert_dataset"],
                    max_steps=0,
                )


if __name__ == "__main__":
    unittest.main()
