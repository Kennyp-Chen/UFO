from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from humanoidverse.command_encoder_play_legacy import (
    AMP_STAGE2_TASK,
    DEFAULT_LEGACY_REPO,
    _resolved_file,
    _validate_command,
    checkpoint_task_is_compatible,
    install_legacy_imports,
    main,
)


class ResolvedFileTest(unittest.TestCase):
    def test_existing_file_resolves(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset.onnx"
            path.touch()
            self.assertEqual(_resolved_file(path, "asset"), path.resolve())

    def test_missing_file_raises(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                _resolved_file(path, "checkpoint")


class InstallLegacyImportsTest(unittest.TestCase):
    def tearDown(self) -> None:
        for name in ("humanoidverse", "humanoidverse.fake_sub"):
            sys.modules.pop(name, None)
        if sys.path and sys.path[0].endswith("legacy_repo"):
            sys.path.pop(0)

    def test_missing_repo_raises(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                install_legacy_imports(Path(directory) / "missing")

    def test_prepends_repo_and_drops_cached_package(self) -> None:
        with TemporaryDirectory() as directory:
            repo = Path(directory) / "legacy_repo"
            repo.mkdir()
            cached = types.ModuleType("humanoidverse")
            sys.modules["humanoidverse"] = cached
            sys.modules["humanoidverse.fake_sub"] = cached
            install_legacy_imports(repo)
            self.assertEqual(sys.path[0], str(repo))
            self.assertNotIn("humanoidverse", sys.modules)
            self.assertNotIn("humanoidverse.fake_sub", sys.modules)


class CheckpointTaskCompatibleTest(unittest.TestCase):
    def test_accepts_amp_stage2_task(self) -> None:
        self.assertTrue(checkpoint_task_is_compatible({"task": AMP_STAGE2_TASK}))

    def test_accepts_missing_task_metadata(self) -> None:
        self.assertTrue(checkpoint_task_is_compatible({}))

    def test_rejects_other_task(self) -> None:
        self.assertFalse(checkpoint_task_is_compatible({"task": "speed_stage2_piplus_22dof"}))


class ValidateCommandTest(unittest.TestCase):
    def test_accepts_in_range_command(self) -> None:
        _validate_command((0.4, 0.0, 0.0))
        _validate_command((1.0, 0.5, 1.0))
        _validate_command((-0.5, -0.5, -1.0))

    def test_rejects_out_of_range_command(self) -> None:
        for command in ((1.5, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, 0.0, 1.1), (-0.6, 0.0, 0.0)):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    _validate_command(command)


class MainFailFastTest(unittest.TestCase):
    """Error paths that must fail before any HT_BFM import (testable standalone)."""

    def _run_main(self, argv: list[str]) -> int:
        with patch("sys.argv", ["command_encoder_play_legacy", *argv]):
            return main()

    def test_missing_checkpoint_raises(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self._run_main(["--checkpoint", os_fspath(Path(directory) / "missing.pt"), "--output", "out.mp4"])

    def test_out_of_range_command_raises(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "c.pt"
            checkpoint.touch()
            with self.assertRaises(ValueError):
                self._run_main(
                    ["--checkpoint", os_fspath(checkpoint), "--output", "out.mp4", "--fixed-command", "2.0", "0.0", "0.0"]
                )

    def test_non_positive_max_steps_raises(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "c.pt"
            checkpoint.touch()
            with self.assertRaises(ValueError):
                self._run_main(["--checkpoint", os_fspath(checkpoint), "--output", "out.mp4", "--max-steps", "0"])

    def test_default_htbfm_repo_is_legacy_repo(self) -> None:
        self.assertEqual(DEFAULT_LEGACY_REPO, Path("/root/autodl-tmp/chenyupeng/HT_BFM"))


def os_fspath(path: Path) -> str:
    import os

    return os.fspath(path)


if __name__ == "__main__":
    unittest.main()
