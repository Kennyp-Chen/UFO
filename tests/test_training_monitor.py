from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from humanoidverse.monitor import (
    as_float,
    build_dashboard,
    format_count,
    format_duration,
    read_csv_rows,
    resolve_run_paths,
    sparkline,
    status_for_log,
)


class TrainingMonitorTest(unittest.TestCase):
    def test_resolve_run_directory_and_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "example"
            run_dir.mkdir()
            log_path = run_dir / "train_log.txt"
            log_path.write_text("timestep,FPS\n1,2\n")

            from_directory = resolve_run_paths(run_dir)
            from_file = resolve_run_paths(log_path)

            self.assertEqual(from_directory.train_log, log_path.resolve())
            self.assertEqual(from_file.run_dir, run_dir.resolve())

    def test_auto_discovery_selects_newest_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "older" / "train_log.txt"
            newer = root / "newer" / "train_log.txt"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.write_text("timestep\n1\n")
            newer.write_text("timestep\n2\n")
            older.touch()
            newer.touch()
            older_mtime = newer.stat().st_mtime - 10
            os.utime(older, (older_mtime, older_mtime))

            paths = resolve_run_paths(None, root)

            self.assertEqual(paths.train_log, newer.resolve())

    def test_read_rows_limits_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_log.txt"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["timestep", "FPS"])
                writer.writeheader()
                for index in range(5):
                    writer.writerow({"timestep": index, "FPS": index * 10})

            rows = read_csv_rows(path, history=2)

            self.assertEqual([row["timestep"] for row in rows], ["3", "4"])

    def test_read_rows_ignores_partial_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_log.txt"
            path.write_text("timestep,FPS\n1,2\n3")

            rows = read_csv_rows(path)

            self.assertEqual(rows, [{"timestep": "1", "FPS": "2"}])

    def test_read_rows_falls_back_to_train_log_after_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            csv_path = run_dir / "train_log.txt"
            csv_path.write_text("timestep,FPS\n1,2\n3,4,extra\n")
            (run_dir / "train.log").write_text(
                "INFO:train:{'timestep': 42, 'FPS': 9.5, 'actor_loss': -1.2}\n"
            )

            rows = read_csv_rows(csv_path)

            self.assertEqual(rows, [{"timestep": "42", "FPS": "9.5", "actor_loss": "-1.2"}])

    def test_read_json_config_shape_is_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"num_env_steps": 192_000_000}))
            self.assertEqual(json.loads(path.read_text())["num_env_steps"], 192_000_000)

    def test_format_helpers(self) -> None:
        self.assertEqual(format_count(1_536_000), "1.54M")
        self.assertEqual(format_duration(3661), "1h 01m 01s")
        self.assertIsNone(as_float("nan"))
        self.assertGreater(len(sparkline([1.0, 2.0, 3.0])), 1)

    def test_status_uses_configured_log_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_log.txt"
            path.touch()
            old_mtime = path.stat().st_mtime - 400
            os.utime(path, (old_mtime, old_mtime))

            status, _, _ = status_for_log(path, 1000, 100, expected_update_seconds=420)

            self.assertEqual(status, "ACTIVE")

    def test_dashboard_renders_training_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = resolve_run_paths(Path(directory))
            paths.train_log.write_text("timestep,FPS,actor_loss\n1536000,921.4,1022.4\n")
            rows = read_csv_rows(paths.train_log)
            output = StringIO()
            console = Console(file=output, width=120, color_system=None)

            console.print(build_dashboard(paths, rows, {"num_env_steps": 192_000_000}, {}, []))

            rendered = output.getvalue()
            self.assertIn("UFO Training Monitor", rendered)
            self.assertIn("1.54M", rendered)
            self.assertIn("921", rendered)


if __name__ == "__main__":
    unittest.main()
