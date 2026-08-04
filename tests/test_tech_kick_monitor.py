from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from humanoidverse.tech_kick_monitor import (
    build_dashboard,
    read_metric_rows,
    resolve_kick_run,
)


class TechKickMonitorTest(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        run = root / "tech_kick_stage2"
        log_dir = run / "torchrunx" / "2026-08-04T00:00:00"
        log_dir.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps({"iterations": 30000, "num_envs": 1024, "world_size": 8}))
        (log_dir / "localhost[0].log").write_text(
            "INFO: {\"iteration\": 1, \"reward_mean\": 0.2, \"contact_rate\": 0.1}\n"
            "INFO: {\"iteration\": 2, \"reward_mean\": 0.4, \"contact_rate\": 0.2}\n"
        )
        (run / "checkpoint_2.pt").write_bytes(b"checkpoint")
        return run

    def test_discovers_latest_stage2_run_and_reads_json_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._run(root)
            paths = resolve_kick_run(None, root)
            self.assertEqual(paths.run_dir, run.resolve())
            rows = read_metric_rows(paths.metric_log)
            self.assertEqual(rows[-1]["iteration"], "2")
            self.assertEqual(rows[-1]["reward_mean"], "0.4")

    def test_dashboard_renders_stage2_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(Path(directory))
            paths = resolve_kick_run(run)
            rows = read_metric_rows(paths.metric_log)
            output = StringIO()
            console = Console(file=output, width=140, color_system=None)
            console.print(build_dashboard(paths, rows, {"iterations": 30000, "num_envs": 1024, "world_size": 8}, 30000, []))
            rendered = output.getvalue()
            self.assertIn("TeCH Kick Stage2 Monitor", rendered)
            self.assertIn("contact_rate", rendered)
            self.assertIn("2", rendered)


if __name__ == "__main__":
    unittest.main()
