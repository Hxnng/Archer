import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "table1_archer.json"
INJECTED_FLAGS = {
    "--benchmark-root",
    "--model-path",
    "--model-repository",
    "--model-revision",
    "--output-dir",
    "--run-name",
}


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def argument_value(run, flag):
    index = run["args"].index(flag)
    return run["args"][index + 1]


class ExperimentConfigTests(unittest.TestCase):
    def test_config_schema_and_pinned_model(self):
        config = load_config()
        self.assertEqual(config["schema_version"], 1)
        self.assertTrue(config["paper_target"])
        self.assertEqual(len(config["runs"]), 3)
        for run in config["runs"]:
            self.assertEqual(run["module"], "archer.benchmark")
            self.assertFalse(INJECTED_FLAGS.intersection(run["args"]))
            self.assertRegex(run["model"]["revision"], r"^[0-9a-f]{40}$")
            self.assertNotIn("--variants", run["args"])

    def test_table1_defaults(self):
        runs = {run["id"]: run for run in load_config()["runs"]}
        expected = {
            "table1_archer_mbpp": ("mbpp", "11"),
            "table1_archer_livecodebench": ("livecodebench", "10"),
            "table1_archer_humaneval": ("humaneval", "8"),
        }
        self.assertEqual(set(runs), set(expected))
        for run_id, (dataset, radius) in expected.items():
            self.assertEqual(argument_value(runs[run_id], "--dataset"), dataset)
            self.assertEqual(argument_value(runs[run_id], "--refresh-radius"), radius)

    def test_refresh_override_changes_run_name(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiment.py"),
                str(CONFIG),
                "--run",
                "table1_archer_mbpp",
                "--refresh-radius",
                "16",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("--refresh-radius 16", completed.stdout)
        self.assertIn("--run-name table1_archer_mbpp_k16", completed.stdout)

    def test_runner_renders_all_commands_without_models(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiment.py"),
                str(CONFIG),
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.stdout.count("archer.benchmark"), 3)
        self.assertIn("--model-revision", completed.stdout)


if __name__ == "__main__":
    unittest.main()
