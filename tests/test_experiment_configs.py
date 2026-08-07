import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
INJECTED_FLAGS = {
    "--benchmark-root",
    "--model-path",
    "--model-repository",
    "--model-revision",
    "--output-dir",
    "--run-name",
}


def load(name):
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def argument_value(run, flag):
    index = run["args"].index(flag)
    return run["args"][index + 1]


class ExperimentConfigTests(unittest.TestCase):
    def test_all_configs_have_valid_schema_and_immutable_models(self):
        for path in sorted(CONFIG_ROOT.glob("*.json")):
            with self.subTest(path=path.name):
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["schema_version"], 1)
                self.assertTrue(config["paper_target"])
                self.assertTrue(config["runs"])
                ids = [run["id"] for run in config["runs"]]
                self.assertEqual(len(ids), len(set(ids)))
                for run in config["runs"]:
                    self.assertTrue(run["module"].startswith("archer."))
                    self.assertEqual(
                        INJECTED_FLAGS.intersection(run["args"]),
                        set(),
                    )
                    self.assertRegex(run["model"]["revision"], r"^[0-9a-f]{40}$")
                    self.assertTrue(run["model"]["repository"])
                    self.assertTrue(run["model"]["path_env"].startswith("ARCHER_"))

    def test_table1_uses_paper_refresh_radii(self):
        runs = {run["id"]: run for run in load("table1_main.json")["runs"]}
        expected = {
            "table1_mbpp_core": "11",
            "table1_livecodebench_core": "10",
            "table1_humaneval_core": "8",
        }
        for run_id, radius in expected.items():
            self.assertEqual(argument_value(runs[run_id], "--refresh-radius"), radius)

    def test_default_reproduction_contains_only_table1_archer_runs(self):
        config = load("table1_archer.json")
        runs = {run["id"]: run for run in config["runs"]}
        expected = {
            "table1_archer_mbpp": ("mbpp", "11"),
            "table1_archer_livecodebench": ("livecodebench", "10"),
            "table1_archer_humaneval": ("humaneval", "8"),
        }
        self.assertEqual(set(runs), set(expected))
        for run_id, (dataset, radius) in expected.items():
            run = runs[run_id]
            self.assertEqual(argument_value(run, "--dataset"), dataset)
            self.assertEqual(argument_value(run, "--refresh-radius"), radius)
            variants_index = run["args"].index("--variants")
            self.assertEqual(run["args"][variants_index + 1], "archer")
            self.assertTrue(run["args"][variants_index + 2].startswith("--"))

    def test_refresh_radius_override_is_isolated_in_run_name(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_experiment.py"),
                str(CONFIG_ROOT / "table1_archer.json"),
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

    def test_table2_backbone_radii(self):
        runs = {run["id"]: run for run in load("table2_backbones.json")["runs"]}
        expected = {
            "table2_llada_mbpp_archer": "11",
            "table2_llada_humaneval_archer": "15",
            "table2_dream_mbpp_archer": "17",
            "table2_dream_humaneval_archer": "9",
            "table2_diffucoder_mbpp_archer": "8",
            "table2_diffucoder_humaneval_archer": "15",
        }
        self.assertEqual(len(runs), 12)
        for run_id, radius in expected.items():
            self.assertEqual(argument_value(runs[run_id], "--refresh-radius"), radius)

    def test_table2_runs_each_method_in_a_fresh_process(self):
        runs = load("table2_backbones.json")["runs"]
        for run in runs:
            index = run["args"].index("--variants")
            self.assertFalse(run["args"][index + 1].startswith("--"))
            self.assertTrue(run["args"][index + 2].startswith("--"))

    def test_table2_non_llada_runs_copy_server_timing_flags(self):
        runs = load("table2_backbones.json")["runs"]
        non_llada = [
            run
            for run in runs
            if run["model"]["path_env"] != "ARCHER_LLADA_MODEL_PATH"
        ]
        self.assertEqual(len(non_llada), 8)
        for run in non_llada:
            self.assertIn("--skip-warmup", run["args"])
            self.assertIn("--continue-on-error", run["args"])

    def test_tables3_and_4_are_fixed_to_k11(self):
        for name in (
            "table3_feedback_k11.json",
            "table4_cache_validity_k11.json",
        ):
            run = load(name)["runs"][0]
            self.assertEqual(argument_value(run, "--refresh-radius"), "11")

    def test_figure2_infinity_is_unreachable_radius(self):
        runs = load("figure2_k_sweep.json")["runs"]
        infinity = [run for run in runs if run["id"].endswith("_kinfinity")]
        self.assertEqual(len(infinity), 3)
        for run in infinity:
            self.assertEqual(argument_value(run, "--refresh-radius"), "257")

    def test_livecodebench_reproduces_server_timing_setup(self):
        runs = load("table1_main.json")["runs"]
        livecodebench = [
            run for run in runs if argument_value(run, "--dataset") == "livecodebench"
        ]
        self.assertEqual(len(livecodebench), 4)
        for run in livecodebench:
            self.assertIn("--skip-warmup", run["args"])

    def test_runner_can_render_every_config_without_models(self):
        for path in sorted(CONFIG_ROOT.glob("*.json")):
            with self.subTest(path=path.name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_experiment.py"),
                        str(path),
                        "--dry-run",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertIn("--model-revision", completed.stdout)


class ArtifactPrivacyTests(unittest.TestCase):
    def test_anonymity_scan_passes(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_anonymity.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
