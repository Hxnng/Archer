import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archer.benchmark import WORK_METRICS, evaluate_rows, variant_call


class BenchmarkGlueTests(unittest.TestCase):
    def test_archer_dispatches_paper_parameters_and_diagnostics(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 11, 12, {"extra_refreshes": 2}

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=8,
        )
        model = SimpleNamespace(config=SimpleNamespace(mask_token_id=126336))
        result = variant_call(
            "archer", model, object(), args, {"generate_archer": generate}
        )

        self.assertEqual(result, ("tokens", 11, 12, {"extra_refreshes": 2}))
        self.assertEqual(calls[0]["refresh_radius"], 8)
        self.assertEqual(calls[0]["gen_length"], 256)
        self.assertEqual(calls[0]["mask_id"], 126336)
        self.assertEqual(WORK_METRICS, {"archer": "nfe"})

    def test_non_archer_variant_is_rejected(self):
        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=8,
        )
        with self.assertRaisesRegex(ValueError, "unsupported variant"):
            variant_call("saber", object(), object(), args, {})

    def test_humaneval_preserves_evaluation_order_and_input(self):
        original = [{"index": 0, "completion": "return 1"}]

        def standard(rows, **_):
            rows[0]["standard_ran"] = True
            return {"pass@1": 0.5}

        def extended(rows, **_):
            self.assertTrue(rows[0]["standard_ran"])
            return {"pass@1": 0.25}

        runtime = {
            "evaluate_solution": standard,
            "evaluate_solution_et": extended,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_rows(
                "humaneval",
                original,
                Path("standard.jsonl"),
                Path("extended.jsonl"),
                Path(directory),
                runtime,
            )

        self.assertEqual(result, (0.5, 0.25))
        self.assertNotIn("standard_ran", original[0])

    def test_mbpp_uses_isolated_evaluator_and_removes_temp_files(self):
        seen_paths = []

        def evaluate(temp_path, problem_file, _run_dir, workers):
            self.assertTrue(temp_path.exists())
            rows = [json.loads(line) for line in temp_path.read_text().splitlines()]
            self.assertEqual(rows[0]["completion"], "return 1")
            seen_paths.append(temp_path)
            self.assertEqual(workers, 8)
            return {"pass_at_1": 0.75 if "ET" not in problem_file.name else 0.5}

        with tempfile.TemporaryDirectory() as directory:
            with patch("archer.benchmark.evaluate_mbpp_in_subprocess", side_effect=evaluate):
                result = evaluate_rows(
                    "mbpp",
                    [{"index": 0, "completion": "return 1"}],
                    Path("MBPP.jsonl"),
                    Path("MBPP_ET.jsonl"),
                    Path(directory),
                    {},
                )

        self.assertEqual(result, (0.75, 0.5))
        self.assertEqual(seen_paths[0], seen_paths[1])
        self.assertFalse(seen_paths[0].exists())

    def test_livecodebench_uses_official_evaluator_without_et(self):
        rows = [{"index": 0, "question_id": "q1", "completion": "print(1)"}]
        expected = {"pass_at_1": 0.25, "detail": {"pass@1": {"0": 0.0}}}
        with tempfile.TemporaryDirectory() as directory:
            detail = Path(directory) / "detail.json"
            with patch(
                "archer.benchmark.evaluate_livecodebench_in_subprocess",
                return_value=expected,
            ) as evaluate:
                result = evaluate_rows(
                    "livecodebench",
                    rows,
                    Path("test.jsonl"),
                    Path("test.jsonl"),
                    Path(directory),
                    {},
                    livecodebench_repo=Path("LiveCodeBench"),
                    livecodebench_eval_workers=4,
                    livecodebench_timeout=6,
                    detail_output=detail,
                )
            self.assertEqual(result, (0.25, None))
            self.assertEqual(json.loads(detail.read_text()), expected)
            evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
