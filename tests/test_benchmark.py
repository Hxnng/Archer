import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archer.benchmark import WORK_METRICS, evaluate_rows, variant_call


class BenchmarkGlueTests(unittest.TestCase):
    def test_llada_dispatches_fixed_full_forward_schedule(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 256

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=8,
            fast_dllm_block_length=32,
            fast_dllm_threshold=0.9,
        )
        result = variant_call(
            "llada", object(), object(), args, {"generate_llada": generate}
        )

        self.assertEqual(result, ("tokens", 256, 256, {}))
        self.assertEqual(calls[0]["steps"], 256)
        self.assertEqual(calls[0]["gen_length"], 256)
        self.assertEqual(calls[0]["block_length"], 256)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(WORK_METRICS["llada"], "nfe")

    def test_fast_dllm_v1_dispatches_paper_configuration(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 17

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=8,
            fast_dllm_block_length=32,
            fast_dllm_threshold=0.9,
        )
        result = variant_call(
            "fast_dllm_v1",
            object(),
            object(),
            args,
            {"generate_fast_dllm_v1": generate},
        )

        self.assertEqual(result, ("tokens", 17, 17, {}))
        self.assertEqual(
            calls[0],
            {"gen_length": 256, "block_length": 32, "threshold": 0.9},
        )
        self.assertEqual(WORK_METRICS["fast_dllm_v1"], "nfe")

    def test_fast_dllm_dispatches_saber_cache_only_configuration(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 256

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=8,
            fast_dllm_block_length=32,
            fast_dllm_threshold=0.9,
        )
        result = variant_call(
            "fast_dllm",
            object(),
            object(),
            args,
            {"generate_fast_dllm_cache_only": generate},
        )

        self.assertEqual(result, ("tokens", 256, 256, {}))
        self.assertEqual(
            calls[0],
            {
                "steps": 256,
                "gen_length": 256,
                "block_length": 256,
                "temperature": 0.0,
            },
        )
        self.assertEqual(WORK_METRICS["fast_dllm"], "nfe")

    def test_archer_dispatches_refresh_radius_and_diagnostics(self):
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
            fast_dllm_block_length=32,
            fast_dllm_threshold=0.9,
        )
        result = variant_call(
            "archer", object(), object(), args, {"generate_archer": generate}
        )

        self.assertEqual(result, ("tokens", 11, 12, {"extra_refreshes": 2}))
        self.assertEqual(calls[0]["refresh_radius"], 8)
        self.assertEqual(WORK_METRICS["archer"], "nfe")

    def test_saber_uses_the_model_mask_token(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 17

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
        )
        model = SimpleNamespace(config=SimpleNamespace(mask_token_id=151666))
        result = variant_call(
            "saber", model, object(), args, {"generate_saber": generate}
        )

        self.assertEqual(result, ("tokens", 17, 17, {}))
        self.assertEqual(calls[0]["mask_id"], 151666)
        self.assertEqual(calls[0]["gen_length"], 256)
        self.assertEqual(calls[0]["mu"], 2)

    def test_dream_family_uses_model_specific_samplers(self):
        calls = []

        def dream_saber(_model, _input_ids, **kwargs):
            calls.append(("saber", kwargs))
            return "saber_tokens", 17

        def dream_archer(_model, _input_ids, **kwargs):
            calls.append(("archer", kwargs))
            return "archer_tokens", 18, 21, {"extra_refreshes": 3}

        args = SimpleNamespace(
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            refresh_radius=9,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="Dream", mask_token_id=151666)
        )
        runtime = {
            "is_dream_family": lambda _model: True,
            "generate_dream_saber": dream_saber,
            "generate_dream_archer": dream_archer,
        }

        saber_result = variant_call("saber", model, object(), args, runtime)
        archer_result = variant_call("archer", model, object(), args, runtime)

        self.assertEqual(saber_result, ("saber_tokens", 17, 17, {}))
        self.assertEqual(
            archer_result,
            ("archer_tokens", 18, 21, {"extra_refreshes": 3}),
        )
        self.assertEqual(calls[0][1]["mask_id"], 151666)
        self.assertEqual(calls[1][1]["mask_id"], 151666)
        self.assertEqual(calls[1][1]["refresh_radius"], 9)

    def test_humaneval_preserves_legacy_evaluation_order(self):
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

    def test_mbpp_evaluation_uses_isolated_evaluator_and_removes_temp_files(self):
        seen_paths = []

        def evaluate(temp_path, problem_file, run_dir, workers):
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

    def test_livecodebench_has_no_et_metric_and_uses_official_evaluator(self):
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
