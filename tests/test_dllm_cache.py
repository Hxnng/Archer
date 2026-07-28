import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archer.benchmark import WORK_METRICS, variant_call
from archer.dllm_cache import generate_dllm_cache


class DLLMCacheAdapterTests(unittest.TestCase):
    def test_dispatch_preserves_method_specific_block_length(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 256

        args = SimpleNamespace(
            dllm_cache_repo=Path("baseline/dLLM-cache"),
            dllm_cache_block_length=32,
            gen_length=256,
            temperature=0.0,
        )
        result = variant_call(
            "dllm_cache", object(), object(), args, {"generate_dllm_cache": generate}
        )

        self.assertEqual(result, ("tokens", 256, 256, {}))
        self.assertEqual(calls[0]["steps"], 256)
        self.assertEqual(calls[0]["gen_length"], 256)
        self.assertEqual(calls[0]["block_length"], 32)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(WORK_METRICS["dllm_cache"], "nfe")

    def test_adapter_delegates_to_official_sampler(self):
        calls = []

        class FakeTensor:
            shape = (1, 3)

            def new_ones(self, shape):
                return ("ones", shape)

        def official_generate(**kwargs):
            calls.append(kwargs)
            return "response"

        fake_torch = SimpleNamespace(cat=lambda tensors, dim: (tensors, dim))
        with patch(
            "archer.dllm_cache._load_official_components",
            return_value=(object, object, object, official_generate),
        ), patch.dict("sys.modules", {"torch": fake_torch}):
            tokens, nfe = generate_dllm_cache(
                "model",
                FakeTensor(),
                repo_path="baseline/dLLM-cache",
                steps=256,
                gen_length=256,
                block_length=32,
            )

        self.assertEqual(nfe, 256)
        self.assertEqual(tokens[1], 1)
        self.assertEqual(calls[0]["remasking"], "low_confidence")
        self.assertEqual(calls[0]["cfg_scale"], 0.0)

    def test_rejects_invalid_block_configuration(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            generate_dllm_cache(
                object(),
                object(),
                repo_path="baseline/dLLM-cache",
                steps=256,
                gen_length=256,
                block_length=30,
            )


if __name__ == "__main__":
    unittest.main()
