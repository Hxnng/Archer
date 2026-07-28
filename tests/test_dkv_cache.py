import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archer.benchmark import WORK_METRICS, variant_call
from archer.dkv_cache import generate_dkv_cache_decode, generate_dkv_cache_greedy


class DKVCacheAdapterTests(unittest.TestCase):
    def test_dispatch_uses_greedy_paper_configuration(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 256

        args = SimpleNamespace(
            dkv_cache_repo=Path("baseline/dKV-Cache"),
            dkv_cache_greedy_reloading_step=2,
            dkv_cache_greedy_window_size=4,
            dkv_cache_greedy_seed=42,
            gen_length=256,
            block_length=256,
            temperature=0.0,
        )
        result = variant_call(
            "dkv_cache_greedy",
            object(),
            object(),
            args,
            {"generate_dkv_cache_greedy": generate},
        )

        self.assertEqual(result, ("tokens", 256, 256, {}))
        self.assertEqual(calls[0]["cache_reloading_step"], 2)
        self.assertEqual(calls[0]["window_size"], 4)
        self.assertEqual(calls[0]["seed"], 42)
        self.assertEqual(WORK_METRICS["dkv_cache_greedy"], "nfe")

    def test_dispatch_uses_decode_paper_configuration(self):
        calls = []

        def generate(_model, _input_ids, **kwargs):
            calls.append(kwargs)
            return "tokens", 256

        args = SimpleNamespace(
            dkv_cache_repo=Path("baseline/dKV-Cache"),
            dkv_cache_reloading_step=8,
            gen_length=256,
            block_length=256,
            temperature=0.0,
        )
        result = variant_call(
            "dkv_cache_decode",
            object(),
            object(),
            args,
            {"generate_dkv_cache_decode": generate},
        )

        self.assertEqual(result, ("tokens", 256, 256, {}))
        self.assertEqual(calls[0]["steps"], 256)
        self.assertEqual(calls[0]["gen_length"], 256)
        self.assertEqual(calls[0]["block_length"], 256)
        self.assertEqual(calls[0]["cache_reloading_step"], 8)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(WORK_METRICS["dkv_cache_decode"], "nfe")

    def test_adapter_delegates_to_official_decode(self):
        calls = []

        def official_generate(model, tokenizer, prompt, **kwargs):
            calls.append((model, tokenizer, prompt, kwargs))
            return "generated"

        with patch(
            "archer.dkv_cache._load_official_components",
            return_value=(object, official_generate),
        ):
            tokens, nfe = generate_dkv_cache_decode(
                "model",
                "prompt",
                repo_path="baseline/dKV-Cache",
                steps=256,
                gen_length=256,
                block_length=256,
                temperature=0.0,
                cache_reloading_step=8,
            )

        self.assertEqual(tokens, "generated")
        self.assertEqual(nfe, 256)
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[0][3]["remasking"], "low_confidence")
        self.assertTrue(calls[0][3]["enable_cache"])
        self.assertEqual(calls[0][3]["cache_reloading_step"], 8)

    def test_rejects_invalid_refresh_interval(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            generate_dkv_cache_decode(
                object(),
                object(),
                repo_path="baseline/dKV-Cache",
                steps=4,
                gen_length=4,
                block_length=4,
                cache_reloading_step=0,
            )

    def test_greedy_adapter_delegates_to_official_configuration(self):
        calls = []
        seeds = []

        def official_generate(model, tokenizer, prompt, **kwargs):
            calls.append((model, tokenizer, prompt, kwargs))
            return "generated"

        with patch(
            "archer.dkv_cache._load_official_greedy_components",
            return_value=(object, official_generate, seeds.append),
        ):
            tokens, nfe = generate_dkv_cache_greedy(
                "model",
                "prompt",
                repo_path="baseline/dKV-Cache",
                steps=256,
                gen_length=256,
                block_length=256,
                cache_reloading_step=2,
                window_size=4,
                seed=42,
            )

        self.assertEqual(tokens, "generated")
        self.assertEqual(nfe, 256)
        self.assertEqual(seeds, [42])
        self.assertEqual(calls[0][3]["remasking"], "random")
        self.assertTrue(calls[0][3]["enable_cache"])
        self.assertEqual(calls[0][3]["cache_reloading_step"], 2)
        self.assertEqual(calls[0][3]["window_size"], 4)


if __name__ == "__main__":
    unittest.main()
