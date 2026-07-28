import unittest
from types import SimpleNamespace

import torch

from archer.fast_dllm_v1 import (
    generate_fast_dllm_v1,
    select_confident_tokens,
)


class _FakeCacheModel:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, *, use_cache, past_key_values=None):
        self.calls.append(
            {
                "length": tokens.shape[1],
                "cached": past_key_values is not None,
                "cache_length": 0
                if past_key_values is None
                else past_key_values[0][0].shape[2],
            }
        )
        # Uniform logits remain below threshold, so exactly one token advances
        # per forward via the mandatory highest-confidence fallback.
        logits = torch.zeros((*tokens.shape, 5), dtype=torch.float32)
        sequence_length = tokens.shape[1]
        key = torch.zeros((tokens.shape[0], 1, sequence_length, 1))
        value = torch.zeros_like(key)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


class FastDLLMV1Tests(unittest.TestCase):
    def test_threshold_selection_always_makes_progress(self):
        logits = torch.zeros((1, 3, 4))
        mask = torch.tensor([[False, True, True]])
        tokens = torch.tensor([[3, 3, 3]])

        predictions, selected = select_confident_tokens(
            logits, mask, tokens, threshold=0.9
        )

        self.assertEqual(int(selected.sum()), 1)
        self.assertFalse(bool(selected[0, 0]))
        self.assertEqual(predictions.shape, tokens.shape)

    def test_cache_refreshes_once_per_block_and_reuses_prefix_inside_block(self):
        model = _FakeCacheModel()
        prompt = torch.tensor([[2]])

        generated, nfe = generate_fast_dllm_v1(
            model,
            prompt,
            gen_length=4,
            block_length=2,
            threshold=0.9,
            mask_id=4,
        )

        self.assertEqual(generated.shape, (1, 5))
        self.assertFalse(bool((generated == 4).any()))
        self.assertEqual(nfe, 4)
        self.assertEqual(
            model.calls,
            [
                {"length": 5, "cached": False, "cache_length": 0},
                {"length": 4, "cached": True, "cache_length": 1},
                {"length": 5, "cached": False, "cache_length": 0},
                {"length": 2, "cached": True, "cache_length": 3},
            ],
        )

    def test_rejects_non_divisible_block_length(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            generate_fast_dllm_v1(
                _FakeCacheModel(),
                torch.tensor([[1]]),
                gen_length=5,
                block_length=2,
                mask_id=4,
            )


if __name__ == "__main__":
    unittest.main()
