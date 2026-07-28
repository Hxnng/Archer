import unittest

import torch

from archer.fast_dllm_saber import select_one_by_confidence


class SaberFastDLLMTests(unittest.TestCase):
    def test_cache_only_rule_commits_exactly_one_masked_token(self):
        logits = torch.tensor(
            [[[0.0, 1.0], [4.0, 0.0], [0.0, 3.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[True, False, True]])
        current = torch.tensor([[9, 8, 9]])

        predictions, selected = select_one_by_confidence(
            logits, mask, current, temperature=0.0
        )

        self.assertEqual(int(selected.sum()), 1)
        self.assertTrue(bool(selected[0, 2]))
        self.assertEqual(int(predictions[0, 1]), 8)


if __name__ == "__main__":
    unittest.main()
