import unittest

try:
    import torch
except ModuleNotFoundError:  # Local source-only checks do not install the GPU runtime.
    torch = None


@unittest.skipIf(torch is None, "ARCHER sampler tests require torch")
class ArcherSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from types import SimpleNamespace

        from archer.saber import generate_saber
        from archer.state_distance import (
            generate_archer,
            generation_hamming_distance,
            summarize_cache_diagnostics,
        )

        cls.SimpleNamespace = SimpleNamespace
        cls.generate_saber = staticmethod(generate_saber)
        cls.generate_archer = staticmethod(generate_archer)
        cls.generation_hamming_distance = staticmethod(generation_hamming_distance)
        cls.summarize_cache_diagnostics = staticmethod(summarize_cache_diagnostics)

    class TinyModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.calls = []

        def __call__(self, input_ids, past_key_values=None, use_cache=False):
            self.calls.append(
                {
                    "length": input_ids.shape[1],
                    "uses_cache": use_cache,
                    "has_past": past_key_values is not None,
                }
            )
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            logits[..., 0] = 1.0
            key = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], 1)
            return ArcherSamplerTests.SimpleNamespace(
                logits=logits, past_key_values=[(key, key)]
            )

    @staticmethod
    def _prompt():
        return torch.tensor([[1]], dtype=torch.long)

    def test_hamming_distance_counts_net_token_changes(self):
        snapshot = torch.tensor([[3, 3, 3]])
        current = torch.tensor([[0, 3, 1]])
        self.assertEqual(self.generation_hamming_distance(current, snapshot), 2)
        self.assertEqual(self.generation_hamming_distance(snapshot, snapshot), 0)
        with self.assertRaises(ValueError):
            self.generation_hamming_distance(current, snapshot[:, :2])

    def test_threshold_one_refreshes_after_each_changed_state(self):
        model = self.TinyModel()
        tokens, steps, nfe, diagnostics = self.generate_archer(
            model,
            self._prompt(),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            mask_id=3,
            refresh_radius=1,
        )

        self.assertEqual(tokens.tolist(), [[1, 0, 0, 0]])
        self.assertEqual(steps, 3)
        self.assertEqual(nfe, 4)
        self.assertEqual(diagnostics["cache_builds"], 3)
        self.assertEqual(diagnostics["extra_refreshes"], 2)
        self.assertEqual(diagnostics["cached_steps"], 1)
        self.assertEqual(diagnostics["max_distance"], 1)

    def test_threshold_zero_has_the_same_tokens_as_full_saber(self):
        saber_model = self.TinyModel()
        archer_model = self.TinyModel()
        saber_tokens, saber_steps = self.generate_saber(
            saber_model,
            self._prompt(),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            mask_id=3,
        )
        archer_tokens, archer_steps, _, diagnostics = self.generate_archer(
            archer_model,
            self._prompt(),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            mask_id=3,
            refresh_radius=0,
        )

        self.assertTrue(torch.equal(archer_tokens, saber_tokens))
        self.assertEqual(archer_steps, saber_steps)
        self.assertEqual(diagnostics["extra_refreshes"], saber_steps - 1)

    def test_summary_uses_global_denominators(self):
        summary = self.summarize_cache_diagnostics(
            [
                {
                    "cache_builds": 2,
                    "extra_refreshes": 1,
                    "cached_steps": 3,
                    "distance_checks": 2,
                    "distance_sum": 3,
                    "max_distance": 2,
                    "refresh_distance_sum": 2,
                    "max_refresh_distance": 2,
                },
                {
                    "cache_builds": 4,
                    "extra_refreshes": 0,
                    "cached_steps": 7,
                    "distance_checks": 4,
                    "distance_sum": 12,
                    "max_distance": 5,
                    "refresh_distance_sum": 0,
                    "max_refresh_distance": 0,
                },
            ]
        )

        self.assertEqual(summary["mean_cache_builds"], 3.0)
        self.assertEqual(summary["mean_snapshot_distance"], 2.5)
        self.assertEqual(summary["mean_refresh_distance"], 2.0)
        self.assertEqual(summary["max_snapshot_distance"], 5)
