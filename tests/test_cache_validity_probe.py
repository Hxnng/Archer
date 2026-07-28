import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "cache-validity tests require torch")
class CacheValidityProbeTests(unittest.TestCase):
    class TinyModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, input_ids, past_key_values=None, use_cache=False):
            del past_key_values, use_cache
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            logits[..., 0] = 2.0
            key = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], 1)
            return SimpleNamespace(logits=logits, past_key_values=[(key, key)])

    @staticmethod
    def prompt():
        return torch.tensor([[1]], dtype=torch.long)

    def test_shadow_probe_does_not_change_archer_trajectory_or_main_nfe(self):
        from archer.cache_validity_probe import generate_archer_with_validity_probe
        from archer.state_distance import generate_archer

        baseline = generate_archer(
            self.TinyModel(),
            self.prompt(),
            n=1,
            mu=2,
            gen_length=4,
            block_length=4,
            mask_id=3,
            refresh_radius=3,
        )
        probed = generate_archer_with_validity_probe(
            self.TinyModel(),
            self.prompt(),
            n=1,
            mu=2,
            gen_length=4,
            block_length=4,
            mask_id=3,
            refresh_radius=3,
            probe_every=1,
            max_probes=8,
        )

        self.assertTrue(torch.equal(probed[0], baseline[0]))
        self.assertEqual(probed[1], baseline[1])
        self.assertEqual(probed[2], baseline[2])
        self.assertEqual(probed[3], baseline[3])
        self.assertEqual(probed[5], len(probed[4]))
        self.assertGreater(probed[5], 0)
        self.assertTrue(
            all(not item["next_action_disagrees"] for item in probed[4])
        )

    def test_probe_restores_rng_before_real_decoder_step(self):
        from archer.cache_validity_probe import generate_archer_with_validity_probe

        torch.manual_seed(2026)
        generate_archer_with_validity_probe(
            self.TinyModel(),
            self.prompt(),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            temperature=0.1,
            mask_id=3,
            refresh_radius=3,
            probe_every=1,
            max_probes=3,
        )
        value_after_probe = torch.rand(1)

        torch.manual_seed(2026)
        from archer.state_distance import generate_archer

        generate_archer(
            self.TinyModel(),
            self.prompt(),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            temperature=0.1,
            mask_id=3,
            refresh_radius=3,
        )
        value_after_baseline = torch.rand(1)
        self.assertTrue(torch.equal(value_after_probe, value_after_baseline))

    def test_summary_distinguishes_distance_from_age(self):
        from archer.cache_validity_probe import summarize_validity_probes

        probes = []
        for distance, age in zip((0, 1, 2, 3), (0, 3, 1, 2)):
            probes.append(
                {
                    "anchor_distance": distance,
                    "cache_age": age,
                    "logit_mae": float(distance),
                    "logit_max_error": float(distance),
                    "top1_disagreement_rate": distance / 3,
                    "next_state_hamming": distance,
                    "next_action_disagrees": bool(distance),
                }
            )
        summary = summarize_validity_probes(probes, refresh_radius=4)

        self.assertEqual(summary["probe_count"], 4)
        self.assertAlmostEqual(
            summary["correlations"]["logit_mae"]["distance_spearman"], 1.0
        )
        self.assertLess(
            summary["correlations"]["logit_mae"]["age_spearman"], 1.0
        )
        self.assertIn(
            "distance_partial_controlling_age",
            summary["correlations"]["next_action_disagrees"],
        )
        self.assertEqual(summary["by_anchor_distance"]["3-3"]["count"], 1)
        self.assertEqual(
            summary["by_normalized_anchor_distance"]["0.75-1.00"]["count"], 1
        )


class CacheValidityCliTests(unittest.TestCase):
    def test_mbpp_probe_defaults_to_k11(self):
        from archer.cache_validity_experiment import parse_args

        args = parse_args(["--model-path", "/model", "--run-name", "smoke"])
        self.assertEqual(args.refresh_radius, 11)
        self.assertEqual(args.probe_every, 4)
        self.assertEqual(args.max_probes, 12)


if __name__ == "__main__":
    unittest.main()
