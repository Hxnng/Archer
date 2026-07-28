import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "feedback probe tests require torch")
class FeedbackProbeTests(unittest.TestCase):
    class TinyModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, input_ids, past_key_values=None, use_cache=False):
            del past_key_values, use_cache
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            logits[..., 0] = 2.0
            key = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], 1)
            return SimpleNamespace(logits=logits, past_key_values=[(key, key)])

    def test_probe_forks_one_shared_commit_state(self):
        from archer.feedback_probe import generate_prompt_feedback_probe

        fresh, cached, probe = generate_prompt_feedback_probe(
            self.TinyModel(),
            torch.tensor([[1]], dtype=torch.long),
            n=1,
            mu=2,
            gen_length=3,
            block_length=3,
            mask_id=3,
            refresh_radius=2,
            confidence_threshold=0.0,
            horizon=2,
        )

        self.assertTrue(probe["eligible"])
        self.assertEqual(probe["event_step"], 1)
        self.assertEqual(probe["target_token"], 0)
        self.assertEqual(probe["fresh_margin"], probe["cached_margin"])
        self.assertEqual(fresh["tokens"].shape, cached["tokens"].shape)

    def test_probe_validates_intervention_parameters(self):
        from archer.feedback_probe import generate_prompt_feedback_probe

        with self.assertRaises(ValueError):
            generate_prompt_feedback_probe(
                self.TinyModel(),
                torch.tensor([[1]], dtype=torch.long),
                gen_length=3,
                block_length=3,
                refresh_radius=0,
            )


@unittest.skipIf(torch is None, "feedback experiment imports the GPU runtime")
class FeedbackSummaryTests(unittest.TestCase):
    def test_summary_reports_paired_disagreements(self):
        from archer.feedback_experiment import summarize

        rows = [
            {
                "task_id": "a",
                "wall_time_s": 2.0,
                "probe": {
                    "eligible": True,
                    "fresh_margin": 3.0,
                    "cached_margin": 1.0,
                    "fresh_probability": 0.9,
                    "cached_probability": 0.7,
                    "fresh_revised_within_horizon": False,
                    "cached_revised_within_horizon": True,
                    "fresh_revision_step": None,
                    "cached_revision_step": 2,
                },
            },
            {"task_id": "b", "wall_time_s": 4.0, "probe": {"eligible": False}},
        ]

        def evaluation(a_passed, b_passed):
            return {
                "standard": {
                    "pass_at_1": 0.5,
                    "task_results": [
                        {"task_id": "a", "passed": a_passed},
                        {"task_id": "b", "passed": b_passed},
                    ],
                },
                "extended": {"pass_at_1": 0.25},
            }

        result = summarize(rows, evaluation(False, True), evaluation(True, False))
        self.assertEqual(result["eligible_events"], 1)
        self.assertEqual(result["mean_fresh_minus_cached_margin"], 2.0)
        self.assertEqual(result["cached"]["revision_rate_within_horizon"], 1.0)
        self.assertEqual(result["paired_outcomes"]["eligible_cached_only_pass"], 1)


if __name__ == "__main__":
    unittest.main()
