import base64
import json
import pickle
import unittest
import zlib

from archer.livecodebench_evaluator import evaluation_sample


class LiveCodeBenchEvaluatorTests(unittest.TestCase):
    def test_builds_official_input_output_contract(self):
        private = [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
        encoded_private = base64.b64encode(
            zlib.compress(pickle.dumps(json.dumps(private)))
        ).decode()
        problem = {
            "public_test_cases": json.dumps(
                [{"input": "1 1\n", "output": "2\n", "testtype": "stdin"}]
            ),
            "private_test_cases": encoded_private,
            "metadata": json.dumps({"func_name": None}),
        }

        sample = json.loads(evaluation_sample(problem)["input_output"])

        self.assertEqual(sample["inputs"], ["1 1\n", "2 3\n"])
        self.assertEqual(sample["outputs"], ["2\n", "5\n"])
        self.assertIsNone(sample["fn_name"])


if __name__ == "__main__":
    unittest.main()
