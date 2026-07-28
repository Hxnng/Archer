import hashlib
import unittest
from pathlib import Path

from archer.datasets import (
    extract_humaneval_answer,
    extract_livecodebench_answer,
    extract_mbpp_answer,
    humaneval_doc_to_text,
    livecodebench_doc_to_text,
    load_jsonl,
    load_mbpp,
    mbpp_doc_to_text,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = {
    "benchmarks/humaneval/HumanEval.jsonl": (
        164,
        "f4c47d5090c07ba3e393c4ab4a22779b269fda6fcfc1e3f8fddb08911a8199c6",
    ),
    "benchmarks/humaneval/HumanEval_ET.jsonl": (
        164,
        "c9d9ccd7e299897a423a013ddd59e248b1befb68068d9e74811f1cfad49be0c5",
    ),
    "benchmarks/mbpp/mbpp_sanitized.jsonl": (
        427,
        "502db01bcc1e911d691d5212bd3d9c60ee52bbc2bd29457a520ace424bb2709d",
    ),
    "benchmarks/mbpp/MBPP_ET.jsonl": (
        974,
        "7c4b41c363f25a8b5a764478c8ca8e14fd1cf35f86b6f665a092f8bde14e41f9",
    ),
}


class DatasetTests(unittest.TestCase):
    def test_benchmark_snapshots(self):
        for relative_path, (expected_records, expected_hash) in SNAPSHOTS.items():
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                self.assertEqual(len(load_jsonl(path)), expected_records)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash
                )

    def test_prompt_contracts(self):
        humaneval_doc = {
            "task_id": "HumanEval/0",
            "entry_point": "identity",
            "prompt": 'def identity(value):\n    """Return value."""',
        }
        context, ground_truth, trailing = humaneval_doc_to_text(humaneval_doc)
        self.assertIs(ground_truth, humaneval_doc)
        self.assertTrue(context[0]["content"].startswith("Write a solution"))
        self.assertTrue(trailing.endswith(humaneval_doc["prompt"] + "\n"))
        self.assertTrue(
            extract_humaneval_answer(
                "    return value\n```", humaneval_doc
            ).endswith("    return value")
        )

        mbpp_doc = {
            "task_id": 1,
            "entry_point": "identity",
            "prompt": (
                ">>> Problem:\nReturn a value.\n>>> Test Cases:\n"
                "assert identity(1) == 1\n"
            ),
        }
        context, ground_truth, trailing = mbpp_doc_to_text(mbpp_doc)
        self.assertIs(ground_truth, mbpp_doc)
        self.assertTrue(
            context[0]["content"].startswith("You are an expert Python programmer")
        )
        self.assertEqual(trailing, ">>> Code:\n```python\n")
        generated = "```python\ndef identity(value):\n    return value\n```"
        self.assertEqual(
            extract_mbpp_answer(generated, mbpp_doc),
            "def identity(value):\n    return value",
        )

        livecodebench_doc = {
            "question_id": "atcoder_1",
            "question_content": "Print the sum of two integers.",
            "starter_code": "",
        }
        context, ground_truth, trailing = livecodebench_doc_to_text(
            livecodebench_doc
        )
        self.assertIs(ground_truth, livecodebench_doc)
        self.assertEqual(trailing, "")
        self.assertIn("Read the inputs from stdin", context[0]["content"])
        self.assertEqual(
            extract_livecodebench_answer(
                "Explanation\n```python\nprint(sum(map(int, input().split())))\n```",
                livecodebench_doc,
            ),
            "print(sum(map(int, input().split())))",
        )

        starter_doc = {
            **livecodebench_doc,
            "starter_code": "class Solution:\n    pass",
        }
        context, _, _ = livecodebench_doc_to_text(starter_doc)
        self.assertIn("class Solution:", context[0]["content"])


if __name__ == "__main__":
    unittest.main()
