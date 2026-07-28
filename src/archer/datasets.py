"""Benchmark loading, prompt construction, and answer extraction."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_mbpp_problem(question: str, tests: Iterable[str]) -> str:
    return ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(
        question.strip(), "\n".join(tests)
    )


def load_mbpp(path: Path) -> List[Dict[str, Any]]:
    docs = []
    for index, example in enumerate(load_jsonl(path)):
        tests = example.get("test_list")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"MBPP example {index} has no test_list")
        candidates = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", tests[0])
        if not candidates:
            raise ValueError(f"cannot find MBPP entry point in example {index}")
        docs.append(
            {
                "task_id": example.get("task_id", str(index)),
                "prompt": format_mbpp_problem(example["prompt"], tests),
                "entry_point": candidates[-1],
                "examples": [],
            }
        )
    return docs


def humaneval_doc_to_text(
    doc: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    prompt = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{doc['prompt']}\n```"
    )
    context = [{"role": "user", "content": prompt}]
    trailing = f"Here is the completed function:\n```python\n{doc['prompt']}\n"
    return context, doc, trailing


def mbpp_doc_to_text(
    doc: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    prompt = (
        "You are an expert Python programmer. Please write a python function to solve "
        f"the following problem:\n{doc['prompt']}"
    )
    return [{"role": "user", "content": prompt}], doc, ">>> Code:\n```python\n"


def livecodebench_doc_to_text(
    doc: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    """Reproduce the zero-shot LiveCodeBench prompt used by Saber."""
    prompt = (
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program.\n\n"
        f"Question:\n{doc['question_content']}\n"
    )
    starter_code = doc.get("starter_code", "")
    if starter_code:
        prompt += (
            "You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters.\n"
            f"```python\n{starter_code}\n```\n\n"
        )
    else:
        prompt += (
            "Read the inputs from stdin solve the problem and write the answer to "
            "stdout (do not directly test on the sample inputs). Enclose your code "
            "within delimiters as follows. Ensure that when the python program runs, "
            "it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return [{"role": "user", "content": prompt}], doc, ""


def extract_humaneval_answer(generated_text: str, doc: Dict[str, Any]) -> str:
    text_with_prefix = f"```python\n{doc['prompt']}\n" + generated_text
    pattern = re.compile(
        rf"def\s+{re.escape(doc['entry_point'])}.*?:\n(.*?)\n```", re.DOTALL
    )
    match = pattern.search(text_with_prefix)
    if match:
        return match.group(1).rstrip()
    return re.split(r"```", generated_text, maxsplit=1)[0].rstrip()


def extract_mbpp_answer(generated_text: str, doc: Dict[str, Any]) -> str:
    pattern = re.compile(
        rf"(def\s+{re.escape(doc['entry_point'])}\s*\(.*?\):\n.*?)(?=^```)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(generated_text)
    if match:
        return match.group(1).rstrip()
    return textwrap.indent(generated_text, " " * 4)


def extract_livecodebench_answer(generated_text: str, doc: Dict[str, Any]) -> str:
    """Extract the first Python fence exactly as Saber's dataset handler does."""
    del doc
    match = re.search(r"```python\s*(.*?)\s*```", str(generated_text), re.DOTALL)
    return match.group(1).strip() if match else str(generated_text).strip()
