import textwrap
from typing import List

from datasets import load_dataset
from gsm8k import GSM8KDataset

MBPP_SYSTEM_PROMPT = """You are a helpful Python coding assistant. When given a programming problem, respond with self-contained Python 3 code that solves the task and passes the provided tests. Output only a single ```python ...``` code block with no additional commentary."""

MBPP_PROMPT_TEMPLATE = """
Problem:
{problem}

Details:
{description}

Unit tests your solution must pass:
{tests}

Setup code that runs before the tests:
{setup}

Write the full solution now. Remember to return only Python code inside a single fenced block.
""".strip()


def _format_lines(lines):
    if not lines:
        return "None"
    cleaned = [textwrap.dedent(line).strip() for line in lines if line.strip()]
    return "\n".join(cleaned) if cleaned else "None"


class MBPPBaseDataset(GSM8KDataset):
    HF_DATASET_NAME = None

    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = False,
        system_prompt: str = MBPP_SYSTEM_PROMPT,
        subsample: int = -1,
        split: str = "test",
        include_challenge_tests: bool = True,
    ):
        self.split = split
        self.include_challenge_tests = include_challenge_tests
        super().__init__(
            tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
        )


    def load_test_dataset(self):
        if self.HF_DATASET_NAME is None:
            raise ValueError("HF_DATASET_NAME must be set in a subclass.")
        self.dataset = load_dataset(self.HF_DATASET_NAME, split=self.split, trust_remote_code=True)

    def _build_question(self, entry):
        problem = entry.get("prompt") or entry.get("text") or ""
        description = entry.get("text") or entry.get("description") or problem

        base_tests = entry.get("test_list") or []
        challenge_tests = entry.get("challenge_test_list") or []
        if self.include_challenge_tests:
            all_tests = base_tests + challenge_tests
        else:
            all_tests = base_tests

        setup_sections = [
            entry.get("test_setup_code") or "",
            entry.get("challenge_test_setup_code") or "",
        ]
        setup_code = "\n\n".join(
            [
                section.strip()
                for section in setup_sections
                if section and section.strip()
            ]
        )
        if not setup_code:
            setup_code = "None"

        tests_formatted = _format_lines(all_tests)

        rendered_prompt = MBPP_PROMPT_TEMPLATE.format(
            problem=problem.strip(),
            description=(
                description.strip()
                if description
                else "No additional details."
            ),
            tests=tests_formatted,
            setup=setup_code,
        )
        return rendered_prompt

    def __getitem__(self, idx):
        sample = self.dataset[self.subsample[idx].item()]
        question = self._build_question(sample)
        prompt = self.create_prompt(question)
        ground_truth = {
            "task_id": sample.get("task_id"),
            "test_list": sample.get("test_list") or [],
            "test_setup_code": sample.get("test_setup_code") or "",
            "challenge_test_list": sample.get("challenge_test_list") or [],
            "challenge_test_setup_code": (
                sample.get("challenge_test_setup_code") or ""
            ),
        }
        return prompt, question, ground_truth

class MBPPDataset(MBPPBaseDataset):
    HF_DATASET_NAME = "Muennighoff/mbpp"

class MBPPPlusDataset(MBPPBaseDataset):
    HF_DATASET_NAME = "evalplus/mbppplus"