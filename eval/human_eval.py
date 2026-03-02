from datasets import load_dataset
from gsm8k import GSM8KDataset

# dllm prompt for generating the reasoning (as comments)
HUMANEVAL_SYSTEM_PROMPT = """
You are helping a separate autoregressive code model solve HumanEval-style Python tasks.

Given the exact task prompt below (signature + docstring + examples), produce a short, concrete implementation guide.

REQUIREMENTS:
- Do NOT write any Python code.
- Do NOT include Markdown fences or explanations outside the hint block.
- Output 6-12 lines total.
- Each line must start with "# " (hash + space), with no other leading text.
- Focus on: algorithm choice, key steps, edge cases, complexity, and any Python gotchas.
- Prefer actionable steps over prose. Do not restate the entire prompt.

TASK PROMPT:
"""

class HumanEvalBaseDataset(GSM8KDataset):
    HF_DATASET_NAME = None
    HF_SPLIT = "test"

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=False,  
        system_prompt=HUMANEVAL_SYSTEM_PROMPT,
        subsample=-1,
    ):
        assert num_examples == 0, "Few-shot is disabled for HumanEval datasets"
        # super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)
        super().__init__(
            tokenizer,
            num_examples=num_examples,
            add_reasoning=add_reasoning,
            system_prompt=system_prompt,
            subsample=subsample,
        )


    def load_few_shot_examples(self):
        return []

    def create_few_shot_prompt(self):
        self.few_shot_prompt = ""

    def load_test_dataset(self):
        if self.HF_DATASET_NAME is None:
            raise ValueError("HF_DATASET_NAME must be set in a subclass.")
        self.dataset = load_dataset(self.HF_DATASET_NAME, split=self.HF_SPLIT)

    # def create_prompt(self, prompt_text: str) -> str:
    #     messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt_text}]
    #     return self.tokenizer.apply_chat_template(
    #         messages, add_generation_prompt=True, tokenize=False
    #     )

    def __getitem__(self, idx):
        row = self.dataset[int(self.subsample[idx])]

        question = row["prompt"]
        prompt = self.create_prompt(question)

        ground_truth = {
            "task_id": row.get("task_id"),
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
            "canonical_solution": row.get("canonical_solution", ""),
        }
        return prompt, question, ground_truth

class HumanEvalDataset(HumanEvalBaseDataset):
    HF_DATASET_NAME = "openai/openai_humaneval"

class HumanEvalPlusDataset(HumanEvalBaseDataset):
    HF_DATASET_NAME = "evalplus/humanevalplus"