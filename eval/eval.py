from __future__ import annotations

import argparse
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import re
from dataclasses import dataclass
from functools import partial
from collections import defaultdict
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from acecoder import AceCodeRM
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from misc_utils import extract_question_and_steps, extract_question_and_steps_coding, extract_hint_lines
from prm_utils import score_steps_qwen_prm, score_code

from countdown import CountdownDataset, COUNTDOWN_SYSTEM_PROMPT
from generate import generate as generate_og
from generate_fastdllm import generate as generate_fastdllm, generate_with_prefix_cache
from generate_ar import generate as generate_ar
from generate_fastdllm import generate as generate_fastdllm
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset, MATH500_SYSTEM_PROMPT
from mbpp import MBPP_SYSTEM_PROMPT, MBPPDataset, MBPPPlusDataset
from human_eval import HUMANEVAL_SYSTEM_PROMPT, HumanEvalDataset, HumanEvalPlusDataset
from model.modeling_llada import LLaDAModelLM as LLaDAModelLMKVCache


class StopAfterBoxed(StoppingCriteria):
    """Stops generation once a full \\boxed{ ... } (with balanced braces) has been generated.
    """
    def __init__(self, tokenizer, start=r"\boxed{", rolling_chars=256):
        self.tok = tokenizer
        self.start = start
        self.rolling_chars = rolling_chars

        self.prev_len = 0
        self.buffer = ""          # rolling decoded text (for detecting start across token boundaries)
        self.in_box = False
        self.brace_depth = 0      # counts { } after the start

    def __call__(self, input_ids, scores, **kwargs):
        # batch size 1 assumed
        ids = input_ids[0].tolist()

        # decode only newly generated tokens since last step
        new_ids = ids[self.prev_len:]
        self.prev_len = len(ids)
        if not new_ids:
            return False

        new_text = self.tok.decode(new_ids, skip_special_tokens=False)

        if not self.in_box:
            # keep a rolling window so "\boxed{" can be found even if split across steps
            self.buffer = (self.buffer + new_text)[-self.rolling_chars:]
            idx = self.buffer.find(self.start)
            if idx == -1:
                return False

            # we just entered a boxed segment
            self.in_box = True
            self.brace_depth = 1  # the "{" in "\boxed{"

            # process any text that already came after "\boxed{"
            after = self.buffer[idx + len(self.start):]
            for ch in after:
                if ch == "{":
                    self.brace_depth += 1
                elif ch == "}":
                    self.brace_depth -= 1
                    if self.brace_depth == 0:
                        return True

            # clear buffer once we've found the start to avoid re-detecting it
            self.buffer = ""
            return False

        # if we're inside the box, keep counting braces until balanced
        for ch in new_text:
            if ch == "{":
                self.brace_depth += 1
            elif ch == "}":
                self.brace_depth -= 1
                if self.brace_depth == 0:
                    return True

        return False
    
DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CountdownDataset,
    "mbpp": MBPPDataset,
    "mbppplus": MBPPPlusDataset,
    "humaneval": HumanEvalDataset,
    "humanevalplus": HumanEvalPlusDataset
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

@dataclass
class StepItem:
    path: int
    step: int
    text: str
    confidence = None

def cleanup_ddp():
    dist.destroy_process_group()

def load_prm(prm_name: str, device: torch.device):
    if prm_name.startswith("Qwen/Qwen2.5-Math-PRM"):
        tok = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            prm_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        return ("qwen_prm", model, tok)

    if prm_name == "TIGER-Lab/AceCodeRM-7B":
        tok = AutoTokenizer.from_pretrained(prm_name, trust_remote_code=True)
        model = AceCodeRM.from_pretrained(prm_name, torch_dtype=torch.bfloat16).to(device).eval()
        return ("acecode_rm", model, tok)

    raise ValueError(f"Unknown PRM: {prm_name}")


def score_steps(args, prm_kind, model_p, tokenizer_p, question, steps):
    if prm_kind == "qwen_prm":
        return score_steps_qwen_prm(args, model_p, tokenizer_p, question, steps)
    
    raise ValueError(prm_kind)

@torch.no_grad()
def generate(
    args,
    model,
    tokenizer,
    input_ids,
    attention_mask,
    mask_id,
    cfg_scale
):
    if args.kv_cache and not args.fast_dllm_sampling:
        raise ValueError("KV cache is only supported with confidence based decoding.")

    if args.fast_dllm_sampling:
        # confidence based
        generate_fn = generate_with_prefix_cache if args.kv_cache else generate_fastdllm
        out, num_steps = generate_fn(
            model,
            input_ids,
            steps=64,  
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking="low_confidence",
            confidence=args.confidence,
            mask_id=mask_id
        )
    else:
        out, num_steps = generate_og(
            model,
            input_ids,
            tokenizer,
            steps=args.gen_length // 2,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            mask_id=mask_id
        )

    return out, num_steps

def format_and_tokenize_verifier_input(tokenizer, text, device, add_reasoning_tags=True):
    if add_reasoning_tags:
        matches = list(re.finditer(re.escape("</reasoning>"), text))
        if len(matches) >= 3:
            text = text[:matches[2].start()]
        text = text + "</reasoning><answer>"

    enc = tokenizer(text, return_tensors="pt")
    return text, enc["input_ids"].to(device), enc["attention_mask"].to(device)

def evaluate(
    args,
    model,
    tokenizer,
    dataloader,
    verifier=None,
    cfg_scale=0
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    all_generations = []
    num_fwds = []
    num_fwds_v = []
    device = model.device

    tokenizer_v = AutoTokenizer.from_pretrained(args.verifier_base, trust_remote_code=True)
    prm_kind, model_p, tokenizer_p = load_prm(args.prm_name, device)

    if args.model_base == "GSAI-ML/LLaDA-8B-Instruct":
        mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        assert mask_id == 126336, f"Unexpected id with value {mask_id} for <|mdm_mask|>"

    else:
        if not args.ar_model:
            raise Exception(f"Unknown base model: {args.model_base}")
        
    def _set_rollout_seed(base_seed, batch_idx, rollout_id):
        # different sample per (example, rollout)
        s = int(base_seed + 100000 * batch_idx + rollout_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)

    def _run_single_rollout(path_idx):
        """
        One and only implementation of a rollout.
        Returns everything rank0 needs for selection/stitching/verifier.
        """
        _set_rollout_seed(args.seed, batch_idx, path_idx)

        out_ids, cur_num_steps = generate(
            args,
            model,
            tokenizer,
            input_ids,
            attention_mask,
            mask_id,
            cfg_scale
        )

        full_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

        # extract question and answer from text, this is very chat template specific
        parts = full_text.split("assistant\n\n")
        question = parts[0].split("user\n\n")[1]
        answer_txt = parts[1]

        SYSTEM_PROMPTS = {
            "countdown": COUNTDOWN_SYSTEM_PROMPT,
            "mbpp": MBPP_SYSTEM_PROMPT, 
            "mbppplus": MBPP_SYSTEM_PROMPT,
            "humaneval": HUMANEVAL_SYSTEM_PROMPT, 
            "humanevalplus": HUMANEVAL_SYSTEM_PROMPT
        }
        BASE_SYSTEM_PROMPT = MATH500_SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPTS.get(args.dataset, BASE_SYSTEM_PROMPT)

        if args.dataset in ["humaneval", "humanevalplus"]:
            # for humaneval we just add in extra reasoning/explanation as comments
            # to make code completion easier for the AR model
            hints = extract_hint_lines(answer_txt) 
            comments = [c for c in hints if isinstance(c, str) and c.lstrip().startswith("#")]
            extracted = {}
            # just use the comments of the function prompt as the question
            pattern = r'("""|\'\'\')([\s\S]*?)\1'
            matches = [m.group(2) for m in re.finditer(pattern, question)]
            extracted["question"] = matches[0]
            # some extra cleanup
            extracted["question"] = extracted["question"].split(">>>")[0].strip().replace("\n", " ")
            extracted["question"] = re.sub(r'\s+', ' ', extracted["question"]).strip()
            # the comments generated need to be evaluated
            extracted["steps"] = [c.lstrip()[1:].lstrip() for c in comments if isinstance(c, str) and c.lstrip().startswith("#")]

            # give the extra information in the system prompt
            system_prompt = system_prompt + "\n\n" + "\n".join(extracted["steps"])
            # we need to remove the dLLM system prompt to get the question
            question = question.split("TASK PROMPT:")[1].strip()

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # build teacher-forced AR text for step extraction
        prefix_ids = tokenizer_v.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)

        ans_ids = torch.tensor(
            tokenizer_v.encode(answer_txt, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        )

        teacher_ids = torch.cat([prefix_ids.unsqueeze(0), ans_ids.unsqueeze(0)], dim=1)
        ar_txt = tokenizer_v.decode(teacher_ids[0], add_special_tokens=True)

        # tokenise verifier input
        is_code = args.dataset in ["mbpp","mbppplus","humaneval","humanevalplus"]
        _, ar_ids, attn_mask = format_and_tokenize_verifier_input(tokenizer_v, ar_txt, device, add_reasoning_tags=not is_code)

        # extract steps + score them
        if args.dataset in ["humaneval", "humanevalplus"]:
            # keep the extracted["steps"] you already built from "# " hints
            pass
        
        elif args.dataset in ["mbpp", "mbppplus"]:
            extracted = extract_question_and_steps_coding(ar_txt)
            # remove final step if it is just an end token like </thought> or </solution>
            if len(extracted["steps"]) > 0:
                if extracted["steps"][-1] in ("</thought>", "</solution>"):
                    extracted["steps"] = extracted["steps"][:-1]

        else:
            extracted = extract_question_and_steps(ar_txt)

        # different ways of calculating confidences of each step 
        if prm_kind == "acecode_rm" and args.dataset in ["mbpp", "mbppplus"]:
            conf = score_code(args, model_p, tokenizer_p, extracted["question"], extracted["code"])
        else:
            conf = score_steps(args, prm_kind, model_p, tokenizer_p, extracted["question"], extracted["steps"])

        conf = np.array(conf, dtype=np.float32)

        return {
            "path_idx": int(path_idx),
            "num_steps": int(cur_num_steps),
            "question_txt": question,
            "answer_txt": answer_txt,
            "steps": extracted["steps"],
            "conf": conf.tolist(),
            "ar_ids": ar_ids,
            "msgs": msgs
        }
    
    def _collect_rollouts(run_fn, num_rollouts):
        """
        Old behavior when parallel_rollouts=False:
        - run_fn called num_rollouts times on this rank.

        Parallel behavior when parallel_rollouts=True:
        - run_fn called once on each rank, gathered into a list on every rank.
        - ASSUMES world_size == num_rollouts and rank == path_idx.
        """
        if not args.parallel_rollouts:
            return [run_fn(i) for i in range(num_rollouts)]

        world = dist.get_world_size()
        rank = dist.get_rank()
        assert world == num_rollouts, f"parallel_rollouts requires world_size==num_rollouts, got {world} vs {num_rollouts}"

        my = run_fn(rank)
        gathered = [None] * world
        dist.all_gather_object(gathered, my)
        return gathered

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0))):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        if args.ar_model:
            out, num_steps = generate_ar(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            # this is not the verifier, but just reusing this metric here
            num_steps_v = num_steps
            num_steps = [0] 

        else:
            # accumulate number of steps used
            num_steps_v = 0
            num_steps = [0] * args.num_rollouts

            # keep track of the rationale steps and their confidences
            all_steps          = [None] * args.num_rollouts
            all_conf           = [None] * args.num_rollouts
            all_question_txt   = [None] * args.num_rollouts
            all_answer_txt     = [None] * args.num_rollouts
            all_ar_ids         = [None] * args.num_rollouts
            all_attention_mask = [None] * args.num_rollouts
            all_msgs           = [None] * args.num_rollouts

            best_conf = -math.inf
            best_path_idx = 0
            generated_texts = [""]
            rollouts = _collect_rollouts(_run_single_rollout, args.num_rollouts)

            for r in rollouts:
                pi                     = r["path_idx"]
                num_steps[pi]          = r["num_steps"]
                all_question_txt[pi]   = r["question_txt"]
                all_answer_txt[pi]     = r["answer_txt"]
                all_steps[pi]          = r["steps"]
                all_conf[pi]           = np.array(r["conf"], dtype=np.float32)
                all_ar_ids[pi]         = r["ar_ids"]
                all_msgs[pi]           = r["msgs"]

            def geo_mean(iterable):
                a = np.array(iterable, dtype=np.float32)
                return float(a.prod() ** (1.0 / max(len(a), 1)))
            
            for pi in range(args.num_rollouts):
                cur = all_conf[pi]
                if geo_mean(cur) > best_conf:
                    best_conf = geo_mean(cur)
                    best_path_idx = pi

            if args.do_stitching:
                _BOXED_RE = re.compile(r"\\boxed\s*{\s*((?:[^{}]|{[^{}]*})*)\s*}")

                def extract_boxed(text: str):
                    # returns last boxed payload if multiple
                    m = None
                    for m in _BOXED_RE.finditer(text):
                        pass
                    return m.group(1) if m else None

                def norm_ans(a: str):
                    if a is None:
                        return None
                    a = a.strip()
                    # drop common wrappers/spaces to make matching robust
                    a = re.sub(r"\s+", "", a)
                    a = a.replace(r"\left", "").replace(r"\right", "")
                    a = a.strip("$")
                    return a
                
                # 1) per-path final boxed answer (scan steps from the end so we catch the "final" one)
                path_ans = []
                for ss in all_steps:
                    ans = None
                    for t in reversed(ss):
                        ans = extract_boxed(t)
                        if ans is not None:
                            break
                    path_ans.append(norm_ans(ans))

                best_ans = path_ans[best_path_idx]

                # 2) decide which paths "agree" with best final answer
                agree = [(pi == best_path_idx) or (best_ans is not None and path_ans[pi] == best_ans)
                        for pi in range(len(all_steps))]
                
                # re-make the ar ids using stitched reasoning steps
                # this works by simply unrolling all the steps and keeping
                # those that are above a threshold
                thr = args.stitching_confidence
                cands_by_step = defaultdict(list)
                for pi, (cs, ss) in enumerate(zip(all_conf, all_steps)):
                    # only look at this path if it agrees with the best
                    if not agree[pi] and args.keep_steps_in_agreement:
                        continue
                    
                    n = min(len(cs), len(ss))
                    for si in range(n):
                        c, t = cs[si], ss[si]
                        # keep those above threshold and also the best cot final step
                        keep = (c > thr) or (pi == best_path_idx and si == n - 1)
                        if keep:
                            cands_by_step[si].append({"path": pi, "step": si, "confidence": c, "text": t})

                stitched_steps = []
                for si in sorted(cands_by_step):
                    items = sorted(cands_by_step[si], key=lambda d: d["confidence"], reverse=True)
                    if args.max_steps_per_step > 0:
                        # keep at most N per step (most confident ones)
                        stitched_steps.extend(items[:args.max_steps_per_step])
                    else:
                        stitched_steps.extend(items)

                # when stitching we need to modify the prompt
                def format_steps(stitched_steps, best_path_idx, max_steps=80):
                    # sort by step index then confidence so it reads like a candidate bank
                    stitched_steps = sorted(
                        stitched_steps,
                        key=lambda d: (d["step"], -d["confidence"], d["path"] != best_path_idx)
                    )
                    stitched_steps = stitched_steps[:max_steps]

                    lines = ["Here are candidate reasoning steps from multiple traces.", "Each item: [c=conf]. Use them as evidence; ignore contradictions; prefer higher conf."]
                    for i, s in enumerate(stitched_steps, 1):
                        lines.append(f"[c={s['confidence']:.3f}] {s['text'].strip()}")
                    return "\n".join(lines)


                # condition the AR model on the stitched rationale
                msgs = []
                if args.dataset in ["humaneval", "humanevalplus", "mbpp", "mbppplus"]:
                    msgs.append({"role": "system", "content": "You are an expert Python programmer. You will be given a Python function signature and docstring. Write a correct implementation. Use the candidate steps as evidence. If steps conflict, choose a consistent subset."}) 
                    problem = all_question_txt[best_path_idx]
                    if problem.lstrip().startswith("User:"):
                        problem = problem.split("User:", 1)[1].strip()
                    msgs.append({"role": "user", "content": f"Problem: {problem}"})
                else:
                    msgs.append({"role": "system", "content": "You are a math expert. You will be given a question to solve. Solve the problem using the candidate steps as evidence. If steps conflict, choose a consistent subset. Wrap the final answer in a \\boxed{}.\nRespond in the following format:\n<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>\n"})
                    msgs.append({"role": "user", "content": all_question_txt[best_path_idx][212:]})
                    
                # breakpoint()
                msgs.append({"role": "user", "content": format_steps(stitched_steps, best_path_idx)})
                ar_txt = tokenizer_v.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                _, ar_ids, attention_mask = format_and_tokenize_verifier_input(tokenizer_v, ar_txt, device)

                # after finishing all the reasoning paths
                stopping = StoppingCriteriaList()
                if args.dataset not in ["mbpp", "mbppplus", "humaneval", "humanevalplus"]:
                    stopping = StoppingCriteriaList([StopAfterBoxed(tokenizer_v)])

                # qwen3: We suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
                out = verifier.generate(
                    input_ids=ar_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer_v.pad_token_id,
                    stopping_criteria=stopping
                )

                full_ids = out.sequences[0]
                prompt_len = ar_ids.shape[-1]
                gen_ids = full_ids[prompt_len:]
                num_steps_v = gen_ids.shape[0]
                generated_texts = [tokenizer_v.decode(gen_ids.tolist(), skip_special_tokens=True)]

                if args.dataset in ["humaneval", "humanevalplus"]:
                    generated_texts[0] = generated_texts[0].split("</answer>")[0]

        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        num_fwds.append(num_steps)
        num_fwds_v.append(num_steps_v)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    if not args.parallel_rollouts:
        torch.cuda.synchronize()

    # num_fwd: [dataset size x num_rollouts]
    avg_num_fwds = np.array(num_fwds).sum(1).mean().item()
    max_avg_num_fwds = np.array(num_fwds).max(1).mean().item()
    avg_num_fwds_v = np.array(num_fwds_v).mean(0).item()
    metrics = {
        "generations": all_generations,
        "total_processed": total_processed.item(),
        "avg_num_fwds": avg_num_fwds,
        "max_avg_num_fwds": max_avg_num_fwds,
        "avg_num_fwds_v": avg_num_fwds_v
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


def main(args, local_rank, world_size, verbose=False):
    if verbose:
        print(f"Running job with args={args}")

    if args.parallel_rollouts:
        assert world_size == args.num_rollouts, f"Need world_size==num_rollouts, got {world_size} vs {args.num_rollouts}"

    # args.diffusion_steps = args.gen_length
    num_evals = {
        "gsm8k": -1, 
        "math": -1,
        "mbpp": -1, 
        "mbppplus": -1,
        "humaneval": -1, 
        "humanevalplus": -1,
        "countdown": 256
    }

    """
    Load models
    """
    if args.kv_cache:
        assert args.model_base == "GSAI-ML/LLaDA-8B-Instruct", "KV cache is only tested with vanilla LLaDA."
        config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
        config.flash_attention = True
        base_model = LLaDAModelLMKVCache.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(local_rank)
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

    else:
        base_model = AutoModel.from_pretrained(
            args.model_base,
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(local_rank)

        tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
        model = base_model

    model = model.eval()

    base_model = AutoModelForCausalLM.from_pretrained(args.verifier_base)
    verifier = base_model
    verifier.to(local_rank)
    verifier.to(torch.bfloat16)
    verifier.eval()

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True
    )

    sampler = None if args.parallel_rollouts else CustomDistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        collate_fn=partial(dataset.collate_fn, ar_model=args.ar_model),
    )

    model_name = args.model_base.split("/")
    model_name = model_name[-2] + "_" + model_name[-1]

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{local_rank}_generations.json"
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        args,
        model,
        tokenizer,
        dataloader,
        verifier=verifier
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "total_processed": metrics["total_processed"],
                    },
                    "model_base": args.model_base,
                    "verifier_base": args.verifier_base,
                    "gen_length": args.gen_length,
                    "temperature": args.temperature,
                    "confidence": args.confidence,
                    "block_length": args.block_length,
                    "avg_num_fwds": metrics['avg_num_fwds'],
                    "max_avg_num_fwds": metrics['max_avg_num_fwds'],
                    "avg_num_fwds_v": metrics['avg_num_fwds_v'],
                    "ar_model": args.ar_model,
                    "fast_dllm_sampling": args.fast_dllm_sampling,
                    "do_stitching": args.do_stitching,
                    "stitching_confidence": args.stitching_confidence,
                    "num_rollouts": args.num_rollouts,
                    "kv_cache": args.kv_cache,
                    "prm_name": args.prm_name,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--verifier_base", type=str, default="/data1/shared/Qwen2.5-Math-1.5B/")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, choices=["mmlu", "gsm8k", "math", "mbpp", "mbppplus", "humaneval", "humanevalplus"], default="gsm8k")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--ar_model", action="store_true")
    parser.add_argument("--fast_dllm_sampling", action="store_true")
    parser.add_argument("--do_stitching", action="store_true")
    parser.add_argument("--stitching_confidence", type=float, default=0.9)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--parallel_rollouts", action="store_true")
    parser.add_argument("--keep_steps_in_agreement", action="store_true")
    parser.add_argument("--max_steps_per_step", type=int, default=0)
    parser.add_argument("--prm_name", default= "Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    verbose = True
    main(args, verbose=verbose)