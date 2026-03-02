import torch
import numpy as np

@torch.no_grad()
def score_steps_qwen_prm(args, model, tokenizer, question: str, steps: list[str]) -> list[float]:
    STEP_TOKEN = "<extra_0>"
    assistant_content = STEP_TOKEN.join(steps) + STEP_TOKEN
    system = "Please reason step by step, and put your final answer within \\boxed{}."

    if args.dataset in ["mbpp", "mbppplus", "humaneval", "humanevalplus"]:
        system = "You are solving a programming problem. Reason step by step."

    if args.dataset in ["mmlu"]:
        system = "Please put your final choice within \\boxed{}."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]
    conv = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(conv, return_tensors="pt").to(model.device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out[0]
    step_sep_id = tokenizer.encode(STEP_TOKEN)[0]
    mask = (input_ids == step_sep_id)  # (B,T)
    probs = logits.softmax(dim=-1)
    step_scores = probs[mask][:, 1].float().tolist()  # P(positive) at each marker
    return step_scores

@torch.no_grad()
def score_code(args, model, tokenizer, question, program):
    if args.dataset in ["mbpp", "mbppplus"]:
        question_q = question.split('Problem:\n')[1].split('Details')[0]
        unit_tests = question.split('Unit test')[1].split('Setup code')[0].split('assert')[1:]
        unit_tests = [ut.strip().strip('\n') for ut in unit_tests if '==' in ut]

    else:
        raise NotImplementedError(f"Code scoring not implemented for dataset {args.dataset}")

    new_question = f"""\
    {question_q}\n
    For example:
    """

    try:
        inputs = [unit_test.split('==')[0].strip() for unit_test in unit_tests]
        outputs = [unit_test.split('==')[1].strip() for unit_test in unit_tests]
        for idx, entry in enumerate(inputs):
            new_question += f"Input: {entry}\nOutput: {outputs[idx]}\n"
            break

    except Exception as e:
        print(e)
        print("Problem parsing unit tests. Skipping few-shot examples.")
        pass
    
    program_chats = [
        [
            {
                "content": new_question,
                "role": "user",
            },
            {
                "role": "assistant",
                "content": program_eg.strip('```python')
            }
        ] for program_eg in [program]
    ]

    input_tokens = tokenizer.apply_chat_template(
        program_chats,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_tokens = {k: v.to(model.device) for k, v in input_tokens.items()}
    rm_scores = model(
        **input_tokens,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,    
    )
    rm_scores = float(rm_scores.detach().to("cpu").item())

    return np.array([rm_scores], dtype=np.float32)