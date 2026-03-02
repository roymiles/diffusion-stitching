import os
username = os.environ.get('USER')
import sys
sys.path.insert(0, f"/home/{username}/diffusion_stitching_suppl/")
from eval import main as main_eval, init_seed, setup_ddp, cleanup_ddp
import torch.distributed as dist
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="One or more evaluation tasks to run.", default=["math", "gsm8k"])
    parser.add_argument("--gen_lengths", nargs="+", type=int, help="Generation lengths for each task.", default=[128, 256, 512])
    parser.add_argument("--model_base", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--verifier_base", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="out/")
    parser.add_argument("--ar_model", action="store_true")
    parser.add_argument("--fast_dllm_sampling", action="store_true")
    parser.add_argument("--do_stitching", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--stitching_confidence", type=float, default=0.9)
    parser.add_argument("--kv_cache", action="store_true")
    parser.add_argument("--parallel_rollouts", action="store_true")
    parser.add_argument("--keep_steps_in_agreement", action="store_true")
    parser.add_argument("--max_steps_per_step", type=int, default=0)
    parser.add_argument("--prm_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    # NOTE: do not do this inside main_eval
    # else all the reasoning paths will be identical
    init_seed(args.seed)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics (parse_and_get_acc.py).
    local_rank = setup_ddp()
    world_size = dist.get_world_size()

    batch_size = 1
    for task in args.tasks:
        for gen_length in args.gen_lengths:
            if task == 'gsm8k':
                args.temperature = 1.7
                args.confidence = 0.7

            if task == 'math':
                args.temperature = 0.8
                args.confidence = 0.7
                
            args.dataset = task
            args.batch_size = batch_size
            args.gen_length = gen_length
            args.block_length = 32
            main_eval(args, local_rank, world_size, verbose=True)

    cleanup_ddp()