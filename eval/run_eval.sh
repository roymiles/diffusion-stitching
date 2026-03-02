#!/bin/bash

# Just use the maximum number of gpus available
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"
  NUM_GPUS="${#_gpus[@]}"
else
  NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
fi

GPU_IDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_IDS+=("$i")
done

echo "Number of gpus available and gpu ids in use:"
echo $NUM_GPUS
echo "${GPU_IDS[@]}"
MASTER_PORT=$((10000 + RANDOM % 90000))

# Set GPU IDs from command line if provided
if [ $# -gt 0 ]; then
  # Clear default GPU list and add provided GPUs
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
  --nproc_per_node $NUM_GPUS \
  --master_port $MASTER_PORT \
  eval/run_eval.py \
  --output_dir "out/math_seed12" \
  --tasks "math" \
  --gen_lengths 512 \
  --model_base "GSAI-ML/LLaDA-8B-Instruct" \
  --verifier_base "Qwen/Qwen2.5-Math-7B-Instruct" \
  --num_rollouts 4 \
  --fast_dllm_sampling \
  --stitching_confidence 0.90 \
  --do_stitching \
  --kv_cache \
  --seed 12

CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
  --nproc_per_node $NUM_GPUS \
  --master_port $MASTER_PORT \
  eval/run_eval.py \
  --output_dir "out/gsm8k_seed13" \
  --tasks "gsm8k" \
  --gen_lengths 512 \
  --model_base "GSAI-ML/LLaDA-8B-Instruct" \
  --verifier_base "Qwen/Qwen2.5-Math-7B-Instruct" \
  --num_rollouts 4 \
  --fast_dllm_sampling \
  --stitching_confidence 0.90 \
  --do_stitching \
  --kv_cache \
  --seed 13

CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
  --nproc_per_node $NUM_GPUS \
  --master_port $MASTER_PORT \
  eval/run_eval.py \
  --output_dir "out/humaneval_seed14" \
  --tasks "humaneval" "humanevalplus" \
  --gen_lengths 512 \
  --model_base "GSAI-ML/LLaDA-8B-Instruct" \
  --verifier_base "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --num_rollouts 4 \
  --fast_dllm_sampling \
  --stitching_confidence 0.90 \
  --do_stitching \
  --kv_cache \
  --seed 14

CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
  --nproc_per_node $NUM_GPUS \
  --master_port $MASTER_PORT \
  eval/run_eval.py \
  --output_dir "out/mbpp_seed15" \
  --tasks "mbpp" "mbppplus" \
  --gen_lengths 512 \
  --model_base "GSAI-ML/LLaDA-8B-Instruct" \
  --verifier_base "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --num_rollouts 4 \
  --fast_dllm_sampling \
  --stitching_confidence 0.90 \
  --do_stitching \
  --kv_cache \
  --seed 15 \
  --prm_name TIGER-Lab/AceCodeRM-7B