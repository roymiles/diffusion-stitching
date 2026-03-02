# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F

def top_p_filtering(logits, top_p=0.95, min_tokens_to_keep=1):
    # logits: [..., vocab]
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    # mask tokens once cumulative prob exceeds top_p
    sorted_mask = cumprobs > top_p
    # shift so we always keep the first token that crosses top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # optionally keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_mask[..., :min_tokens_to_keep] = False

    # scatter mask back to original vocab positions
    mask = torch.zeros_like(sorted_mask).scatter(-1, sorted_idx, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))

def sample_gumbel_top_p(logits, temperature=1.0, top_p=0.95, eps=1e-20):
    # 1) temperature
    logits = logits / temperature

    # 2) top-p filter (based on the temperature-scaled distribution)
    logits = top_p_filtering(logits, top_p=top_p)

    # 3) gumbel-max
    u = torch.rand_like(logits).clamp_(eps, 1 - eps)
    g = -torch.log(-torch.log(u))
    return torch.argmax(logits + g, dim=-1)

def create_blockwise_causal_mask(seq_len, block_size=32, device=None, dtype=torch.bool):
    num_blocks = (seq_len + block_size - 1) // block_size
    block_mask = torch.triu(torch.ones((num_blocks, num_blocks), dtype=dtype), diagonal=1)
    mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    mask = mask[:seq_len, :seq_len]
    return mask.to(device) if device else mask

def create_causal_mask(seq_len, device=None, dtype=torch.bool, diagonal=1):
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=dtype), diagonal=diagonal)
    return mask.to(device) if device else mask

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def generate(
    model, prompt,
    steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking='low_confidence', mask_id=126336, confidence=None, 
    factor=None):
    '''
    Args:
        model: Masked diffusion language model.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''

    # sanity check
    n_emb = model.get_input_embeddings().num_embeddings
    xmin = int(prompt.min().item())
    xmax = int(prompt.max().item())
    assert 0 <= xmin, xmin
    assert xmax < n_emb, (xmax, n_emb)
    assert mask_id < n_emb

    # fastdllm does not support multi-batch, so we have to change a few things
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0

        while True:
            nfe += 1
            mask_index = (x == mask_id)

            out = model(x, output_hidden_states=True)
            
            # llada 1.0, 1.5
            logits = out.logits
            z = out.hidden_states[-1]

            
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0

            x0, transfer_index = get_transfer_index(
                logits,
                temperature, 
                remasking, 
                mask_index, 
                x, 
                num_transfer_tokens[:, i] if confidence is None else None, confidence
            )

            # update at transfer index
            x[transfer_index] = x0[transfer_index]
            i += 1

            # if no more masked tokens
            # i.e. there are no more tokens that are masked
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break

    return x, nfe

@torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, confidence=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if confidence is None else None, confidence)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                    x[:, current_block_start:], num_transfer_tokens[:, i] if confidence is None else None, confidence)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    x0 = sample_gumbel_top_p(logits, temperature=temperature, top_p=0.90)  # 0.95 before
 
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        
    elif remasking == 'auto_regressive':
        # only keep probability of unmasking as the first token 
        idx = mask_index.nonzero(as_tuple=False)[0]
        x0_p = torch.zeros((x0.shape[0], x0.shape[1]), dtype=torch.bool, device=x0.device)
        x0_p[idx[0], idx[1]] = True

    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index