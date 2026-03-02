import torch

@torch.no_grad()
def generate(
    model,
    attention_mask,
    pad_token_id,
    input_ids=None,
    inputs_embeds=None,
    max_length=None
):
    out = model.generate(
        input_ids=input_ids, 
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_length,
        use_cache=True,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )
    prompt_len = input_ids.shape[-1]
    n_steps = out.shape[-1] - prompt_len 
    return out, n_steps
