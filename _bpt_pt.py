import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from functools import partial
from einops import rearrange



def blockwise_compute_ffn(cell, inputs, chunk_size):
    return cell(inputs)
    inputs = torch.split(inputs, chunk_size, dim=-2)
    outputs = [ cell(inputs[i]) for i in range(len(inputs)) ]
    return torch.concat(outputs, dim=-2)



## memory efficient attention
def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index = qk_start_indices
    q_chunk_size, k_chunk_size, device = q.shape[-2], k.shape[-2], q.device

    # print(">>>", qk_start_indices) # DEBUG
    # >>> (  0, 0) >>> (  0, 512) >>> (  0, 1024) >>> (  0, 1536)
    # >>> (512, 0) >>> (512, 512) >>> (512, 1024) >>> (512, 1536)
    # => q, k được chia thành các block 512 x 512
    # print(">>>", q_chunk_size, k_chunk_size) # DEBUG
    # >>> 512, 512

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if attn_bias_chunk is not None:
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if mask is not None:
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device)
        causal_mask = causal_mask.triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()
    exp_weight = F.dropout(exp_weight, p = dropout)
    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')



def memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8,
    dropout = 0.,
    training = False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # chunk all the inputs
    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)

    if mask is not None:    mask_chunks = mask.split(k_bucket_size, dim = -1)
    else:                   mask_chunks = ((None,) * len(k_chunks))

    if attn_bias is not None:
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # loop through all chunks and accumulate
    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            if attn_bias is None: attn_bias_chunk = None
            else:                 attn_bias_chunk = attn_bias_chunks[q_index][k_index]

            if q.requires_grad or k.requires_grad or v.requires_grad:
                # https://pytorch.org/docs/stable/checkpoint.html
                exp_weight_chunk, weighted_value_chunk, weight_max_chunk = checkpoint(summarize_qkv_chunk,
                    q_chunk, k_chunk, v_chunk,
                    mask_chunk, attn_bias_chunk, causal,
                    (q_start_index, k_start_index),
                    dropout if training else 0.
                )
            else:
                exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_chunk(
                    q_chunk, k_chunk, v_chunk,
                    mask_chunk, attn_bias_chunk, causal,
                    (q_start_index, k_start_index),
                    dropout if training else 0.
                )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim = -1)

        weighted_values = torch.stack(weighted_values, dim = -1)
        exp_weights = torch.stack(exp_weights, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)



if __name__ == "__main__":
    ## regular attention
    def attention(
        q, k, v,
        mask = None,
        causal = False,
        attn_bias = None,
        **kwargs
    ):
        scale = q.shape[-1] ** -0.5
        q = q * scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if attn_bias is not None:
            sim = sim + attn_bias

        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


    from transformers.activations import ACT2FN
    class GPTNeoXMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_h_to_4h = nn.Linear(512, 2048)
            self.dense_4h_to_h = nn.Linear(2048, 512)
            self.act = ACT2FN["gelu"]

        def forward(self, hidden_states):
            hidden_states = self.dense_h_to_4h(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.dense_4h_to_h(hidden_states)
            return hidden_states

    # Blocked mem stuff
    k = torch.rand(2, 16, 2048, 128)
    v = torch.rand(2, 16, 2048, 128)
    bias = torch.rand(2, 1, 512, 2048) # bias có shape x y i j, với x <= b, y <= h
    # weight =  einsum('b h i d, b h j d -> b h i j', q, k)
    # weight = weight + bias

    ## Test attn
    for bucket_size in [512, 256, 128, 64]:
        q = torch.rand(2, 16, 512, 128)
        q.requires_grad = True
        y_attn = attention(q, k, v, attn_bias=bias)

        q1 = q.detach().clone(); q1.requires_grad = True
        y_pt_mem = memory_efficient_attention(q1, k, v, 
            attn_bias=bias, q_bucket_size=bucket_size, k_bucket_size=bucket_size)

        # Test forward
        result = (y_attn - y_pt_mem).sum()
        print(f"bucket_size {bucket_size}, {result}")
        assert abs(result) < 0.01

        # Test backward
        result.backward()
        delta = (q.grad + q1.grad).sum()
        if not abs(delta) < 0.015:
            print(q.grad[0][0], "\n", q1.grad[0][0], delta)
            assert False

    # Test forward no-checkpoint
    q = torch.rand(2, 16, 512, 128)
    y_attn = attention(q, k, v, attn_bias=bias)
    bucket_size = 32
    y_pt_mem = memory_efficient_attention(q, k, v, 
        attn_bias=bias, q_bucket_size=bucket_size, k_bucket_size=bucket_size)
    result = (y_attn - y_pt_mem).sum()
    print(f"bucket_size {bucket_size}, {result}")
    assert abs(result) < 0.01


    ## Test ffn
    x = torch.rand(2, 256, 512); x.requires_grad = True # đầu vào x
    fnn = GPTNeoXMLP()
    y_pt_ffn = blockwise_compute_ffn(fnn, x, 256)
    x1 = x.detach().clone(); x1.requires_grad = True
    y_fnn = fnn(x1)

    # Test forward
    result = (y_fnn - y_pt_ffn).sum()
    print(result)
    assert abs(result) < 0.01

    # Test backward
    result.backward()
    delta = (x.grad + x1.grad).sum()
    # print(x.grad, "\n", x1.grad, delta)
    assert abs(delta) < 0.01
