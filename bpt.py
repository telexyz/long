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



from transformers.models.bloom.modeling_bloom import dropout_add
from typing import Optional, Tuple

class BPTAttentionWrapperWithAlibi(torch.nn.Module):
    def __init__(self, attention, query_chunk_size=512, key_chunk_size=1024 ):
        super().__init__()
        self.attention = attention
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size
        self.dropout_p = 0.0

    def forward(self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        ):
        fused_qkv = self.attention.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self.attention._split_heads(fused_qkv)
        batch_size, q_length, _, _ = query_layer.shape
                
        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.attention.num_heads, q_length, self.attention.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.attention.num_heads, self.attention.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.attention.num_heads, q_length, self.attention.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        reshaped_query_layer = query_layer.reshape(batch_size, self.attention.num_heads, query_layer.shape[1], query_layer.shape[2]).permute(0, 2, 1, 3)
        reshaped_key_layer = key_layer.reshape(batch_size, self.attention.num_heads, key_layer.shape[1], key_layer.shape[2]).permute(0, 3, 1, 2)
        reshaped_value_layer = value_layer.reshape(batch_size, self.attention.num_heads, value_layer.shape[1], value_layer.shape[2]).permute(0, 2, 1, 3)
        offset_key_layer = self.attention.inv_norm_factor * reshaped_key_layer + self.attention.beta * (torch.linalg.pinv(reshaped_query_layer.permute(0,2,1,3).float()) * alibi.view(batch_size, alibi.shape[0]//batch_size, alibi.shape[1], alibi.shape[2])).permute(0, 3, 1, 2).half()

        context_layer = memory_efficient_attention(reshaped_query_layer, offset_key_layer, reshaped_value_layer, q_bucket_size=self.query_chunk_size, k_bucket_size=self.key_chunk_size, dropout=self.dropout_p)
        context_layer = torch.flatten(context_layer, start_dim = 2)
        
        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.attention.pretraining_tp > 1 and self.attention.slow_but_exact:
            slices = self.attention.hidden_size / self.attention.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.attention.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.attention.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.attention.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.attention.hidden_dropout, self.attention.training)

        outputs = (output_tensor, present)
        return outputs


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


    ## Test BPTAttentionWrapperWithAlibi
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    model_path_or_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name).cpu()

    attn0 = model.transformer.h[0].self_attention
    attn1 = BPTAttentionWrapperWithAlibi(attn0)

    print(attn0)
    # >>> torch.Size([1, 519, 1024]) torch.Size([16, 1, 519]) torch.Size([1, 1, 519, 519]) torch.Size([1, 519, 1024])
    # >>> torch.float32 torch.float32 torch.bool torch.float32
    x0 = torch.rand(1, 519, 1024).cpu()#; x.requires_grad = True # đầu vào x
    x1 = x0.detach().clone().cpu()
    alibi = torch.rand(16, 1, 519).cpu()
    attention_mask = torch.rand(1, 1, 519, 519).bool().cpu()
    residual = torch.rand(1, 519, 1024).cpu()
    y0 = attn0(x0, residual=residual, alibi=alibi, attention_mask=attention_mask)
    y1 = attn1(x1, residual=residual, alibi=alibi, attention_mask=attention_mask)
    print(y0[0] ,"\n", y1[0])
    result = (y0[0] - y1[0]).sum()
    print(f"bloom forward {result}")
    assert abs(result) < 0.01