Source: https://github.com/arnavdantuluri/long-context-transformers

```sh
python3 -m venv ~/venv/long

source ~/venv/long/bin/activate # alias long

pip3 install -r requirements.txt

## Với cuda 12 cần cài torch phù hợp
#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

python3 qlora.py --model_name_or_path bigscience/bloom-560m --bits 4 --per_device_train_batch_size 8

python3 qlora_bpt_attn.py --model_name_or_path bigscience/bloom-560m --bits 4 --per_device_train_batch_size 8
#old_bpt     `0%|        | 2/10000 [04:06<341:09:43, 122.84s/it]`
#new_memeff  `0%|        | 1/10000 [08:24<1401:24:16, 504.56s/it]`
```

- 560m params model mất 13 ngày để train qlora + BPT cho 10k interations => Quá chậm!
- Hiệu quả thu đc chỉ là 2x - 4x longer context length so với Flash Attention

TODOS:
- [ ] Speedup [Blockwise Parallel Transformer](https://arxiv.org/abs/2305.19370) (BPT) bằng Triton hoặc CUDA


- - -


> Flash Attention is still quadratic growth, so 2x length means 4x memory, while BPT is sqrt(N)
> if we wanted to we could simply replaced blocked attention in BPT with flash attention
> it’s worth at least a couple of optimizations for BPT since i’m using simple python loops with no optimizations yet.
> We could get a baseline for time and memory by using torch 2.0 for attention and then maybe jitting the blocked feed forward network
> BPT for now is probably the best option for longer context.

> Is memory_efficient_attention is the same as blockwise attention?

> yup it should be, algorithmically it seems to be the exact same and I also tested the diffs between https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/blockwise_parallel.py and https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/memeff.py and they evaluated to the same.


yup it should be, algorithmically it seems to be the exact same and I also tested the diffs between https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/blockwise_parallel.py and https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/memeff.py and they evaluated to the same.
