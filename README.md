Source: https://github.com/arnavdantuluri/long-context-transformers

```sh
python3 -m venv ~/venv/long

source ~/venv/long/bin/activate # alias long

pip3 install -r requirements.txt

## Với cuda 12 cần cài torch phù hợp
#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# python3 qlora.py --model_name_or_path EleutherAI/pythia-70m
python3 qlora.py --model_name_or_path bigscience/bloom-560m --bits 4 --per_device_train_batch_size 8
```

- 560m params model mất 13 ngày để train qlora + BPT cho 10k interations => Quá chậm!
- Hiệu quả thu đc chỉ là 2x - 4x longer context length so với Flash Attention

> Flash Attention is still quadratic growth, so 2x length means 4x memory, while BPT is sqrt(N)
> if we wanted to we could simply replaced blocked attention in BPT with flash attention
> it’s worth at least a couple of optimizations for BPT since i’m using simple python loops with no optimizations yet.
> We could get a baseline for time and memory by using torch 2.0 for attention and then maybe jitting the blocked feed forward network
> BPT for now is probably the best option for longer context.

TODOS:
- [ ] Speedup [Blockwise Parallel Transformer](https://arxiv.org/abs/2305.19370) (BPT) bằng Triton hoặc CUDA
