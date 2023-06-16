Source: https://github.com/arnavdantuluri/long-context-transformers

```sh
python3 -m venv ~/venv/long

source ~/venv/long/bin/activate # alias long

pip3 install -r requirements.txt

python3 qlora.py
```

TODOS:
- [ ] Speedup [Blockwise Parallel Transformer](https://arxiv.org/abs/2305.19370) (BPT) bằng Triton hoặc CUDA
