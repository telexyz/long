Source: https://github.com/arnavdantuluri/long-context-transformers

```sh
python3 -m venv ~/venv/long

source ~/venv/long/bin/activate # alias long

pip3 install -r requirements.txt

## Với cuda 12 cần cài torch phù hợp
#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

## Chạy thử
python3 test_patch.py

## Huấn luyện thật
python3 qlora.py --model_name_or_path bigscience/bloom-560m --bits 4 --per_device_train_batch_size 8
#old_bpt     `0%|        | 2/10000 [04:06< 341:09:43, 122.84s/it]`
#new_memeff  `0%|        | 1/10000 [08:24<1401:24:16, 504.56s/it]`
```

Thử nghiệm với nhiều chunk size khác nhau.
```sh
# QUERY_CHUNK_SIZE=512
# KEY_CHUNK_SIZE=1024
# 0%|                                                                             | 1/10000 [09:10<1528:05:16, 550.17s/it]

# QUERY_CHUNK_SIZE=512
# KEY_CHUNK_SIZE=512
# 0%|                                                                             | 1/10000 [08:17<1381:39:29, 497.45s/it]

# QUERY_CHUNK_SIZE=2048
# KEY_CHUNK_SIZE=2048
# 0%|                                                                             | 6/10000 [47:14<1293:02:59, 465.78s/it]
```

- 560m params model mất 13 ngày để train qlora + BPT cho 10k interations => Quá chậm!
- Hiệu quả thu đc chỉ là 2x - 4x longer context length so với Flash Attention

TODOS:
- [ ] Speedup [Blockwise Parallel Transformer](https://arxiv.org/abs/2305.19370) (BPT) bằng Triton hoặc CUDA
- [ ] https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html
  - All implementations are enabled by default. Scaled dot product attention attempts to automatically select the most optimal implementation based on the inputs. In order to provide more fine-grained control over what implementation is used, the following functions are provided for enabling and disabling implementations. The context manager is the preferred mechanism:
    - `torch.backends.cuda.enable_flash_sdp(bool)`: Enables or Disables FlashAttention.
    - `torch.backends.cuda.enable_mem_efficient_sdp(bool)`: Enables or Disables Memory-Efficient Attention.

- - -


> Flash Attention is still quadratic growth, so 2x length means 4x memory, while BPT is sqrt(N)
> if we wanted to we could simply replaced blocked attention in BPT with flash attention
> it’s worth at least a couple of optimizations for BPT since i’m using simple python loops with no optimizations yet.
> We could get a baseline for time and memory by using torch 2.0 for attention and then maybe jitting the blocked feed forward network
> BPT for now is probably the best option for longer context.

> Is memory_efficient_attention is the same as blockwise attention?

> yup it should be, algorithmically it seems to be the exact same and I also tested the diffs between https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/blockwise_parallel.py and https://github.com/lhao499/blockwise-parallel-transformer/blob/main/bpt/blocks/memeff.py and they evaluated to the same.

- - -


https://ontocord.slack.com/archives/C053B9FSK0D/p1686794722960569
> Why’s torch2.0 a dependency a bad thing? Is there anything in the infra that’s currently tied to torch < 2.0? Torch2.0 actually does a good job at fusion, and their benchmarking results look pretty decent. Also, PyTorch 2 has nvgraph as its default fusion engine, so lots of the efficiency coming from kernel launching can get removed.
> I’m not sure if CUDA can do any magic to sequential. Sequential is kind of an inherent thing to the network / algorithm itself. Lower-level languages like CUDA / Triton can get memory tiling / fusion better. However, if the bottleneck is purely due to the sequential, then I am not confident how much CUDA / Triton can bring to the table. **Or, is it the case that by rewriting the kernels in CUDA / Triton, we can change its sequential part to parallel?** That part I’m missing. I could be missing the big picture here and any directions are welcome!

> If I'm not mistaken Longformer has a kernel for sliding windows which is more efficient than regular looping, I'm really just hoping for some speed improvements with looping and attention calculation. If we could also get lower memory overhead to use larger chunk sizes that would be an extra bonus. But if this can't work I think it's possible to use any efficient attention mechanism in place of this.. **the real novelty of the paper is blockwise ffn calculation**. I'm not very experienced with CUDA/Triton and how they work and what they can do so I'll leave that decision up to you

> It could be possible to simply replace memory efficient attention with Flash Attention + block the feed forward + QLoRa and get same if not longer results as BPT

- - -

https://ontocord.slack.com/archives/C053B9FSK0D/p1686872087221869?thread_ts=1686794722.960569&cid=C053B9FSK0D

- I will start by profiling the attention part based on https://github.com/arnavdantuluri/long-context-transformers, and get some sense of where the overhead is spent.

- Based on the profiling results, I’m likely first going to try to fix it by auto-fusion through nvGraph w/ jax / torch. Hopefully fusion can make BPT work!

- If nvGraph is not cutting it, I will proceed with surgery at lower-level, Triton / CUDA level (these two are the same abstraction level, Triton is more research-ish), and see if these tricks can work.


> I think this part is likely where it gets the most slowdowns: https://github.com/arnavdantuluri/long-context-transformers/blob/41558f56b5f884dd0c3c33bca8af6770b0b2ff2d/Mem_helpers/BPT/bpt_pt.py#L192.
> A for loop in python incurs way too many overheads around inner kernels (tensors gets transferred, control gets handed over to host land, and host start the next iteration over PCIe). Compilers like JAX also don’t have visibility over pure python for loop, which makes hard for JAX to run optimization.
> In JAX, there’s a device-level loop (I need to search for it a bit… I believe it is something like this: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop). If you write your loop like this, the loop is recognized by JAX as an optimizable, none-python loop, and JAX can do loop-level pipelining and scheduling for you automatically on the device side (instead of pushing everything to host). (edited) 

> haha, yea I pretty much assumed the python loop would cause a lot of memory issues, I think the jax optimization you were talking about is this? https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html but unfortunately nothing yet like this for pytorch so for loops were a necessary evil

> Yeah, for torch you have to do some tricks with jitting compile. I will kick these tricks in and hopefully bring back some numbers for further discussion!
