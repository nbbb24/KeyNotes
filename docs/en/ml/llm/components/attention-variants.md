# FlashAttention, PagedAttention, and RadixAttention

These names concern different bottlenecks. Understanding the resource each one manages is more useful than memorizing them as interchangeable attention variants.

| Technique | Main resource | Main idea |
| --- | --- | --- |
| FlashAttention | HBM traffic and intermediate tensors | Tile attention and accumulate softmax online |
| PagedAttention | KV-cache allocation | Map logical token blocks to noncontiguous physical blocks |
| RadixAttention | Repeated prefix computation | Find and reuse cached token prefixes using a radix tree |

## FlashAttention: avoid materializing the score matrix

A straightforward implementation writes an $n\times n$ score/probability matrix to high-bandwidth memory. Tiled computation instead keeps small pieces in faster on-chip memory. For a running row maximum $m$, denominator $l$, and unnormalized value sum $o$, a new score tile $s$ updates:

$$
m'=\max(m,\max s),\quad
l'=e^{m-m'}l+\sum_j e^{s_j-m'},
$$

$$
o'=e^{m-m'}o+\sum_j e^{s_j-m'}v_j.
$$

The final output is $o/l$. Rescaling is what makes the online accumulation stable. This computes exact attention mathematically, with normal floating-point differences from operation ordering. It reduces intermediate storage and memory reads/writes; dense attention's quadratic arithmetic remains.

## PagedAttention: separate logical sequence from physical memory

Suppose a request needs 33 tokens and blocks hold 16. It uses three blocks, with 15 unused slots in its last block, rather than reserving an entire maximum-length contiguous allocation. A block table can map logical blocks 0, 1, 2 to physical blocks 17, 3, 25. Attention kernels follow this mapping.

Shared prompt blocks can serve several sampled continuations. Reference counts and copy-on-write protect shared contents when a branch diverges. Paging improves allocation and sharing; it does not mean device memory is unlimited or that all cache automatically moves to host RAM.

## RadixAttention: reuse computation for identical prefixes

Requests with the same system instructions and few-shot examples often repeat expensive prefill. A radix tree indexes token prefixes and their KV tensors. A new request looks up the longest reusable prefix, then computes only its unmatched suffix. Eviction recovers memory when capacity is limited.

Text similarity alone is insufficient: token IDs, model weights, adapter state, positions, and other computation-relevant settings must match. A shared paragraph appearing after a different prefix generally cannot reuse the same causal KV states because its hidden representations depend on earlier context.

## Choosing and measuring

These techniques can coexist. Measure time to first token, generation latency, throughput, cache hit rate, peak memory, and queue delay separately. High prefix reuse may reduce prefill time while decode remains memory-bandwidth bound.

**Exercise.** Two prompts share their final 500 tokens but have different first tokens. Can ordinary prefix caching reuse those 500 tokens? No: they are a common suffix, and causal states depend on the preceding context.

References: [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention](https://arxiv.org/abs/2309.06180), [SGLang and RadixAttention](https://arxiv.org/abs/2312.07104).
