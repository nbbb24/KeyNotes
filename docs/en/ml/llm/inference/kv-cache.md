# KV cache: reuse the past during generation

In a causal Transformer, earlier token states do not change when future tokens are appended. Their attention keys and values can therefore be cached. The next step computes projections for the new token and attends to the stored history.

## Prefill versus decode

Prefill processes the prompt and creates its cache. Decode repeatedly adds new tokens. For dense attention, a new token's attention work grows with context length: approximately $O(n)$ at history length $n$ per layer, ignoring fixed dimensions. It is **not constant per token**. Over N generated steps without a fixed window, cumulative attention work is quadratic in N. Caching avoids repeatedly recomputing the full prefix's projections, FFNs, and attention.

## Count the bytes

For standard K/V storage:

$$
M=2BLnH_{KV}d_hs,
$$

where B is batch size, L layers, n cached tokens, $H_{KV}$ key/value heads, $d_h$ head width, and s bytes per element. Use KV heads, not query heads, for grouped-query attention.

For B=1, L=32, n=2048, $H_{KV}=40$, $d_h=128$, and FP16:

$$
M=1{,}342{,}177{,}280\text{ bytes}=1.25\text{ GiB}\approx1.342\text{ GB}.
$$

```python
# runnable-example
memory = 2 * 1 * 32 * 2048 * 40 * 128 * 2
assert memory == 1_342_177_280
assert memory / 2**30 == 1.25
```

FP16 and BF16 both occupy two bytes per element, though their numerical ranges differ. Allocation rounding, block tables, scales for quantized caches, and runtime buffers add overhead. MLA uses a different compressed representation and needs a different calculation.

## Sharing, eviction, and correctness

Beam search branches need logically distinct histories, but common prefix blocks may be shared until divergence. Paged allocation helps manage growing requests. Prefix caching across requests requires compatible model state, token IDs, masks, and positions.

A sliding window bounds storage but changes accessible context unless it is part of the model's intended attention design. Dropping old cache entries arbitrarily can break quality or positional assumptions. Test long-range tasks after any eviction policy change.

**Exercise.** Replacing 40 KV heads with 8 at the same other dimensions reduces this standard cache by what factor? Five. Query-head count alone does not determine cache size.

References: [multi-query attention](https://arxiv.org/abs/1911.02150), [PagedAttention](https://arxiv.org/abs/2309.06180). Continue with [cache layouts](layers-pages.md).
