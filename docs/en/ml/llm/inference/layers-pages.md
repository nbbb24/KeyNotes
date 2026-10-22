# KV-cache layouts: layers, pages, and physical buffers

A logical cache is indexed by request, layer, token, head, and feature. Physical layout determines which values are contiguous in memory. Computation and cache transfer may prefer different layouts, so a shape alone is not a performance explanation.

## Three illustrative orders

Assume row-major contiguous storage and a separate leading K/V axis:

| Layout | Example axes | Natural grouping |
| --- | --- | --- |
| Layer-first | `[KV, layer, page, token_in_page, head, dim]` | One layer's token blocks |
| Token-first | `[KV, page, token_in_page, layer, head, dim]` | One token across layers within a page |
| Page then layer | `[KV, page, layer, token_in_page, head, dim]` | One page's per-layer blocks |

For two layers and four tokens per page, page-then-layer stores `L0:t0,t1,t2,t3` followed by `L1:t0,t1,t2,t3` inside each K/V page region. Token-first interleaves `t0:L0,L1`, then `t1:L0,L1`, and so on. These contain the same logical values but have different strides.

A layer-specific attention kernel often benefits from layer-oriented access. Offloading a complete prefix block across layers may favor page-oriented grouping. K and V may still be separate buffers, so “one page” need not mean one contiguous transfer containing everything.

## MLA uses different stored components

An MLA cache can store a compressed latent state and a separate rotary-key component rather than full per-head K/V tensors. For an illustrative latent width 512 and rotary width 64, two buffers might have shapes `[pages, layers, page_size, 1, 512]` and `[pages, layers, page_size, 1, 64]`. Names such as `k_buffer` and `v_buffer` in an implementation do not prove that their contents are ordinary attention keys and values.

Terms such as `page_first_direct` or `page_first_kv_split` are framework/version-specific identifiers. Verify the actual strides, data type, allocation, and kernel contract in the version being used. A system's “L3 cache tier” may mean remote/storage-backed KV cache, not the CPU's hardware L3 cache.

## Measure the complete path

Include layout-conversion kernels, gather/scatter operations, DMA granularity, alignment, and synchronization. A faster storage transfer can lose its benefit if repacking before every attention call is expensive. Use representative prefix lengths and reuse rates.

**Exercise.** Does permuting a tensor's logical dimensions always move its bytes? No. A view may only change strides; a contiguous conversion performs the actual data movement.

Reference: [SGLang hierarchical cache design](https://docs.sglang.ai/advanced_features/hicache_design.html). Review [KV-cache size](kv-cache.md) and [data movement](../../../infra/data-movement.md).
