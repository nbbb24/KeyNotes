!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Attention Type

## Flash Attention

**FlashAttention 的核心思想是：不要把完整 attention matrix 存下来。**

它把 Q/K/V 分块搬到 GPU 片上 SRAM 里，一块一块算，同时用 online softmax 维护中间结果。这样可以显著减少 HBM ↔ SRAM 的读写量。原论文把它定义为一种 **IO-aware exact attention algorithm**，也就是说它不是近似 attention，而是结果等价，只是计算顺序和内存访问方式更聪明。

你可以理解成：

```
普通 attention:
QK^T 全量算出来 -> 存到 HBM -> softmax -> 再读写 -> 乘 V

FlashAttention:
分块读 Q/K/V -> 在 SRAM 中局部计算 -> online softmax -> 直接累积输出
```

所以 FlashAttention 主要解决：

```
attention 计算过程中的 HBM 读写太多
```

它不主要解决 KV cache 的管理，也不负责判断两个请求有没有共享前缀。

## PagedAttention

LLM 推理时，每个请求都会产生 KV cache。问题是每个请求长度不同，而且 decode 过程中还会不断增长。

传统方式如果给每个 request 分配一大块连续显存，会有很多浪费：

```
Request A: 需要 300 tokens，但分了 512 tokens 空间
Request B: 需要 700 tokens，但分了 1024 tokens 空间
Request C: 还在增长，需要重新扩容
```

PagedAttention 的想法来自操作系统里的 virtual memory / paging：

**不要要求一个 request 的 KV cache 在物理显存里连续存放，而是拆成固定大小的 block/page。**

逻辑上，一个请求的 KV cache 是连续的：

```
token 0, token 1, token 2, ..., token n
```

但物理显存中可以是不连续的：

```
logical block 0 -> physical block 17
logical block 1 -> physical block 3
logical block 2 -> physical block 25
```

中间靠一个 **block table** 做映射。PagedAttention 论文明确说，它允许 attention keys 和 values 存在非连续的 paged memory 中，从而减少 KV cache 的碎片和重复拷贝。

你可以理解成：

```
PagedAttention = 给 KV cache 做“页表管理”
```

它解决的是：

```
KV cache 显存碎片、动态增长、不同请求长度不一致、并行采样共享 KV 的问题
```

比如 beam search / parallel sampling 里，多个输出共享同一个 prompt，PagedAttention 可以让它们共享 prompt 部分的 physical blocks，并用 reference count / copy-on-write 管理。vLLM 的介绍里也强调了 block table、共享物理 blocks 和 copy-on-write。

## RadixAttention

RadixAttention 是 SGLang 里的核心优化。它解决的是另一类问题：

很多 LLM 请求之间有相同前缀，比如：

```
Request 1:
System prompt + few-shot examples + Question A

Request 2:
System prompt + few-shot examples + Question B

Request 3:
System prompt + few-shot examples + Question C
```

如果每次都重新 prefill `System prompt + few-shot examples`，会浪费大量计算。

RadixAttention 的核心思想是：

> 把已经算过的 token prefix 和对应的 KV cache 存在 radix tree 里。
>
>
> 新请求来了以后，先在 radix tree 里找最长公共前缀，命中的部分直接复用 KV cache，只计算后面的 suffix。
>

SGLang 官方博客说，RadixAttention 会在请求结束后保留 prompt 和 generation result 的 KV cache，并用 radix tree 支持 prefix search、insertion 和 eviction。

简单例子：

```
已有缓存：
root
 └── "You are a helpful assistant. Example 1 ... Example 2 ..."
       ├── "Question A"
       └── "Question B"

新请求：
"You are a helpful assistant. Example 1 ... Example 2 ... Question C"
```

RadixAttention 会发现：

```
shared prefix = "You are a helpful assistant. Example 1 ... Example 2 ..."
```

于是：

```
前缀部分：直接复用 KV cache
Question C：重新 prefill
```

所以它解决的是：

```
跨 request / 跨 generation call 的 prefix KV cache 复用
```

这和 PagedAttention 不一样。PagedAttention 管的是 KV cache 的物理内存布局；RadixAttention 管的是“哪些 token prefix 对应哪些 KV cache 可以复用”。
