# 🏗️ Transformer architecture advances: identify the bottleneck first

Modern Transformer variants modify FFNs, attention state, normalization, and positions. These changes address different resources or learning dynamics. Compare them at matched parameter, compute, data, and context budgets where possible.

## 🏗️ Feed-forward networks

![Transformer architecture advances: identify the bottleneck first, original figure 1](../../../../../assets/notion/040-00.png)
![Transformer architecture advances: identify the bottleneck first, original figure 2](../../../../../assets/notion/040-01.png)

*The first slides compare activations and gated/bilinear FFNs. GELU is $x\Phi(x)$ with the Gaussian cumulative distribution, not $x\phi(x)$ with its density; “Switch” in the activation label should be read as Swish.*

A standard FFN uses two projections around an activation. A gated FFN multiplies two projected branches, one nonlinear, before a down projection. Because that adds a matrix, reduce intermediate width when seeking a matched parameter budget. See [LLaMA's SwiGLU](../../models/llama.md).

## 🧠 Attention heads and cache storage

![Transformer architecture advances: identify the bottleneck first, original figure 3](../../../../../assets/notion/040-02.png)
![Transformer architecture advances: identify the bottleneck first, original figure 4](../../../../../assets/notion/040-03.png)
![Transformer architecture advances: identify the bottleneck first, original figure 5](../../../../../assets/notion/040-04.png)
![Transformer architecture advances: identify the bottleneck first, original figure 6](../../../../../assets/notion/040-05.png)

*The progression is MHA → shared KV groups → one KV head → latent KV compression. Mean-pooling projection weights is an initialization/conversion strategy, not an exact equivalence to the original multi-head model.*

MHA uses separate KV heads for each query head. GQA shares KV heads across query groups. MQA shares one KV head across all query heads. MLA changes the cache representation using a latent state and positional components. Reducing KV heads or latent width can save decode bandwidth and capacity, with quality depending on training and configuration.

For 32 query heads and 8 KV heads, four query heads share each KV head. The standard cache is one quarter the corresponding 32-KV-head cache at equal other dimensions. Attention kernels still compute distinct query-head outputs.

## 📌 Normalization and residual paths

![Transformer architecture advances: identify the bottleneck first, original figure 7](../../../../../assets/notion/040-06.png)
![Transformer architecture advances: identify the bottleneck first, original figure 8](../../../../../assets/notion/040-07.png)

*Pre-norm and post-norm place normalization on different sides of a residual addition. RMSNorm rescales without centering; the plots reflect particular experiments, not universal convergence guarantees.*

## 📍 Position information

![Transformer architecture advances: identify the bottleneck first, original figure 9](../../../../../assets/notion/040-08.png)
![Transformer architecture advances: identify the bottleneck first, original figure 10](../../../../../assets/notion/040-09.png)
![Transformer architecture advances: identify the bottleneck first, original figure 11](../../../../../assets/notion/040-10.png)
![Transformer architecture advances: identify the bottleneck first, original figure 12](../../../../../assets/notion/040-11.png)

*The slides move from added absolute sinusoids to relative score biases and rotary Q/K transformations. These are different mechanisms for injecting order.*

The same model may combine positional choices with long-context adaptation, but extending a position formula does not prove long-context capability. Evaluate retrieval and reasoning at different distances and positions.

**Exercise.** Which change directly reduces stored KV-head count: FlashAttention or GQA? GQA. FlashAttention primarily changes the attention computation's memory traffic; the two optimizations can coexist.

References: [GQA](https://arxiv.org/abs/2305.13245), [GLU variants](https://arxiv.org/abs/2002.05202), [RoFormer](https://arxiv.org/abs/2104.09864). Continue with [evaluation](evaluation.md).
