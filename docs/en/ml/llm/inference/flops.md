# 📊 FLOPs, memory traffic, and latency

FLOPs count floating-point operations. FLOP/s measures a rate. Neither alone is elapsed time: a workload can be limited by arithmetic, memory bandwidth, communication, or software overhead.

## 🧭 Start with a matrix multiplication

Multiplying an $m\times k$ matrix by a $k\times n$ matrix uses approximately $2mkn$ FLOPs when a multiply and an addition count separately. Some specifications count a fused multiply-add differently, so state the convention.

For a linear layer processing B sequences of length S, with input/output widths $d_{in},d_{out}$:

$$
F\approx2BSd_{in}d_{out}.
$$

A $1\times4096$ vector multiplied by a $4096\times4096$ weight requires about 33.6 million FLOPs. Reading the FP16 weight once requires about 33.6 MB. At batch one, that low arithmetic intensity can make weight bandwidth more important than peak matrix throughput.

## 🧮 Transformer estimates

Dense attention's QK and weighted-value products each scale as $O(S^2d)$ during full-sequence processing. Projection and FFN costs scale approximately linearly with S and quadratically with hidden width. For many dense-model training estimates, $6ND$ is a useful rough count for N parameters and D training tokens, assuming forward/backward matrix operations dominate. It omits important details such as long-context attention, recomputation, embeddings, sparsity, and communication.

For MoE, total parameter count is not the same as parameters activated per token. For inference, separate prefill from cached decode and include KV reads. A smaller FLOP count can still be slower if kernels are inefficient or communication dominates.

## 🧮 A roofline estimate

$$
\text{attainable FLOP/s}\leq
\min(\text{peak compute},\ \text{bandwidth}\times\text{arithmetic intensity}).
$$

Arithmetic intensity is operations per byte moved at the memory level being analyzed. This is an upper-bound model, not an exact runtime predictor. Measure actual latency, tokens/s, memory usage, and synchronization costs.

**Exercise.** If an operation needs 100 GB of memory traffic on a system sustaining 1 TB/s, its memory-time lower bound is 0.1 seconds. Can its runtime be less solely because peak FLOP/s doubles? Not if that traffic remains the bottleneck.

Reference: [Roofline model](https://www.osti.gov/biblio/1164365). Connect to [Infra performance tools](../../../infra/performance-tools.md).
