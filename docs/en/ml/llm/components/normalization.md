# LayerNorm and RMSNorm

Deep residual networks repeatedly add transformed features. Normalization controls feature scale so optimization does not depend too strongly on drifting activation magnitudes. In a Transformer, normalization usually acts on the hidden features of each token independently.

For a tensor with shape `[batch, sequence, hidden]`, normalizing the last dimension uses one mean/variance per token. “Normalize columns” is ambiguous without stating tensor axes.

## LayerNorm centers and rescales

For a token vector $x\in\mathbb R^d$:

$$
\mu=\frac1d\sum_i x_i,\quad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,\quad
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

The learned scale and bias allow useful feature distributions after normalization. Unlike BatchNorm, LayerNorm does not estimate statistics across examples and normally has no train/evaluation running-statistics switch.

## RMSNorm does not subtract the mean

$$
\operatorname{RMSNorm}(x)_i=
\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_j x_j^2+\epsilon}}.
$$

Removing only LayerNorm's bias does **not** produce RMSNorm. RMSNorm also omits centering. For $x=[1,3]$, LayerNorm before learned affine parameters is approximately $[-1,1]$, while RMSNorm gives $[1/\sqrt5,3/\sqrt5]$. Its mean is not zero. Fewer reductions can improve runtime, although actual speed depends on the kernel and hardware.

```python
# runnable-example
import math
x = [1.0, 3.0]
mean = sum(x) / len(x)
var = sum((v - mean) ** 2 for v in x) / len(x)
ln = [(v - mean) / math.sqrt(var) for v in x]
rms = math.sqrt(sum(v * v for v in x) / len(x))
rn = [v / rms for v in x]
assert ln == [-1.0, 1.0]
assert abs(sum(v * v for v in rn) / 2 - 1) < 1e-12
```

## Placement and numerical care

Post-norm computes $\operatorname{LN}(x+F(x))$; pre-norm computes $x+F(\operatorname{LN}(x))$. Pre-norm preserves an explicit identity route through the residual stream, which often helps optimization of deep networks. Placement affects training dynamics and is part of the architecture, not a cosmetic refactor of trained weights.

A small epsilon prevents division by zero but changes behavior near zero. Mixed-precision implementations often accumulate statistics in higher precision. Check the normalized shape, epsilon, affine parameters, and accumulation type before comparing two implementations.

**Exercise.** Does adding a constant to every feature leave RMSNorm unchanged? No. LayerNorm removes this common shift before scaling; RMSNorm preserves information about it.

References: [Layer Normalization](https://arxiv.org/abs/1607.06450), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). Compare [BatchNorm](../../mle/dropout-batchnorm.md).
