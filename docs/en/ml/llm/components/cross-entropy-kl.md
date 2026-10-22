# Entropy, cross-entropy, and KL divergence

These quantities answer related but distinct questions: how uncertain a source is, how costly a prediction is, and how much two distributions differ in a particular direction.

For discrete distributions $P$ and $Q$ on the same outcomes:

$$
H(P)=-\sum_iP_i\log P_i,\qquad
H(P,Q)=-\sum_iP_i\log Q_i,
$$

$$
D_{KL}(P\Vert Q)=\sum_iP_i\log\frac{P_i}{Q_i}
=H(P,Q)-H(P).
$$

Natural logarithms give nats; base-2 logarithms give bits. Entropy measures intrinsic uncertainty under $P$. Cross-entropy measures the expected negative log probability assigned by $Q$ when observations follow $P$. KL is the extra expected coding cost relative to using $P$ itself.

## Work through a prediction

Let the true distribution over two outcomes be $P=(0.75,0.25)$ and the model predict $Q=(0.5,0.5)$. Then $H(P)\approx0.5623$, cross-entropy $\approx0.6931$, and KL $\approx0.1308$ nats. Improving $Q$ toward $P$ reduces cross-entropy to the irreducible entropy of $P$, not necessarily to zero.

```python
# runnable-example
import math
p, q = [0.75, 0.25], [0.5, 0.5]
h = -sum(a * math.log(a) for a in p)
ce = -sum(a * math.log(b) for a, b in zip(p, q))
kl = sum(a * math.log(a / b) for a, b in zip(p, q))
assert abs(ce - h - kl) < 1e-12
assert kl > 0
```

For a one-hot training label $y$, cross-entropy is simply $-\log Q_y$. Averaging it across sampled observations estimates the population objective. Label smoothing and distillation replace that one-hot target with a soft distribution, so target entropy is no longer zero.

## Details that matter in advanced work

KL is nonnegative but asymmetric and is not a distance metric. If $P_i>0$ while $Q_i=0$, forward KL is infinite. Terms with $P_i=0$ contribute zero by continuity. Forward and reverse KL weight errors under different distributions; statements about mode covering or mode seeking depend on the approximation family and optimization setting.

Minimizing cross-entropy is equivalent to minimizing $D_{KL}(P\Vert Q)$ only when $P$ is fixed with respect to the optimized parameters. In [RLHF](../alignment/rlhf.md), the policy distribution itself changes, so be explicit about KL direction and which distribution generates samples. Library KL functions may expect log probabilities as inputs and use reductions that differ by batch or token count.

**Exercise.** Is a cross-entropy of 0.4 always better than 0.6 from another dataset? No. Vocabulary, task, tokenization, target distribution, and averaging convention must be comparable.

Related: [loss functions](../../mle/loss-functions.md), [distillation](../adaptation/distillation.md). Reference: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
