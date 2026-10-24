# 🖼️ SigLIP: pairwise sigmoid training for image–text alignment

SigLIP keeps the dual-encoder idea of CLIP but changes the loss. Instead of normalizing every image against all text candidates with a softmax, it treats selected image–text pairs as binary matching examples.

## 🎯 From similarity to a pairwise objective

Let $v_i$ and $t_j$ be normalized embeddings and $s_{ij}=a\,v_i^\top t_j+b$ a scaled and shifted similarity. With labels $y_{ij}\in\{-1,+1\}$, a pair contributes

$$
\ell_{ij}=\log(1+e^{-y_{ij}s_{ij}}).
$$

This is binary logistic loss, equivalently binary cross-entropy on a sigmoid. The scale and bias matter, particularly because negative pairs can greatly outnumber positive pairs. Implementations should use a numerically stable logits-based loss.

## ⚖️ A small comparison

For one positive pair with score 2, loss is $\log(1+e^{-2})\approx0.127$. For a negative pair with the same score, loss is $\log(1+e^2)\approx2.127$. The objective therefore rewards positive scores for matches and negative scores for mismatches.

```python
# runnable-example
import math
positive = math.log1p(math.exp(-2.0))
negative = math.log1p(math.exp(2.0))
assert positive < 0.13 and negative > 2.12
```

Unlike CLIP softmax, the pair loss needs no global normalization denominator. This supports flexible pair sampling and distributed computation. The losses are pairwise, but the examples are **not statistically independent during learning**: all pairs share encoder parameters.

## 📌 What the change does not guarantee

A sigmoid objective does not automatically fix false negatives. If a valid image–text match is labeled negative, the model still receives contradictory supervision. Batch size, negative sampling, data quality, architecture, and compute budget all affect results. Improvements reported for a particular model are not a theorem that sigmoid is always more stable, multilingual, or accurate.

SigLIP features can support retrieval, zero-shot classification, or a multimodal generator's vision encoder. A generator still needs an interface to language and appropriate training.

**Exercise.** If you double sampled negatives without changing weighting, is the optimization problem unchanged? No. The relative positive/negative contribution changes unless the reduction or weights compensate.

Reference: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). Review [CLIP](clip.md).
