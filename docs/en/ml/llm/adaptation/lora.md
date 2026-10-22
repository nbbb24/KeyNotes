# LoRA: adapt a model through a low-rank weight update

LoRA freezes a pretrained weight matrix and learns a factorized update. Its motivation is that useful task adaptation can often be represented with fewer degrees of freedom than a full dense update.

For $W\in\mathbb R^{d_{out}\times d_{in}}$:

$$
W'=W+\frac\alpha r BA,\qquad
A\in\mathbb R^{r\times d_{in}},\quad
B\in\mathbb R^{d_{out}\times r}.
$$

The trainable count is $r(d_{in}+d_{out})$. For a $4096\times4096$ matrix and rank 8, LoRA uses 65,536 weights rather than 16,777,216. This is a 256-fold reduction for that matrix's trainable weights, not a 256-fold reduction in total GPU memory.

## Initialization and targets

A common initialization makes one factor random and the other zero, so the initial update is zero. Initializing both factors to zero would prevent useful first-order learning of either factor. Attention Q/V projections are common targets, but K/O and feed-forward projections can also be adapted. The best choice is empirical and model-dependent.

Rank controls maximum update rank, while $\alpha/r$ controls scaling in the original formulation. Rank, scaling, learning rate, dropout, target modules, and training data interact; rank alone is not a quality knob.

## Training and deployment

During training, compute $Wx+(\alpha/r)B(Ax)$. For a compatible ordinary dense layer, merge the learned update into $W$ for inference. Keep an unmerged base copy when multiple adapters must be swapped. Quantized backbones require care: merging and requantizing can change numerical error, and not every serving backend supports every combination.

QLoRA trains adapters on a frozen quantized backbone. It reduces base-weight memory as well, but activations and optimizer states still contribute. LoRA itself is an adaptation parameterization, not a loss: it can be used with SFT or preference objectives.

**Exercise.** Does rank 8 mean the adapted full matrix has rank at most 8? No. Only $BA$ has rank at most 8; $W+BA$ can be full rank.

References: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314). Compare [adapters](adapters.md).
