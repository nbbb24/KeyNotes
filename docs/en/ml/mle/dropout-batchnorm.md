# 💧 Dropout, BatchNorm, and LayerNorm

These methods often appear together, but they modify different parts of learning. Dropout injects noise into activations during training. BatchNorm normalizes using statistics across examples and, for convolutional inputs, spatial positions. LayerNorm normalizes selected feature dimensions within an example. None is a universal replacement for the others.

## 🧭 Dropout: preserve the expectation, change the computation

With drop probability $p$, inverted dropout uses a mask $m\sim\operatorname{Bernoulli}(1-p)$:

$$
\tilde x=\frac{mx}{1-p}.
$$

During ordinary evaluation, the operation becomes the identity. This keeps $E[\tilde x]=x$ during training. It discourages brittle reliance on particular intermediate activations, but describing it as an exact ensemble average over all subnetworks is too strong: the inference computation is an approximation.

For $x=8$ and $p=0.25$, an activation is zero with probability 0.25 and $8/0.75$ otherwise. Its expected value is still eight. Do not apply a second test-time rescaling when the implementation already uses inverted dropout.

## 🧩 BatchNorm: specify the axes and mode

For the selected normalization set,

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat x+\beta.
$$

Training commonly uses batch statistics and updates running estimates. Evaluation normally uses those running estimates, depending on configuration. The learned scale and shift let the model change the normalized representation.

For a convolutional tensor $[N,C,H,W]$, standard BatchNorm computes statistics per channel over $N,H,W$. For a dense batch $[N,D]$, it typically normalizes each feature over $N$. “Normalize over the batch” is therefore a shorthand, not a complete axis specification.

BatchNorm **does not require Gaussian inputs**. Reducing internal covariate shift was the original motivation, but it is not a complete explanation of its benefits; subsequent work emphasizes effects on the optimization landscape. Avoid presenting an unsettled mechanism as a single proven cause.

## 🏗️ LayerNorm: normalize each token's features

For a token vector $x\in\mathbb R^d$, LayerNorm computes mean and variance over the chosen $d$ features. It does not depend on which other examples share the batch. This is convenient for variable-length sequences and autoregressive inference. The normalization calculation is the same in training and evaluation, although other parts of the network, such as dropout, may still change behavior.

| Property | Dropout | BatchNorm | LayerNorm |
| --- | --- | --- | --- |
| Main operation | Random masking | Batch/spatial statistics | Per-example feature statistics |
| Typical train/eval difference | Masking off at evaluation | Running versus batch statistics | Usually none in the normalization itself |
| Sensitive to batch composition | Not through a shared mean | Yes | Not through other examples |

## ⚠️ A practical debugging example

A model performs well during training but poorly on validation. First verify the evaluation mode. Leaving dropout active makes outputs stochastic; leaving BatchNorm in training mode can make each prediction depend on the other validation examples. Conversely, freezing running statistics too early can make them unrepresentative.

A batch containing one example is not automatically impossible for convolutional BatchNorm if spatial dimensions provide multiple values, but the statistics may still be poor. Gradient accumulation does not automatically create the same BatchNorm statistics as a single large physical batch.

## 🚀 Going further

`Conv → BatchNorm → ReLU` is a common pattern, not a theorem. Pre-activation residual networks use other orders. Combining strong dropout before BatchNorm can shift the distributions seen by running statistics; evaluate the specific placement. For tiny or highly correlated batches, consider whether LayerNorm or GroupNorm fits the architecture.

**Question:** does normalization remove the need for a learning-rate schedule? **Answer:** no. Better-conditioned optimization still depends on step size, data, and model scale.

## 📚 Sources and further reading

- [BatchNorm paper](https://arxiv.org/abs/1502.03167), [optimization analysis](https://arxiv.org/abs/1805.11604), and [LayerNorm paper](https://arxiv.org/abs/1607.06450).
- [PyTorch: Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) documents inverted scaling and evaluation behavior.
