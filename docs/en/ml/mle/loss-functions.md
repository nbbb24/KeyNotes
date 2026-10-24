# 🎯 Loss functions: choose the error signal you want to optimize

A loss assigns a cost to a prediction and provides the signal used to update parameters. It should match the target type and the consequences of errors. A reporting metric can differ from the training loss because it may be discontinuous, expensive, or only meaningful over a whole dataset.

## 🧮 Regression: what counts as a large mistake?

For residual $r=y-\hat y$, squared error grows as $r^2$ and absolute error as $|r|$. Minimizing expected squared error targets the conditional mean; minimizing expected absolute error targets a conditional median. They can produce different optimal predictions even with unlimited model capacity.

Consider targets `[1, 2, 30]` with no input features. The squared-error optimum is the mean, 11. The absolute-error optimum is the median, 2. That difference follows from the objective, not an optimizer malfunction.

Huber loss is quadratic near zero and linear for large residuals:

$$
L_\delta(r)=\begin{cases}\frac12r^2,&|r|\leq\delta\\
\delta(|r|-\frac12\delta),&|r|>\delta.\end{cases}
$$

The threshold has the units of the residual. Frameworks may define Smooth L1 as a rescaled Huber loss; in PyTorch, the relationship depends on `beta`. They coincide numerically at the common unit-threshold setting, not for every configuration.

## 🧩 Classification: likelihood and logits

For a binary label $y$ and positive-class probability $p$, binary cross-entropy is

$$
-y\log p-(1-y)\log(1-p).
$$

For a mutually exclusive class $y$, multiclass cross-entropy is $-\log p_y$. Soft target distributions generalize it to $-\sum_kq_k\log p_k$. Libraries commonly combine these losses with the sigmoid or log-softmax calculation, accepting **unnormalized logits** for stability.

A true-class probability of 0.8 gives loss $-\log0.8\approx0.223$; a confident wrong probability of 0.01 gives about 4.605. Cross-entropy therefore distinguishes confidence levels that accuracy ignores.

```python
# runnable-example
from math import log
assert abs(-log(0.8) - 0.2231435513) < 1e-9
assert -log(0.01) > 4.6
```

## 🎯 Other objectives and their assumptions

| Loss | Typical use | What to watch |
| --- | --- | --- |
| Hinge $\max(0,1-yf(x))$ | Margin-based classification, $y\in\{-1,1\}$ | Scores are not probabilities |
| Focal $-\alpha_t(1-p_t)^\gamma\log p_t$ | Reduce the contribution of easy classified examples | Label noise can also be “hard”; calibration can change |
| Poisson NLL $\lambda-y\log\lambda$ | Nonnegative counts, up to target-only terms | Positive rate; count-model assumptions |
| L1/L2 penalties | Prefer sparse or small parameters | Scale and optimizer conventions matter |
| Elastic net | Combine L1 and L2 preferences | Tune both penalty strength and mixture |

Here $p_t$ is the probability assigned to the true binary outcome. Class weights and focal modulation change the effective optimization problem; they are not guaranteed improvements for every imbalanced dataset.

## 🎯 Going further: reductions are part of the objective

A sum and a mean differ by a scale factor, which changes effective gradient magnitude. In sequence learning, decide whether to average per token or per example. Otherwise long examples may receive more weight than intended. Padding tokens must be excluded from both the loss numerator and its normalization count.

In multitask models, adding raw losses with different units can let one task dominate. Inspect gradients or use a deliberate weighting scheme. An apparently decreasing total loss can hide deterioration on one component.

**Exercise:** two sequences have 10 and 100 valid tokens. Does averaging every token weight the sequences equally? **Answer:** no. The second contributes ten times as many terms. Average each sequence first if equal sequence weighting is the intended objective.

Continue with [cross-entropy and KL](../llm/components/cross-entropy-kl.md), [metrics](metrics.md), and [optimizers](optimizers.md).

## 📚 Sources and further reading

- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and [SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html).
- [Focal Loss paper](https://arxiv.org/abs/1708.02002) gives the original dense-detection motivation.
