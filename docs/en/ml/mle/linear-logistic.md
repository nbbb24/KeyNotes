# 📉 Linear and logistic regression: similar scores, different targets

Both models start with a linear score $z=w^Tx+b$. Their difference is what that score means and which objective estimates it. Linear regression predicts a numerical response directly; logistic regression models the probability of a binary outcome through a nonlinear link.

## 🧭 Linear regression

For a continuous target, the simplest model is $\hat y=w^Tx+b$. Least squares minimizes the sum of squared residuals. This objective emphasizes large errors and, under a Gaussian noise model with constant variance, corresponds to maximum likelihood.

A coefficient describes how the prediction changes with one feature while the other encoded features stay fixed. That is a model interpretation, not automatically a causal effect. Correlated inputs can make individual coefficients unstable even when predictions are stable.

## 🧩 Logistic regression

For $y\in\{0,1\}$, logistic regression applies the sigmoid:

$$
p(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The log-odds satisfy $\log(p/(1-p))=z$. Thus a one-unit feature increase multiplies modeled odds by $e^{w_j}$, holding other inputs fixed. It does not increase probability by a constant amount.

Binary cross-entropy follows from the Bernoulli likelihood:

$$
\ell(y,z)=\log(1+e^z)-yz.
$$

Use a numerically stable logits-based implementation rather than computing `log(sigmoid(z))` directly for extreme scores.

## 🧩 Worked example

Suppose $z=-2+0.5x$, with $x$ measured in a fixed unit. At $x=2$, $z=-1$ and $p\approx0.269$. At $x=4$, $z=0$ and $p=0.5$. At $x=6$, $z=1$ and $p\approx0.731$. The score changes linearly, but the probability curve flattens near zero and one.

```python
# runnable-example
from math import exp
sigmoid = lambda z: 1 / (1 + exp(-z))
assert abs(sigmoid(-1) - 0.2689414214) < 1e-9
assert sigmoid(0) == 0.5
assert abs(sigmoid(1) - 0.7310585786) < 1e-9
```

With threshold 0.5, the boundary is $w^Tx+b=0$, a hyperplane. Nonlinear feature transformations can create nonlinear boundaries in the original input coordinates while keeping the model linear in its parameters.

## 🚀 Going further

Perfect separation can drive unregularized logistic-regression coefficients toward infinity; regularization is more than a cosmetic preference. Standardize features when comparing coefficient magnitudes or applying a uniform penalty, fitting the scaler only on training data.

For multiclass prediction, softmax regression models mutually exclusive classes jointly. Independent sigmoids instead suit multilabel tasks where several labels may be true. These are different target assumptions, not interchangeable activation choices.

A probability threshold should reflect application costs and validation results. Logistic regression does not guarantee calibrated probabilities under misspecification, distribution shift, or altered class sampling.

## ✅ Check your understanding

**Question:** why not fit binary labels with ordinary least squares? **Answer:** it can be a baseline, but predictions need not lie in $[0,1]$, and its error assumptions differ from a Bernoulli model. The probabilistic interpretation is weaker.

Continue with [activation functions](activation-functions.md), [loss functions](loss-functions.md), and [regularization](overfitting.md).

## 📚 Sources and further reading

- [scikit-learn: linear models](https://scikit-learn.org/stable/modules/linear_model.html) covers least squares, logistic regression, and regularization conventions.
- [PyTorch: binary cross-entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) documents the stable combined operation.
