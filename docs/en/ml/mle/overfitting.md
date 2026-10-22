# Overfitting, regularization, and trustworthy validation

Overfitting occurs when a fitting procedure captures patterns specific to its training sample that do not reliably help on new data. The practical symptom is often excellent training performance with disappointing held-out performance, but that gap can also reflect distribution shift or leakage in the experimental design.

## Follow the whole selection process

Separate training, validation, and final testing. Training fits parameters; validation selects hyperparameters and checkpoints; a final test estimates the performance of the selected procedure. Repeatedly inspecting test results turns the test set into another validation set.

For repeated users, devices, or documents, split by group. For forecasting, train on the past and evaluate on later observations. A random row split can hide memorization of the same entity or future information.

## Why regularization helps

Regularization adds a preference beyond fitting the observed targets. With parameters $w$ and data loss $L(w)$, common objectives are

$$
L(w)+\lambda\sum_j|w_j|\qquad\text{and}\qquad
L(w)+\lambda\sum_jw_j^2.
$$

These are **sum of absolute values** and **sum of squares**, not absolute value or square of the sum. Opposite-signed coefficients must not cancel the penalty.

L1 can produce sparse solutions because its nonsmooth geometry permits exact zeros. Ordinary small-step gradient descent alone does not guarantee exact sparsity; proximal methods or appropriate solvers handle it explicitly. L2 shrinks coefficients continuously and usually keeps them nonzero. Feature scaling changes the meaning of either penalty.

## Worked example: the cost of a cancellation error

Let $w=[3,-3]$. The L1 penalty is 6 and the squared L2 penalty is 18. But $|3-3|$ and $(3-3)^2$ are both zero, incorrectly treating large opposing coefficients as unregularized.

```python
# runnable-example
w = [3, -3]
assert sum(abs(v) for v in w) == 6
assert sum(v * v for v in w) == 18
assert abs(sum(w)) == 0
```

This small arithmetic check is useful because apparently minor formula errors can completely change an optimization objective.

## Diagnose before adding complexity

| Observation | Possible interpretation | Useful next experiment |
| --- | --- | --- |
| Both train and validation losses are high | Underfitting, poor optimization, or wrong inputs | Fit a tiny subset; inspect targets and gradients |
| Training improves while validation worsens | Overfitting or validation mismatch | Early stopping; stronger regularization; inspect split |
| Validation is implausibly good | Leakage or duplicates | Audit preprocessing, identifiers, and timestamps |
| Offline good, online poor | Shift or metric mismatch | Evaluate recent data and deployment slices |

More data help when they add relevant information. Repeating near-duplicates does not provide the same benefit as collecting new conditions. Augmentation should preserve the target semantics: rotating a digit can change its class.

## Going further

Cross-validation estimates the performance of a training-and-selection procedure. Every learned transform belongs inside the fold. If you use the same folds to select a large number of models and report the best score, selection bias remains; nested cross-validation or a final untouched holdout can address it.

Early stopping regularizes through the training trajectory. Dropout adds training noise. Weight decay modifies updates; for adaptive optimizers it is not generally equivalent to adding an L2 term. See [optimizers](optimizers.md).

**Question:** does a zero train–test gap prove a model is useful? **Answer:** no. Both scores may be poor, the metric may be irrelevant, or both sets may share leakage.

## Sources and further reading

- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) covers grouped and temporal evaluation.
- [scikit-learn: linear-model regularization](https://scikit-learn.org/stable/modules/linear_model.html) describes Ridge and Lasso objectives.
