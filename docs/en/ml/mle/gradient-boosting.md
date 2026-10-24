# 📈 Gradient boosting: learn an additive correction

Gradient boosting builds a predictor in stages. Instead of fitting every tree independently to the original labels, it asks: **what additional function would reduce the loss of the current model?** This connects an ensemble of trees to gradient-based optimization.

## 🎯 The training loop

Start with a constant prediction $F_0$ that minimizes the chosen loss. At stage $m$, compute pseudo-residuals

$$
r_{im}=-\left.\frac{\partial\ell(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}.
$$

Fit a small tree to these targets, determine its leaf outputs according to the objective, and update $F_m=F_{m-1}+\eta h_m$. The shrinkage factor $\eta$ controls each stage's contribution.

For squared error $\frac12(y-F)^2$, the negative gradient is $y-F$, the familiar residual. For logistic loss, the gradient is related to $y-p$ while the additive score is a logit. “Fit raw residuals” is therefore a special case, not the complete definition.

## 🧩 Worked example

Targets are `[1, 3, 5]`. The best constant squared-error prediction is 3, giving residuals `[-2, 0, 2]`. Suppose a shallow tree predicts corrections `[-1, -1, 2]`. With learning rate 0.5, the updated predictions are `[2.5, 2.5, 4]`. Squared error falls from 8 to 3.5.

```python
# runnable-example
y = [1, 3, 5]
old = [3, 3, 3]
correction = [-1, -1, 2]
new = [a + 0.5 * b for a, b in zip(old, correction)]
sse = lambda pred: sum((a - b) ** 2 for a, b in zip(y, pred))
assert sse(old) == 8
assert sse(new) == 3.5
```

The correction is imperfect, but useful. Boosting does not require each weak learner to solve the entire original problem.

```mermaid
flowchart LR
    A[Current predictions] --> B[Compute loss gradients]
    B --> C[Fit a correction tree]
    C --> D[Scale and add correction]
    D --> A
```

## 📌 Regularization is part of the method

Tune learning rate together with stage count. A smaller rate usually requires more stages; it does not guarantee better generalization if the budget is insufficient or the objective is mismatched. Shallow trees constrain interactions; leaf-size limits, row/feature subsampling, and penalties add other controls.

Use validation-based early stopping and record the selected number of stages. The test set must not decide when to stop. Noisy labels and outliers can attract repeated corrections, especially under losses that emphasize large errors.

## 🚀 Going further

XGBoost uses a regularized second-order approximation to score tree structures and leaf values. LightGBM uses efficient histogram and tree-growth techniques. Both belong to a broader gradient-boosting family; their implementation choices do not redefine the fundamental additive procedure.

Training stages are sequential, but calculations within a stage can be parallelized. Serving cost scales with the number and structure of trees; benchmark the actual ensemble, not just one weak learner.

**Question:** why is a boosting tree's output not always a target value? **Answer:** it is a contribution to the current additive score, such as a logit correction in classification.

## 📚 Sources and further reading

- [scikit-learn: gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) covers losses and regularization.
- [Friedman's gradient-boosting paper](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) develops the functional-gradient view.
