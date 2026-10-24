# 🌲 Bagging versus boosting: two reasons to combine models

An ensemble combines multiple predictors, but the training relationship between them matters. Bagging fits models on perturbed datasets and aggregates them. Boosting constructs an additive model sequentially, choosing each new component to improve the current objective.

## 🎯 Bagging: make unstable learners more stable

A bootstrap sample draws $n$ observations with replacement from a dataset of size $n$. Train a base learner on each sample, then aggregate predictions. Individual trees may vary greatly, while their average can be more stable.

For $B$ equally variable predictions with variance $\sigma^2$ and pairwise correlation $\rho$, a simplified variance calculation gives

$$
\operatorname{Var}(\bar f)=\sigma^2\left(\rho+\frac{1-\rho}{B}\right).
$$

More models reduce the independent part of the variance, but cannot average away a shared error. This is why diversity matters, and why copying the same fitted model a hundred times does not help.

## 🧩 Boosting: fit the next useful correction

An additive model has the form $F_M(x)=F_0(x)+\sum_{m=1}^M\eta h_m(x)$. Gradient boosting chooses $h_m$ to approximate a useful direction in function space, typically the negative gradient of the loss evaluated at current predictions.

AdaBoost emphasizes misclassified examples through weight updates. That description should not be used as the definition of all boosting: squared-error gradient boosting fits residuals, and other losses have other gradient targets.

![Bagging versus boosting: two reasons to combine models, original figure 1](../../../assets/notion/081-00.png)

The arrows distinguish independently trained bootstrap models from sequential corrections. “Parallel” describes training dependence; actual runtime still depends on hardware, tree count, and implementation.

## ⚖️ Worked comparison

Three regression trees predict 8, 10, and 12 for the same input. Bagging averages them to 10; none was trained to repair the specific errors of the previous one.

A boosting model might instead start at 8, fit a correction of 4, and add only $0.25\times4=1$, reaching 9. The next tree is trained against the updated model. Its contribution is not a standalone prediction of the final target.

| Question | Bagging | Boosting |
| --- | --- | --- |
| Can base models be trained independently? | Usually yes | Successive stages depend on earlier stages |
| Main intuition | Reduce instability through averaging | Improve an additive objective step by step |
| Common controls | Number of models, sampling, base complexity | Number of stages, learning rate, tree complexity |
| Typical risk | Correlated errors remain | Too many aggressive corrections fit noise |

## 🚀 Going further

Bagging often reduces variance for unstable learners such as deep trees. It need not help a very stable learner much. Boosting can reduce approximation error and also overfit; the outcome depends on data, base learners, and regularization. Neither has an unconditional advantage.

Parallelizing feature and split calculations inside a boosting stage does not remove the dependence between stages. Conversely, prediction latency for a large bagging ensemble may remain substantial even if training was easy to parallelize.

For evaluation, compare complete pipelines under the same split and budget. A thousand deep trees versus ten shallow trees is not a clean test of the ensemble principle.

**Question:** why does random forest add feature randomness on top of bootstrapping? **Answer:** to reduce correlation among trees, making averaging more effective.

Continue with [random forests](random-forests.md) and [gradient boosting](gradient-boosting.md).

## 📚 Sources and further reading

- [scikit-learn: ensembles](https://scikit-learn.org/stable/modules/ensemble.html) distinguishes bagging, forests, AdaBoost, and gradient boosting.
- [Friedman: Greedy Function Approximation](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) develops gradient boosting.
