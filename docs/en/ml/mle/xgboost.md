# 🚀 XGBoost: regularized trees and second-order optimization

XGBoost is an implementation of gradient-boosted models with explicit regularization and engineering for efficient training. To understand it, first understand [gradient boosting](gradient-boosting.md): every new tree adds a correction to the current score.

## 💡 Why use both gradients and curvature?

At a boosting stage, approximate each observation's loss after adding $f(x_i)$ by

$$
\ell_i(F+f)\approx\ell_i(F)+g_if(x_i)+\frac12h_if(x_i)^2,
$$

where $g_i$ and $h_i$ are first and second derivatives with respect to the current score. The second derivative describes local curvature. This is an approximation around the current model, not a guarantee that every large update improves the exact loss.

For a tree with $T$ leaves and leaf values $w_j$, a common regularizer is $\gamma T+\frac12\lambda\sum_jw_j^2$. Additional L1 regularization is also supported. Leaf-count and leaf-weight penalties discourage unnecessarily complex or extreme corrections.

## 🧮 Derive a leaf value

Let $G_j=\sum_{i\in j}g_i$ and $H_j=\sum_{i\in j}h_i$. Without the L1 term, the approximate optimum for a fixed leaf is

$$
w_j^*=-\frac{G_j}{H_j+\lambda}.
$$

If $G_j=-6$, $H_j=3$, and $\lambda=1$, the unshrunk leaf update is 1.5. With learning rate 0.1, its contribution to the additive score is 0.15. Increasing $\lambda$ shrinks the update even though the gradient direction stays the same.

A candidate split compares the regularized improvement of its children with the parent, subtracting the cost of adding complexity. This explains why a split with a small raw impurity improvement may be rejected.

## 🛠️ Practical controls

| Control | Main effect | Common misunderstanding |
| --- | --- | --- |
| Learning rate and number of rounds | Size and count of corrections | Tuning one while ignoring the other |
| Depth / leaf constraints | Interaction complexity | More depth always helps |
| Minimum child weight | Minimum Hessian-related mass | It always means a raw row count |
| Row and feature sampling | Stochastic regularization and cost | Sampling fixes a leaking split |
| L1/L2 and split penalties | Regularize leaf values and structure | All penalties have the same meaning |

Histogram methods bin continuous features to reduce split-search cost. Missing values can use learned default directions. Exact behavior depends on the selected booster and tree method; keep configuration and library version with experiment results.

## 🚀 Going further

The statement “XGBoost always grows level by level” is too broad. Growth policies and supported options vary by configuration, including loss-guided growth. Compare XGBoost and LightGBM at equal validation and resource budgets rather than using one algorithm slogan as a universal ranking.

Built-in handling of missing values does not make every missingness mechanism safe. Changes in data collection can alter default-branch behavior. Likewise, categorical support does not remove the need to understand category encoding and unseen values.

**Exercise:** if $G_j=0$ and $H_j+\lambda>0$, what is the optimal leaf value under this approximation? **Answer:** zero; there is no first-order incentive to move that leaf's predictions.

## 📚 Sources and further reading

- [XGBoost: introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) derives the regularized objective.
- [XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html) specifies growth policies and parameter meanings.
