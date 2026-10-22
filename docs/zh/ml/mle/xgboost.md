!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# XGBoost

- **Regularization (L1 & L2):**

    Adds penalties to leaf weights to prevent overfitting — a major improvement over classic GBDT.

    $$
    \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2
    $$

- **Second-order optimization:**

    Uses both **first and second derivatives** of the loss function (gradients & Hessians) for more accurate updates.

- **Parallelized tree construction:**

    Splits are computed in parallel, making training much faster.
