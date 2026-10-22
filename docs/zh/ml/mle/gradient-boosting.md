!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# GBDT

Boosting

builds multiple **decision trees sequentially**, where each new tree **learns to correct the errors** made by the previous ones.

### Steps

- Train the first tree on the data to make initial predictions.
- Compute **residuals** — the difference between predictions and true labels.
- Train the next tree to predict these residuals.
- Update predictions by adding the new tree’s output (scaled by a learning rate).
- Repeat for many iterations until performance converges.

# FQA

- **How does GBDT optimize its loss?**

    → By adding trees in the direction of the **negative gradient** of the loss function.

- **What’s the role of the learning rate?**

    → It controls how much each tree contributes — smaller values improve generalization but need more trees.

- **How do you prevent overfitting?**

    → Use early stopping, smaller learning rate, fewer/depth-limited trees, and regularization parameters.
