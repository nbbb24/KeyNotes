!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Random Forest

1. **Ensemble (Bagging)**: reduce overfitting and learn more generalized predictions
2. **Reduced Variance**: averaging trees, make it more stable
3. **Robust to outliers**: multiple trees and bootstrapping

### Each Tree

Each tree is trained on a **bootstrap sample** (random sampling with replacement) and uses a **random subset of features** when splitting nodes.

### Classification VS Regression

During inference:

- For **classification**, the model takes the **majority vote** across trees.
- For **regression**, it averages their predictions.
