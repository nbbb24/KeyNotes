!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# LightGBM

### Feature

- **Leaf-wise tree growth (best-first):**

Instead of growing trees level by level (like XGBoost), LightGBM grows the **leaf with the largest loss reduction first** , which improves accuracy but risks overfitting on small data.

- **Histogram-based splitting:**

Continuous features are bucketed into discrete bins, **reducing memory** and **speeding up training**.

### Disadvantage

- Can **overfit** small datasets due to aggressive leaf-wise growth.
- Slightly harder to interpret and tune.

### GOSS

Gradient-based One-Side Sampling:

keeps data points with large gradients (hard samples) and randomly drops some easy ones to speed up training.

### EFB

Exclusive Feature Bundling:

combines mutually exclusive sparse features to reduce dimensionality.

**When would you prefer LightGBM over XGBoost?**

For **large-scale, high-dimensional** data where training speed and memory efficiency matter most.
