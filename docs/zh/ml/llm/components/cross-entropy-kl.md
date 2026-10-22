!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Cross-Entropy/ KL Divergence

## Cross-Entropy

$$

H(P, Q) = - \sum_{i} P(i) \log Q(i)

$$

## KL-Divergence

**Entropy** of distribution P:

$$

H(P) = - \sum_{i} P(i) \log P(i)
$$

Now subtract cross_entropy with entropy of P

$$

= - \sum_{i} P(i) \log Q(i) + \sum_{i} P(i) \log P(i)
$$

$$
D_{\mathrm{KL}}(P \parallel Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$
