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