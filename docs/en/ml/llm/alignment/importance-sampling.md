# 🎲 Importance sampling: learn from a different sampling distribution

Importance sampling estimates an expectation under a target distribution p using samples from a proposal distribution q. It is useful when sampling from p is difficult or when data were collected under an earlier policy.

$$
\mathbb E_{x\sim p}[f(x)]=
\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right].
$$

The identity requires q to be positive wherever p contributes to the expectation. If the proposal never samples an important region, weights cannot recover information that is absent.

## 🧭 A discrete example

Let p assign probabilities `[0.8,0.2]`, q assign `[0.5,0.5]`, and f take values `[1,3]`. The target expectation is $0.8+0.6=1.4$. Under q, weights are `[1.6,0.4]`, giving $0.5\cdot1.6\cdot1+0.5\cdot0.4\cdot3=1.4$.

```python
# runnable-example
p, q, f = [0.8, 0.2], [0.5, 0.5], [1.0, 3.0]
target = sum(a * v for a, v in zip(p, f))
weighted = sum(b * (a / b) * v for a, b, v in zip(p, q, f))
assert abs(target - weighted) < 1e-12
```

This exact finite sum illustrates the identity. A finite Monte Carlo sample has sampling error, and a few very large weights can dominate it.

## 🎯 In policy optimization

A policy likelihood ratio compares the current and data-collecting policies at the same state/action:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

For language models, the state is the prompt plus preceding generated tokens, and the action is the next token. Whole-sequence ratios multiply many token ratios and can have high variance. PPO uses token/action ratios in a surrogate objective; that does not imply an exact correction of every changed state-visitation distribution.

Clipping weights or self-normalizing them can reduce variance but generally introduces bias. Effective sample size, often estimated as $(\sum_iw_i)^2/\sum_iw_i^2$, is a useful weight-degeneracy diagnostic, not a proof that the estimator is accurate.

**Exercise.** If q assigns zero probability to an outcome with positive p and nonzero f, can unlimited samples from q estimate its contribution? No. The support condition fails.

Reference: [PPO](https://arxiv.org/abs/1707.06347). Continue with [PPO's clipping objective](ppo.md).
