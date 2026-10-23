# 熵、交叉熵与 KL 散度

这些量回答了相关但不同的问题：一个源的不确定性，预测的成本，以及两个分布在特定方向上的差异。

对于离散分布 $ P $ 和 $ Q $ 在相同的结果上：

$$
H(P)=-\sum_iP_i\log P_i,\qquad
H(P,Q)=-\sum_iP_i\log Q_i,
$$

$$
D_{KL}(P\Vert Q)=\sum_iP_i\log\frac{P_i}{Q_i}
=H(P,Q)-H(P).
$$

自然对数给出纳特（nats），以2为底的对数给出比特（bits）。熵衡量在 $ P $ 下的内在不确定性。交叉熵衡量当观察遵循 $ P $ 时，$ Q $ 分配的期望负对数概率。KL是相对于使用 $ P $ 本身额外的预期编码成本。

## 做一个预测

假设两个结果的真实分布为 $ P=(0.75,0.25) $，模型预测为 $ Q=(0.5,0.5) $。那么 $ H(P)\approx0.5623 $，交叉熵 $\approx0.6931$，KL $\approx0.1308$ 纳特。改进 $ Q $ 向 $ P $ 接近会将交叉熵减少到 $ P $ 的不可约熵，不一定为零。

```python
# runnable-example
import math
p, q = [0.75, 0.25], [0.5, 0.5]
h = -sum(a * math.log(a) for a in p)
ce = -sum(a * math.log(b) for a, b in zip(p, q))
kl = sum(a * math.log(a / b) for a, b in zip(p, q))
assert abs(ce - h - kl) < 1e-12
assert kl > 0
```

对于一个独热训练标签 $ y $，交叉熵仅仅是 $-\log Q_y$。在采样观察的平均值估计总体目标。标签平滑和蒸馏用软分布替换那个独热目标，因此目标熵不再是零。

## 高级工作中需要注意的细节

KL是非负但不对称的，并且不是一个距离度量。如果 $ P_i>0 $ 而 $ Q_i=0 $，前向 KL 是无穷大。具有 $ P_i=0 $ 的项通过连续性贡献零。前向和后向 KL 在不同的分布下加权错误；关于模式覆盖或模式寻求的陈述取决于近似家族和优化设置。

最小化交叉熵仅当 $ P $ 对于优化参数是固定的时，才等价于最小化 $ D_{KL}(P\Vert Q) $。在 [RLHF](../alignment/rlhf.md) 中，策略分布本身会发生变化，因此要明确 KL 方向和哪个分布生成样本。库的 KL 函数可能期望对数概率作为输入，并使用不同的减少方式，这些方式因批次或标记计数而异。

**练习。** 0.4 的交叉熵是否总是比另一个数据集中的 0.6 好？不是。词汇表、任务、分词、目标分布和平均约定必须可比较。

相关：[损失函数](../../mle/loss-functions.md)，[蒸馏](../adaptation/distillation.md)。参考：[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
