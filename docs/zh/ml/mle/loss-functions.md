# 损失函数：选择真正要优化的误差

损失为预测分配代价，并产生更新参数的信号。它应匹配目标类型及错误后果。报告指标可以不同于训练损失，因为指标可能不连续、计算昂贵，或者只有在整个数据集上才有意义。

## 回归：什么算严重错误

对残差 $r=y-\hat y$，平方误差按 $r^2$ 增长，绝对误差按 $|r|$ 增长。最小化期望平方误差得到条件均值，最小化期望绝对误差得到条件中位数。即便模型容量无限，最优预测也可能不同。

假设没有输入特征，目标为 `[1, 2, 30]`。平方误差最优常数是均值 11，绝对误差最优常数是中位数 2。这是目标函数造成的区别，不是优化器故障。

Huber Loss 在零附近为二次形式，大残差时为线性形式：

$$
L_\delta(r)=\begin{cases}\frac12r^2,&|r|\leq\delta\\
\delta(|r|-\frac12\delta),&|r|>\delta.\end{cases}
$$

阈值与残差具有相同单位。框架可能将 Smooth L1 定义为缩放后的 Huber；PyTorch 中两者关系取决于 `beta`。常用单位阈值下数值一致，不代表所有配置都相同。

## 分类：似然与 Logits

二元标签 $y$、正类概率 $p$ 的交叉熵为：

$$
-y\log p-(1-y)\log(1-p).
$$

互斥类别 $y$ 的多分类交叉熵为 $-\log p_y$，软标签分布将其推广为 $-\sum_kq_k\log p_k$。库通常将损失与 Sigmoid 或 Log-Softmax 合并，接收**未归一化 Logits**，以提高数值稳定性。

真实类别概率为 0.8，损失约为 $-\log0.8=0.223$；若自信地预测错误，真实类别概率只有 0.01，损失约为 4.605。因此交叉熵能区分准确率忽略的置信程度。

```python
# runnable-example
from math import log
assert abs(-log(0.8) - 0.2231435513) < 1e-9
assert -log(0.01) > 4.6
```

## 其他目标及其前提

| 损失 | 典型用途 | 注意点 |
| --- | --- | --- |
| Hinge：$\max(0,1-yf(x))$ | 基于间隔的分类，$y\in\{-1,1\}$ | 分数不是概率 |
| Focal：$-\alpha_t(1-p_t)^\gamma\log p_t$ | 降低易分类样本的贡献 | 噪声标签也可能很“难”，校准会改变 |
| Poisson NLL：$\lambda-y\log\lambda$ | 非负计数，省略仅依赖目标的项 | 率必须为正，并满足计数模型假设 |
| L1/L2 惩罚 | 偏好稀疏或较小参数 | 尺度与优化器约定有影响 |
| Elastic Net | 结合 L1 与 L2 | 同时调节总强度及混合比例 |

$p_t$ 是赋给真实二元结果的概率。类别权重和 Focal 调制会改变实际优化问题，不保证对每个不平衡数据集都有帮助。

## 进阶：归约方式也是目标的一部分

求和与求均值相差比例因子，会改变梯度尺度。序列学习要明确按词元还是按样本平均，否则长样本可能获得超出预期的权重。Padding 必须同时从损失分子和归一化计数中排除。

多任务模型若直接相加单位不同的损失，某任务可能支配训练。应检查梯度或制定合理权重。总损失下降可能掩盖其中一个任务恶化。

**练习：**两条序列分别有 10 和 100 个有效词元，直接对全部词元平均，是否给予序列相等权重？**答案：**不是，第二条贡献十倍项数。如果需要序列等权，应先对每条序列平均。

继续阅读[交叉熵与 KL](../llm/components/cross-entropy-kl.md)、[指标](metrics.md)和[优化器](optimizers.md)。资料：[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)、[Focal Loss 论文](https://arxiv.org/abs/1708.02002)。
