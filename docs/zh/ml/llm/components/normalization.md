# ⚖️ LayerNorm 与 RMSNorm

深度残差网络反复将变换后的特征相加。归一化控制特征的尺度，使得优化不依赖于激活幅度的变化。在Transformer中，归一化通常独立地对每个标记的隐藏特征进行操作。

对于形状为 `[batch, sequence, hidden]` 的张量，对最后一个维度进行归一化时，使用每个标记的一个均值/方差。“归一化列”在没有明确指出张量轴的情况下是模糊的。

## 🏗️ 层归一化的中心和缩放

对于一个标记向量 $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_i x_i,\quad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,\quad
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

学习到的尺度和偏置允许在归一化后有用特征分布。与BatchNorm不同，层归一化不跨示例估计统计量，并且通常没有训练/评估运行统计量开关。

## 🧩 RMS归一化不减去均值

$$
\operatorname{RMSNorm}(x)_i=
\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_j x_j^2+\epsilon}}.
$$

仅移除层归一化的偏置**不会**产生RMS归一化。RMS归一化还省略了中心化。对于 $x=[1,3]$，在学习到的仿射参数之前进行层归一化时，大约为 $[-1,1]$，而RMS归一化给出 $[1/\sqrt5,3/\sqrt5]$。它的均值不为零。较少的减少可以提高运行时间，尽管实际速度取决于内核和硬件。

```python
# runnable-example
import math
x = [1.0, 3.0]
mean = sum(x) / len(x)
var = sum((v - mean) ** 2 for v in x) / len(x)
ln = [(v - mean) / math.sqrt(var) for v in x]
rms = math.sqrt(sum(v * v for v in x) / len(x))
rn = [v / rms for v in x]
assert ln == [-1.0, 1.0]
assert abs(sum(v * v for v in rn) / 2 - 1) < 1e-12
```

## 📍 放置和数值注意

后归一化计算 $\operatorname{LN}(x+F(x))$；前归一化计算 $x+F(\operatorname{LN}(x))$。前归一化保留了残差流中显式的恒等路径，这通常有助于深度网络的优化。放置会影响训练动态，并且是架构的一部分，而不是对已训练权重的美观重构。

一个很小的epsilon可以防止除以零，但会在接近零时改变行为。混合精度实现通常在更高精度下累积统计量。在比较两个实现之前，请检查归一化形状、epsilon、仿射参数和累积类型。

**练习。** 在每个特征上加一个常数是否会使RMS归一化保持不变？不。层归一化在缩放之前移除这个常见的偏移；RMS归一化保留了关于它的信息。

参考文献：[层归一化](https://arxiv.org/abs/1607.06450)，[根均方层归一化](https://arxiv.org/abs/1910.07467)。与[批归一化](../../mle/dropout-batchnorm.md)比较。
