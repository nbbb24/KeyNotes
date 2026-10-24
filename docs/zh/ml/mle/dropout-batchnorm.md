# 💧 Dropout、BatchNorm 与 LayerNorm

它们经常一起出现，却修改学习过程的不同部分。Dropout 在训练时向激活加入噪声。BatchNorm 跨样本计算统计量，卷积输入还包括空间位置。LayerNorm 在单个样本内部选定特征维度上归一化。三者不能普遍互相替代。

## 🧮 Dropout：保持期望，改变计算

丢弃概率为 $p$ 时，Inverted Dropout 使用掩码 $m\sim\operatorname{Bernoulli}(1-p)$：

$$
\tilde x=\frac{mx}{1-p}.
$$

普通评价模式下操作变为恒等映射，训练时保持 $E[\tilde x]=x$。它减少模型对个别中间激活的脆弱依赖，但“精确平均所有子网络”说得过强，推理计算只是近似。

若 $x=8$、$p=0.25$，激活以 0.25 概率变成零，否则为 $8/0.75$，期望仍是 8。实现已经使用 Inverted Dropout 时，测试阶段不要再次缩放。

## 🧩 BatchNorm：说清轴和模式

对选定归一化集合：

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat x+\beta.
$$

训练通常使用批内统计量并更新运行估计；评价通常使用运行估计，具体取决于配置。可学习缩放和平移，让模型调整归一化后的表征。

卷积张量 $[N,C,H,W]$ 的标准 BatchNorm，按通道在 $N,H,W$ 上计算统计量。全连接输入 $[N,D]$ 通常对每个特征沿 $N$ 归一化。因此“沿 Batch 归一化”只是简写，并不是完整轴定义。

BatchNorm **不要求输入服从高斯分布**。减少内部协变量偏移是原始动机，但并不是效果的完整解释；后续研究强调其对优化景观的影响，不应把存在讨论的机制讲成唯一已证明原因。

## 🏗️ LayerNorm：归一化每个词元的特征

对词元向量 $x\in\mathbb R^d$，LayerNorm 在选定的 $d$ 个特征上计算均值和方差，不依赖同批其他样本。这适合变长序列和自回归推理。归一化本身在训练、评价时计算一致，但 Dropout 等其他模块仍可能切换行为。

| 属性 | Dropout | BatchNorm | LayerNorm |
| --- | --- | --- | --- |
| 主要操作 | 随机掩码 | 批内与空间统计量 | 单样本特征统计量 |
| 常见训练/评价差异 | 评价时关闭掩码 | 批内统计与运行统计切换 | 归一化本身通常无差异 |
| 对批次组成是否敏感 | 不通过共享均值产生依赖 | 是 | 不依赖其他样本 |

## 🧩 一个排障例子

模型训练表现好，验证很差，先检查是否切换评价模式。Dropout 未关闭会使输出随机；BatchNorm 仍处于训练模式，会让每个预测依赖其他验证样本。反过来，过早冻结运行统计量也可能让它们不具代表性。

卷积 BatchNorm 的一个批次只有一个样本，不一定无法计算，因为空间维度还可能提供多个值，但统计量可能不可靠。梯度累积也不会自动产生与单个物理大批次相同的 BatchNorm 统计量。

## 🚀 进阶

`Conv → BatchNorm → ReLU` 是常见模式，不是定理，预激活残差网络采用其他顺序。在 BatchNorm 前加入较强 Dropout，可能改变运行统计量所见分布，应实测具体位置。对于极小或高度相关的批次，可以考虑 LayerNorm、GroupNorm 是否适合架构。

**问题：**归一化是否使学习率调度不再必要？**回答：**不是。优化条件改善后，仍依赖步长、数据和模型规模。

资料：[BatchNorm 论文](https://arxiv.org/abs/1502.03167)、[优化分析](https://arxiv.org/abs/1805.11604)、[LayerNorm 论文](https://arxiv.org/abs/1607.06450)和 [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)。
