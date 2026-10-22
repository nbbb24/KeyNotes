!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Optimizer

权重初始化不能全部为0

前向传播时，所有神经元输出都一样。反向传播时，它们的梯度也完全一样。

# Optimizer 面试资料

## 1. SGD (Stochastic Gradient Descent)

**基本原理**

SGD 是最基础的优化算法。它通过对每个 mini-batch 的梯度进行估计，更新参数。

公式：

$$
w_{t+1} = w_t - \eta \nabla L(w_t; x_i, y_i)
$$

**优点**

- 实现简单，内存消耗小。
- 在凸优化中有良好理论保证。
- 泛化性能往往比复杂优化器更好。

**缺点**

- 收敛速度慢。
- 对学习率敏感，需要调参。

**适用场景**

- 数据量大，模型较简单。
- 需要良好的泛化能力时。

---

## RMSProp（解决“不同参数尺度不同”）

### 核心思想

> 每个参数用自己的学习率
>

### 更新

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t
$$

### 直觉

- 梯度大 → 学习率自动变小
- 梯度小 → 学习率变大

👉 非常适合 **RNN / 非平稳 loss**

## 2. Adam (Adaptive Moment Estimation)

**基本原理**

结合了 **Momentum** 和 **RMSProp** 思想：

- 一阶动量：梯度的指数滑动平均（类似动量），**平滑以及加速收敛**
- 二阶动量：梯度平方的指数滑动平均，**控制步长避免梯度爆炸消失**

更新规则：

$$
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
 \hat{m_t} = m_t / (1 - \beta_1^t), \quad \hat{v_t} = v_t / (1 - \beta_2^t)
$$

$$
 w_{t+1} = w_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
$$

**优点**

- 收敛快，适合非凸问题。
- 对学习率不敏感，调参相对容易。
- 适合稀疏梯度问题（如 NLP）。

**缺点**

- 有时泛化性能不如 SGD。
- 在某些任务中可能收敛到较差的解。

**适用场景**

- 深度学习任务（CNN, RNN, Transformer）。
- 稀疏特征场景。

---

## 3. AdamW (Adam with Decoupled Weight Decay)

**基本原理**

AdamW 是对 Adam 的改进，把 **L2 正则化 (weight decay)** 从梯度更新中解耦：

$$
 w_{t+1} = w_t - \eta \left( \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} + \lambda w_t \right)
$$

区别：

- Adam 原始版本中 L2 正则化等效于梯度缩放，不是真正的 weight decay。
- AdamW 则是直接在更新时衰减参数，避免过拟合。

**优点**

- 泛化性能更好。
- 更适合大规模预训练模型（如 BERT, GPT）。
- 与学习率调度配合更稳定。

**缺点**

- 计算量与 Adam 相同，仍比 SGD 大。

**适用场景**

- Transformer 等大模型训练。
- 需要正则化效果时。

---

## 总结对比

| Optimizer | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| SGD | 简单，泛化好 | 收敛慢，调参难 | 大数据，泛化任务 |
| Adam | 收敛快，适合稀疏梯度 | 泛化差，可能收敛到坏解 | 深度学习，NLP |
| AdamW | 泛化好，正则化效果佳 | 计算量大 | 大模型 (BERT, GPT) |

---

✅ **面试建议**

- 先讲 **SGD**，作为基准。
- 再讲 **Adam**，说明其改进之处。
- 最后讲 **AdamW**，突出其在大模型中的应用。
