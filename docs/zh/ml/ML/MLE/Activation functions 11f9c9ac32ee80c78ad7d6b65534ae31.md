# Activation functions

introduce non-linearity

determine whether a neuron should be activated

## Sigmoid

![image.png](Activation%20functions/image.png)

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

特点：

- 输出范围：$(0,1)$
- 适合用在**二分类问题的输出层**
- 缺点：容易出现梯度消失（在极大或极小输入时，导数接近 0）

## Tanh

![image.png](Activation%20functions/image%201.png)

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

特点：

- 输出范围：$(-1,1)$
- 常用于**隐藏层**，比 Sigmoid 更好（数据均值更接近 0，收敛更快）
- 仍可能存在梯度消失问题

## RELU

![image.png](Activation%20functions/image%202.png)

$$
f(x) = \max(0, x)
$$

特点：

- 简单且计算高效
- 在正区间不会饱和，梯度消失问题较少
- 缺点：可能出现 **“神经元死亡”**（输入始终小于 0 时，梯度为 0，权重无法更新）

## Softmax

![image.png](Activation%20functions/image%203.png)

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

特点：

- 输出为概率分布（每个类别的概率总和为 1）
- 常用于**多分类任务的输出层**

## SWISS

![image.png](Activation%20functions/image%204.png)

$$
f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

特点：

- 谷歌提出的新型激活函数
- 在一些深度网络中效果优于 ReLU
- 平滑且可微，能避免 ReLU 的“神经元死亡”问题