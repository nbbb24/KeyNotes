# 📈 梯度提升：逐步学习加法修正

梯度提升分阶段构建预测器。它不让所有树各自拟合原始标签，而是问：**增加怎样的函数，可以降低当前模型的损失？**这把树集成与梯度优化联系起来。

## 🎯 训练循环

从使选定损失最小的常数预测 $F_0$ 开始。在第 $m$ 阶段计算伪残差

$$
r_{im}=-\left.\frac{\partial\ell(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}.
$$

用一棵小树拟合这些目标，按损失函数确定叶节点输出，再更新 $F_m=F_{m-1}+\eta h_m$。收缩系数 $\eta$ 控制每个阶段的贡献。

对平方误差 $\frac12(y-F)^2$，负梯度就是熟悉的残差 $y-F$。对逻辑损失，梯度与 $y-p$ 有关，而累加的分数是 logit。因此，“拟合原始残差”只是特例，不是完整定义。

## 🧩 算例

目标为 `[1, 3, 5]`。平方误差下最优常数预测是 3，残差为 `[-2, 0, 2]`。假设浅树给出的修正为 `[-1, -1, 2]`，学习率为 0.5，更新后的预测便是 `[2.5, 2.5, 4]`。平方误差和从 8 降至 3.5。

```python
# runnable-example
y = [1, 3, 5]
old = [3, 3, 3]
correction = [-1, -1, 2]
new = [a + 0.5 * b for a, b in zip(old, correction)]
sse = lambda pred: sum((a - b) ** 2 for a, b in zip(y, pred))
assert sse(old) == 8
assert sse(new) == 3.5
```

修正并不完美，却有帮助。Boosting 不要求每个弱学习器单独解决整个原始问题。

```mermaid
flowchart LR
    A[当前预测] --> B[计算损失梯度]
    B --> C[拟合修正树]
    C --> D[缩放并加上修正]
    D --> A
```

## 📌 正则化是方法的一部分

学习率要和阶段数一起调整。较小学习率通常需要更多阶段；如果预算不足或目标不合适，也不保证泛化更好。浅树限制交互复杂度，叶节点样本数限制、行与特征子采样以及惩罚项提供其他控制。

使用验证集早停并记录最终阶段数，不能让测试集决定何时停止。噪声标签和异常值可能吸引反复修正，尤其当损失强调大误差时。

## 🔍 进阶理解

XGBoost 用带正则化的二阶近似评价树结构和叶节点数值。LightGBM 使用高效的直方图和树生长技术。它们都属于梯度提升家族；实现上的选择没有改变逐步构建加法模型的基本过程。

训练阶段彼此依赖，但阶段内部的计算可以并行。服务成本随树的数量和结构增长，应测量完整集成，而非只测一棵弱学习器。

**思考：**为什么一棵 Boosting 树的输出不一定是目标值？**答案：**它是对当前加法分数的贡献，例如分类任务中的 logit 修正。

## 📚 来源与延伸阅读

- [scikit-learn：梯度提升](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)：损失与正则化。
- [Friedman 的梯度提升论文](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)：函数梯度视角。
