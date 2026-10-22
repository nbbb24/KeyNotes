!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Loss function

## 面试要点：损失函数

### 定义与作用

损失函数（也称代价函数）用于度量模型预测与真实值之间的差异，从而评估模型表现，并为参数优化提供目标。

### 常见损失函数

- 均方误差（MSE）
    - 回归常用；放大大误差影响
    - 公式：

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

- 平均绝对误差（MAE）
    - 回归常用；对离群点更稳健
    - 公式：

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n \lvert y_i - \hat{y}_i \rvert
$$

- 二元交叉熵（Binary Cross-Entropy, BCE）
    - 二分类；p 为正类概率，p∈(0,1)
    - 公式：

$$
\mathcal{L}_{\mathrm{BCE}} = - \big( y\,\log p + (1-y)\,\log(1-p) \big)
$$

- 多分类交叉熵（Categorical Cross-Entropy）
    - 多分类；p_k 为类别 k 的 softmax 概率，y_k 为 one-hot
    - 公式：

        ```python
        import numpy as np

        def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
            """
            y_true: shape (n, K)，one-hot 标签
            y_pred: shape (n, K)，预测概率 (softmax 输出)
            """
            y_pred = np.clip(y_pred, eps, 1 - eps)  # 避免 log(0)
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            return loss

        # 示例
        y_true = np.array([[1,0,0], [0,1,0], [0,0,1]])  # one-hot
        y_pred = np.array([[0.7,0.2,0.1], [0.1,0.8,0.1], [0.2,0.2,0.6]])
        print("Categorical Cross-Entropy Loss:", categorical_cross_entropy(y_true, y_pred))
        ```


$$
\mathcal{L} = - \sum_{k=1}^K y_k \, \log p_k,\quad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

- 合页损失（Hinge Loss）
    - 线性分类与 SVM；y ∈ {−1,+1}
    - 公式：

$$
\mathcal{L}_{\mathrm{hinge}} = \max\big(0,\, 1 - y\, f(x)\big)
$$

- Huber 损失
    - 兼顾 MSE 与 MAE，对异常点更稳健；δ>0 为阈值
    - 公式：

$$
\mathcal{L}_\delta(r) =
\begin{cases}
\tfrac{1}{2} r^2, & \lvert r \rvert \le \delta \\
\delta\,\lvert r \rvert - \tfrac{1}{2}\delta^2, & \text{否则}
\end{cases},\quad r = y-\hat{y}
$$

- Smooth L1（L1 平滑）
    - 常用于检测框回归；等价于 δ=1 的 Huber
    - 公式：

$$
\mathrm{smooth\text{-}L1}(r) =
\begin{cases}
\tfrac{1}{2} r^2, & \lvert r \rvert < 1 \\
\lvert r \rvert - \tfrac{1}{2}, & \text{否则}
\end{cases}
$$

- Focal Loss（用于类别极不平衡）
    - 在 BCE/交叉熵上增加聚焦项，抑制易样本、聚焦难样本；γ≥0，α∈(0,1)
    - 公式：

$$
\mathcal{L}_{\mathrm{focal}} = -\alpha\,(1-p)^{\gamma}\, y\,\log p - (1-\alpha)\, p^{\gamma}\, (1-y)\,\log(1-p)
$$

- 泊松负对数似然（Poisson NLL）
    - 计数型回归；hat{lambda} > 0
    - 公式：

$$
\mathcal{L} = \hat{\lambda} - y\,\log \hat{\lambda}\quad(\text{忽略常数项})
$$

### 将正则项并入损失

- L1 正则（Lasso）：

$$
\lambda \sum_j \lvert w_j \rvert
$$

- L2 正则（Ridge）：

$$
\lambda \sum_j w_j^2
$$

- 弹性网（Elastic Net）：

$$
\lambda_1 \sum_j \lvert w_j \rvert + \lambda_2 \sum_j w_j^2
$$

### 选择准则

- 任务类型：回归或分类
- 对离群点的敏感性
- 计算效率与可优化性
- 收敛性质与数值稳定性
- 具体问题的业务需求

### 面试常见问题与参考回答

1. 解释 MSE 与 MAE 的差异，各自适用场景是什么？
    - 差异：MSE 对大误差二次放大，梯度连续且更平滑；MAE 对离群点更稳健，但在零点处不可导（可用近似如 Huber）。
    - 适用：噪声近似高斯且关心大误差时选 MSE；存在离群点或长尾噪声时选 MAE 或 Huber。
2. 多分类的交叉熵是如何定义与计算的？
    - 定义与计算：

$$
\mathcal{L} = - \sum_{k=1}^K y_k \log p_k, \quad p_k = \mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

实现时使用 log-sum-exp 做数值稳定化；对每个样本等价于取真实类别的 $-log p_y$。

1. 针对含有大量离群点的回归任务，选择何种损失更合适？为何？
    - 首选 MAE 或 Huber。原因：对大残差的惩罚增长更缓，梯度不被离群点主导；Huber 在小残差区保持二次性质，优化更稳定。
2. 将正则项并入损失如何帮助抑制过拟合？
    - 在经验风险后添加结构风险项：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda\,\Omega(w)
$$

L1 促稀疏，起到特征选择作用；L2 限制权重范数，降低方差并提升数值稳定性；弹性网兼具二者优点。

1. 损失函数的选择如何影响训练动态与最终性能？
    - 影响梯度规模、鲁棒性与收敛速度。例如 MSE 梯度随残差线性增大，易受异常点影响；MAE 梯度恒定，收敛可能较慢；交叉熵与 softmax 匹配良好，分类边界清晰；Focal Loss 在极不平衡数据上提升困难样本的学习权重。
