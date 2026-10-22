# Support Vectors Machine

![image.png](Support%20Vectors%20Machine/image.png)

find hyperplane to classifies

**support vectors**: data points nearest to the hyperplane

**kernel**: non linear data

### 软间隔：

$$
min\frac{1}{2}∥w∥^2+C∑ξi
$$

- **C 大**：
    - 更在乎训练误差
    - 容易过拟合
- **C 小**：
    - 允许分错点
    - margin 更大，泛化更强

### 核函数

核函数通过隐式地将数据映射到高维空间，使原本线性不可分的数据在高维空间线性可分。

$$
K(x,z)=ϕ(x)⋅ϕ(z)
$$

### ⭐ 1. 线性核（Linear Kernel）

$$
K(x,z)=x⊤z
$$

**适合：**

- 特征维度高
- 数据本身近似线性可分

👉 等价于 **线性分类器**

---

### ⭐ 2. 多项式核（Polynomial Kernel）

$$
K(x,z) = (x^\top z + c)^d
$$

**特点：**

- 能学习特征之间的组合关系
- 次数 ddd 越高 → 越复杂（越容易过拟合）

---

### ⭐ 3. 高斯核 / RBF 核（最常用🔥）

$$
K(x,z) = \exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right)
$$

**直觉：**

- 距离近 → 相似度高
- 距离远 → 几乎没关系

**特点：**

- 无限维特征空间
- 非常强大
- 默认首选

---

### ⭐ 4. Sigmoid 核（较少用）

$$
K(x,z) = \tanh(\alpha x^\top z + c)
$$

- 类似神经网络
- 不总是满足 Mercer 条件
- 实际用得少

---

### 📌 面试速记表

| 核函数 | 使用场景 |
| --- | --- |
| Linear | 高维稀疏特征 |
| Polynomial | 特征交互明显 |
| RBF | **通用首选（最常考）** |
| Sigmoid | 很少用 |