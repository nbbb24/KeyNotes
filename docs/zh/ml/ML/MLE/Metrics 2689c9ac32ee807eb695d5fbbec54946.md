# Metrics

# Confusion Matrix

![image.png](Metrics/image.png)

**False Positive**: Actual Negative, Predicted Positive

**False Negative**: Actual Positive, Predicted Negative

| **Precision** | predicted positive and true positive / predicted positive |
| --- | --- |
| **Recall/Sensitivity** | predicted positive and true positive / actual positive |
| **False Positive Rate** | predicted positive but actual negative /actual negative |
| **Specificity** | predicted negative and actual negative / actual negative |

# ROC

- **ROC 曲线**：展示不同阈值下，模型在 **真正例率（TPR）** 与 **假正例率（FPR）** 之间的权衡。
- **横轴**：FPR（假正例率）
- **纵轴**：TPR（真正例率）

## **特点**

- ROC 曲线通过遍历所有可能的阈值，绘制出 TPR 对 FPR 的关系。
- 对比不同模型时，ROC 曲线 **越靠近左上角**，说明模型性能越好。
- ROC 是 **阈值无关指标**，能反映整体分类能力。

![image.png](Metrics/image%201.png)

IF FP cost is expensive, choose lower FPR even if TPR reduces

## **AUC（Area Under the Curve）**

- AUC 即 ROC 曲线下的面积，用来量化模型整体分类能力。
- 它等价于：一个随机正样本分数大于随机负样本分数的概率。

$$
AUC = P(\text{score}(x^+) > \text{score}(x^-))
$$

## **计算方法**

**1. 梯形法则（Trapezoidal Rule）**

- 将 ROC 曲线分成小梯形，逐段累加面积：

$$
AUC = \sum_{i=1}^{n-1} (FPR_{i+1} - FPR_i) \cdot \frac{TPR_{i+1} + TPR_i}{2}
$$

- **复杂度**：
    - 需要先排序阈值 → O(n \log n)
    - 遍历计算梯形面积 → O(n)
    - 总复杂度：O(n \log n)、

```python
def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    使用梯形法计算 AUC。要求 fpr 单调非降。
    复杂度: O(m) ，m 为 ROC 点数
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    # 手写梯形（不依赖 np.trapz，便于教学/面试）
    area = 0.0
    for i in range(len(fpr) - 1):
        width = fpr[i+1] - fpr[i]
        height = (tpr[i+1] + tpr[i]) / 2.0
        area += width * height
    return float(area)
```

---

**2. Mann–Whitney U 统计量**

- 把所有样本分数排序，计算正样本的秩和 R^+：

$$
AUC = \frac{R^+ - \frac{n_+(n_+ + 1)}{2}}{n_+ \cdot n_-}
$$

- **复杂度**：
    - 排序 → O(n \log n)
    - 秩和计算 → O(n)
    - 总复杂度：O(n \log n)

## 八、ROC / AUC 的优点（面试高频）

✅ **优点**

- 不依赖阈值
- 对类别不平衡相对稳健
- 衡量排序能力（比 accuracy 更靠谱）

---

## 九、ROC / AUC 的局限（一定要会说）

❌ **缺点**

1. **不关注正负样本比例**
    - 实际业务中 FP 代价可能很高
2. **不能反映某个固定阈值下的效果**
3. 在 **极度不平衡** 场景下
    
    👉 PR-AUC 往往更有意义