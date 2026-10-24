# 🎭 注意力掩码：控制信息可见范围

注意力掩码指定允许的信息流动。它不是dropout，通常不会从张量中移除一个标记。屏蔽是必要的，因为训练序列包含未来答案，这些答案必须对自回归预测保持不可见。

## 🔍 按行阅读矩阵

行代表查询，列代表键。四个位置的加性因果掩码为：

$$
M=\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}.
$$

在softmax之前添加负无穷大会使禁止的权重变为零。位置2可以使用位置0、1和2。位置2的训练标签通常是在输入和目标移位后标记3。因此，对角线是允许的，不会揭示下一个标记的标签。

| 掩码 | 它阻止的内容 | 典型用途 |
| --- | --- | --- |
| 因果 | 访问未来键 | 自回归语言模型 |
| 填充 | 访问填充键位置 | 混合长度批次 |
| 前缀 | 允许双向提示访问时输出泄漏 | 前缀语言模型 |
| 段落 | 访问独立打包文档之间的内容 | 效率高的打包训练 |
| 局部窗口 | 超出邻域的访问 | 上下文限制注意力 |

掩码可以组合。仅使用因果掩码并不阻止一个打包训练示例读取另一个放在它前面的示例。

## 🧩 一个小的例子

输入为 `The cat sleeps`，解码器在每个位置上被训练来预测下一个标记。对于 `The` 的状态必须不能读取 `cat`，尽管两个张量在同一并行训练调用中计算。在推理时，一个单一的新查询可以关注所有有效的缓存过去键；查询和键的长度不同，因此盲目应用平方三角形掩码是错误的。

```python
# runnable-example
length = 4
allowed = [[key <= query for key in range(length)] for query in range(length)]
assert allowed[0] == [True, False, False, False]
assert allowed[3] == [True, True, True, True]
```

## ⚠️ 实现陷阱

布尔掩码约定在不同API之间有所不同。在PyTorch缩放点积注意力中，`True`表示允许；在几个填充掩码接口中，它表示排除。应以具体 API 文档为准。完全屏蔽的一行可以创建未定义的softmax值。显式处理空/填充查询。屏蔽填充键也不自动排除填充预测位置的损失；损失屏蔽是一个独立的操作。

**练习。** 一个三标记提示是双向的，两个响应标记是因果的。响应位置3可能读取哪些键？位置0到3。提示行可以读取位置0到2，但永远不会读取响应位置。

继续阅读 [注意力](attention.md) 和 [KV缓存](../inference/kv-cache.md)。参考：[PyTorch缩放点积注意力](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
