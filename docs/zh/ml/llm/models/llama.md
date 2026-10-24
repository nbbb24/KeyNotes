# 🦙 LLaMA：理解关键架构选择

原始的LLaMA论文描述了由经过仔细训练的标准组件构建的解码器语言模型。本章将解释这些组件；后续的LLama发布版本有自己的配置，不应被视为相同的架构。

## 🧭 预归一化和RMSNorm

预归一化残差块计算$x+F(\operatorname{Norm}(x))$。恒等路径有助于梯度在深层堆栈中流动。原始Transformer使用了后归一化排列，但单独的放置并不能保证或防止梯度消失/爆炸。

LLaMA使用RMSNorm，它通过根均方值重新缩放，而不减去特征均值。请参阅[归一化](../components/normalization.md)以获取确切公式。将其替换为LayerNorm会改变计算，并不能假设与训练权重兼容。

## 🧩 SwiGLU，保持投影分开

一种常见的无偏形式是

$$
\operatorname{FFN}(x)=W_{down}\left[\operatorname{SiLU}(W_{gate}x)\odot(W_{up}x)\right],
\qquad \operatorname{SiLU}(u)=u\sigma(u).
$$

一个投影提供门控非线性特征；另一个提供用于调制的值。下投影恢复隐藏宽度。这与普通GLU不同，后者使用没有额外乘以门预激活的sigmoid门。乘积是一个学习到的特征交互，而不是对序列位置的注意力。

![LLaMA：理解关键架构选择：原图 1](../../../../assets/notion/050-00.png)

第一张图展示了小SwiGLU示例中的两个投影分支。显示的2x2排列是分支值的展示，不是原始权重矩阵的形状。

![LLaMA：理解关键架构选择：原图 2](../../../../assets/notion/050-01.png)

第二张图将SiLU/Swish逐元素应用于门分支。它使用了$\beta=1$的形式；一般Swish参数会改变sigmoid参数。

![LLaMA：理解关键架构选择：原图 3](../../../../assets/notion/050-02.png)

第三张图将值和激活后的门逐元素相乘。然后应用输出投影，一个完整的Transformer前馈块会应用输出投影；Llama家族的实现通常使用无偏投影，与示例中的$Wx+b$不同。

*使用原始激活图来区分sigmoid门控、Swish/SiLU以及SwiGLU的两个投影分支。最终的下投影也是FFN的一部分。*

## 📌 RoPE和生成

旋转位置嵌入修改Q/K坐标对，使注意力分数编码相对位移。模型保持因果性，并在生成过程中受益于KV缓存。这些设计选择与训练数据、优化和规模相互作用；命名组件本身并不能预测基准质量。

**练习。** 如果两个FFN分支各自将宽度d扩展到m，参数数量是否与一个宽度为m的两矩阵ReLU FFN相同？不是。SwiGLU有门、上和下矩阵，因此中间宽度通常会调整以实现可比的预算。

参考：[LLaMA](https://arxiv.org/abs/2302.13971)。继续阅读[Llama 3.1](llama-3-1.md)。
