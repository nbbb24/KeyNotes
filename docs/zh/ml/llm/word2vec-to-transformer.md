# 🔤 从 Word2Vec 到 Seq2Seq 与 Transformer

这个进展是关于消除表示和信息流中的瓶颈。Word2Vec学习有用的词向量；递归的序列到序列模型学习条件生成；注意力提供对源状态的直接访问；Transformers围绕注意力构建序列处理。

## 🎯 1. Word2Vec：学习一个可重用的查找表

大小为$V$的词汇表和嵌入宽度$d$定义了一个矩阵$E\in\mathbb R^{V\times d}$。将一个独热行乘以$E$选择一行；实现使用高效的查找代替。Skip-gram从中心词预测上下文词，而CBOW从周围上下文预测中心词。

对于`the cat sat on the mat`，围绕`sat`的窗口创建正中心-上下文对。负采样通过将观察到的对与采样的替代品进行对比，避免了每个对的全词汇softmax。输入和输出嵌入表扮演不同的角色。基本的Word2Vec投影中没有隐藏的非线性，但这并不意味着所有的语义关系都是线性的或精确的。

## 🎯 2. 序列到序列：学习一个输入-输出映射

编码器RNN压缩源为状态；解码器生成目标。最早的固定向量设计将整个源压缩成一个摘要，使得长输入难以处理。注意力允许每个解码步骤读取所有编码器状态的加权组合。这减少了固定摘要的瓶颈，尽管递归计算仍然遵循顺序依赖链。

## 📌 3. Transformer：跟踪形状

对于批次$B$、序列长度$L$、隐藏宽度$d$和$h$个头部：

| 操作 | 输出形状 |
| --- | --- |
| 词查找和位置信息 | $B\times L\times d$ |
| Q/K/V投影和头部拆分 | $B\times h\times L\times d_k$ 当 $d_k=d/h$ |
| Q 与 K 的转置相乘 | $B\times h\times L\times L$ |
| softmax 加权求和 | $B\times h\times L\times d_k$ |
| 合并头部和输出投影 | $B\times L\times d$ |
| 位置前馈网络 | $B\times L\times d$ |

原始的前馈块将512个特征扩展到2048，应用ReLU，然后重新投影。每个位置共享相同的FFN参数。残差连接保留了信息和梯度的路径；规范化稳定了特征尺度。注意力分数表示内容相关的兼容性，而不仅仅是相对位置。

在原始的编码器-解码器Transformer中，解码器有因果自注意力和交叉注意力。交叉注意力从解码器状态获取查询，并从编码器输出获取键/值。最后的词表投影产生 logits（未归一化分数）；softmax 将其转为词元概率。

## 🚀 原始图表，逐步展示

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 1](../../../assets/notion/074-00.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 2](../../../assets/notion/074-01.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 3](../../../assets/notion/074-02.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 4](../../../assets/notion/074-03.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 5](../../../assets/notion/074-04.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 6](../../../assets/notion/074-05.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 7](../../../assets/notion/074-06.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 8](../../../assets/notion/074-07.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 9](../../../assets/notion/074-08.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 10](../../../assets/notion/074-09.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 11](../../../assets/notion/074-10.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 12](../../../assets/notion/074-11.png)

![从 Word2Vec 到 Seq2Seq 与 Transformer：原图 13](../../../assets/notion/074-12.png)

**图示指南。** 图1是本学习路径的目标Transformer。图2-5显示了Word2Vec的上下文对、独热标识符和学习到的嵌入表。图6-7展示了循环编码器—解码器流程以及隐藏状态形状的变化。图8-10从离散输入ID移动到嵌入和位置向量。图11-12显示了多个注意力头部以及解码器的自注意力与编码器-解码器注意力：解码器状态提供查询，编码器状态在交叉注意力块中提供键和值。图13将隐藏状态投影到词表 logits；softmax 随后可将 logits 转为概率。这些图表来自不同的实现，因此请比较它们的信息流，而不是假设相同的规范化顺序。

使用图表来追踪查找、递归编码、编码器自注意力和解码器交叉注意力。原始架构中用于词嵌入的$\sqrt d$因子与注意力分数中使用的$1/\sqrt{d_k}$因子是分开的。

## ✅ 检查并扩展

**练习。** 当$d=512$，有8个头部和20个标记时，一个头部的得分矩阵是什么？它是$20\times20$，每个Q/K向量宽度为64。得分矩阵的宽度不是64。

训练可以并行计算所有因果位置，因为正确的前一个标记是可用的。自回归生成仍然逐个标记进行，除非额外的解码方法改变这种工作流程。

参考文献：[Word2Vec](https://arxiv.org/abs/1301.3781)，[神经翻译中的注意力](https://arxiv.org/abs/1409.0473)，[Transformer](https://arxiv.org/abs/1706.03762)。
