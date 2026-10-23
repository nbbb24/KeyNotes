# 多模态机器学习：实践学习路线

这条路线遵循原始CMU多模态课程的广泛主题：表示、对齐、交互、生成和评估。目标是解释在何时多个模态提供有用信息以及如何证明这种贡献。

## 前提条件和顺序

首先学习[学习范式](../../mle/learning-paradigms.md)、[损失函数](../../mle/loss-functions.md)和[注意力](../components/attention.md)。然后研究：

1. [多模态基础](../multimodal/foundations.md): 区分核心问题。
2. [CNNs](../models/cnn.md)，[ViT](../models/vit.md) 和 [词表示](advanced-nlp/word-representations.md): 理解模态特定的编码器。
3. [对齐](../multimodal/alignment.md) 和 [CLIP](../models/clip.md): 连接对应示例。
4. [融合](../multimodal/fusion.md): 模型加性和交互信息。
5. [BLIP-2](../models/blip.md)，[Flamingo](../models/flamingo.md) 和 [LLaVA](../models/llava.md): 与生成接口进行比较。
6. [语音评估](../speech/asr.md): 将指标匹配到实际任务。

## 项目：测试第二个模态是否有助于

选择一个配对数据集，明确任务。单独训练或评估每个模态、简单后期融合和一个交互模型。保持数据分割和评估条件固定。包括缺失模态和打乱对的测试。如果交互模型改进，识别额外模态提供第一个中缺少信息的例子。

对于视频中的情感分析，单词可能传达命题，而语音和表情则对其进行修饰。具有强大文本捷径的数据集可能会显示出其他模态的需求不大。这是一个需要测量的属性，而不是假设模型学习了交互的借口。

## 高级扩展

测试时间错位、噪声观测和领域转移。报告平均质量和失败子集。添加模态会增加收集、同步和服务成本；改进必须在这些成本和可用性约束下进行解释。

**练习。** 如果在测试时移除视频不改变预测，可能的两个解释是什么？模型忽略了视频，或者任务/数据不需要它。有针对性的例子有助于区分它们。

原始资源：[CMU多模态机器学习，2023年秋季](https://cmu-multicomp-lab.github.io/mmml-course/fall2023/schedule/)。
