# 🤖 语言模型学习路线

语言模型工程将表示、目标、优化、生成和系统连接起来。研究这些联系：记住模型名称不会解释为什么一个模型会幻觉，为什么一个长提示会耗尽内存，或者为什么微调无法教给缺失的事实。

## 🏗️ 跟随一个序列通过模型

从[基础](foundations.md)和[Word2Vec到Transformer](word2vec-to-transformer.md)开始。学习如何标记成为向量，如何使用[注意力](components/attention.md)结合信息，以及如何使用[位置](components/position.md)和[掩码](components/attention-masks.md)约束它。[归一化](components/normalization.md)，[交叉熵和KL](components/cross-entropy-kl.md)，以及[注意力变体](components/attention-variants.md)解释了主要的计算组件。

```mermaid
flowchart LR
    A[数据和标记] --> B[表示和架构]
    B --> C[预训练目标]
    C --> D[适应和对齐]
    D --> E[解码和工具]
    E --> F[评估和部署]
```

对于每一条箭头，问一下会发生什么变化：数据、可训练参数、目标或推理程序。检索会改变可用的证据；LoRA会改变参数更新的表示方式；采样会改变输出的选择方式。它们解决不同的问题。

## 🏗️ 架构和模型家族

| 问题 | 章节 |
| --- | --- |
| 序列和图像表示是如何发展的？ | [RNN](models/rnn.md)，[树RNN](models/tree-rnn.md)，[CNN](models/cnn.md)，[ELMo](models/elmo.md)，[ViT](models/vit.md) |
| 编码器、解码器还是两者都有？ | [BERT](models/bert.md)，[GPT](models/gpt.md)，[T5/BART](models/t5-bart.md)，[Llama](models/llama.md)，[Llama与GPT的比较](models/llama-vs-gpt.md) |
| 后续模型报告中发生了什么变化？ | [Llama 3.1](models/llama-3-1.md)，[DeepSeek-V3](models/deepseek-v3.md)，[DeepSeek-R1](models/deepseek-r1.md)，[Qwen2.5](models/qwen-2-5.md)，[Qwen2.5-VL](models/qwen-2-5-vl.md) |
| 图像如何与文本相遇？ | [CLIP](models/clip.md)，[SigLIP](models/siglip.md)，[ALBEF](models/albef.md)，[BLIP](models/blip.md)，[Flamingo](models/flamingo.md)，[LLaVA](models/llava.md) |
| 语音是如何转录的？ | [Whisper](speech/whisper.md)，[ASR评估](speech/asr.md) |

模型章节描述了特定的研究家族和发布，而不是一个不断更新的排行榜。在将报告中的改进转移到另一个设置之前，请比较证据和假设。

## 📌 选择正确的干预措施

对于任务行为，研究[SFT](adaptation/sft.md)，[LoRA](adaptation/lora.md)，[适配器](adaptation/adapters.md)，[提示/前缀调优](adaptation/prompt-tuning.md)和[蒸馏](adaptation/distillation.md)。对于偏好优化，遵循[RLHF](alignment/rlhf.md) → [重要性采样](alignment/importance-sampling.md) → [PPO](alignment/ppo.md)，然后比较[DPO](alignment/dpo.md)和[GRPO](alignment/grpo.md)。

对于生成和部署，连接[解码](inference/decoding.md)，[KV缓存](inference/kv-cache.md)，[FLOPs](inference/flops.md)，[量化](inference/quantization.md)，[层和页面](inference/layers-pages.md)和[多标记预测](inference/multi-token-prediction.md)。[MoE](components/moe.md)是一种架构和路由技术；它本身不是一个代理框架。

对于外部信息和行动，研究[RAG](rag.md)，[思维树](inference/tree-of-thoughts.md)，[MCP](tools/mcp.md)，[LangChain](tools/langchain.md)和[DeepSpeed](tools/deepspeed.md)。区分模型推理、搜索、工具编排和分布式训练基础设施。

## 🚀 结构化课程和多模态扩展

课程轨迹提供了更长的序列：[高级NLP](courses/advanced-nlp/index.md)，[LLM方法与应用](courses/llm/index.md)，[多模态机器学习](courses/multimodal.md)和[高级多模态主题](courses/advanced-multimodal.md)。用[多模态基础](multimodal/foundations.md)，[融合](multimodal/fusion.md)和[对齐](multimodal/alignment.md)补充它们。

原始阅读路线图也命名为VisualBERT、ViLBERT、LXMERT、DALL·E、Parti、Imagen和Kosmos-1。将它们作为扩展主题：比较联合编码与分离的视觉语言编码，以及图像理解与图像生成。它们在这里不被呈现为独立完成的章节。CLIP、BLIP、Flamingo和LLaVA现在有自己的解释在上面。

一个顶点项目：构建一个小的基于证据的问题回答实验，分别评估检索和生成，测量延迟和内存，然后解释你的下一个改进是否应该改变数据、模型、检索器或系统。
