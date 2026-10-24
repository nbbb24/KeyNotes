# 🦙 Llama 3.1 and the Llama 3 technical report

The Llama 3 report covers text models and additional multimodal experiments. Released Llama 3.1 text checkpoints should not be described as if they automatically include every image, video, and speech component explored in the report.

## 🎯 Text architecture and post-training

The reported text models use a decoder-only Transformer with grouped-query attention. Sharing KV heads among query heads reduces cache storage; the published configurations use eight KV heads. Model size changes layer count and hidden dimensions, so compute cache size from the exact checkpoint configuration.

![Llama 3.1 and the Llama 3 technical report, original figure 1](../../../../assets/notion/051-00.png)

*The architecture table separates model size, attention heads, and KV heads. They are different inputs to memory and compute estimates.*

Post-training iterates data collection, reward modeling, rejection sampling, SFT, and DPO. Rejection sampling generates candidates and selects high-scoring responses for training. The report's DPO recipe masks special formatting tokens and adds chosen-response NLL with coefficient 0.2; these are recipe-specific modifications, not universal DPO requirements.

![Llama 3.1 and the Llama 3 technical report, original figure 2](../../../../assets/notion/051-01.png)

*The post-training diagram is a data-and-model iteration loop, not a single pass through one dataset.*

## 🛠️ Tool use and deployment

![Llama 3.1 and the Llama 3 technical report, original figure 3](../../../../assets/notion/051-02.png)

*Tool-use examples include a structured call, a tool result, and a final response. A model-generated call still needs host-side validation and execution.*

The report also describes selective FP8 inference choices, including leaving first/last Transformer layers unquantized. Such results depend on kernels and hardware; “FP8” is not a promise of a fixed speedup for every model or batch.

## 📌 Multimodal experiments are separate scope

![Llama 3.1 and the Llama 3 technical report, original figure 4](../../../../assets/notion/051-03.png)

*The vision experiment connects an image encoder to language through an adapter. It should be read as an experimental extension rather than a property of every text checkpoint.*

The report studies separately pretrained image and speech encoders, modality adapters, and video/speech extensions. Training which component is frozen or updated is stage-specific. For actual use, check the named checkpoint's modality support, preprocessing, context configuration, and model card.

**Exercise.** Can a text-only checkpoint read an image merely because its technical report contains a vision section? No. The checkpoint and runtime must contain and support the visual path.

Reference: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). Review [GQA and cache accounting](../inference/kv-cache.md).
