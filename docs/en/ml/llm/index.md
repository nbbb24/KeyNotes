# 🤖 Language models: a learning map

Language-model engineering connects representations, objectives, optimization, generation, and systems. Study the connections: memorizing model names will not explain why a model hallucinates, why a long prompt runs out of memory, or why fine-tuning fails to teach a missing fact.

## 🔄 Follow one sequence through the model

Start with [foundations](foundations.md) and [Word2Vec to Transformer](word2vec-to-transformer.md). Learn how tokens become vectors, how [attention](components/attention.md) combines information, and how [positions](components/position.md) and [masks](components/attention-masks.md) constrain it. [Normalization](components/normalization.md), [cross-entropy and KL](components/cross-entropy-kl.md), and [attention variants](components/attention-variants.md) explain the main computational pieces.

```mermaid
flowchart LR
    A[Data and tokens] --> B[Representation and architecture]
    B --> C[Pretraining objective]
    C --> D[Adaptation and alignment]
    D --> E[Decoding and tools]
    E --> F[Evaluation and deployment]
```

For each arrow, ask what changes: data, trainable parameters, objective, or inference procedure. Retrieval changes available evidence; LoRA changes how parameter updates are represented; sampling changes how outputs are selected. They solve different problems.

## 🏗️ Architecture and model families

| Question | Chapters |
| --- | --- |
| How did sequence and image representations develop? | [RNN](models/rnn.md), [Tree RNN](models/tree-rnn.md), [CNN](models/cnn.md), [ELMo](models/elmo.md), [ViT](models/vit.md) |
| Encoder, decoder, or both? | [BERT](models/bert.md), [GPT](models/gpt.md), [T5/BART](models/t5-bart.md), [Llama](models/llama.md), [Llama versus GPT](models/llama-vs-gpt.md) |
| What changed in later model reports? | [Llama 3.1](models/llama-3-1.md), [DeepSeek-V3](models/deepseek-v3.md), [DeepSeek-R1](models/deepseek-r1.md), [Qwen2.5](models/qwen-2-5.md), [Qwen2.5-VL](models/qwen-2-5-vl.md) |
| How do images meet text? | [CLIP](models/clip.md), [SigLIP](models/siglip.md), [ALBEF](models/albef.md), [BLIP](models/blip.md), [Flamingo](models/flamingo.md), [LLaVA](models/llava.md) |
| How is speech transcribed? | [Whisper](speech/whisper.md), [ASR evaluation](speech/asr.md) |

Model chapters describe specific research families and releases, not a continuously updated leaderboard. Compare evidence and assumptions before transferring a reported improvement to another setting.

## 📌 Choose the right intervention

For task behavior, study [SFT](adaptation/sft.md), [LoRA](adaptation/lora.md), [adapters](adaptation/adapters.md), [prompt/prefix tuning](adaptation/prompt-tuning.md), and [distillation](adaptation/distillation.md). For preference optimization, follow [RLHF](alignment/rlhf.md) → [importance sampling](alignment/importance-sampling.md) → [PPO](alignment/ppo.md), then compare [DPO](alignment/dpo.md) and [GRPO](alignment/grpo.md).

For generation and deployment, connect [decoding](inference/decoding.md), [KV cache](inference/kv-cache.md), [FLOPs](inference/flops.md), [quantization](inference/quantization.md), [layers and pages](inference/layers-pages.md), and [multi-token prediction](inference/multi-token-prediction.md). [MoE](components/moe.md) is an architecture and routing technique; it is not itself an agent framework.

For external information and actions, study [RAG](rag.md), [Tree of Thoughts](inference/tree-of-thoughts.md), [MCP](tools/mcp.md), [LangChain](tools/langchain.md), and [DeepSpeed](tools/deepspeed.md). Distinguish model reasoning, search, tool orchestration, and distributed training infrastructure.

## 🚀 Structured courses and multimodal extensions

The course tracks provide longer sequences: [Advanced NLP](courses/advanced-nlp/index.md), [LLM methods and applications](courses/llm/index.md), [Multimodal Machine Learning](courses/multimodal.md), and [Advanced Multimodal Topics](courses/advanced-multimodal.md). Supplement them with [multimodal foundations](multimodal/foundations.md), [fusion](multimodal/fusion.md), and [alignment](multimodal/alignment.md).

The original reading roadmap also named VisualBERT, ViLBERT, LXMERT, DALL·E, Parti, Imagen, and Kosmos-1. Keep them as extension topics: compare joint versus separate vision-language encoding, and image understanding versus image generation. They are not presented here as completed standalone chapters. CLIP, BLIP, Flamingo, and LLaVA now have their own explanations above.

A capstone: build a small evidence-grounded question-answering experiment, evaluate retrieval separately from generation, measure latency and memory, then explain whether your next improvement should change the data, model, retriever, or system.
