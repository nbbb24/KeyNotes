# ⚖️ Comparing Llama and GPT without confusing family names

“Llama” and “GPT” name evolving model families, not two fixed architectures. A useful comparison names exact checkpoints or service versions, their access conditions, and the workload. Historical GPT-style and Llama text models share the decoder-only autoregressive pattern; many important differences are in data, training, architecture details, and deployment.

## ⚖️ Compare on explicit axes

| Axis | Questions to answer |
| --- | --- |
| Architecture | Dense or MoE? Attention/KV configuration? Modalities? |
| Training | Base, instruction-tuned, or reasoning-oriented? Known data and objectives? |
| Access | Downloadable weights or hosted endpoint? Which operations are exposed? |
| Deployment | Local hardware, quantization, batching, and context limits? |
| Quality | Which task, language, prompt format, and decoding settings? |
| Cost | Training/adaptation cost, per-request cost, operational overhead? |
| Reproducibility | Fixed checkpoint/version, seed, runtime, and evaluation data? |

Do not infer unpublished internals from a product name. Downloadable weights improve some forms of inspection and adaptation, but license terms and complete training-data availability are separate questions. Check the specific release's documentation when making a deployment decision.

## 🧩 A fair experiment

For document question answering, use the same held-out questions and evidence. Apply each model's intended chat format. Measure correctness, supported citations, refusal when evidence is absent, latency, and cost. Then run the same task under the actual deployment constraints. A self-hosted model and a hosted service may differ in queueing, tooling, and preprocessing as well as raw model quality.

Parameter count alone does not settle the comparison. Context-window limits do not establish effective long-context understanding. A single leaderboard average can conceal weak performance on the languages or document types the application uses.

## ⚖️ What the historical comparison teaches

LLaMA's documented RMSNorm, SwiGLU, and RoPE choices provide concrete engineering examples. GPT-style next-token modeling illustrates the general causal-generation framework. Understanding these components is more durable than memorizing a claim that one family is always faster or better.

**Exercise.** Two models score differently after one receives retrieved evidence and the other does not. Is this a controlled architecture comparison? No. The system inputs differ.

References: [original LLaMA](https://arxiv.org/abs/2302.13971), [GPT-3](https://arxiv.org/abs/2005.14165). See [evaluation methodology](../courses/llm/evaluation.md).
