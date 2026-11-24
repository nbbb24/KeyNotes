# 🧠 GLM-5.2：相同主干宽度，跨层复用 Indexer

本页绑定 `zai-org/GLM-5.2@cf457fa734ab` 的 [官方配置](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json)。与 GLM-5.1 比较时，最重要的结构差异是 **IndexShare** 与上下文配置，而不是另造一组 Attention 宽度。实现依据固定版本的 [Attention](https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)与[配置类](https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py)。

## 🗺️ 配置与 GLM-5.1 的关系

| 项目 | GLM-5.1 | GLM-5.2 |
| --- | --- | --- |
| 主干层 / 隐藏宽度 / 词表 | 78 / 6144 / 154880 | 相同 |
| Dense 层 / MoE 层 | 0–2 / 3–77 | 相同；配置显式列出 `mlp_layer_types` |
| 专家数 / top-k / 共享专家 / 单专家 F | 256 / 8 / 1 / 2048 | 相同 |
| Q 头 / 主 QK 维度 / V 维度 | 64 / 256 / 256 | 相同 |
| Q 低秩 / KV 潜在维度 / RoPE 维度 | 2048 / 512 / 64 | 相同 |
| Indexer 头 / 头宽 / top-k | 32 / 128 / 2048 | 相同 |
| 本次配置中的独立 Indexer 层 | 78 | 21，其余 57 层复用选中位置 |
| 最大位置数 / RoPE theta | 202752 / 1000000 | 1048576 / 8000000 |
| MTP 层数 | 1 | 1；预测效果/展开步数不由这个字段单独决定 |

模型卡将此改动称为 [IndexShare](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/README.md)，相应方法见[原始论文](https://arxiv.org/abs/2603.12201)。下面的逐层表直接记录配置中的 78 层及其索引复用关系。

## 📐 一层的主投影 Shape

| 模块 | 逻辑权重 `[out,in]` | 输出 |
| --- | --- | --- |
| Embedding / 未绑定 LM Head | 各 `[154880,6144]` | 隐藏 `[B,S,6144]` / logits `[B,S,154880]` |
| `q_a` → Q norm | `[2048,6144]`，norm `[2048]` | `[B,S,2048]` |
| `q_b` | `[16384,2048]` | `[B,S,64,256]` |
| `kv_a` → KV norm | `[576,6144]`，norm `[512]` | 潜在 512 + RoPE 64 |
| `kv_b` | `[28672,512]` | K-NoPE `[B,S,64,192]`，V `[B,S,64,256]` |
| `o_proj` | `[6144,16384]` | `[B,S,6144]` |
| Dense gate/up → down | 各 `[12288,6144]` → `[6144,12288]` | 12288 → 6144 |
| Router / 单专家 gate/up → down | `[256,6144]` / 各 `[2048,6144]` → `[6144,2048]` | top-8 路由与共享专家 |

与 [GLM-5.1 模块表](glm-5-1.md)相同，完整 K 仍是 NoPE 192 与 RoPE 64 拼接。配置中的 `head_dim=192` 不是主 Q/K 的总宽度，不能据此把 `q_b` 写成 `64×192`。

## 🔎 IndexShare 逐层复用的是 Top-k，不是主 KV

本次配置显式给出 `indexer_types`：层 0、1、2，以及 `6,10,14,…,74` 是 `full`；其余层为 `shared`。例如：

| 层 | 索引行为 | 主 MLA KV |
| --- | --- | --- |
| 2 | 自己计算 Top-k，发布选中位置 | 自己的潜在 KV |
| 3、4、5 | 复用层 2 的选中位置 | 每层仍有自己的潜在 KV |
| 6 | 重新计算 Top-k | 自己的潜在 KV |
| 7、8、9 | 复用层 6 的选中位置 | 每层仍有自己的潜在 KV |
| 74 → 75、76、77 | 74 计算，后 3 层复用 | 各自的主 KV |

`full` 表示该层运行 Indexer，不表示主注意力变成 Dense Full Attention；`shared` 也不表示复用整个 Attention 输出或隐藏状态。参考实现仅在需要时实例化 Indexer，并通过 `prev_topk_indices` 传递索引。独立 Indexer 的投影维度仍为 `wq_b=[4096,2048]`、`wk=[128,6144]`、`weights_proj=[32,6144]`。

## 🧠 缓存与扩上下文的实际影响

主 MLA 缓存的每层潜在维度仍为 512+64；不能因为只有 21 个独立 Indexer 就把 78 层主缓存也缩成 21 层。Indexer Key 与 Top-k 中间结果则由索引复用策略和后端缓存实现决定。最大位置数增加，也不意味着部署时已经预分配了 1M Token 的全部缓存。

相同 BF16 主缓存假设下，B=1、T=4096 的载荷仍为 351 MiB；T 增长时需要增加容量预算。实际支持长度还受内存、稀疏注意力内核和配置限制。参见[Shape 约定](shape-conventions.md)和[KV 内存预算](../infra/serving-frameworks.md)。

## 🪜 完整主干层表

<!-- BEGIN GENERATED LAYERS glm-5-2 -->
??? info "GLM-5.2 · 78 主干层"

    | 层 | 层间输入/输出 | FFN | Indexer / 索引来源 | 主 KV（每层自有） |
    | --- | --- | --- | --- | --- |
    | 0 | `[B,S,6144]` | Dense F=12288 | full; L0 | own L0: `[B,T,512]` + `[B,T,64]` |
    | 1 | `[B,S,6144]` | Dense F=12288 | full; L1 | own L1: `[B,T,512]` + `[B,T,64]` |
    | 2 | `[B,S,6144]` | Dense F=12288 | full; L2 | own L2: `[B,T,512]` + `[B,T,64]` |
    | 3 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L2 | own L3: `[B,T,512]` + `[B,T,64]` |
    | 4 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L2 | own L4: `[B,T,512]` + `[B,T,64]` |
    | 5 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L2 | own L5: `[B,T,512]` + `[B,T,64]` |
    | 6 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L6 | own L6: `[B,T,512]` + `[B,T,64]` |
    | 7 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L6 | own L7: `[B,T,512]` + `[B,T,64]` |
    | 8 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L6 | own L8: `[B,T,512]` + `[B,T,64]` |
    | 9 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L6 | own L9: `[B,T,512]` + `[B,T,64]` |
    | 10 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L10 | own L10: `[B,T,512]` + `[B,T,64]` |
    | 11 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L10 | own L11: `[B,T,512]` + `[B,T,64]` |
    | 12 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L10 | own L12: `[B,T,512]` + `[B,T,64]` |
    | 13 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L10 | own L13: `[B,T,512]` + `[B,T,64]` |
    | 14 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L14 | own L14: `[B,T,512]` + `[B,T,64]` |
    | 15 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L14 | own L15: `[B,T,512]` + `[B,T,64]` |
    | 16 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L14 | own L16: `[B,T,512]` + `[B,T,64]` |
    | 17 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L14 | own L17: `[B,T,512]` + `[B,T,64]` |
    | 18 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L18 | own L18: `[B,T,512]` + `[B,T,64]` |
    | 19 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L18 | own L19: `[B,T,512]` + `[B,T,64]` |
    | 20 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L18 | own L20: `[B,T,512]` + `[B,T,64]` |
    | 21 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L18 | own L21: `[B,T,512]` + `[B,T,64]` |
    | 22 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L22 | own L22: `[B,T,512]` + `[B,T,64]` |
    | 23 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L22 | own L23: `[B,T,512]` + `[B,T,64]` |
    | 24 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L22 | own L24: `[B,T,512]` + `[B,T,64]` |
    | 25 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L22 | own L25: `[B,T,512]` + `[B,T,64]` |
    | 26 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L26 | own L26: `[B,T,512]` + `[B,T,64]` |
    | 27 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L26 | own L27: `[B,T,512]` + `[B,T,64]` |
    | 28 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L26 | own L28: `[B,T,512]` + `[B,T,64]` |
    | 29 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L26 | own L29: `[B,T,512]` + `[B,T,64]` |
    | 30 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L30 | own L30: `[B,T,512]` + `[B,T,64]` |
    | 31 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L30 | own L31: `[B,T,512]` + `[B,T,64]` |
    | 32 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L30 | own L32: `[B,T,512]` + `[B,T,64]` |
    | 33 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L30 | own L33: `[B,T,512]` + `[B,T,64]` |
    | 34 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L34 | own L34: `[B,T,512]` + `[B,T,64]` |
    | 35 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L34 | own L35: `[B,T,512]` + `[B,T,64]` |
    | 36 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L34 | own L36: `[B,T,512]` + `[B,T,64]` |
    | 37 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L34 | own L37: `[B,T,512]` + `[B,T,64]` |
    | 38 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L38 | own L38: `[B,T,512]` + `[B,T,64]` |
    | 39 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L38 | own L39: `[B,T,512]` + `[B,T,64]` |
    | 40 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L38 | own L40: `[B,T,512]` + `[B,T,64]` |
    | 41 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L38 | own L41: `[B,T,512]` + `[B,T,64]` |
    | 42 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L42 | own L42: `[B,T,512]` + `[B,T,64]` |
    | 43 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L42 | own L43: `[B,T,512]` + `[B,T,64]` |
    | 44 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L42 | own L44: `[B,T,512]` + `[B,T,64]` |
    | 45 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L42 | own L45: `[B,T,512]` + `[B,T,64]` |
    | 46 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L46 | own L46: `[B,T,512]` + `[B,T,64]` |
    | 47 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L46 | own L47: `[B,T,512]` + `[B,T,64]` |
    | 48 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L46 | own L48: `[B,T,512]` + `[B,T,64]` |
    | 49 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L46 | own L49: `[B,T,512]` + `[B,T,64]` |
    | 50 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L50 | own L50: `[B,T,512]` + `[B,T,64]` |
    | 51 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L50 | own L51: `[B,T,512]` + `[B,T,64]` |
    | 52 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L50 | own L52: `[B,T,512]` + `[B,T,64]` |
    | 53 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L50 | own L53: `[B,T,512]` + `[B,T,64]` |
    | 54 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L54 | own L54: `[B,T,512]` + `[B,T,64]` |
    | 55 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L54 | own L55: `[B,T,512]` + `[B,T,64]` |
    | 56 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L54 | own L56: `[B,T,512]` + `[B,T,64]` |
    | 57 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L54 | own L57: `[B,T,512]` + `[B,T,64]` |
    | 58 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L58 | own L58: `[B,T,512]` + `[B,T,64]` |
    | 59 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L58 | own L59: `[B,T,512]` + `[B,T,64]` |
    | 60 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L58 | own L60: `[B,T,512]` + `[B,T,64]` |
    | 61 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L58 | own L61: `[B,T,512]` + `[B,T,64]` |
    | 62 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L62 | own L62: `[B,T,512]` + `[B,T,64]` |
    | 63 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L62 | own L63: `[B,T,512]` + `[B,T,64]` |
    | 64 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L62 | own L64: `[B,T,512]` + `[B,T,64]` |
    | 65 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L62 | own L65: `[B,T,512]` + `[B,T,64]` |
    | 66 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L66 | own L66: `[B,T,512]` + `[B,T,64]` |
    | 67 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L66 | own L67: `[B,T,512]` + `[B,T,64]` |
    | 68 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L66 | own L68: `[B,T,512]` + `[B,T,64]` |
    | 69 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L66 | own L69: `[B,T,512]` + `[B,T,64]` |
    | 70 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L70 | own L70: `[B,T,512]` + `[B,T,64]` |
    | 71 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L70 | own L71: `[B,T,512]` + `[B,T,64]` |
    | 72 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L70 | own L72: `[B,T,512]` + `[B,T,64]` |
    | 73 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L70 | own L73: `[B,T,512]` + `[B,T,64]` |
    | 74 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L74 | own L74: `[B,T,512]` + `[B,T,64]` |
    | 75 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L74 | own L75: `[B,T,512]` + `[B,T,64]` |
    | 76 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L74 | own L76: `[B,T,512]` + `[B,T,64]` |
    | 77 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | shared; L74 | own L77: `[B,T,512]` + `[B,T,64]` |
<!-- END GENERATED LAYERS glm-5-2 -->
