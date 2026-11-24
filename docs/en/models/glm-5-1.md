# 🧠 GLM-5.1: shapes across 78 MLA, DSA, and MoE layers

This page pins the [official config](https://huggingface.co/zai-org/GLM-5.1/blob/26e1bd6e011feb778d25ae34b09b07074139d92d/config.json) at `zai-org/GLM-5.1@26e1bd6e011f`, with modules interpreted through a pinned [Transformers implementation](https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py). Shapes are logical and unpartitioned by TP/EP; weights use `[out,in]`.

## 🗺️ Configuration and layer organization

| Setting | Value |
| --- | --- |
| Vocabulary / hidden width / backbone layers | 154880 / 6144 / 78 |
| FFN placement | Dense at 0–2, MoE at 3–77 |
| Dense / per-expert intermediate width | 12288 / 2048 |
| Routed experts / top-k / shared experts | 256 / 8 / 1 |
| Query heads / NoPE + RoPE / value width | 64 / 192 + 64 / 256 |
| Q rank / KV latent width | 2048 / 512 |
| DSA indexer | 32 heads × 128, selecting up to 2048 positions |
| Maximum positions / MTP layers | 202752 / 1, outside the 78-layer backbone |

Each layer applies normalization, MLA/DSA attention and a residual, then normalization, a dense FFN or MoE, and a residual. Inter-layer input/output is `[B,S,6144]`, followed by final normalization and the vocabulary head. Embedding and untied LM head each have weight `[154880,6144]`. Full-position logits are `[B,S,154880]`; serving may compute only needed positions.

## 📐 Per-layer attention projections

| Module | Logical weight | Output activation |
| --- | --- | --- |
| `q_a_proj` | `[2048,6144]` | `[B,S,2048]` |
| Q RMSNorm | `[2048]` | `[B,S,2048]` |
| `q_b_proj` | `[16384,2048]` | `[B,S,64,256]` |
| Q NoPE / RoPE slices | No separate projection | `[B,S,64,192]` / `[B,S,64,64]` |
| `kv_a_proj_with_mqa` | `[576,6144]` | Latent `[B,S,512]` + shared RoPE K `[B,S,64]` |
| KV RMSNorm | `[512]` | `[B,S,512]` |
| `kv_b_proj` | `[28672,512]` | Logical K-NoPE `[B,S,64,192]` + V `[B,S,64,256]` |
| K after shared RoPE concatenation | No extra weight | `[B,S,64,256]` |
| Concatenated attention output | No extra weight | `[B,S,16384]` |
| `o_proj` | `[6144,16384]` | `[B,S,6144]` |

`64×(192+256)=28672`, rather than `64×256×2`. The 256-dimensional Q/K scoring axis and 512-dimensional KV latent are different. JSON `head_dim=64` does not replace the main attention width of 256; the reference configuration class handles RoPE fields separately. See the [config implementation](https://github.com/huggingface/transformers/blob/d8f10a82213a3a9db48385aba3a4b57ab535de50/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py).

## 🔎 DSA index heads are a side path

The indexer projects the Q latent into `[B,S,32,128]` using `wq_b=[4096,2048]`. `wk=[128,6144]` produces one 128-dimensional key per position, followed by LayerNorm; `weights_proj=[32,6144]` supplies head weights. Selected indices have shape `[B,S,min(2048,T)]`, then main attention accesses the selected KV.

All 78 layers run their own indexer in this snapshot. Its 32 heads do not replace the main 64 heads, and top-k does not mean retaining only 2048 historical tokens. A simple reference may materialize full scores; optimized sparse kernels can use different intermediate storage.

## 🧩 Dense and MoE FFNs

| Layer/module | Logical weight | Main activation |
| --- | --- | --- |
| Layers 0–2: dense gate / up | Each `[12288,6144]` | Each `[B,S,12288]` |
| Layers 0–2: dense down | `[6144,12288]` | `[B,S,6144]` |
| Layers 3–77: router | `[256,6144]` | Scores `[N,256]`, selected IDs/weights `[N,8]` |
| One routed expert gate / up | Each `[2048,6144]` | `[N_e,2048]` |
| One routed expert down | `[6144,2048]` | `[N_e,6144]` |
| Shared expert | Same dimensions as one routed expert | Processes every token in that layer |
| Fused routed gate+up / down | `[256,4096,6144]` / `[256,6144,2048]` | May use grouped GEMM |

Each token visits 8 routed experts plus the shared expert, rather than all 256. Actual routing determines `N_e`; `N/256` is not a fixed runtime shape. Fused shapes describe the cited implementation, while checkpoints or other backends may keep separate expert keys.

## 🧠 KV accounting

Main MLA cache can be expressed per layer as latent `[B,T,512]` and RoPE K `[B,T,64]`; interfaces may retain a singleton head axis. Expanded `[B,T,64,256]` K/V are logical attention views and need not all persist. Indexer keys, indices, pages, and workspaces require separate accounting.

Assuming BF16 unsharded main latents only, each token/layer costs `(512+64)×2=1152` bytes. With 78 layers, 4096 tokens, and B=1, the payload is **351 MiB**. This excludes indexer state, quantization metadata, allocation rounding, and parallel replication; it is not total device memory.

## 🪜 Complete backbone map

The common module tables define repeated shapes; the map below enumerates ownership. MTP is an auxiliary prediction branch rather than a mandatory 79th backbone layer in every ordinary forward. Analyze speculative execution separately.

<!-- BEGIN GENERATED LAYERS glm-5-1 -->
??? info "GLM-5.1 · 78 backbone layers"

    | Layer | Hidden in/out | FFN | Indexer / selection source | Main KV (owned per layer) |
    | --- | --- | --- | --- | --- |
    | 0 | `[B,S,6144]` | Dense F=12288 | full; L0 | own L0: `[B,T,512]` + `[B,T,64]` |
    | 1 | `[B,S,6144]` | Dense F=12288 | full; L1 | own L1: `[B,T,512]` + `[B,T,64]` |
    | 2 | `[B,S,6144]` | Dense F=12288 | full; L2 | own L2: `[B,T,512]` + `[B,T,64]` |
    | 3 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L3 | own L3: `[B,T,512]` + `[B,T,64]` |
    | 4 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L4 | own L4: `[B,T,512]` + `[B,T,64]` |
    | 5 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L5 | own L5: `[B,T,512]` + `[B,T,64]` |
    | 6 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L6 | own L6: `[B,T,512]` + `[B,T,64]` |
    | 7 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L7 | own L7: `[B,T,512]` + `[B,T,64]` |
    | 8 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L8 | own L8: `[B,T,512]` + `[B,T,64]` |
    | 9 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L9 | own L9: `[B,T,512]` + `[B,T,64]` |
    | 10 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L10 | own L10: `[B,T,512]` + `[B,T,64]` |
    | 11 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L11 | own L11: `[B,T,512]` + `[B,T,64]` |
    | 12 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L12 | own L12: `[B,T,512]` + `[B,T,64]` |
    | 13 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L13 | own L13: `[B,T,512]` + `[B,T,64]` |
    | 14 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L14 | own L14: `[B,T,512]` + `[B,T,64]` |
    | 15 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L15 | own L15: `[B,T,512]` + `[B,T,64]` |
    | 16 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L16 | own L16: `[B,T,512]` + `[B,T,64]` |
    | 17 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L17 | own L17: `[B,T,512]` + `[B,T,64]` |
    | 18 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L18 | own L18: `[B,T,512]` + `[B,T,64]` |
    | 19 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L19 | own L19: `[B,T,512]` + `[B,T,64]` |
    | 20 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L20 | own L20: `[B,T,512]` + `[B,T,64]` |
    | 21 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L21 | own L21: `[B,T,512]` + `[B,T,64]` |
    | 22 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L22 | own L22: `[B,T,512]` + `[B,T,64]` |
    | 23 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L23 | own L23: `[B,T,512]` + `[B,T,64]` |
    | 24 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L24 | own L24: `[B,T,512]` + `[B,T,64]` |
    | 25 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L25 | own L25: `[B,T,512]` + `[B,T,64]` |
    | 26 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L26 | own L26: `[B,T,512]` + `[B,T,64]` |
    | 27 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L27 | own L27: `[B,T,512]` + `[B,T,64]` |
    | 28 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L28 | own L28: `[B,T,512]` + `[B,T,64]` |
    | 29 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L29 | own L29: `[B,T,512]` + `[B,T,64]` |
    | 30 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L30 | own L30: `[B,T,512]` + `[B,T,64]` |
    | 31 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L31 | own L31: `[B,T,512]` + `[B,T,64]` |
    | 32 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L32 | own L32: `[B,T,512]` + `[B,T,64]` |
    | 33 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L33 | own L33: `[B,T,512]` + `[B,T,64]` |
    | 34 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L34 | own L34: `[B,T,512]` + `[B,T,64]` |
    | 35 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L35 | own L35: `[B,T,512]` + `[B,T,64]` |
    | 36 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L36 | own L36: `[B,T,512]` + `[B,T,64]` |
    | 37 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L37 | own L37: `[B,T,512]` + `[B,T,64]` |
    | 38 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L38 | own L38: `[B,T,512]` + `[B,T,64]` |
    | 39 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L39 | own L39: `[B,T,512]` + `[B,T,64]` |
    | 40 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L40 | own L40: `[B,T,512]` + `[B,T,64]` |
    | 41 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L41 | own L41: `[B,T,512]` + `[B,T,64]` |
    | 42 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L42 | own L42: `[B,T,512]` + `[B,T,64]` |
    | 43 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L43 | own L43: `[B,T,512]` + `[B,T,64]` |
    | 44 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L44 | own L44: `[B,T,512]` + `[B,T,64]` |
    | 45 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L45 | own L45: `[B,T,512]` + `[B,T,64]` |
    | 46 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L46 | own L46: `[B,T,512]` + `[B,T,64]` |
    | 47 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L47 | own L47: `[B,T,512]` + `[B,T,64]` |
    | 48 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L48 | own L48: `[B,T,512]` + `[B,T,64]` |
    | 49 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L49 | own L49: `[B,T,512]` + `[B,T,64]` |
    | 50 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L50 | own L50: `[B,T,512]` + `[B,T,64]` |
    | 51 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L51 | own L51: `[B,T,512]` + `[B,T,64]` |
    | 52 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L52 | own L52: `[B,T,512]` + `[B,T,64]` |
    | 53 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L53 | own L53: `[B,T,512]` + `[B,T,64]` |
    | 54 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L54 | own L54: `[B,T,512]` + `[B,T,64]` |
    | 55 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L55 | own L55: `[B,T,512]` + `[B,T,64]` |
    | 56 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L56 | own L56: `[B,T,512]` + `[B,T,64]` |
    | 57 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L57 | own L57: `[B,T,512]` + `[B,T,64]` |
    | 58 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L58 | own L58: `[B,T,512]` + `[B,T,64]` |
    | 59 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L59 | own L59: `[B,T,512]` + `[B,T,64]` |
    | 60 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L60 | own L60: `[B,T,512]` + `[B,T,64]` |
    | 61 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L61 | own L61: `[B,T,512]` + `[B,T,64]` |
    | 62 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L62 | own L62: `[B,T,512]` + `[B,T,64]` |
    | 63 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L63 | own L63: `[B,T,512]` + `[B,T,64]` |
    | 64 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L64 | own L64: `[B,T,512]` + `[B,T,64]` |
    | 65 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L65 | own L65: `[B,T,512]` + `[B,T,64]` |
    | 66 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L66 | own L66: `[B,T,512]` + `[B,T,64]` |
    | 67 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L67 | own L67: `[B,T,512]` + `[B,T,64]` |
    | 68 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L68 | own L68: `[B,T,512]` + `[B,T,64]` |
    | 69 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L69 | own L69: `[B,T,512]` + `[B,T,64]` |
    | 70 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L70 | own L70: `[B,T,512]` + `[B,T,64]` |
    | 71 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L71 | own L71: `[B,T,512]` + `[B,T,64]` |
    | 72 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L72 | own L72: `[B,T,512]` + `[B,T,64]` |
    | 73 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L73 | own L73: `[B,T,512]` + `[B,T,64]` |
    | 74 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L74 | own L74: `[B,T,512]` + `[B,T,64]` |
    | 75 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L75 | own L75: `[B,T,512]` + `[B,T,64]` |
    | 76 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L76 | own L76: `[B,T,512]` + `[B,T,64]` |
    | 77 | `[B,S,6144]` | MoE E=256, k=8, F=2048 | full; L77 | own L77: `[B,T,512]` + `[B,T,64]` |
<!-- END GENERATED LAYERS glm-5-1 -->

Compare [GLM-5.2 IndexShare](glm-5-2.md), then map shapes through [TP/EP](../ml/llm/inference/parallelism.md) and [layerwise KV movement](../ml/llm/inference/layerwise-inference.md).
