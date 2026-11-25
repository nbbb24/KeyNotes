# 🧠 DeepSeek-R1: 61 MLA layers with equal per-layer caches

The full DeepSeek-R1 backbone has **61 layers**: three dense-FFN layers followed by 58 MoE layers. All layers use the same MLA cache dimensions. For one 128-token BF16 block, each layer contributes **128 KiB of latent KV plus 16 KiB of RoPE keys**.

??? note "Config, implementation, and scope"

    This page binds to the [official config](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/config.json) at `deepseek-ai/DeepSeek-R1@56d4cbbb4d29`, preserving its [original config bytes](../../../../assets/model-specs/deepseek-r1.config.json). Expanded and absorbed MLA cache layouts follow the pinned [official DeepSeek-V3 implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py), corresponding to the config's `DeepseekV3ForCausalLM` architecture. Shapes are logical, before TP/EP partitioning; weights use `[out,in]`. These dimensions do not describe R1 distillations based on Qwen or Llama.

## 🗺️ Global dimensions and layer responsibilities

| Config | Value |
| --- | --- |
| Vocabulary / hidden width / backbone depth | 129280 / 7168 / 61 |
| FFN placement | Layers 0–2 dense; layers 3–60 MoE |
| Dense / individual expert intermediate width | 18432 / 2048 |
| Routed experts / selected per token / shared experts | 256 / 8 / 1 |
| Query heads / NoPE + RoPE / value head width | 128 / 128 + 64 / 128 |
| Query low rank / latent KV width | 1536 / 512 |
| Maximum positions / additional prediction layers | 163840 / 1, outside the 61-layer backbone |

Each layer applies normalization, MLA, and a residual addition, then normalization, dense FFN or MoE, and another residual addition. Hidden input and output remain `[B,S,7168]`. Changing the FFN type does not change the main cache size for the same token block. R1 has neither GLM's separate DSA indexer keys nor V4's alternating CSA/HCA compression ratios.

Embedding and untied LM-head weights are each `[129280,7168]`. Keeping logits for all current positions yields `[B,S,129280]`.

## 📐 MLA projections and activations

The table uses module names from the cited DeepSeek reference. Other implementations may use equivalent names such as `q_a_proj`.

| Module | Logical weight shape | Output or logical activation shape |
| --- | --- | --- |
| `wq_a` / query RMSNorm | `[1536,7168]` / `[1536]` | `[B,S,1536]` |
| `wq_b` | `[24576,1536]` | `[B,S,128,192]` |
| Query NoPE / RoPE slices | No separate projection | `[B,S,128,128]` / `[B,S,128,64]` |
| `wkv_a` | `[576,7168]` | Latent KV `[B,S,512]` plus shared RoPE K `[B,S,64]` |
| KV RMSNorm | `[512]` | `[B,S,512]` |
| `wkv_b` | `[32768,512]` | Logical K-NoPE `[B,S,128,128]` plus V `[B,S,128,128]` |
| K after concatenating shared RoPE | No additional weight | `[B,S,128,192]` |
| Attention output with merged heads | No additional weight | `[B,S,16384]` |
| `wo` | `[7168,16384]` | `[B,S,7168]` |

`24576 = 128 × (128 + 64)`, `32768 = 128 × (128 + 128)`, and `16384 = 128 × 128`. Having 128 query heads does not require 128 persistent latent-KV copies. Absorbed MLA retains shared latent vectors and shared RoPE keys, combining them with projection weights for per-head scoring and output computation.

## 🧩 Dense and MoE FFNs

In layers 0–2, dense gate/up weights are each `[18432,7168]`, down is `[7168,18432]`, and intermediate activations are `[B,S,18432]`. In layers 3–60, the router weight is `[256,7168]` and selects eight routed experts per token.

Individual expert gate/up weights are each `[2048,7168]`; down is `[7168,2048]`. Expert intermediate activations are `[N_e,2048]`, where `N_e` counts tokens actually routed to that expert. The shared expert has the same dimensions as one routed expert and runs for all tokens in the layer. Expert activations do not enter the KV-transfer byte formula below.

## 📦 Why `61 × (128k + 16k)`

Here `k` means KiB. Assume absorbed MLA, BF16 cache storage, and 128 original tokens per key:

| Cache owned by each model layer | Logical block shape, omitting batch | Payload bytes |
| --- | --- | ---: |
| Latent KV | `[128,512]` | `128 × 512 × 2 = 128 KiB` |
| RoPE K | `[128,64]` | `128 × 64 × 2 = 16 KiB` |

Each layer contributes `144 KiB`; all 61 contribute `8784 KiB = 8.578125 MiB = 8,994,816 bytes`. **Every model layer has the same total cache size. The 128k and 16k components coexist within each layer.**

```text
Layer 0:  [128k latent KV, 16k RoPE K]
Layer 1:  [128k latent KV, 16k RoPE K]
…
Layer 60: [128k latent KV, 16k RoPE K]
```

The [pinned MMC benchmark](https://github.com/nbbb24/self_test/blob/2a7ca1333ac3ce5710764ba84d75850b1586b520/mmc_benchmark/model_profiles.py) splits each layer into two segments. One key has sizes `[131072,16384] × 61`, totaling **122 buffer segments**. Equal-sized segments can share two `uint8` tensors, `[61,B,131072]` and `[61,B,16384]`, where B counts keys in a batch IO. Combining both components into one buffer per layer gives 61 segments of 144 KiB each: equal payload, different address segmentation.

The 128k segment is not a full K tensor, and the 16k segment is not a full V tensor. Together they represent compressed latent KV and the separate RoPE key. Do not add another K/V factor of two; the two bytes per BF16 element are already included.

## 🔄 Physical layouts and deployment boundaries

The reference's `naive` path stores expanded per-head K/V. Its `absorb` path stores `[B,T,512]` and `[B,T,64]`; the formula above assumes this path. FP8 weight metadata does not establish FP8 cache storage.

The formula excludes quantization scales, padding, alignment, page tables, workspaces, MTP, and runtime replicas. With PP, count only layers owned by the stage. Whether TP replicates shared latent KV depends on the backend; do not divide the entire formula mechanically by the TP degree.

The benchmark models random bytes and scattered-address put/get calls without running attention or FFNs. Compare all models under the same assumptions in the [structure overview](index.md).

## 🪜 Complete backbone layer map

The common MLA and FFN shapes are defined above. Layer indices start at zero; the additional prediction layer is excluded.

<!-- BEGIN GENERATED LAYERS deepseek-r1 -->
??? info "DeepSeek-R1 · 61 backbone layers"

    | Layer | Hidden in/out | FFN | Main KV (owned per layer) |
    | --- | --- | --- | --- |
    | 0 | `[B,S,7168]` | Dense F=18432 | own L0: `[B,T,512]` + `[B,T,64]` |
    | 1 | `[B,S,7168]` | Dense F=18432 | own L1: `[B,T,512]` + `[B,T,64]` |
    | 2 | `[B,S,7168]` | Dense F=18432 | own L2: `[B,T,512]` + `[B,T,64]` |
    | 3 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L3: `[B,T,512]` + `[B,T,64]` |
    | 4 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L4: `[B,T,512]` + `[B,T,64]` |
    | 5 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L5: `[B,T,512]` + `[B,T,64]` |
    | 6 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L6: `[B,T,512]` + `[B,T,64]` |
    | 7 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L7: `[B,T,512]` + `[B,T,64]` |
    | 8 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L8: `[B,T,512]` + `[B,T,64]` |
    | 9 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L9: `[B,T,512]` + `[B,T,64]` |
    | 10 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L10: `[B,T,512]` + `[B,T,64]` |
    | 11 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L11: `[B,T,512]` + `[B,T,64]` |
    | 12 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L12: `[B,T,512]` + `[B,T,64]` |
    | 13 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L13: `[B,T,512]` + `[B,T,64]` |
    | 14 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L14: `[B,T,512]` + `[B,T,64]` |
    | 15 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L15: `[B,T,512]` + `[B,T,64]` |
    | 16 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L16: `[B,T,512]` + `[B,T,64]` |
    | 17 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L17: `[B,T,512]` + `[B,T,64]` |
    | 18 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L18: `[B,T,512]` + `[B,T,64]` |
    | 19 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L19: `[B,T,512]` + `[B,T,64]` |
    | 20 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L20: `[B,T,512]` + `[B,T,64]` |
    | 21 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L21: `[B,T,512]` + `[B,T,64]` |
    | 22 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L22: `[B,T,512]` + `[B,T,64]` |
    | 23 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L23: `[B,T,512]` + `[B,T,64]` |
    | 24 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L24: `[B,T,512]` + `[B,T,64]` |
    | 25 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L25: `[B,T,512]` + `[B,T,64]` |
    | 26 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L26: `[B,T,512]` + `[B,T,64]` |
    | 27 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L27: `[B,T,512]` + `[B,T,64]` |
    | 28 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L28: `[B,T,512]` + `[B,T,64]` |
    | 29 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L29: `[B,T,512]` + `[B,T,64]` |
    | 30 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L30: `[B,T,512]` + `[B,T,64]` |
    | 31 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L31: `[B,T,512]` + `[B,T,64]` |
    | 32 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L32: `[B,T,512]` + `[B,T,64]` |
    | 33 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L33: `[B,T,512]` + `[B,T,64]` |
    | 34 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L34: `[B,T,512]` + `[B,T,64]` |
    | 35 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L35: `[B,T,512]` + `[B,T,64]` |
    | 36 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L36: `[B,T,512]` + `[B,T,64]` |
    | 37 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L37: `[B,T,512]` + `[B,T,64]` |
    | 38 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L38: `[B,T,512]` + `[B,T,64]` |
    | 39 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L39: `[B,T,512]` + `[B,T,64]` |
    | 40 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L40: `[B,T,512]` + `[B,T,64]` |
    | 41 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L41: `[B,T,512]` + `[B,T,64]` |
    | 42 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L42: `[B,T,512]` + `[B,T,64]` |
    | 43 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L43: `[B,T,512]` + `[B,T,64]` |
    | 44 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L44: `[B,T,512]` + `[B,T,64]` |
    | 45 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L45: `[B,T,512]` + `[B,T,64]` |
    | 46 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L46: `[B,T,512]` + `[B,T,64]` |
    | 47 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L47: `[B,T,512]` + `[B,T,64]` |
    | 48 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L48: `[B,T,512]` + `[B,T,64]` |
    | 49 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L49: `[B,T,512]` + `[B,T,64]` |
    | 50 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L50: `[B,T,512]` + `[B,T,64]` |
    | 51 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L51: `[B,T,512]` + `[B,T,64]` |
    | 52 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L52: `[B,T,512]` + `[B,T,64]` |
    | 53 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L53: `[B,T,512]` + `[B,T,64]` |
    | 54 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L54: `[B,T,512]` + `[B,T,64]` |
    | 55 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L55: `[B,T,512]` + `[B,T,64]` |
    | 56 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L56: `[B,T,512]` + `[B,T,64]` |
    | 57 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L57: `[B,T,512]` + `[B,T,64]` |
    | 58 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L58: `[B,T,512]` + `[B,T,64]` |
    | 59 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L59: `[B,T,512]` + `[B,T,64]` |
    | 60 | `[B,S,7168]` | MoE E=256, k=8, F=2048 | own L60: `[B,T,512]` + `[B,T,64]` |
<!-- END GENERATED LAYERS deepseek-r1 -->

For reinforcement learning, R1-Zero, and distillation, read the [DeepSeek-R1 model guide](../models/deepseek-r1.md).
