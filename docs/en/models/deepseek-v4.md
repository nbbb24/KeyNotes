# 🧠 DeepSeek-V4 Flash / Pro: mHC, compressed attention, and expert shapes

This page separately pins the official [Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) and [Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json). Module shapes are derived from the corresponding [official reference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py). Flash-0731, Pro-0813, and moving API aliases require separate verification.

## 🗺️ Global dimensions

| Setting | V4-Flash | V4-Pro |
| --- | ---: | ---: |
| Backbone layers / H | 43 / 4096 | 61 / 7168 |
| Vocabulary | 129280 | 129280 |
| Q heads / head width | 64 / 512 | 128 / 512 |
| NoPE / RoPE widths | 448 / 64 | 448 / 64 |
| Q rank | 1024 | 1536 |
| O groups / rank per group | 8 / 1024 | 16 / 1024 |
| Routed experts / top-k / shared experts | 256 / 6 / 1 | 384 / 6 / 1 |
| Per-expert FFN width F | 2048 | 3072 |
| Hash-routed layers | 0, 1, 2 | 0, 1, 2 |
| Sliding window / mHC expansion | 128 / 4 | 128 / 4 |
| Index heads × width / top-k | 64 × 128 / 512 | 64 × 128 / 1024 |
| Maximum positions / extra MTP layers | 1048576 / 1 | 1048576 / 1 |

The model card reports Flash backbone parameters as 284B with 13B active per token, and Pro as 1.6T with 49B active. Stored experts are not all activated every step. See the [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md) and [V4 report](https://arxiv.org/abs/2606.19348).

## 🪜 Residual streams versus sublayer input

Embedding produces `[B,S,H]`, expanded into `[B,S,4,H]` before the backbone. mHC collapses the streams into `[B,S,H]` for attention/FFN, then merges the result into four-stream residuals. Inter-layer shapes are `[B,S,4,4096]` for Flash and `[B,S,4,7168]` for Pro.

Each `hc_attn_fn` and `hc_ffn_fn` has shape `[24,4H]`, since `(2+4)×4=24`: `[24,16384]` for Flash and `[24,28672]` for Pro. Coefficients split into four pre-weights, four post-weights, and a 4×4 combination matrix. These control dimensions do not change expert input width H.

## 📐 Main attention projections

| Module | Flash weight `[out,in]` | Pro weight `[out,in]` | Logical output |
| --- | --- | --- | --- |
| `wq_a` | `[1024,4096]` | `[1536,7168]` | `[B,S,Rq]` |
| `wq_b` | `[32768,1024]` | `[65536,1536]` | `[B,S,A,512]` |
| `wkv` | `[512,4096]` | `[512,7168]` | Shared latent KV `[B,S,512]` |
| Grouped `wo_a`, storage view | `[8192,4096]` | `[16384,4096]` | `[B,S,G,1024]` |
| Grouped `wo_a`, operation view | `[8,1024,4096]` | `[16,1024,4096]` | Each group handles only its heads |
| `wo_b` | `[4096,8192]` | `[7168,16384]` | `[B,S,H]` |
| Embedding / untied head | Each `[129280,4096]` | Each `[129280,7168]` | `[B,S,H]` / selected-position logits |

`wo_a` is grouped/block-diagonal, **not an arbitrary dense `[G×1024,A×512]` matrix**. Each group's input width is `A×512/G=4096`. Q norm has width Rq, KV norm 512, and layer norms H. Reference attention also inverse-rotates the output's RoPE slice before O projection.

## 🔄 Layer placement of SWA, CSA, and HCA

Every layer has a local SWA path. `compress_ratios[l]` determines the extra global path:

| Model / layer indices | Ratio | Attention |
| --- | ---: | --- |
| Flash 0, 1 | 0 | SWA only |
| Flash even 2–42 | 4 | SWA + CSA; indexer selects compressed positions |
| Flash odd 3–41 | 128 | SWA + HCA; no CSA indexer |
| Pro 0, 1 and odd 3–59 | 128 | SWA + HCA |
| Pro even 2–60 | 4 | SWA + CSA |

Flash has 2 pure SWA, 21 CSA, and 20 HCA backbone layers. Pro has 30 CSA and 31 HCA layers. The ratio arrays include an extra MTP entry: Flash index 43 and Pro index 61 are zero and do not add backbone layers.

For compressed layers, main logical capacity is `[B,128+floor(T_max/r),512]`; r=0 has only `[B,128,512]`. Effective compressed length follows completed groups. CSA also holds compressed index keys of capacity `[B,floor(T_max/4),128]`. Top-k bounds attended positions rather than deleting all unselected historical state.

The ratio-4 main compressor uses overlapping gated pooling: reference `wkv/wgate` output width is 1024, versus 512 at ratio 128. These are separate modules from local per-layer `wkv`; identical leaf names do not establish shared weights.

## 🧩 Hash-routed layers are still MoE

| Module | Flash | Pro |
| --- | --- | --- |
| Router weight | `[256,4096]` | `[384,7168]` |
| Selected IDs / weights | `[N,6]` | `[N,6]` |
| Token-to-expert table, layers 0–2 | Integer `[129280,6]` | Same |
| One expert gate / up | Each `[2048,4096]` | Each `[3072,7168]` |
| One expert down | `[4096,2048]` | `[7168,3072]` |
| Shared expert | One extra expert with the same F/H | Same |

The first three layers derive selected expert IDs from token IDs; expert computation still occurs. They are not GLM's three initial dense layers. Later layers select by score. FP4 packing, scales, fusion, and TP alter physical storage beyond these logical shapes.

## 🪜 Complete maps and deployment boundaries

Global KV is owned by its layer here, unlike [V4.1 cross-layer sharing](deepseek-v4-1.md). Reference buffers can retain BF16 storage while simulating quantized values, so their allocated bytes are not a production packed-cache budget.

<!-- BEGIN GENERATED LAYERS deepseek-v4-flash -->
??? info "DeepSeek-V4-Flash · 43 backbone layers"

    | Layer | Four-stream residual in/out | Attention | FFN | Global KV owner / valid length | Indexer source |
    | --- | --- | --- | --- | --- | --- |
    | 0 | `[B,S,4,4096]` | SWA | Hash-MoE | — | — |
    | 1 | `[B,S,4,4096]` | SWA | Hash-MoE | — | — |
    | 2 | `[B,S,4,4096]` | SWA + CSA, r=4 | Hash-MoE | own L2: `[B,floor(T/4),512]` | own L2 |
    | 3 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L3: `[B,floor(T/128),512]` | — |
    | 4 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L4: `[B,floor(T/4),512]` | own L4 |
    | 5 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L5: `[B,floor(T/128),512]` | — |
    | 6 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L6: `[B,floor(T/4),512]` | own L6 |
    | 7 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L7: `[B,floor(T/128),512]` | — |
    | 8 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L8: `[B,floor(T/4),512]` | own L8 |
    | 9 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L9: `[B,floor(T/128),512]` | — |
    | 10 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L10: `[B,floor(T/4),512]` | own L10 |
    | 11 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L11: `[B,floor(T/128),512]` | — |
    | 12 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L12: `[B,floor(T/4),512]` | own L12 |
    | 13 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L13: `[B,floor(T/128),512]` | — |
    | 14 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L14: `[B,floor(T/4),512]` | own L14 |
    | 15 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L15: `[B,floor(T/128),512]` | — |
    | 16 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L16: `[B,floor(T/4),512]` | own L16 |
    | 17 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L17: `[B,floor(T/128),512]` | — |
    | 18 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L18: `[B,floor(T/4),512]` | own L18 |
    | 19 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L19: `[B,floor(T/128),512]` | — |
    | 20 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L20: `[B,floor(T/4),512]` | own L20 |
    | 21 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L21: `[B,floor(T/128),512]` | — |
    | 22 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L22: `[B,floor(T/4),512]` | own L22 |
    | 23 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L23: `[B,floor(T/128),512]` | — |
    | 24 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L24: `[B,floor(T/4),512]` | own L24 |
    | 25 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L25: `[B,floor(T/128),512]` | — |
    | 26 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L26: `[B,floor(T/4),512]` | own L26 |
    | 27 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L27: `[B,floor(T/128),512]` | — |
    | 28 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L28: `[B,floor(T/4),512]` | own L28 |
    | 29 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L29: `[B,floor(T/128),512]` | — |
    | 30 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L30: `[B,floor(T/4),512]` | own L30 |
    | 31 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L31: `[B,floor(T/128),512]` | — |
    | 32 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L32: `[B,floor(T/4),512]` | own L32 |
    | 33 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L33: `[B,floor(T/128),512]` | — |
    | 34 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L34: `[B,floor(T/4),512]` | own L34 |
    | 35 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L35: `[B,floor(T/128),512]` | — |
    | 36 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L36: `[B,floor(T/4),512]` | own L36 |
    | 37 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L37: `[B,floor(T/128),512]` | — |
    | 38 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L38: `[B,floor(T/4),512]` | own L38 |
    | 39 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L39: `[B,floor(T/128),512]` | — |
    | 40 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L40: `[B,floor(T/4),512]` | own L40 |
    | 41 | `[B,S,4,4096]` | SWA + HCA, r=128 | MoE | own L41: `[B,floor(T/128),512]` | — |
    | 42 | `[B,S,4,4096]` | SWA + CSA, r=4 | MoE | own L42: `[B,floor(T/4),512]` | own L42 |
<!-- END GENERATED LAYERS deepseek-v4-flash -->

<!-- BEGIN GENERATED LAYERS deepseek-v4-pro -->
??? info "DeepSeek-V4-Pro · 61 backbone layers"

    | Layer | Four-stream residual in/out | Attention | FFN | Global KV owner / valid length | Indexer source |
    | --- | --- | --- | --- | --- | --- |
    | 0 | `[B,S,4,7168]` | SWA + HCA, r=128 | Hash-MoE | own L0: `[B,floor(T/128),512]` | — |
    | 1 | `[B,S,4,7168]` | SWA + HCA, r=128 | Hash-MoE | own L1: `[B,floor(T/128),512]` | — |
    | 2 | `[B,S,4,7168]` | SWA + CSA, r=4 | Hash-MoE | own L2: `[B,floor(T/4),512]` | own L2 |
    | 3 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L3: `[B,floor(T/128),512]` | — |
    | 4 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L4: `[B,floor(T/4),512]` | own L4 |
    | 5 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L5: `[B,floor(T/128),512]` | — |
    | 6 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L6: `[B,floor(T/4),512]` | own L6 |
    | 7 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L7: `[B,floor(T/128),512]` | — |
    | 8 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L8: `[B,floor(T/4),512]` | own L8 |
    | 9 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L9: `[B,floor(T/128),512]` | — |
    | 10 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L10: `[B,floor(T/4),512]` | own L10 |
    | 11 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L11: `[B,floor(T/128),512]` | — |
    | 12 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L12: `[B,floor(T/4),512]` | own L12 |
    | 13 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L13: `[B,floor(T/128),512]` | — |
    | 14 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L14: `[B,floor(T/4),512]` | own L14 |
    | 15 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L15: `[B,floor(T/128),512]` | — |
    | 16 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L16: `[B,floor(T/4),512]` | own L16 |
    | 17 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L17: `[B,floor(T/128),512]` | — |
    | 18 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L18: `[B,floor(T/4),512]` | own L18 |
    | 19 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L19: `[B,floor(T/128),512]` | — |
    | 20 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L20: `[B,floor(T/4),512]` | own L20 |
    | 21 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L21: `[B,floor(T/128),512]` | — |
    | 22 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L22: `[B,floor(T/4),512]` | own L22 |
    | 23 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L23: `[B,floor(T/128),512]` | — |
    | 24 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L24: `[B,floor(T/4),512]` | own L24 |
    | 25 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L25: `[B,floor(T/128),512]` | — |
    | 26 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L26: `[B,floor(T/4),512]` | own L26 |
    | 27 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L27: `[B,floor(T/128),512]` | — |
    | 28 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L28: `[B,floor(T/4),512]` | own L28 |
    | 29 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L29: `[B,floor(T/128),512]` | — |
    | 30 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L30: `[B,floor(T/4),512]` | own L30 |
    | 31 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L31: `[B,floor(T/128),512]` | — |
    | 32 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L32: `[B,floor(T/4),512]` | own L32 |
    | 33 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L33: `[B,floor(T/128),512]` | — |
    | 34 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L34: `[B,floor(T/4),512]` | own L34 |
    | 35 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L35: `[B,floor(T/128),512]` | — |
    | 36 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L36: `[B,floor(T/4),512]` | own L36 |
    | 37 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L37: `[B,floor(T/128),512]` | — |
    | 38 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L38: `[B,floor(T/4),512]` | own L38 |
    | 39 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L39: `[B,floor(T/128),512]` | — |
    | 40 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L40: `[B,floor(T/4),512]` | own L40 |
    | 41 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L41: `[B,floor(T/128),512]` | — |
    | 42 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L42: `[B,floor(T/4),512]` | own L42 |
    | 43 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L43: `[B,floor(T/128),512]` | — |
    | 44 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L44: `[B,floor(T/4),512]` | own L44 |
    | 45 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L45: `[B,floor(T/128),512]` | — |
    | 46 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L46: `[B,floor(T/4),512]` | own L46 |
    | 47 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L47: `[B,floor(T/128),512]` | — |
    | 48 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L48: `[B,floor(T/4),512]` | own L48 |
    | 49 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L49: `[B,floor(T/128),512]` | — |
    | 50 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L50: `[B,floor(T/4),512]` | own L50 |
    | 51 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L51: `[B,floor(T/128),512]` | — |
    | 52 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L52: `[B,floor(T/4),512]` | own L52 |
    | 53 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L53: `[B,floor(T/128),512]` | — |
    | 54 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L54: `[B,floor(T/4),512]` | own L54 |
    | 55 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L55: `[B,floor(T/128),512]` | — |
    | 56 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L56: `[B,floor(T/4),512]` | own L56 |
    | 57 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L57: `[B,floor(T/128),512]` | — |
    | 58 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L58: `[B,floor(T/4),512]` | own L58 |
    | 59 | `[B,S,4,7168]` | SWA + HCA, r=128 | MoE | own L59: `[B,floor(T/128),512]` | — |
    | 60 | `[B,S,4,7168]` | SWA + CSA, r=4 | MoE | own L60: `[B,floor(T/4),512]` | own L60 |
<!-- END GENERATED LAYERS deepseek-v4-pro -->

PP boundaries can carry four-stream residuals. TP must respect head groups, while EP distinguishes global and local expert sets. See [parallelism](../ml/llm/inference/parallelism.md) and [layerwise transfer](../ml/llm/inference/layerwise-inference.md).
