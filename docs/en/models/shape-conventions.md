# 📐 Shape conventions: weights, activations, and caches

One model can have multiple layouts, precisions, and parallel partitions. This reference starts with **unpartitioned logical dimensions** and then explains implementation views. Unless stated otherwise, numbers exclude padding, scales, alignment, page metadata, and workspaces.

## 🧩 Symbols and matrix orientation

| Symbol | Meaning |
| --- | --- |
| B | Number of sequences in a batch |
| S | Tokens computed in this invocation; often 1 for ordinary decode, larger for prefill |
| T | Retained/accessible history length; inclusion of the current position follows the interface |
| N = B × S | Flattened current tokens |
| H / F | Model hidden width / one FFN's intermediate width |
| A / D | Main query-head count / head width; do not assume D = H/A |
| E / k | Routed expert count / selected routed experts per token |
| r | Sequence compression ratio; r=0 means a window-only path in these DeepSeek configs |

For `Linear(in, out)`, weights are **`[out,in]`**, matching PyTorch `weight`. Input `[B,S,in]` produces `[B,S,out]`, mathematically $Y=XW^T$. Papers may use `[in,out]`; reconcile orientation before comparing. See [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).

GLM `q_b_proj.weight=[16384,2048]` maps `[B,S,2048]` to `[B,S,16384]`, then reshapes into `[B,S,64,256]`. A backend may instead use `[B,64,S,256]`: the element count agrees, while strides and kernel interfaces differ.

For prefill with B=2 and S=128, GLM layer input is `[2,128,6144]` and Q is `[2,128,64,256]`. V4.1 residuals are `[2,128,4,5120]` and main Q is `[2,128,64,512]`. Ordinary one-step decode changes S to 1 without reducing historical cache length T to 1.

## 🧠 Four different shape questions

| Object | Example | Purpose |
| --- | --- | --- |
| Logical weight | Expert gate `[2304,5120]` | Projection and parameter count |
| Quantized physical weight | Packing two FP4 values per byte can yield payload `[2304,2560]` | Storage; scales and backend layout remain separate |
| Activation | Expert input `[N_e,5120]`, where N_e is its actual assignment count | Compute and dispatch/combine |
| Cache | Shared main KV `[B,floor(T/r),512]` | Persistent state across iterations |

Reference code using fake quantization can store quantized numerical values back in a BF16 buffer. **Simulated low-precision values do not establish packed memory usage.** A dtype name alone is not a production HBM budget.

## 🪜 Layers, heads, and residual streams

GLM inter-layer hidden states are `[B,S,6144]`. DeepSeek V4/V4.1 mHC maintains `[B,S,4,H]` residuals, collapsed to `[B,S,H]` at attention/FFN inputs. `hc_mult=4` does not make every expert matrix accept `4H`, or imply four independent Transformer executions per sublayer.

MLA, shared latent KV, sliding windows, and cross-layer sharing require explicit cache accounting rather than blindly applying `2 × layers × KV heads × head width × T`. Identify the stored representation, its owning layer, and any aliases. See [KV fundamentals](../ml/llm/inference/kv-cache.md).

## 🔄 From global to per-device shapes

Column-parallel projections may partition `out`; row-parallel projections partition `in` with corresponding reductions. EP primarily partitions experts; PP partitions layers. Norms, shared latent KV, Engram tables, and grouped output projections need not use the same partition.

For example, head-sharded TP=4 can turn GLM `q_b` into local weight `[4096,2048]` and Q `[B,S,16,256]`. It does not authorize dividing `kv_a`, every norm, and every cache by four. DeepSeek grouped O projections also constrain compatible group/TP layouts. Check [parallel groups](../ml/llm/inference/parallelism.md) and the backend.

Weight dimensions determine logical parameter count; bytes also depend on precision, scales, replication, and partitioning. `[2304,5120]` has 11,796,480 parameters, with a 5,898,240-byte FP4 payload before scales. These are calculations under explicit assumptions, not measurements.
