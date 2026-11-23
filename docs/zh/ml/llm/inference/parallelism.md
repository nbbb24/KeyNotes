# 🧭 大模型推理并行：TP、PP、DP、EP 与 CP

模型放不下、单请求太慢、并发上不去、长上下文 KV 爆满，是不同问题。并行策略决定**哪些权重、Token、层和缓存由哪些 Rank 持有，哪些中间结果必须通信**。本章讨论推理；训练中的梯度 AllReduce、反向流水线和优化器状态不能直接套进推理成本。

## 🗺️ 先按切分对象理解

| 策略 | 切什么 | 每个 Rank 主要持有什么 | 典型通信 / 代价 |
| --- | --- | --- | --- |
| TP，Tensor Parallelism | 一层内部的矩阵、头或特征 | 每层的权重分片，及相应 KV | 层内 AllReduce、AllGather 或 ReduceScatter；通信频繁 |
| PP，Pipeline Parallelism | 模型深度，若干层组成 Stage | 分配给该 Stage 的层和 KV | Stage 间传激活；启动/排空气泡与负载不均 |
| DP，Data Parallelism | 请求/批次 | 一个完整逻辑模型副本，可在内部使用 TP/PP | 普通 Dense 副本间通常不逐层交换激活；路由与负载均衡 |
| EP，Expert Parallelism | MoE 的专家集合 | 本 Rank 的专家；Attention 可采用另一种并行 | Token Dispatch / Combine，常见 All-to-All；路由偏斜 |
| CP，Context Parallelism | 同一请求的上下文/序列维 | 部分上下文激活或 KV，取决于实现 | Attention 需要跨分片组合信息；通信与长上下文收益权衡 |

TP、PP 的部署入口可查 [vLLM 并行文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)；DP 的含义还应结合模型类型看 [DP 部署文档](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)。上表是职责地图，不规定所有后端使用相同通信算法。

## 🧮 TP：让多个 Rank 一起算一层

用简化两层 FFN 说明：$Y=\phi(XW_1)W_2$，其中 $X\in\mathbb{R}^{N\times H}$，$W_1\in\mathbb{R}^{H\times F}$，$W_2\in\mathbb{R}^{F\times H}$。将 $W_1$ 按输出列切成 $p$ 片，将 $W_2$ 按对应输入行切分：

$$Z_i=\phi(XW_{1,i}),\qquad Y=\sum_{i=0}^{p-1}Z_iW_{2,i}.$$

每个 Rank 先算局部扩展特征，再生成一个局部输出贡献；若下一步需要复制的完整 $Y$，可用 AllReduce 求和。如果后续采用分片布局，也可能选用 ReduceScatter 等组合。真实 SwiGLU 还要一致切分 gate/up 分支；注意力、Embedding 与 LM Head 又有各自的切分方式。[Megatron-LM](https://arxiv.org/abs/1909.08053)提供了层内矩阵并行的基础思路。

TP 减少每卡权重和计算量，但每层的通信启动成本会进入逐 Token 的关键路径。通常先在高速互联域内评估 TP；跨节点也可以做，但应实际测量小消息延迟和有效带宽。TP 数增加后，矩阵过小、归约开销和同步等待可能抵消计算收益。

**KV 不一定严格除以 TP。** MHA/GQA 的 KV 头在可兼容切分时可分给不同 Rank；KV 头数少、TP 过大时，某些实现会复制 KV 头。MQA、MLA、量化或特殊布局另算。核对实际每卡 KV 形状，不能只用总大小除以卡数。

## 🏗️ PP：按层分 Stage，用不同请求填流水线

32 层模型做 PP=4 时，可示意为 Stage 0 持有层 0–7，Stage 1 持有层 8–15，依次类推。每个 Token 的激活必须依次走完各 Stage；对应层的历史 KV 通常留在所属 Stage，跨 Stage 传的是中间激活，不是每步迁移全部 KV。

```mermaid
flowchart LR
    R[微批 A / B / C] --> S0[Stage 0: 层 0-7]
    S0 -->|激活| S1[Stage 1: 层 8-15]
    S1 -->|激活| S2[Stage 2: 层 16-23]
    S2 -->|激活| S3[Stage 3: 层 24-31]
    S3 --> O[输出与采样]
```

单个请求的下一个自回归 Token 通常要等当前 Token 完成末层并采样，再反馈给首 Stage；因此不能把 PP=4 理解成单请求延迟直接缩为四分之一。不同请求/微批可以处于不同 Stage，提高系统吞吐。分层微批的思想可参见训练论文 [GPipe](https://arxiv.org/abs/1811.06965)；推理还需要考虑自回归反馈和在线调度。

在忽略通信、反馈且各 Stage 耗时相等的**单轮前向**模型中，$m$ 个独立微批经过 $p$ 个 Stage 需要 $m+p-1$ 个时间槽，平均利用率为：

$$U=\frac{m}{m+p-1}.$$

例如 $p=4,m=8$，利用率约 72.7%；这是理想流水线算例，不是在线 Decode 的通用利用率公式。Stage 应按运行时间和内存峰值平衡，而非只按层数均分；Embedding、MoE 和 LM Head 都可能造成倾斜。

## 📦 DP：增加请求吞吐和服务容量

Dense 推理中，DP=2 可以看作两个逻辑模型副本分别接不同请求。每个副本还可由 TP/PP 组成，因此 DP Rank 不一定等于“单卡装下整个模型”。普通副本间没有训练那样的梯度同步；它们各自分配活跃请求的 KV。

DP 适合在单副本已满足容量与时延要求后扩吞吐。吞吐不会保证线性增长：路由偏斜、公共前缀分散、不同请求长度和某副本的 KV 用尽，都会造成不均衡。把请求迁往另一副本还可能需要搬 KV 或重算，不能只改一个路由地址。

MoE 中的 Attention DP 可能与跨副本的专家组耦合，因此这类“DP”不代表完全独立的请求执行器。[vLLM DP 文档](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)明确区分 Dense 和 MoE 行为。

## 🧩 EP：Token 去找专家，再把结果送回来

一个 MoE 层的典型过程是 `Router → 按目标专家打包 → Dispatch → 专家 FFN → Combine → 按路由权重合并`。EP 切专家集合，TP 切矩阵内部；系统也可能让一个专家在更小组内做 TP。共享专家、Attention 与路由器的放置要单独分析。先读 [MoE 原理](../components/moe.md)。

例如 64 个专家分布在 EP=8 上，每个 Rank 持有 8 个专家。128 个 Token 做 top-2，会产生 256 个专家分配，平均每专家 4 个；实际分布可能很偏。对这一批，若隐藏维 4096、BF16，且所有分配均远程、不做去重，Dispatch 与 Combine 的隐藏向量载荷粗估为：

$$V\approx 2\times k\times N\times H\times b=4\ \text{MiB}.$$

这里两倍表示去与回；不含路由元数据、padding、协议开销。实际网络流量取决于本地专家比例、同一目标 Rank 的合并以及通信实现。Top-k 大、专家负载偏斜、批量过小都可能拖慢尾延迟；增加专家容量不等于每个 Token 都计算全部专家。[vLLM EP 文档](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)说明了专家组和 Attention 组的组合。

## 🔢 混合并行：先画 Rank 组，再计算卡数

对三个独立维度构成的 Dense 部署，$W=DP\times PP\times TP$。取 DP=2、PP=2、TP=4，共 16 个 Rank；示意编号 $r=((d\times2)+p)\times4+t$：

| 逻辑副本 | Stage 0 的 TP 组 | Stage 1 的 TP 组 |
| --- | --- | --- |
| DP 0 | Rank 0–3，层 0–15 | Rank 4–7，层 16–31 |
| DP 1 | Rank 8–11，层 0–15 | Rank 12–15，层 16–31 |

DP 0 的一个请求先在 0–3 内做层内并行，再进入 4–7。Stage 间具体由哪些 Rank 发送，取决于激活布局与后端，不能从全局编号直接推断物理链路。数字编号只是教学约定。

**EP 不能未经确认就再乘一次。** 例如 vLLM 所引 EP 部署方式中，专家组大小为 `TP × DP`；TP=2、DP=4、PP=1 仍是 8 张设备，专家组 EP=8，而非再乘 8 变成 64 张。其他框架的组定义可能不同，先核对配置语义再数卡。[vLLM EP 分组定义](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)。

## 🧠 内存账本：权重、KV、激活分别看

对均匀切分的 Dense 模型，可先估算每卡权重 $M_W\approx P_{param}b_W/(TP\times PP)$。DP 复制逻辑模型，不会继续降低每个副本的每卡权重。另加量化 scale、复制层、工作区、图捕获和分配器预留；MoE 应分开计算共享权重和专家权重。

普通 KV 载荷为 $L\times2\times H_{KV}\times D_{head}\times T\times b_{KV}$。32 层、8 个 KV 头、头维 128、2048 Token、BF16 的全模型单序列 KV 为 256 MiB。若 PP=2 且 TP=4 能完整分 KV 头，每卡该序列约 32 MiB；DP=2 时，同一序列一般只归属其中一个副本，不能再除以 2。[KV 大小](kv-cache.md)与[物理布局](layers-pages.md)解释更多边界。

## 📏 CP、SP 与 PD：与前面是什么关系

CP 把同一序列的上下文分给多个 Rank，Attention 仍需组合其他分片的信息；长 Prefill 与 Decode 的具体分片和通信方案可能不同。SP 是依框架而变化的术语；Megatron 的经典 SP 主要切分部分算子的序列维激活，不能把它等同于完整 CP 或通用“KV 除以 SP”。[Megatron CP 说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)是训练背景的定义参考，推理支持需另看实际框架。

PD 分离按 Prefill/Decode 服务阶段分工；PP 按模型层分工。P 与 D 内部可以各自采用 TP/PP/DP/EP；两边切分不同时，KV 交接还可能需要聚合、重分片或布局转换，前提是连接器支持该组合。Layerwise 则是[逐层执行与搬运的粒度](layerwise-inference.md)，不额外增加一个必然独立的卡数维度。

## 📊 选择与验证

先测单副本的权重/KV/激活峰值及 TTFT、ITL；再决定是 TP/PP 解决容量，DP 增加吞吐，EP 放置专家，还是 CP 处理长上下文。将 TP 归约、PP 激活、EP Dispatch/Combine、PD KV 迁移分别计时，结合[通信路径](../../../infra/interconnects.md)观察链路争用。记录模型、硬件、并行组、输入/输出长度和并发；只报告“用了 16 卡”无法解释收益。
