# 🔄 数据搬运：H2D、DMA 与异步复制

一个张量已经能被 CPU 访问，并不代表加速器内核已经能使用它。搬到设备全局内存是一环，把一个计算块加载进片上存储又是一环。区分两者，才能解释很多看似属于“计算”的瓶颈。

```mermaid
sequenceDiagram
    participant H as Host DRAM
    participant G as 设备全局内存
    participant L as 片上局部内存
    participant C as 计算单元
    H->>G: H2D 复制
    G->>L: 计算块 Copy-in
    L->>C: 提供操作数
    C->>L: 写入结果
    L->>G: 计算块 Copy-out
    G->>H: 需要时进行 D2H
```

H2D 完成不意味着所有操作数已经位于 AI Core 的局部缓冲区。内核调度和内部搬运仍然影响执行。

## 🔄 方向、引擎和路径分别是什么

| 术语 | 它说明什么 | 它没有说明什么 |
| --- | --- | --- |
| H2D / D2H | Host 到 Device / Device 到 Host 的方向 | 具体链路、引擎和同步语义 |
| D2D | 设备内或设备之间复制，取决于 API | 是否涉及另一张物理卡 |
| H2H | 两端都是 Host 内存 | 两端是否在同一台机器 |
| P2P | 对等设备访问或传输能力 | 最终选中哪条路径，是否发生回退 |
| DMA | 专用引擎搬运，避免 CPU 逐字节复制 | 初始化、协调和完成处理不需要 CPU |
| SDMA | 昇腾场景中的设备搬运引擎或路径术语 | 它与 AI Core 的 MTE 完全相同 |
| MTE | AI Core 存储层级之间的搬运及支持的转换 | 它是通用网络传输协议 |
| RDMA | 对通信所需内存准备完成后进行远程访问 | 任意 Device 指针都能自动被远端访问 |

常见 H2D 会经过 PCIe 等 Host I/O 路径。设备传输可能经过 PCIe、专用互联或网络。分析时分别回答：端点在哪里、物理路线是什么、使用什么 API、通过什么机制确认完成。

## 🧷 普通内存为什么会多一次缓冲区拷贝？ {#pageable-staging}

CANN 9.0 [官方《Host 锁页内存使用》](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0006.html)描述的 H2D 路径可以画成下面两条：

```mermaid
flowchart LR
    A[Pageable Host 应用缓冲区] -->|Host 端暂存拷贝| B[Runtime 暂存缓冲区]
    B -->|DMA| D[Device 内存]
    C[Pinned Host 应用缓冲区] -->|DMA| D
```

因此，**调用一个使用 DMA 的复制接口，不代表整条路径都省去了 Host 端拷贝**。在上述普通内存路径中，DMA 搬运的是后半段；暂存步骤仍有额外开销。图中的暂存缓冲区不同于 AI Core 的片上局部缓冲区。

锁页路径省去的是这一步暂存，Host→Device 传输仍然存在，不能称为完全没有数据复制。进一步推论，如果应用先把已有普通缓冲区手动复制到锁页缓冲区，这次 Host 端拷贝仍应计入成本；让数据直接生成在可复用的锁页缓冲区中，才可能省去它。内存类型、注册与释放方式见[锁页内存说明](memory-addressing.md#host-pinned-memory)。

### 同一个 Async 接口，返回时机也要看内存类型 {#cann-copy-completion}

官方[内存管理总述](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0004.html)将 `aclrtMemcpy` 与 `aclrtMemcpyAsync` 分别用于同步和异步拷贝。但 9.0 的 [`memcpy_async` 接口参考（含 `aclrtMemcpyAsync` C 原型）](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/runtimeapi/aclpythondevg_01_0121.html)还按 Host 内存类型细分返回语义：

| 调用场景 | 返回语义 |
| --- | --- |
| `aclrtMemcpy` | 同步拷贝；普通 Host 内存仍可能走暂存路径 |
| `aclrtMemcpyAsync` + 锁页 Host 内存 | 成功表示任务已提交，完成状态仍需另行确认 |
| `aclrtMemcpyAsync` + 非锁页 Host 内存 | 按所引 9.0 文档，复制任务完成后才返回 |

这说明 **Async 名称、Host 缓冲区类型、实际完成点必须一起判断**。所引接口还要求源、目的地址 64 字节对齐；产品形态、复制方向及版本支持应以对应接口文档为准。这里的 64 字节约束属于拷贝接口，前一章的 4KB 首地址约束属于 `aclrtHostRegisterV2`，两者不能互相替代。

锁页 H2D 的生命周期可以记为：`aclrtMallocHost → 填充数据 → aclrtMemcpyAsync → aclrtSynchronizeStream 确认完成 → 在所有使用者结束后复用或 aclrtFreeHost`。等待前不要改写或释放 DMA 仍可能读取的源缓冲区；D2H 则还需在完成后才能消费目标数据。

## 🧩 异步意味着必须管理生命周期

同步接口有文档规定的完成点，具体保证仍应查阅接口定义。异步接口可能在操作入队后就返回。应用必须让源、目标缓冲区持续有效，并在相应操作完成前避免冲突访问。

流水线的目标依赖关系是 `填充输入 → 复制 → 计算 → 消费输出`。Stream 或 Queue 保证部分顺序，Event 用于连接跨 Stream 依赖。提交后立即复用 Host 缓冲区，可能与 DMA 读取产生竞争。只测入队调用的时间，得到的是提交开销，不是传输带宽。

页锁定 Host 内存可以减少准备开销，并支持高效 DMA 路径；它也是有限资源，适合用有界内存池复用，而不是无限制锁页。重叠执行还要求硬件引擎支持、任务独立和依赖正确，仅仅使用异步接口并不能证明发生了重叠。

## 🧮 估算瓶颈

一阶模型是 $T(S)=L+S/B$，其中 $L$ 为固定开销，$S$ 为字节数，$B$ 为持续带宽。若 $L=10\,\mu s$、$B=20$ GB/s，复制 4 KB 约需 $10.2\,\mu s$，复制 20 MB 约需 $1.01$ ms。小传输适合通过批处理摊薄开销，大传输更容易暴露带宽限制。

有效带宽不会超过参与路径中最慢的内存、链路或引擎；争用还会进一步降低它。NUMA 放置、共享 PCIe 上行链路、内存注册和同步都会贡献开销。

记录方向、大小、次数、缓冲区类型、NUMA 放置、预热和完成计时方式。对比稳态大块复制与大量小复制，检查是否存在多余往返。继续阅读[性能测量](performance-tools.md)。

来源：[昇腾 AI Core 架构与 MTE](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/tbeaicpudevg/atlasopdev_10_0008.html)。文档描述特定版本的架构，实际引擎行为需要对应产品的编程指南。
