# 📖 Infra 术语表：先判断它属于哪一层

画数据传输路径时，可以随时查阅本页。每遇到一个术语，先判断它描述的是物理对象、地址、操作、网络角色，还是软件抽象。很多排障误区，来自把不同类别的术语当成可以互换的概念。

## 🧭 硬件与身份

| 术语 | 含义 | 排查时要问 |
| --- | --- | --- |
| Die | 一块物理硅片 | 这里是物理 Die 编号还是逻辑编号？ |
| Package | 将一个或多个 Die 组装成的封装 | 包含几个 Die，内部怎样连接？ |
| Card | 承载器件的板卡 | 与运行时 Device 如何对应？ |
| NPU | 针对神经网络、张量计算优化的处理器 | 具体型号和架构代际是什么？ |
| Device | 运行时可见的加速器对象 | 这个编号对应哪个物理资源？ |
| Rank | 通信组中参与者的编号 | 对应哪个进程、主机、设备和端点？ |
| AI Core | 加速器计算资源 | 采用哪种矩阵、向量计算架构？ |
| AICPU | 设备侧辅助处理器 | 哪些任务在这里执行，而不是在 AI Core 上？ |

## 🧠 存储与地址

| 术语 | 含义与区别 |
| --- | --- |
| Host Memory | CPU 侧内存，通常是 DRAM |
| Device Global Memory / GM | 设备全局存储，具体产品可能使用 HBM 或 DDR |
| Local Memory | 靠近计算单元的小容量片上缓冲区，例如 L0、L1、UB |
| HBM | High Bandwidth Memory；容量和带宽是不同限制 |
| VA / PA | 虚拟地址 / 物理地址 |
| IOVA / IOMMU | 设备 I/O 虚拟地址 / 对应的地址转换与隔离机制 |
| MMU | 执行地址转换和保护的内存管理单元 |
| NUMA | 访问代价取决于处理器与内存所在位置 |
| SVM | Shared Virtual Memory；共享地址不代表不需要同步 |
| GVA | 在 MemFabric 中是 Global Virtual Address：受管理片段的逻辑身份，并非到处可解引用的指针 |
| HVA / DVA | 某个进程的 Host 虚拟地址 / 某个设备上下文中的 Device 虚拟地址；仍受映射和生命周期约束 |
| 预留 / 后端内存 / 映射 | 占用虚拟地址范围 / 取得物理存储 / 把存储关联到地址范围，是不同步骤 |
| MR / Registration | Memory Region，以及为通信引擎准备访问条件的注册操作 |
| Endpoint / EID | 通信端点 / 端点标识符 |
| Key / Access Descriptor | 后端定义的访问授权或远端资源定位信息 |

## 🔄 搬运与通信

| 术语 | 含义与区别 |
| --- | --- |
| H2D / D2H / H2H | Host→Device / Device→Host / Host→Host 的方向 |
| D2D / P2P | 设备复制 / 对等访问；精确定义依 API 而定 |
| DMA / SDMA / MTE | 数据搬运机制 / 设备路径术语 / AI Core 搬运引擎 |
| PCIe / PCIe Switch | I/O 互联 / 扩展并共享连接能力的交换器件 |
| HCCS | 昇腾高速互联，具体能力取决于产品 |
| RDMA / RoCE | 远程直接内存访问 / 在以太网上承载 RDMA |
| UB / URMA / UBoE | Unified Bus 体系 / 对应远程内存访问接口与机制 / UB over Ethernet |
| Collective | AllReduce、AllGather、Broadcast 等组内操作 |
| HCCL / HCOMM | 集合通信库 / 受支持的底层通信资源与操作能力 |
| Scale-up / Scale-out | 紧耦合域内扩展 / 通过网络连接系统进行扩展，边界取决于架构 |
| Completion / Notification | 操作状态 / 给另一参与者的信号；它们都不自动包含所有可见性保证 |
| WR / WQE | 软件工作请求 / 硬件消费的工作队列项；不是同一个结构 |
| SQ / RQ / QP | RDMA 发送队列 / 接收队列 / 关联队列与传输状态的 Queue Pair |
| SQE | Submission Queue Entry，或具体系统中的 Send Queue Entry；先确认所属队列 |
| CQ / CQE / WC | 完成队列 / 硬件完成项 / verbs 返回的软件完成结果 |
| Doorbell / DBR | 向硬件发布新工作的门铃 / 部分实现使用的内存门铃记录 |
| Device-initiated / Proxy | 设备提出通信请求 / 代理执行提交；与载荷是否直达是不同维度 |
| PD 分离 | 输入 Prefill 与逐步 Decode 分别由不同 Worker 执行，并交接 KV Cache |

AI Core 文档中的 UB 还可能表示 Unified Buffer。同一个缩写，不代表片上缓冲区就是 Unified Bus 互联。

队列关系与完成边界见[工作队列与门铃](work-queues.md)，提交方与实际搬运方的区别见[设备发起传输](device-initiated-transfer.md)。模型层的 TP、PP、DP、EP、CP 则见[推理并行策略](../ml/llm/inference/parallelism.md)。

## 📊 网络平面与性能

控制面和数据面按职责分类：前者建立、管理通信，后者搬运载荷。参数面、存储面、业务面、带外管理面按流量用途分类。出框口则表示物理边界，而不是协议。

带宽衡量每秒字节数或比特数；延迟衡量耗时；吞吐量衡量单位时间完成的有效工作。IOPS 统计操作次数，P50/P99 表示延迟分位数。Chunk Size 是每块字节数，Batch Size 是一批操作的数量，Queue Depth 是在途操作数。全双工允许双向同时传输，双向带宽之和不能当成单向带宽。

试着分析“Rank 3 通过 RDMA 发送 H2D 数据”这句话。哪些信息还不明确？Rank 3 标识参与者，H2D 描述内存端点，RDMA 描述访问语义；只有实际拓扑与 API，才能说明这个组合是否受支持。详细解释与资料入口见[学习路线](index.md)。
