# Infrastructure glossary: identify the layer first

Use this page while drawing a transfer path. For every term, ask whether it names a physical object, address, operation, network role, or software abstraction. Many debugging mistakes come from treating terms in different categories as interchangeable.

## Hardware and identity

| Term | Meaning | Diagnostic question |
| --- | --- | --- |
| Die | One physical piece of silicon | Physical die identifier or logical identifier? |
| Package | One or more dies assembled as a component | How many dies and what internal connections? |
| Card | Board carrying components | How does it map to runtime devices? |
| NPU | Processor optimized for neural/tensor workloads | Which model and architecture generation? |
| Device | Runtime-visible accelerator object | Which physical resource does this ID name? |
| Rank | Participant's ID within a communication group | Which process, host, device, and endpoint? |
| AI Core | Accelerator compute resource | Which matrix/vector architecture applies? |
| AICPU | Device-side auxiliary processor | Which tasks run there rather than on AI Core? |

## Storage and addresses

| Term | Meaning and distinction |
| --- | --- |
| Host memory | CPU-side memory, commonly DRAM |
| Device global memory / GM | Device-wide storage, often HBM or DDR depending on product |
| Local memory | Smaller on-chip buffers near compute, such as L0/L1/UB |
| HBM | High Bandwidth Memory; capacity and bandwidth are different limits |
| VA / PA | Virtual address / physical address |
| IOVA / IOMMU | Device I/O virtual address / its translation and isolation mechanism |
| MMU | Memory-management unit implementing address translation and protection |
| NUMA | Access cost depends on processor and memory location |
| SVM | Shared Virtual Memory; sharing addresses does not erase synchronization requirements |
| GVA | Context-dependent term, sometimes Global Virtual Address; inspect the implementation |
| MR / registration | Memory region and preparation for communication-engine access |
| Endpoint / EID | Communication endpoint / endpoint identifier |
| Key or access descriptor | Backend-defined information authorizing or locating remote access |

## Movement and communication

| Term | Meaning and distinction |
| --- | --- |
| H2D / D2H / H2H | Host→device / device→host / host→host directions |
| D2D / P2P | Device copy / peer access; precise scope depends on API |
| DMA / SDMA / MTE | Data-movement mechanism / device-path term / AI Core transfer engine |
| PCIe / PCIe switch | I/O interconnect / device expanding and sharing its connectivity |
| HCCS | Ascend high-speed interconnect; exact capabilities depend on product |
| RDMA / RoCE | Remote direct memory access / RDMA transport over Ethernet |
| UB / URMA / UBoE | Unified Bus ecosystem / its remote-memory-access interface and mechanisms / UB over Ethernet |
| Collective | Group operation such as AllReduce, AllGather, or Broadcast |
| HCCL / HCOMM | Collective communication library / supported lower-level communication resources and operations |
| Scale-up / scale-out | Expansion within a tightly coupled domain / across networked systems; boundaries depend on architecture |
| Completion / notification | Operation status / signal to another participant; neither implies every possible visibility guarantee |

“UB” also means Unified Buffer in AI Core documentation. The same abbreviation does not make the on-chip buffer the Unified Bus interconnect.

## Planes and performance

Control and data planes describe responsibilities: establishing/managing communication versus moving payloads. Parameter, storage, business, and out-of-band networks describe traffic purposes. An external enclosure port describes a physical boundary, not a protocol.

Bandwidth measures bytes or bits per second; latency measures elapsed time; throughput measures useful work per time. IOPS counts operations, and P50/P99 describe latency quantiles. Chunk size is bytes per piece, batch size counts grouped work, and queue depth counts outstanding work. Full duplex permits simultaneous traffic in both directions; bidirectional aggregate bandwidth must not be reported as one-way bandwidth.

Try classifying “rank 3 sends H2D over RDMA.” Which parts require clarification? Rank 3 identifies a participant, H2D describes memory endpoints, and RDMA describes access semantics; only the actual topology and API tell whether that combination is supported. See [the learning map](index.md) for detailed chapters and sources.
