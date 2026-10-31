# 🌐 How GVA unified addressing works

[Page tables and huge pages](page-tables-huge-pages.md) explain how a virtual address in one address space translates to a physical address. This chapter asks the next question: how can an application describe and locate memory distributed across processes, devices, and nodes? Start with GVA, then follow resource lookup, access-information resolution, and the actual access.

## 1. What is GVA?

GVA (Global Virtual Address) is a unified memory-addressing mechanism that brings memory resources from different processes, devices, and nodes into a common address space.

More precisely, a GVA is the global address representation used by this mechanism; “GVA unified addressing” refers to the resource-management and access mechanisms built around that representation.

In MemFabric, pooled resources can include:

- **Host DRAM:** CPU-side memory, usually accessed through an HVA (Host Virtual Address).
- **Device HBM:** NPU-side memory, usually accessed through a DVA (Device Virtual Address).

Traditionally, different processes and devices have their own virtual address spaces. For example, process A's HVA cannot directly access process B's memory, and an ordinary HVA does not directly let the CPU access NPU HBM. This describes the situation without a corresponding shared or device mapping; access still depends on valid mappings, permissions, and hardware capabilities in the current context.

GVA provides a common global address representation for these scattered resources. Applications can describe and locate pooled memory through GVA without directly managing the underlying HVAs, DVAs, and device locations.

Here, “global” means **a consistent address representation among resources participating in the same pool or management domain**. It does not give every machine or process direct access. See the [MemFabric project overview](https://github.com/Ascend/memfabric_hybrid/blob/master/README.md) for its pooling and global addressing design.

## 2. How does GVA unify addressing?

Suppose two processes manage different DRAM and HBM resources:

```text
                  Global GVA address space
┌─────────────────────────────────────────────────┐
│ GVA region A → process A's Host DRAM             │
│ GVA region B → process A's Device HBM            │
│ GVA region C → process B's Host DRAM             │
│ GVA region D → process B's Device HBM            │
└─────────────────────────────────────────────────┘
                         │
                         ▼
              Global memory mapping information
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Process A             Process B
         ┌────────┐            ┌────────┐
         │  DRAM  │            │  DRAM  │
         │  HBM   │            │  HBM   │
         └────────┘            └────────┘
```

GVA does not mean that all memory moves to one location. Global address management organizes memory in different physical locations into a common logical address space.

The core relationship is:

**GVA → memory resource location information → underlying address or memory handle → actual data access**

Read the diagram by asking three questions:

1. **Which resource does the address identify?** For example, region D corresponds to Device HBM managed by process B.
2. **Where does that resource live?** Mapping information describes its node, process, device, memory type, region, and related identity.
3. **How can the current participant reach it?** Access also needs backend-supported address mappings, access handles, or transport resources.

“Global memory mapping information” represents a logical relationship. It does not require one centralized table or a network lookup for every read and write. Keep resource identity separate from the actual access path when learning the model.

## 3. How is memory accessed through GVA?

Suppose an application needs to access a remote NPU's HBM through a GVA. The conceptual steps are:

1. **Obtain the GVA:** identify the target memory region and offset.
2. **Locate the resource:** use the GVA to find the node, device, memory type, and region.
3. **Resolve access information:** obtain the local or remote addresses, registered-memory information, access handles, and other information needed for transfer.
4. **Choose a transfer path:** select a suitable mechanism based on source and destination placement and system configuration, such as an SDMA, RDMA, or URMA-related path.
5. **Perform the access:** the underlying transfer engine reads, writes, or copies the data.

A GVA does not necessarily convert to an HVA or DVA on every access. Access may instead use a device mapping, DMA address, or registered-memory handle, depending on the implementation.

**Terminology:** SDMA, RDMA, and URMA remain examples of access paths here, but they are not three network protocols at the same layer. SDMA is a data-movement engine, RDMA is a remote-memory-access mechanism, and URMA provides unified remote-memory-access interfaces and semantics. Remote HBM support depends on hardware, backend, and memory type; see the [MemFabric architecture](https://github.com/Ascend/memfabric_hybrid/blob/master/README.md), [openEuler UMDK's URMA introduction](https://gitee.com/openeuler/umdk/blob/master/README.md), and [Interconnects](interconnects.md).

### Worked example: process A reads process B's HBM

Using the diagram above, let `G_D` denote the GVA base of region D. The application reads `0x1000` bytes starting at offset `0x2000` into process A's local DRAM buffer:

```text
Source identity: GVA = G_D + 0x2000
Source owner:    process B's node → corresponding NPU → region D's Device HBM
Access length:   0x1000 bytes
Destination:     allocated Host DRAM in process A, prepared for backend access

Locate the source
  → Obtain backend access information for source and destination
  → Select a supported direct or staged path
  → Submit the read or copy
  → Wait for required completion and establish visibility
  → Process A consumes the data in its local buffer
```

`G_D` is a symbolic region base for this example, not a definition of actual GVA bit encoding. If the backend establishes a DVA mapping for this region, the location might correspond to `D_B + 0x2000`; another backend might locate it through a registered-memory descriptor. **Knowing a GVA does not imply having a usable DVA or access permission; a backend using registered-memory descriptors must instead validate the relevant descriptors.**

At the interface level, MemFabric's BM documentation shows obtaining a GVA and then calling a copy interface. SHM provides supported device-side access, so not every path can be treated as an ordinary CPU pointer read. See the [BM, TRANS, and SHM feature guide](https://github.com/Ascend/memfabric_hybrid/blob/master/doc/feature.md).

Access also has completion and lifetime requirements. Successful submission does not necessarily mean the application can consume the destination data. Memory, mappings, and registration handles must stay valid while operations are in flight. Follow the complete sequence in [End-to-end transfer](end-to-end-transfer.md). For CPU virtual addresses versus device DMA addresses, see the [Linux DMA API guide](https://docs.kernel.org/core-api/dma-api-howto.html).

## 4. How does GVA differ from a page table?

| Concept | Role |
| --- | --- |
| VA | General term for a virtual address |
| HVA | Virtual address used by a host process |
| DVA | Virtual address used on the NPU device side |
| GVA | Global virtual address used to locate pooled memory consistently |
| Page table | Manages mappings between virtual and physical pages |
| PTE (Page Table Entry) | One mapping record in a page table |

Both involve address mappings, but at different abstraction levels:

- **Page tables:** primarily translate CPU or device virtual addresses into physical addresses.
- **GVA unified addressing:** primarily provides consistent resource addressing across processes, devices, and nodes.

GVA is therefore not equivalent to a CPU virtual address and does not let an application dereference an arbitrary GVA through an ordinary CPU pointer.

**Different abstractions can still interact in an implementation.** MemFabric's global memory management can formulate page-table mapping policies and install them through the driver. Access to a resource described by GVA may therefore rely on hardware mappings, while GVA itself remains distinct from a hardware page table. See the [Global Memory Management module description](https://github.com/Ascend/memfabric_hybrid/blob/master/README.md).

If “virtual page → page-table entry → physical page → page offset” is still unfamiliar, return to [Page tables, PTEs, huge pages, and address translation](page-tables-huge-pages.md).

## 5. Summary

GVA unified addressing **establishes a common logical address space for DRAM and HBM distributed across processes, devices, and nodes, locates underlying resources through mappings, and ultimately uses the appropriate transfer mechanisms to access data.**

Understand it at three levels:

- **Unified addressing:** pooled memory in different locations uses a common GVA address system.
- **Resource lookup:** the GVA identifies the target node, device, and memory region.
- **Data access:** HVAs, DVAs, memory registration information, or other underlying mechanisms enable the actual access.

**GVA unifies address representation and resource addressing, rather than the physical location of memory, and it cannot be equated directly with a hardware page table.**

Continue with [Memory and addresses](memory-addressing.md) to connect these ideas with virtual-range reservation, physical allocation, mapping, registration, and reclamation.
