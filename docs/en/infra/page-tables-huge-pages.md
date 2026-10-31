# 🗂️ Page tables, PTEs, huge pages, and address translation

Programs use virtual addresses, while bytes in memory have physical addresses. To understand a memory access, first identify who records mappings and who performs translation. Then split an address into a page number and an offset, and examine how page size changes mapping counts and TLB coverage.

This chapter follows [Memory and addresses](memory-addressing.md) and compares 4KB ordinary pages, 2MB huge pages, and 1GB huge pages. To retain familiar page-size notation, KB, MB, and GB mean binary capacities throughout, corresponding to KiB, MiB, and GiB: $1\text{KB}=2^{10}$ bytes, $1\text{MB}=2^{20}$ bytes, and $1\text{GB}=2^{30}$ bytes. Supported page sizes depend on the processor and operating system.

TB likewise means the binary capacity TiB here, with $1\text{TB}=2^{40}$ bytes. The detailed multilevel examples use **typical x86-64 four-level paging**: first divide the address, then follow how entries at each level complete the mapping.

## 1. Basic concepts and core formulas {#1-what-is-a-page-table}

A program uses a **virtual address (VA)**. To access physical memory, the CPU needs to translate it into a **physical address (PA)**. Put addresses, page numbers, mapping records, and hardware responsibilities together:

| Concept | Full name | Responsibility | How to think about it |
| --- | --- | --- | --- |
| VA | Virtual Address | Address used by the program | The address supplied for a program's memory access |
| PA | Physical Address | Address obtained through translation | Locates the target in physical address space |
| VPN | Virtual Page Number | Identifies the virtual page | The address portion that determines the mapping |
| PFN | Page Frame Number | Identifies the physical frame | Combine it with the page size to obtain the physical page base |
| PTE | Page Table Entry | A record in a page table; a final mapping entry describes the physical page, permissions, and other information | An entry in the table, separate from the data page it maps |
| Page table | Page Table | Records virtual-to-physical address mappings | A mapping table, usually organized into multiple levels |
| MMU | Memory Management Unit | Performs virtual-to-physical translation and checks access permissions | The hardware that executes translation |
| TLB | Translation Lookaside Buffer | Caches address mappings to avoid repeated page-table lookups | A cache of recently used translation results |
| Offset | Page offset | Selects a particular byte within a page | Preserved by the paging translation |

The three core relationships are:

$$
\boxed{\mathrm{VA}=\mathrm{VPN}\mathbin{\Vert}\mathrm{Offset}}
$$

$$
\boxed{\mathrm{VPN}\xrightarrow{\text{page table}}\mathrm{PFN}}
$$

$$
\boxed{\mathrm{PA}=\mathrm{PhysicalPageBase}+\mathrm{Offset}}
$$

The first is also written `VA = VPN + Offset`, where “+” means **field concatenation**, rather than numerical addition of unshifted values. The equation uses $\Vert$ to make concatenation explicit. Adding an offset to a physical page base is numerical addition. With page size $S=2^n$, the numerical address relationship is $\mathrm{VA}=\mathrm{VPN}\times S+\mathrm{Offset}$.

The following flow assumes a valid mapping and permitted access, using a common system with hardware page-table walks as its example:

```mermaid
flowchart TD
    A[CPU provides virtual address VA] --> B[MMU checks TLB]
    B -->|Hit| C[Obtain physical page base address]
    B -->|Miss| D[Walk page tables to find mapping]
    D --> E[Obtain mapping and fill TLB]
    E --> C
    C --> F[Add page offset to obtain physical address PA]
```

**A TLB miss is not a page fault.** A miss means that the cache lacks the translation; the page table may still hold a valid mapping. A page fault involves conditions such as an absent mapping, a nonresident page, or access permissions. See the [Linux page-table documentation](https://docs.kernel.org/mm/page_tables.html) for page-table levels, permissions, and translation caches.

## 2. A complete walk: from malloc to a physical memory access {#malloc-to-physical-address}

Start with a C fragment emphasizing allocation and the first write:

```c
char* p = (char*)malloc(4096);
p[0] = 1;
```

A real program must check allocation success and call `free` after the last use. Follow an actual memory access using typical x86-64 four-level paging and ordinary 4KB pages.

For the arithmetic, fix an **example access address** of `VA = 0x12345678`. It can identify a byte inside an allocation; it is not a claim that a real `malloc` must return this value. Actual return values must satisfy allocator alignment guarantees. Nor does `malloc(4096)` guarantee a 4KB-aligned start or exactly one virtual page: 4096 bytes beginning at page offset `0x678` cross two 4KB pages. See the [Linux malloc manual](https://man7.org/linux/man-pages/man3/malloc.3.html) for allocation and return values, and the [Linux mmap manual](https://man7.org/linux/man-pages/man2/mmap.2.html) for page-aligned mappings.

### 2.1. Step one: malloc supplies usable virtual addresses

```c
char* p = (char*)malloc(4096);
```

The allocator supplies a usable virtual address range. It may reuse managed heap storage or obtain more space from the operating system. A `malloc` call therefore need not create a new mapping, and receiving an address does not mean an equal amount of physical memory was allocated at that moment.

For demand-allocated pages, the first access may raise a **page fault**. If the address belongs to a valid region, the kernel can allocate a physical page, establish or update the mapping, and restart the access. The allocator may already have touched the page, so the first write to `p[0]` need not fault. An access that cannot be handled legally does not automatically become a successful physical memory access. See [Linux page tables and faults](https://docs.kernel.org/mm/page_tables.html#mmu-tlb-and-page-faults).

### 2.2. Step two: split VA into VPN and offset

A 4KB page contains $2^{12}$ bytes. The low 12 bits are the offset, and the remaining effective address bits form the VPN:

```text
VA                    = 0x12345678
Page size             = 0x1000
VPN                   = VA >> 12  = 0x12345
Offset                = VA & 0xFFF = 0x678
Reconstruct VA        = 0x12345 × 0x1000 + 0x678 = 0x12345678
```

**VPN determines the page mapping; the offset selects the byte within it.** This is a logical explanation of address fields, not a requirement that hardware execute each step serially.

### 2.3. Step three: check the TLB first

Think of the TLB as the MMU's translation cache. A simplified illustration for the current address space, expressed in 4KB pages, is:

```text
TLB
┌──────────────┬──────────────┐
│ VPN          │ PFN          │
├──────────────┼──────────────┤
│ 0x12345      │ 0xABCDE      │
│ 0x12346      │ 0x98765      │
└──────────────┴──────────────┘
```

- **Hit with permitted access:** obtain physical frame information without another table walk for this access, then proceed to PA construction.
- **Miss:** walk the tables; a valid translation can then fill the TLB for later accesses.

The diagram shows only VPN and PFN. Real translations also distinguish address space, page size, and permissions; the same VPN in two processes does not automatically identify the same physical memory.

### 2.4. Step four: walk the tables on a TLB miss

The 48 effective bits split as `PGD(9) + PUD(9) + PMD(9) + PTE(9) + Offset(12)`. Starting from the current address space's table root, the MMU uses a different 9-bit index at each level:

```text
PGD: select an entry using the PGD index → locate the PUD table
 ↓
PUD: select an entry using the PUD index → locate the PMD table
 ↓
PMD: select an entry using the PMD index → locate the PTE table
 ↓
PTE: select an entry using the PTE index → obtain frame information and permissions
 ↓
PFN
```

For `0x12345678`, the indices are:

```text
PGD index = (VA >> 39) & 0x1FF = 0x000
PUD index = (VA >> 30) & 0x1FF = 0x000
PMD index = (VA >> 21) & 0x1FF = 0x091
PTE index = (VA >> 12) & 0x1FF = 0x145
```

Three points matter:

1. A **non-leaf entry** identifies the next table and carries control information.
2. A **leaf entry** describes a physical frame and permissions instead of referring to another table.
3. **The index path normally determines the VPN, so a PTE need not store the complete VPN again.** The table and slot containing an entry are part of how the mapping is represented.

### 2.5. Step five: turn PFN into a physical page base

Assume the final mapping is present and permits this access:

```text
VPN = 0x12345
        ↓
PFN recorded by the leaf PTE = 0xABCDE
```

Using this example's 4KB frame units:

$$
0xABCDE\times0x1000=0xABCDE000
$$

`0x1000` is 4KB in hexadecimal. PFN is frame information decoded from the entry; the complete PTE value, including flags, is not itself the physical page address.

### 2.6. Step six: preserve the offset and construct PA

```text
Physical page base: 0xABCDE000
Page offset:        0x00000678
────────────────────────────
Physical address:   0xABCDE678
```

Throughout translation, **the offset remains `0x678`, while VPN `0x12345` maps to PFN `0xABCDE`**. The resulting PA locates a byte backed by DRAM in this example.

Distinguish **translation** from **when DRAM is accessed**. CPU data caches can satisfy hits, and a common write-back policy allows a write to update a cache before reaching memory later. Executing `p[0] = 1` does not imply an immediate DRAM write. The TLB caches translations; data caches hold data. See [Intel's cache write-back explanation](https://edc.intel.com/content/www/us/en/design/ipla/software-development-platforms/client/platforms/alder-lake-desktop/12th-generation-intel-core-processors-datasheet-volume-1-of-2/001/cache-line-write-back/).

## 3. Multilevel tables: address fields, demand allocation, and coverage {#3-how-is-a-virtual-address-divided}

Assume a **48-bit virtual address**. This means 48 effective address bits participate in translation; processors do not all use the same address width.

### 3.1. Start with a virtual page number and offset {#31-start-with-a-virtual-page-number-and-offset}

| Page size | Virtual page number (VPN) | Page offset |
| --- | --- | --- |
| 4KB | 36 bits | 12 bits |
| 2MB | 27 bits | 21 bits |
| 1GB | 18 bits | 30 bits |

Every row satisfies “VPN bits + offset bits = 48”:

```text
4KB page: [             VPN: 36 bits             ][Offset: 12 bits]
2MB page: [          VPN: 27 bits          ][    Offset: 21 bits    ]
1GB page: [       VPN: 18 bits       ][          Offset: 30 bits          ]
```

The translation can be written as:

```text
Virtual address = virtual page number + page offset

Virtual page number -> page-table lookup -> physical frame number

Physical address = physical page base address + page offset
```

The first “+” means **concatenation of bit fields**. Do not numerically add an unshifted VPN to the offset. For page size $S=2^n$ bytes, the numerical relationships are:

$$
\mathrm{VA}=\mathrm{VPN}\times S+\mathrm{Offset}
$$

$$
\mathrm{PA}=\mathrm{PFN}\times S+\mathrm{Offset}
$$

Here, PFN is the physical frame number **in units of this mapping's page size $S$**, so $\mathrm{PFN}\times S$ is the physical page base address. Operating-system interfaces may instead express PFNs in base-page units; check the unit when reading code. Multilevel page tables further divide the VPN into indices for individual levels rather than requiring one flat table indexed by the entire VPN.

In this paging model, translation changes the page's location while **preserving the page offset**. A larger page uses more offset bits and fewer VPN bits.

### 3.2. How 48 address bits relate to four-level paging {#32-how-48-address-bits-relate-to-four-level-paging}

With typical **x86-64 four-level paging**, 48 effective virtual address bits represent $2^{48}$ byte positions: **256TB (256TiB) of virtual address space**. This is virtual address capacity, not installed physical memory, and it does not imply that all addresses are mapped. See the [Linux four-level and five-level paging explanation](https://docs.kernel.org/arch/x86/x86_64/5level-paging.html).

For ordinary 4KB pages, the 36-bit VPN splits into four 9-bit indices:

```text
48-bit virtual address
┌────────┬────────┬────────┬────────┬──────────┐
│  PGD   │  PUD   │  PMD   │  PTE   │  Offset  │
│ 9 bits │ 9 bits │ 9 bits │ 9 bits │ 12 bits  │
└────────┴────────┴────────┴────────┴──────────┘

Bits:    [47:39]  [38:30]  [29:21]  [20:12]   [11:0]
```

- **PGD, PUD, PMD, and PTE indices:** each selects an entry at a different table level. Nine bits is the index width, not the size of an entire entry.
- **Page offset:** selects a particular byte in the physical page. A 4KB page has $2^{12}$ byte positions and therefore needs 12 offset bits.
- **All 48 bits belong to the virtual address; none are a physical address field.** The physical page location comes from the selected page-table entry, and physical address width need not match virtual address width.

**Level names:** this chapter uses Linux's PGD, PUD, PMD, and PTE terminology. In four-level paging, these correspond to the hardware's PML4, PDPT, PD, and PT. The diagram's “PTE” field contains address bits indexing the bottom-level PT, rather than the entry's contents. Linux's generic interfaces also include P4D, which is folded in four-level paging and adds no separate lookup. See the [Linux page-table documentation](https://docs.kernel.org/mm/page_tables.html) for levels and folding.

**How can 48 effective bits coexist with a 64-bit pointer?** In the conventional x86-64 four-level canonical address form, the high 16 bits of a 64-bit address replicate bit 47. Valid addresses occupy a low 128TiB region and a high 128TiB region, totaling 256TiB, with a noncanonical hole between them. This is not one contiguous 256TiB range starting at zero, nor does it mean a user process can freely use all that capacity. See the [Linux x86-64 virtual memory layout](https://docs.kernel.org/arch/x86/x86_64/mm.html).

### 3.3. Why could a flat table require 512GiB? {#why-multilevel-page-tables}

With 48 address bits and 4KB pages, the virtual page count is:

$$
\frac{2^{48}}{2^{12}}=2^{36}=68{,}719{,}476{,}736
$$

That is approximately **68.7 billion virtual pages**. Preallocating a complete single-level table with an 8-byte entry for every page would require:

$$
2^{36}\times8=2^{39}\text{ bytes}=512\text{ GiB}
$$

Most programs do not use the entire address space, so preparing every possible entry wastes memory. **Multilevel tables allocate lower levels on demand:** an unmapped range can have an absent lower branch instead of a fully allocated collection of bottom-level entries.

### 3.4. How large is one page table? {#page-table-size}

In this x86-64 paging scheme:

- Each level's index uses **9 bits**, selecting among $2^9=512$ entries.
- Each page-table entry occupies **8 bytes (64 bits)**.
- Therefore, **one table occupies $512\times8=4096$ bytes, or 4KB**.

```text
One index: 9 bits → 512 possible values → select one entry in the current table
One entry: 8 bytes → next-table address or physical page mapping, plus flags
One table: 512 entries × 8 bytes = 4096 bytes = 4KB
```

**The size of a table and the memory range mapped by an entry are different concepts.** A 4KB table does not imply that every entry can map only 4KB; an entry mapping 1GB does not occupy 1GB. A process may use many such tables, so its entire multilevel structure is not limited to 4KB. These sizes apply to this x86-64 example, rather than every architecture. Compare the entry types, 512-entry table sizes, and PMD/PUD range definitions in the [Linux x86-64 page-table definitions](https://github.com/torvalds/linux/blob/master/arch/x86/include/asm/pgtable_64_types.h).

### 3.5. How much does each level cover? {#level-coverage}

Every table in this four-level example has 512 entries, but the virtual coverage depends on its level:

| Level | Coverage of one entry | Coverage of one table |
| --- | --- | --- |
| PGD | 512GiB | 256TiB |
| PUD | 1GiB | 512GiB |
| PMD | 2MiB | 1GiB |
| PTE | 4KiB | 2MiB |

Starting at the bottom, multiply by 512: `4KiB × 512 = 2MiB`, then `1GiB`, `512GiB`, and `256TiB`. **Coverage does not imply that the corresponding amount of physical memory has been allocated.**

For non-leaf entries, coverage describes the virtual range of the lower subtree, rather than a promise to map one physical page of that size. A PGD entry covering 512GiB does not mean this example supports 512GiB pages. Adjacent virtual pages covered by a PTE table also need not map to adjacent physical frames.

## 4. How ordinary and huge pages change translation {#huge-pages}

### 4.1. Ordinary pages versus huge pages {#2-ordinary-pages-versus-huge-pages}

The key difference is **the memory range covered by one mapping**. Dividing the same amount of memory into larger pages requires fewer pages and final mapping entries.

| Comparison | 4KB ordinary page | 2MB huge page | 1GB huge page |
| --- | --- | --- | --- |
| Range covered by one mapping | 4KB | 2MB | 1GB |
| Pages needed to map 1GB | 262144 | 512 | 1 |
| Page offset | 12 bits | 21 bits | 30 bits |
| Page size in hexadecimal | `0x1000` | `0x200000` | `0x40000000` |
| Table walk | PGD → PUD → PMD → PTE | PGD → PUD → PMD | PGD → PUD |
| Physical contiguity | Each 4KB page contiguous and aligned; different pages need not be adjacent | Each 2MB page contiguous and aligned | Entire 1GB contiguous and 1GB-aligned |

Calculate the page count directly. Assume that the 1GB region satisfies the relevant page alignment and is mapped entirely with the same page size:

```text
4KB pages: 1GB / 4KB = 2^30 / 2^12 = 2^18 = 262144 pages
2MB pages: 1GB / 2MB = 2^30 / 2^21 = 2^9  = 512 pages
1GB pages: 1GB / 1GB = 2^30 / 2^30 = 1 page
```

These are counts of data pages, and therefore **final mapping entries**. They do not include upper-level directory entries in a multilevel page table.

**The PTE itself has not become 1GB; the range covered by one mapping has changed.** An entry holds mapping information, while data lives in the mapped physical page. In a multilevel page table, some entries refer to another table and others complete the mapping as leaf entries. Huge pages may be mapped directly by higher-level entries, ending the walk before the lowest level. When reading implementations, distinguish the general term “page-table entry” from PTE as a name specifically for the lowest-level entry; see the [Linux page-table hierarchy](https://docs.kernel.org/mm/page_tables.html).

### 4.2. How do multilevel tables map ordinary and huge pages? {#leaf-levels}

A **leaf page-table entry** completes a physical memory mapping instead of pointing to another table. “Leaf” means this mapping ends here; it need not be at the bottom PTE level.

In this four-level example, the three paths are:

```text
4KB ordinary page:
PGD → PUD → PMD → PTE → 4KB physical page

2MB huge page:
PGD → PUD → PMD ──────→ 2MB physical page

1GB huge page:
PGD → PUD ────────────→ 1GB physical page
```

Compare the leaf level, offset width, and number of leaf mappings needed for 1GB:

| Page size | Mapping completes at | Page offset | Leaf mappings needed for 1GB | Table levels visited |
| --- | --- | --- | --- | --- |
| 4KB | PTE | 12 bits | 262144 | 4 |
| 2MB | PMD | 21 bits | 512 | 3 |
| 1GB | PUD | 30 bits | 1 | 2 |

For a 2MB page, the PMD entry completes the mapping without continuing to a PTE table. For a 1GB page, the PUD entry completes it without continuing to PMD or PTE tables. These are logical walk paths. Actual latency depends on the TLB, page-walk caches, and ordinary caches; the number of levels is not a count of actual DRAM accesses.

### 4.3. Why does a 1GB page shorten the walk? {#huge-page-offset-bits}

Compare the address fields to see where the bits go when a mapping ends earlier.

For a 4KB page:

```text
PGD(9) + PUD(9) + PMD(9) + PTE(9) + Offset(12) = 48 bits
```

For a 2MB page:

```text
PGD(9) + PUD(9) + PMD(9) + Offset(21) = 48 bits
21 bits = 9 bits + 12 bits
```

For a 1GB page:

```text
PGD(9) + PUD(9) + Offset(30) = 48 bits
30 bits = 9 bits + 9 bits + 12 bits
```

The **18 bits previously used as PMD and PTE indices** become part of the offset in a 1GB mapping. Likewise, a 2MB mapping combines the former 9-bit PTE index with the offset to obtain 21 offset bits.

Thus **a 1GB page does not delete 18 bits or change the virtual address width; the mapping ends earlier, at the PUD level**. This example still has a 48-bit effective virtual address. What changes is which bits select table entries and which locate a byte within the physical page.

### 4.4. Why can huge pages improve performance? {#4-why-can-huge-pages-improve-performance}

The mapping counts and walk paths explain three main benefits:

1. **Fewer mapping entries reduce page-table memory overhead.** Mapping 1GB needs 262144 leaf entries with 4KB pages, 512 with 2MB pages, or 1 with a 1GB page. Lower-level tables for that region may also become unnecessary. However, “one leaf entry” does not mean the entire structure occupies only 8 bytes; upper-level tables still exist or are reused.
2. **Fewer table levels are visited.** In this example, 1GB translation completes at PUD, while 2MB translation completes at PMD.
3. **Fewer TLB misses and less TLB pressure.** Each TLB mapping covers a larger virtual range, allowing a suitable working set to fit in fewer entries.

**Accessing one 4KB page does not sequentially inspect 262144 PTEs.** Each level's index selects the required entry directly, and the TLB caches translation results. The number 262144 describes how many leaf mappings cover the entire 1GB region, not how many entries a single translation checks in sequence.

Assume a TLB holds **64 mappings**. For this comparison, separately assume that all 64 entries can serve the page size under discussion:

- **4KB pages:** coverage is $64\times4\text{KB}=256\text{KB}$.
- **1GB pages:** coverage is $64\times1\text{GB}=64\text{GB}$.

This is a theoretical example. Actual TLB capacity varies with hardware and page size; it does not imply that a real processor offers equal entry counts for every page size.

**Huge pages do not make the TLB store more entries; each entry covers more addresses.** For a suitable working set and access pattern, the same data needs fewer mappings, which can reduce TLB misses and page-table walk overhead. Higher-level leaf entries can also end walks earlier.

Allocation costs matter too: huge pages require appropriate contiguous physical memory and alignment, and using only a small portion can waste capacity. They do not guarantee that every program runs faster. See the [Linux HugeTLB documentation](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html) for supported sizes and allocation constraints.

### 4.5. What does configuring 1GB pages actually change? {#huge-page-configuration}

In this x86-64 four-level example, assume the processor supports 1GB pages, memory meets the corresponding alignment and contiguity requirements, and the operating system **actually establishes a 1GB mapping for the region**:

**Configuring and using a 1GB page means the operating system uses a PUD-level leaf entry to map a 1GB virtual address range directly to a 1GB physical address range.**

It changes neither this example's 48-bit effective virtual address width nor the system's basic page-table structure. It changes **the mapping granularity and walk depth for regions using huge pages**. Different regions can use 4KB, 2MB, and 1GB pages simultaneously; one 1GB mapping does not reduce the entire system to two-level tables. Requesting huge pages is also distinct from confirming the mapping granularity actually established.

The core effects are **fewer page-table mapping entries, less TLB pressure, and lower address-translation overhead**.

### 4.6. How Linux manages huge pages with HugeTLB {#linux-hugetlb}

The preceding sections explain hardware mappings. **HugeTLB provides Linux applications with explicit huge-page allocation and management**, using pools for individual page sizes. Official guide: [HugeTLB Pages (hugetlbpage)](https://cdn.kernel.org/doc/html/latest/admin-guide/mm/hugetlbpage.html).

Follow resource preparation, mapping, and observation:

1. **Prepare pools** at boot or runtime. `/proc/sys/vm/nr_hugepages` controls the persistent pool's default-size page count; `/sys/kernel/mm/hugepages/` exposes individual sizes. Physical contiguity and NUMA placement affect allocation success.
2. **Request mappings explicitly** through `mmap(..., MAP_HUGETLB, ...)`, mapped `hugetlbfs` files, or `shmget` / `shmat`. `MAP_HUGETLB` needs no filesystem mount. Combine it with `MAP_HUGE_1GB` on supported systems; mapping and unmapping have huge-page alignment requirements. See [mmap](https://man7.org/linux/man-pages/man2/mmap.2.html).
3. **Track lifetime.** HugeTLB pages cannot be swapped out. Released persistent pages return to the pool, rather than automatically returning to the kernel's ordinary page allocator.

Inspect Linux state without changing configuration:

```sh
cat /proc/meminfo
ls /sys/kernel/mm/hugepages/
```

Distinguish these fields; see also [proc_meminfo](https://man7.org/linux/man-pages/man5/proc_meminfo.5.html):

| Field | Meaning |
| --- | --- |
| `HugePages_Total` | Default-size pool page count |
| `HugePages_Free` | Unallocated pool pages |
| `HugePages_Rsvd` | Committed pages not yet allocated at fault time |
| `HugePages_Surp` | Surplus pages above the persistent target |
| `Hugepagesize` | Default page size, not pool capacity |
| `Hugetlb` | Combined capacity across HugeTLB sizes, as explained in the kernel guide |

Inspect per-size sysfs directories when multiple sizes coexist. **1GiB of pool capacity does not imply one 1GiB page:** 512 × 2MiB has equal capacity but different mapping granularity.

**Distinguish HugeTLB from Transparent Huge Pages (THP).** THP transparently uses, merges, or splits huge pages for eligible mappings according to kernel policy, without using HugeTLB's reserved pool. Ordinary `malloc` regions may benefit; a large allocation alone guarantees neither HugeTLB nor 1GB pages. See [Transparent Hugepage Support](https://cdn.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html).

HugeTLB's non-swappable property also does not replace [device DMA pinning, registration, and mapping requirements](memory-addressing.md#host-memory-registration). Page size, pool management, and device access remain separate concerns.

**CANN measurement: non-swappable does not imply Runtime-pinned.** The reported HugeTLB `aclrtMemcpyBatchAsync` call took about 3.5 ms, followed by 6 µs of synchronization; the `aclrtMallocHost` comparison took 0.33 ms and 2.67 ms. HugeTLB with individual `aclrtMemcpyAsync` calls took about 3.5 ms and 9 µs. This is consistent with the former completing mainly inside the API and the latter progressing asynchronously on the stream after submission. HugeTLB residency alone does not guarantee the Runtime-pinned asynchronous path. See the [measurement analysis](data-movement.md#hugetlb-runtime-pinned) for the full table, timing scope, and inference limits.

## 5. Page size, hexadecimal, and address arithmetic {#page-address-arithmetic}

### 5.1. Why express page sizes in hexadecimal? {#5-why-express-page-sizes-in-hexadecimal}

Because:

$$
16=2^4
$$

Thus **one hexadecimal digit represents 4 bits**. Hexadecimal makes binary address boundaries easier to inspect. It changes the representation, not the actual page size.

For example, the following values are in bytes, and `0x` denotes hexadecimal:

```text
0x10        = 2^4    = 16B
0x100       = 2^8    = 256B
0x1000      = 2^12   = 4KB
0x200000    = 2^21   = 2MB
0x40000000  = 2^30   = 1GB
```

`0x1000` is 1 shifted left by 12 bits; `0x200000` is $2\times16^5=2^{21}$; and `0x40000000` is $4\times16^7=2^{30}$. Hexadecimal can therefore represent 2MB and 1GB clearly even though their offset widths are not multiples of four.

### 5.2. Why can the physical page base address differ? {#6-why-can-the-physical-page-base-address-differ}

**Different page sizes define different memory boundaries.** The same physical address belongs to pages with different starting points when memory is grouped by different page sizes. This compares two partitioning schemes, rather than assigning two page sizes to a single translation.

Consider the same physical address, `0x40001234`:

| Comparison | 4KB page | 1GB page |
| --- | --- | --- |
| Page base address | `0x40001000` | `0x40000000` |
| Page offset | `0x234` | `0x1234` |
| Final address | `0x40001234` | `0x40001234` |

The formulas are:

$$
\text{Page base}=\left\lfloor\frac{\text{Address}}{\text{Page size}}\right\rfloor\times\text{Page size}
$$

$$
\text{Page offset}=\text{Address}\bmod\text{Page size}
$$

Substitution verifies both cases:

```text
Using 4KB pages:
Page size     = 0x1000
Page base     = floor(0x40001234 / 0x1000) × 0x1000 = 0x40001000
Page offset   = 0x40001234 mod 0x1000 = 0x234
Final address = 0x40001000 + 0x234 = 0x40001234

Using 1GB pages:
Page size     = 0x40000000
Page base     = floor(0x40001234 / 0x40000000) × 0x40000000 = 0x40000000
Page offset   = 0x40001234 mod 0x40000000 = 0x1234
Final address = 0x40000000 + 0x1234 = 0x40001234
```

A page base must be a multiple of the page size. A 4KB base has 12 low zero bits; a 1GB base has 30 low zero bits. Although the base and offset differ, their sum reconstructs the same physical address.

### 5.3. Page size and offset width {#7-page-size-and-offset-width}

**A page of $2^n$ bytes needs $n$ offset bits.** It contains $2^n$ byte positions, numbered from 0 through $2^n-1$.

| Page size | Offset bits | Maximum offset |
| --- | --- | --- |
| 4KB | 12 bits | `0xFFF` |
| 2MB | 21 bits | `0x1FFFFF` |
| 1GB | 30 bits | `0x3FFFFFFF` |

For a 1GB page:

- **Page size:** `0x40000000` (4 followed by seven zeros).
- **Maximum offset:** `0x3FFFFFFF`, the page size minus 1.
- **Offset range:** 0 through 1GB−1 bytes, or 0 through 1,073,741,823.

Do not confuse the page size with the offset of its last byte. An offset of `0x40000000` has already reached the beginning of the next 1GB page.

## 6. The complete flow and what to study next {#allocation-to-dram-summary}

Connect allocation, ordinary translation, fault handling, and the subsequent memory access:

```mermaid
flowchart TD
    A[Program malloc / mmap] --> B[Obtain usable virtual addresses VA]
    B --> C[CPU issues a memory access]
    C --> D[MMU distinguishes VPN and offset]
    D --> E[Check TLB in the current address space]
    E -->|Hit| J[Obtain frame information and check permissions]
    E -->|Miss| F[Walk tables using indices]
    F -->|Valid mapping| I[Obtain translation and fill TLB]
    I --> J
    F -->|Absent or invalid mapping| G[Page fault handled by kernel]
    J -->|Permission fault| G
    G -->|Handleable: establish or update mapping and retry| C
    G -->|Cannot handle legally| X[Access failure or exception handling]
    J -->|Access permitted| K[Physical page base + original offset]
    K --> L[Construct physical address PA]
    L --> M[CPU caches and memory hierarchy]
    M -->|Memory access or write-back required by cache policy| N[Access DRAM]
```

The diagram expresses logical dependencies; hardware need not execute every box serially. A fault retries only when the kernel can resolve it legally, rather than supplying a physical page for every arbitrary address.

Three conclusions connect the concepts:

1. **Page tables record mappings at page granularity.** They do not store a complete entry for every virtual byte address: a virtual page maps to a physical frame, and the offset selects the byte.
2. **The MMU executes translation, the TLB caches translations, and page tables store mapping information.** VPN determines an index path; PFN determines a physical page base.
3. **One 1GB leaf mapping covers contiguous 1GB virtual and physical ranges.** The table itself has not become 1GB, and an ordinary-page access does not scan 262144 PTEs. The benefits come from greater coverage, fewer lower tables, and shorter translation paths.

This is a foundation for Linux virtual memory. When studying NPU DVAs, device MMUs, [IOMMUs and DMA addresses](memory-addressing.md), [pinned memory](memory-addressing.md#host-pinned-memory), and MemFabric's [GVA unified addressing](gva-unified-addressing.md), distinguish address space, page granularity, permissions, and transport. Do not assume they use this chapter's x86-64 levels or page sizes.

Connect the ideas in one sentence: **The CPU uses a virtual address, finds the corresponding physical page through the TLB or page tables, and combines it with the page offset to obtain a physical address; the central advantage of huge pages is that each mapping covers more memory, reducing mapping counts and TLB pressure.**

Continue with [How GVA unified addressing works](gva-unified-addressing.md) to connect page-table translation within one address space with resource addressing across processes, devices, and nodes.

Return to [Memory and addresses](memory-addressing.md) to connect a single translation with virtual-range reservation, physical allocation, mapping, and registration.
