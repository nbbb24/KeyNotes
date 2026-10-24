# 📊 Measure transfer performance without confusing the units

A useful benchmark answers a specific question: how long does useful work take under a defined payload, route, concurrency, and completion guarantee? A single “bandwidth” number without these conditions cannot explain a real pipeline.

## 🧮 Units and workload shape

| Quantity | Meaning | Common trap |
| --- | --- | --- |
| Bandwidth | Transferred payload bytes per elapsed second | Mixing bits/bytes or decimal/binary prefixes |
| Latency | Time per operation | Timing submission instead of completion |
| Throughput | Useful work per second | Counting queued rather than completed work |
| IOPS | Completed operations per second | Comparing different operation sizes |
| P50 / P99 | Median / 99th-percentile latency | Reporting averages that hide a long tail |
| Chunk / batch / queue depth | Bytes per piece / grouped operations / outstanding operations | Changing all three together |

One GiB is $2^{30}$ bytes; one GB is $10^9$ bytes. Transferring 1 GiB in 20 ms gives 50 GiB/s, or about 53.69 GB/s. A nominal 400 Gbit/s link corresponds to 50 GB/s before protocol and system overheads. Summing both directions of a full-duplex test does not produce a one-way result.

```python
# runnable-example
size_bytes = 2**30
elapsed_s = 0.020
print(size_bytes / elapsed_s / 1e9)  # 53.6870912 GB/s
assert size_bytes / elapsed_s / 2**30 == 50
```

## 🧩 Separate fixed cost from sustained bandwidth

Fit the rough model $T(S)=L+S/B$ using multiple sizes. Small transfers reveal submission and synchronization overhead; large transfers expose bandwidth. Increasing queue depth can hide latency until a shared resource saturates, after which it can worsen tail latency. Little's law relates average in-flight work to throughput times average latency in a stable system.

Measure cold initialization and steady state separately. Warm up explicitly, time through completion, validate destination contents, and report repetitions and latency distribution. When testing overlap, compare elapsed end-to-end time and a timeline; enqueueing copy and compute asynchronously is not proof that hardware runs them concurrently.

## 🔍 Linux inspection tools

These are observational examples for an appropriately equipped Linux host, not commands to run on every development machine.

| Purpose | Command | Interpretation |
| --- | --- | --- |
| NPU health and identity | `npu-smi info` | Match visible devices to the target job |
| PCIe tree and link details | `lspci -tv`, `lspci -vv` | Inspect shared switches and negotiated capabilities |
| NUMA and CPU topology | `numactl -H`, `lscpu` | Compare device locality with thread and memory placement |
| Network interfaces | `ip -br link` | Confirm intended interface and state |
| RDMA devices | `rdma link show` | Identify available RDMA resources |
| NIC capabilities/counters | `ethtool <interface>`, `ethtool -S <interface>` | Inspect link and device-specific error/congestion counters |

Ascend `ascend-dmi` can provide bandwidth and link diagnostics on supported products. Use the installed version's help and [matching Ascend tool documentation](https://www.hiascend.com/document) for supported H2D/D2H/D2D/P2P modes; do not assume identical options across generations.

## 🚀 A reproducible experiment

Keep hardware, software, backend, direction, buffer type, NUMA placement, and completion rule fixed. Sweep payload size first, then concurrency. Record CPU utilization, memory bandwidth where available, port counters, and P99 alongside payload throughput. A host-staged path can saturate DRAM before the network, while a shared uplink can cap otherwise independent devices.

Finally connect the microbenchmark to application behavior. Faster bulk transfer may not improve token latency if the application mostly sends tiny fragments or waits for a barrier. Use [the end-to-end model](end-to-end-transfer.md) to choose the next measurement.
