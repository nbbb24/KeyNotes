# 🚀 DeepSpeed and ZeRO: fit training states across devices

DeepSpeed is a deep-learning optimization system. ZeRO, its Zero Redundancy Optimizer family, reduces replicated training state across data-parallel workers. The name is DeepSpeed; it is separate from the DeepSeek model family.

## 💡 Why weights are not the whole memory bill

Training stores weights, gradients, optimizer state, activations, and temporary buffers. Adam-like optimizers maintain two moment estimates, often in FP32, and mixed-precision recipes may also keep master weights. A model whose inference weights fit on one device can still exceed its memory during training.

Ordinary data parallelism replicates model state on each worker. ZeRO partitions progressively more state:

| Stage | Partitioned across data-parallel workers |
| --- | --- |
| ZeRO-1 | Optimizer states |
| ZeRO-2 | Optimizer states and gradients |
| ZeRO-3 | Optimizer states, gradients, and parameters |

With eight workers, ideal evenly partitioned state occupies one eighth per worker for the portions being sharded. Activations, buffers, and temporarily gathered parameters do not necessarily shrink by that factor.

## 🧠 Memory savings exchange work with communication

ZeRO-3 gathers parameters when computation needs them and releases/reuses storage according to the execution schedule. Offloading can place selected states in CPU memory or NVMe, adding transfer costs. The best configuration depends on interconnect bandwidth, host memory, storage, batch size, and overlap—not simply the highest stage number.

Activation checkpointing addresses a different memory component by recomputing activations during backward. Tensor and pipeline parallelism partition computation in other ways. These techniques can coexist but require coherent process groups and batch accounting.

## 🧩 An illustrative configuration

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {"stage": 2}
}
```

This is a configuration fragment, not a complete training program. With four data-parallel workers, its effective batch is $2\times8\times4=64$ examples per optimizer step, assuming ordinary data-parallel accumulation. Specify the optimizer, precision, launcher, and model integration for an actual run.

**Exercise.** Does ZeRO-1 remove replicated gradients? No; that is added in ZeRO-2. Does ZeRO-3 make checkpoint saving identical to single-device training? No; ensure the chosen save/consolidation procedure captures the required shards.

References: [ZeRO paper](https://arxiv.org/abs/1910.02054), [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/). Connect to [Infra interconnects](../../../infra/interconnects.md).
