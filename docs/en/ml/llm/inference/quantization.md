# Quantization: trade numeric precision for storage and speed

Quantization represents weights or activations with fewer bits. It can reduce memory traffic and capacity requirements, but its speed and quality depend on the format, scaling scheme, kernels, and workload.

## Fix the units first

One byte is eight bits. One billion FP16 parameters occupy approximately 2 GB, or 1.86 GiB, for weights alone. Ideal 8-bit storage uses 1 GB; ideal 4-bit storage uses 0.5 GB. Scales, zero points, padding, unquantized layers, and runtime buffers add overhead. Decimal GB means $10^9$ bytes; binary GiB means $2^{30}$ bytes.

## An affine quantizer

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\qquad
\hat x=s(q-z).
$$

The scale s sets resolution, and zero point z aligns real zero to an integer. A symmetric signed quantizer often uses z=0. Smaller scales resolve small differences but risk clipping large values; larger scales preserve range but increase rounding error.

For s=0.1 and z=0, x=0.26 becomes q=3 and reconstructs as 0.3, an error of 0.04. Quantization error can accumulate through layers, and outlier activations can make one global scale ineffective.

## Choose what and how to quantize

Per-tensor, per-channel, and per-group scales trade metadata and complexity for accuracy. Weight-only quantization reduces stored weights while activations may remain floating point. Weight-and-activation quantization can reduce more computation/traffic but is often harder to calibrate. KV-cache quantization targets context-dependent memory instead of model weights.

Post-training quantization uses calibration or optimization after training. Quantization-aware training simulates quantization effects during optimization. BF16/FP16 mixed-precision training is related to precision management but is not identical to integer quantization.

## Validate the real deployment

Measure quality on representative and difficult inputs, not just average perplexity. Check that the serving backend has an efficient kernel for the exact format and shape. Dequantization overhead can erase theoretical gains at some batch sizes. Long context and rare numeric tasks may expose errors hidden by short benchmarks.

**Exercise.** Does a 4-bit checkpoint imply total process memory is one quarter of FP16? No. Activations, KV cache, metadata, and other allocations may dominate.

References: [GPTQ](https://arxiv.org/abs/2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438), [QLoRA](https://arxiv.org/abs/2305.14314).
