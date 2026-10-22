# KV cache

### **1. How does the KV cache change the time and memory complexity of autoregressive decoding?**

Without caching, autoregressive decoding requires recomputing attention for all n previous tokens at each step, which is O(n^2). 

With KV caching, each new token only computes attention over already-stored keys/values, making the complexity **linear in sequence length**, O(n), and **constant per token**. 

Memory grows linearly with the number of layers and sequence length.

### **2. What happens to the KV cache when you use techniques like beam search or top-k sampling?**

In **beam search**, each beam must maintain its own KV cache since each beam may follow a different generation path. 

In **top-k/top-p sampling**, the cache simply grows per token — the only complexity is ensuring that the correct sampled path updates the cache.

### **3.If a transformer has 32 layers, 40 heads, head dim 128, and sequence length 2048 in float16, how much memory is needed for the KV cache?**

$$
32 \times 40 \times 2048 \times 128 \times 2 \text{ (K+V)} \times 2 \text{ (bytes)} = 1,310,720,000\ \text{bytes} \approx 1.25\ \text{GB}
$$

### **4. How do you manage/truncate KV cache in streaming inference?**

Implement a **moving window**: once the cache exceeds a limit (say 2048 tokens), drop the oldest keys/values (e.g., first 512) to make space. This trades off some context for constant memory usage.

# During inference

Prefill

Decoding

# Data type

float 16 

Bfloat 16