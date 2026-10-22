# Inference

# Greedy Decoding

Greedy decoding always selects **the token with the highest probability** at that step and appends it to the sequence. 

The process repeats until the model generates an end-of-sequence token or reaches the maximum length limit.

This method is **fast, deterministic, and easy to implement**, since it doesn’t involve random sampling or exploring multiple possible sequences.

However, it can easily get **stuck in local optima** — often produces **repetitive or less diverse outputs** compared to more advanced strategies like beam search or nucleus sampling.

# Beam Search

top-k most likely sequences at each step instead of only one.

![image.png](Inference/image.png)

# Top-k Sampling

the model first keeps only the **top k tokens with the highest probabilities** and **renormalizes** their probabilities so they sum to 1.

Then, it **randomly samples** one token from this reduced set according to the adjusted probabilities.

This makes the output **more diverse** and often **more natural**, since it avoids deterministic repetition like greedy decoding—but it also introduces randomness, so the same prompt can yield different results each time.

# Top-P Sampling

Instead of keeping a fixed number of top-k tokens, it dynamically chooses the **smallest set of tokens whose cumulative probability ≥ p** (for example, p = 0.9).

Then it renormalizes those tokens’ probabilities and randomly samples one from this set.

This way, the model adapts the number of candidate tokens to the uncertainty of the prediction — using more options when the distribution is flat and fewer when it’s confident — producing fluent, diverse, and context-aware text.

# Temperature

**Temperature** controls how “creative” or random a model’s output is.

When **temperature < 1**, the probability distribution becomes sharper — the model favors high-probability tokens, producing more focused and deterministic text.

When **temperature > 1**, the distribution flattens — low-probability tokens get more chance, making the output more diverse and unpredictable.

In short:

> Lower temperature → more stable and precise; higher temperature → more random and creative.
>