# Learning objectives and pre-training data

# Possible objects

Prefix, Masked, Full Language Modeling

![image.png](Learning%20objectives%20and%20pre-training%20data/image.png)

# Preprocessing clean texts

tokenization+batching

![image.png](Learning%20objectives%20and%20pre-training%20data/image%201.png)

# Subword Tokenization

## BPE

1. vocab with every character
2. merge most frequent
3. re-tokenize
4. repeat

![image.png](Learning%20objectives%20and%20pre-training%20data/image%202.png)

## Pros

1. Controlled vocabulary size
2. Learned vocabulary, best compromise between character-level and word-level tokenization
3. Units are well-suited to neural methods

## Further Quandries

1. When treating characters as your basic units, unknown (sub)tokens can still exist
2. Algorithm requires an initial word-level tokenization. This can be tricky
3. Step of finding most frequent pairs has a naive runtime of O(n2)

# Batching Data

![image.png](Learning%20objectives%20and%20pre-training%20data/image%203.png)

**OPTION 1**: Each document gets placed into its own batch.

![image.png](Learning%20objectives%20and%20pre-training%20data/image%204.png)

**OPTION 2**: Documents get patched into batches.

![image.png](Learning%20objectives%20and%20pre-training%20data/image%205.png)