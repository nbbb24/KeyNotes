!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Learning objectives and pre-training data

# Possible objects

Prefix, Masked, Full Language Modeling

![image.png](../../../../../assets/notion/045-00.png)

# Preprocessing clean texts

tokenization+batching

![image.png](../../../../../assets/notion/045-01.png)

# Subword Tokenization

## BPE

1. vocab with every character
2. merge most frequent
3. re-tokenize
4. repeat

![image.png](../../../../../assets/notion/045-02.png)

## Pros

1. Controlled vocabulary size
2. Learned vocabulary, best compromise between character-level and word-level tokenization
3. Units are well-suited to neural methods

## Further Quandries

1. When treating characters as your basic units, unknown (sub)tokens can still exist
2. Algorithm requires an initial word-level tokenization. This can be tricky
3. Step of finding most frequent pairs has a naive runtime of O(n2)

# Batching Data

![image.png](../../../../../assets/notion/045-03.png)

**OPTION 1**: Each document gets placed into its own batch.

![image.png](../../../../../assets/notion/045-04.png)

**OPTION 2**: Documents get patched into batches.

![image.png](../../../../../assets/notion/045-05.png)
