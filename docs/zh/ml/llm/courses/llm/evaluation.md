!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Automatic evaluation

# Perplexity

![image.png](../../../../../assets/notion/041-00.png)

# BLEU (precision)

![image.png](../../../../../assets/notion/041-01.png)

**Pros:**

○ relaxes exact match
○ Handles tasks with multiple target sequences
○ correlation with human preferences (MT)

**Cons:**

○ semantically similar words ignored

# ROUGE (recall)

![image.png](../../../../../assets/notion/041-02.png)

**Pros:**

○ relaxes exact match
○ correlation with human preferences (MDS)

**Cons:**

○ semantically similar words ignored

# Character n-gram precision (chrP)

![image.png](../../../../../assets/notion/041-03.png)

# Character n-gram F-score (chrF)

![image.png](../../../../../assets/notion/041-04.png)

**Pros:**

○ relaxes exact match and captures (some) morphological similarity

**Cons:**

○ does not capture similarity when there is no character overlap

# BERT-based similarity

![image.png](../../../../../assets/notion/041-05.png)

# BERT-based precision and recall

![image.png](../../../../../assets/notion/041-06.png)

# COMET

![image.png](../../../../../assets/notion/041-07.png)

![image.png](../../../../../assets/notion/041-08.png)
