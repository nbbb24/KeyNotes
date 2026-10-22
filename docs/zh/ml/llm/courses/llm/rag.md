!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Retrieval Augmented Generation

![image.png](../../../../../assets/notion/046-00.png)

# REALM

![image.png](../../../../../assets/notion/046-01.png)

KNN research

# RETRO

Incorporating context in intermediate layers

![image.png](../../../../../assets/notion/046-02.png)

![image.png](../../../../../assets/notion/046-03.png)

![image.png](../../../../../assets/notion/046-04.png)

Given the input sequence, it first retrieves a set of relevant documents (embedding of text)

![image.png](../../../../../assets/notion/046-05.png)

Use cross-attention to generate retrieved context-aware representations

![image.png](../../../../../assets/notion/046-06.png)

Concatnate all of the CA output (the size of input H and output CCA(H,E) remains the same

![image.png](../../../../../assets/notion/046-07.png)

![image.png](../../../../../assets/notion/046-08.png)

# KNNLM

![image.png](../../../../../assets/notion/046-09.png)

![image.png](../../../../../assets/notion/046-10.png)

# TRIME

![image.png](../../../../../assets/notion/046-11.png)

# RAG

## limitations

● In-context retrieval-augmented LMs generate what is not fully supported by their citations
● They can easily get distracted by unhelpful context
● Diverse tasks require diffent retrieval needs e.g., content, frequency

---

# Optimizing LLM for rag: training/controlling LLMs with retrieved context

# SAIL

SAIL augments existing instruction-tuning data to teach the LM how to use retrieved context

![image.png](../../../../../assets/notion/046-12.png)

# Self-RAG

Self-RAG teaches LMs to adaptively retrieve and evaluates context & own generation

![image.png](../../../../../assets/notion/046-13.png)

![image.png](../../../../../assets/notion/046-14.png)

---

# Optimizing Retriever for RAG: training retrievers for LLM RAG

# REPLUG

![image.png](../../../../../assets/notion/046-15.png)

# RADIT

![image.png](../../../../../assets/notion/046-16.png)

# introducing additional modules for reranking or filtering

![image.png](../../../../../assets/notion/046-17.png)

---

# Optimizing Prompts for RAG: advanced prompt techniques

# DSPy

![image.png](../../../../../assets/notion/046-18.png)
