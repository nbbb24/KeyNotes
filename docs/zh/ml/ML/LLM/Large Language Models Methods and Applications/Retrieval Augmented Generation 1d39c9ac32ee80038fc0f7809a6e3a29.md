# Retrieval Augmented Generation

![image.png](Retrieval%20Augmented%20Generation/image.png)

# REALM

![image.png](Retrieval%20Augmented%20Generation/image%201.png)

KNN research

# RETRO

Incorporating context in intermediate layers

![image.png](Retrieval%20Augmented%20Generation/image%202.png)

![image.png](Retrieval%20Augmented%20Generation/image%203.png)

![image.png](Retrieval%20Augmented%20Generation/image%204.png)

Given the input sequence, it first retrieves a set of relevant documents (embedding of text)

![image.png](Retrieval%20Augmented%20Generation/image%205.png)

Use cross-attention to generate retrieved context-aware representations

![image.png](Retrieval%20Augmented%20Generation/image%206.png)

Concatnate all of the CA output (the size of input H and output CCA(H,E) remains the same

![image.png](Retrieval%20Augmented%20Generation/image%207.png)

![image.png](Retrieval%20Augmented%20Generation/image%208.png)

# KNNLM

![image.png](Retrieval%20Augmented%20Generation/image%209.png)

![image.png](Retrieval%20Augmented%20Generation/image%2010.png)

# TRIME

![image.png](Retrieval%20Augmented%20Generation/image%2011.png)

# RAG

## limitations

● In-context retrieval-augmented LMs generate what is not fully supported by their citations
● They can easily get distracted by unhelpful context
● Diverse tasks require diffent retrieval needs e.g., content, frequency

---

# Optimizing LLM for rag: training/controlling LLMs with retrieved context

# SAIL

SAIL augments existing instruction-tuning data to teach the LM how to use retrieved context

![image.png](Retrieval%20Augmented%20Generation/image%2012.png)

# Self-RAG

Self-RAG teaches LMs to adaptively retrieve and evaluates context & own generation

![image.png](Retrieval%20Augmented%20Generation/image%2013.png)

![image.png](Retrieval%20Augmented%20Generation/image%2014.png)

---

# Optimizing Retriever for RAG: training retrievers for LLM RAG

# REPLUG

![image.png](Retrieval%20Augmented%20Generation/image%2015.png)

# RADIT

![image.png](Retrieval%20Augmented%20Generation/image%2016.png)

# introducing additional modules for reranking or filtering

![image.png](Retrieval%20Augmented%20Generation/image%2017.png)

---

# Optimizing Prompts for RAG: advanced prompt techniques

# DSPy

![image.png](Retrieval%20Augmented%20Generation/image%2018.png)