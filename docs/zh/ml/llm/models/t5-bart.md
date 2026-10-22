!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# T5 (Text-to-Text Transfer Transformer)/BART (Bidirectional and Auto-Regressive Transformers)

# T5

[https://arxiv.org/pdf/1910.10683](https://arxiv.org/pdf/1910.10683)

Everything—**input and output**—is cast as text

use traditional transformerso

![image.png](../../../../assets/notion/070-00.png)

## prefix

The Corpus of Linguistic Acceptability (cola), binary classification

Semantic Textual Similarity Benchmark (stsb), similarity

# BART

[https://arxiv.org/pdf/1910.13461](https://arxiv.org/pdf/1910.13461)

![image.png](../../../../assets/notion/070-01.png)

Token Masking

Token Deletion

Text Infilling: A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3)

Sentence Permutation

Document Rotation

![image.png](../../../../assets/notion/070-02.png)
