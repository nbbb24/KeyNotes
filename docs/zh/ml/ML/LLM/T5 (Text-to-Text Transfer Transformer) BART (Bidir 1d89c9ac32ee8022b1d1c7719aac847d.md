# T5 (Text-to-Text Transfer Transformer)/BART (Bidirectional and Auto-Regressive Transformers)

# T5

[https://arxiv.org/pdf/1910.10683](https://arxiv.org/pdf/1910.10683)

Everything—**input and output**—is cast as text

use traditional transformerso 

![image.png](T5%20(Text-to-Text%20Transfer%20Transformer)%20BART%20(Bidir/image.png)

## prefix

The Corpus of Linguistic Acceptability (cola), binary classification

Semantic Textual Similarity Benchmark (stsb), similarity

# BART

[https://arxiv.org/pdf/1910.13461](https://arxiv.org/pdf/1910.13461)

![image.png](T5%20(Text-to-Text%20Transfer%20Transformer)%20BART%20(Bidir/image%201.png)

Token Masking

Token Deletion

Text Infilling: A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3)

Sentence Permutation

Document Rotation

![image.png](T5%20(Text-to-Text%20Transfer%20Transformer)%20BART%20(Bidir/image%202.png)