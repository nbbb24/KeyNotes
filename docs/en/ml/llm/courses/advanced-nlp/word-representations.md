# Word representations: lookup tables, CBOW, and context

Words must become numeric inputs before a neural model can process them. An embedding table provides a compact representation while allowing the model to learn which words behave similarly in a particular training objective.

## Lookup is matrix multiplication without the waste

Let $E$ have shape $V\times d$. A one-hot vector for token ID $i$ multiplied by $E$ selects row $E_i$. An embedding lookup performs this selection directly, avoiding a length-$V$ dense vector. Gradients update the selected rows, subject to the optimizer's behavior.

![Original Word Representation figure 1](../../../../../assets/notion/014-00.png)

![Original Word Representation figure 2](../../../../../assets/notion/014-01.png)

## Continuous bag of words

CBOW predicts a center word from its surrounding words. A simple implementation averages context embeddings:

$$
h=\frac1{|C|}\sum_{c\in C}E_c,\qquad
p(w\mid C)=\operatorname{softmax}(Uh)_w.
$$

For `the cat sat on the mat`, a context around `sat` might include `cat` and `on`. Skip-gram reverses the direction: a center word predicts surrounding words. Training choices such as window size, subsampling, and negative sampling affect what relationships the vectors capture.

## What similarity means

Cosine similarity compares direction, not magnitude. Words used in similar contexts can be close even if their meanings oppose each other: `hot` and `cold` occur in similar grammatical environments. Nearest neighbors therefore reveal distributional behavior, not a guaranteed synonym dictionary. Analogy arithmetic is an empirical pattern and can encode biases from the corpus.

A static table supplies one vector per token ID. ELMo and Transformer models produce contextual states, so the same word can differ across sentences. Subword vocabularies improve coverage but require aggregation for word-level tasks.

**Exercise.** With $V=10{,}000$ and $d=128$, how many table parameters are there? 1,280,000. Increasing sentence length changes lookup work and activations, not the table's parameter count.

Reference: [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781). Continue with [ELMo](../../models/elmo.md) and [the Transformer progression](../../word2vec-to-transformer.md).
