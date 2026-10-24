# 🏗️ LLM architectures: from recurrent attention to Transformers

Architecture specifies how information moves through a model. Decoding specifies how outputs are chosen. Keeping those separate prevents confusing a beam-search improvement with a change to the network itself.

## 🔄 Begin with sequence search

![LLM architectures: from recurrent attention to Transformers, original figure 1](../../../../../assets/notion/044-00.png)
![LLM architectures: from recurrent attention to Transformers, original figure 2](../../../../../assets/notion/044-01.png)

*Multiply edge probabilities, or equivalently add log probabilities, to score a path. The second diagram prunes to a beam of two prefixes.*

In the first diagram, greedy selection begins with b at probability 0.4, but the path a→b has probability $0.35\times0.8=0.28$, exceeding b→b at $0.4\times0.5=0.20$. This illustrates why local and sequence-level optima can differ. Beam search is still approximate because pruned prefixes cannot return.

## 🧩 Attention removes the fixed-summary bottleneck

An attention-based decoder forms a context vector from source states:

$$
c_j=\sum_i\alpha_{ji}h_i^{enc},\qquad
\alpha_{ji}=\frac{\exp(e_{ji})}{\sum_k\exp(e_{jk})}.
$$

The sum normalizes over source positions for one decoder query. Index conventions differ across slides, so identify the query and key axes before copying an equation.

![LLM architectures: from recurrent attention to Transformers, original figure 3](../../../../../assets/notion/044-02.png)
![LLM architectures: from recurrent attention to Transformers, original figure 4](../../../../../assets/notion/044-03.png)
![LLM architectures: from recurrent attention to Transformers, original figure 5](../../../../../assets/notion/044-04.png)
![LLM architectures: from recurrent attention to Transformers, original figure 6](../../../../../assets/notion/044-05.png)

*The context is a weighted source-state sum; the decoder combines it with its own state. The normalization denominator must range over the available source positions.*

## 🏗️ Q, K, and V in each architecture

In encoder self-attention, all three come from encoder states. In decoder causal self-attention, all three come from decoder states with a causal mask. In encoder–decoder cross-attention, **queries come from the decoder; keys and values come from the encoder**.

![LLM architectures: from recurrent attention to Transformers, original figure 7](../../../../../assets/notion/044-06.png)

*The compact slide shows raw scaled dot products. Standard scaled dot-product attention applies softmax over those scores before weighting values; omitting it changes the operation.*

![LLM architectures: from recurrent attention to Transformers, original figure 8](../../../../../assets/notion/044-07.png)
![LLM architectures: from recurrent attention to Transformers, original figure 9](../../../../../assets/notion/044-08.png)

*These block diagrams use residual additions, normalization, and FFNs. Function-argument order is notation-dependent; trace the Q/K/V sources instead of assuming the first displayed argument is always Q.*

## 🏗️ Visibility distinguishes model families

![LLM architectures: from recurrent attention to Transformers, original figure 10](../../../../../assets/notion/044-09.png)

*The patterns contrast causal decoding, encoder–decoder flow, and bidirectional-prefix/causal-output behavior. Read rows as receiving positions and columns as visible positions under the chosen convention.*

RNNs have a sequential time dependency; they are not incapable of all parallel computation. Transformers parallelize token-position computation during teacher-forced training, while ordinary autoregressive decoding still advances sequentially.

**Exercise.** With 12 source tokens and 5 decoder queries, cross-attention scores have what shape per head? $5\times12$.

Reference: [Transformer](https://arxiv.org/abs/1706.03762). Continue with [architecture advances](architecture-advances.md).
