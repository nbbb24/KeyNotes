!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# LLM architectures

# Beam search

![image.png](../../../../../assets/notion/044-00.png)

![image.png](../../../../../assets/notion/044-01.png)

# Attention

![image.png](../../../../../assets/notion/044-02.png)

![image.png](../../../../../assets/notion/044-03.png)

![image.png](../../../../../assets/notion/044-04.png)

![image.png](../../../../../assets/notion/044-05.png)

𝛼𝑖,𝑗 =how important the encoder hidden state at position 𝑖 is to the model’s prediction at position j

# Problem of RNN

Recurrent neural networks are slow to train. Computation cannot be parallelized

Recurrent neural networks do poorly with long contexts

# Generic Attention Mechanism

**Values**: previous layer’s hidden states;

for Encoder-Decoder, **values** are final hidden states of encoder, **keys** are hidden states from target sequence

**query** is current focus of attention

![image.png](../../../../../assets/notion/044-06.png)

# Encoder

# Decoder

![image.png](../../../../../assets/notion/044-07.png)

![image.png](../../../../../assets/notion/044-08.png)

# Different Transformer Architectures

![image.png](../../../../../assets/notion/044-09.png)
