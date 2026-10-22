# LLM architectures

# Beam search

![image.png](LLM%20architectures/image.png)

![image.png](LLM%20architectures/image%201.png)

# Attention

![image.png](LLM%20architectures/image%202.png)

![image.png](LLM%20architectures/image%203.png)

![image.png](LLM%20architectures/image%204.png)

![image.png](LLM%20architectures/image%205.png)

𝛼𝑖,𝑗 =how important the encoder hidden state at position 𝑖 is to the model’s prediction at position j

# Problem of RNN

Recurrent neural networks are slow to train. Computation cannot be parallelized

Recurrent neural networks do poorly with long contexts

# Generic Attention Mechanism

**Values**: previous layer’s hidden states; 

for Encoder-Decoder, **values** are final hidden states of encoder, **keys** are hidden states from target sequence

**query** is current focus of attention

![image.png](LLM%20architectures/image%206.png)

# Encoder

# Decoder

![image.png](LLM%20architectures/image%207.png)

![image.png](LLM%20architectures/image%208.png)

# Different Transformer Architectures

![image.png](LLM%20architectures/image%209.png)