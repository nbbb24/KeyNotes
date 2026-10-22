# Architecture Advancement of Transformer

# FFN

![image.png](Architecture%20Advancement%20of%20Transformer/image.png)

### Bilinear FFNs

![image.png](Architecture%20Advancement%20of%20Transformer/image%201.png)

### Efficient Attention

**Grouped-Query Attention:** 

Divide Q in G groups,

and share K, V in the same group

![image.png](Architecture%20Advancement%20of%20Transformer/image%202.png)

**Multi-Query Attention:**

Single K, V for all Q heads

![image.png](Architecture%20Advancement%20of%20Transformer/image%203.png)

![image.png](Architecture%20Advancement%20of%20Transformer/image%204.png)

**Multi-Head Latent Attention:**

 Project K, V into a lower dimension latent vector

![image.png](Architecture%20Advancement%20of%20Transformer/image%205.png)

# Layer Norm

![image.png](Architecture%20Advancement%20of%20Transformer/image%206.png)

![image.png](Architecture%20Advancement%20of%20Transformer/image%207.png)

# Position Encoding

## relative position information

![image.png](Architecture%20Advancement%20of%20Transformer/image%208.png)

## rotational position embedding

![image.png](Architecture%20Advancement%20of%20Transformer/image%209.png)

![image.png](Architecture%20Advancement%20of%20Transformer/image%2010.png)

![image.png](Architecture%20Advancement%20of%20Transformer/image%2011.png)