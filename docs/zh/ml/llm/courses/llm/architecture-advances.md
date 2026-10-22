!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Architecture Advancement of Transformer

# FFN

![image.png](../../../../../assets/notion/040-00.png)

### Bilinear FFNs

![image.png](../../../../../assets/notion/040-01.png)

### Efficient Attention

**Grouped-Query Attention:**

Divide Q in G groups,

and share K, V in the same group

![image.png](../../../../../assets/notion/040-02.png)

**Multi-Query Attention:**

Single K, V for all Q heads

![image.png](../../../../../assets/notion/040-03.png)

![image.png](../../../../../assets/notion/040-04.png)

**Multi-Head Latent Attention:**

 Project K, V into a lower dimension latent vector

![image.png](../../../../../assets/notion/040-05.png)

# Layer Norm

![image.png](../../../../../assets/notion/040-06.png)

![image.png](../../../../../assets/notion/040-07.png)

# Position Encoding

## relative position information

![image.png](../../../../../assets/notion/040-08.png)

## rotational position embedding

![image.png](../../../../../assets/notion/040-09.png)

![image.png](../../../../../assets/notion/040-10.png)

![image.png](../../../../../assets/notion/040-11.png)
