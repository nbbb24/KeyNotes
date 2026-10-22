!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Llama (Decoder-only)

[https://arxiv.org/pdf/2302.13971](https://arxiv.org/pdf/2302.13971)

# Pre-Normalization

Normalization is applied to the input of each sub-layer

Normalization is applied after each sub-layer in transformer

→ gradients vanishing/exploding

## RMSNorm

# SwiGLU (Sigmoid-Weighted Linear Unit)

$$

\text{GLU}(x) = x \ast \sigma(Wx + b)

$$

$$

\text{Swish}(x) = x \cdot \sigma(\beta x)
$$

$$

\text{SwiGLU}(x) = \left(x W_1 + b_1\right)_1 \cdot \sigma\left(\left(x W_1 + b_1\right)_2\right)

= x_1 \cdot \text{Swish}(x_2) = x_1 \cdot \left( x_2 \cdot \sigma(x_2) \right)

$$

![image.png](../../../../assets/notion/050-00.png)

![image.png](../../../../assets/notion/050-01.png)

![image.png](../../../../assets/notion/050-02.png)

### Pros

**1.Gating + Nonlinearity = More Expressiveness**

This makes it **more expressive** than simple activations (e.g., ReLU), because the model can **dynamically scale** and **filter** the features depending on the context.

**2.Smooth Gradients (thanks to Swish)**

The Swish activation is **smooth and differentiable everywhere**, unlike ReLU which has a zero gradient for negative inputs.

**3.Gated Feature Selection**

Like GLU, SwiGLU introduces **gating**, which allows the model to learn **what features to pass through** and **what to suppress** — a form of **attention at the neuron level**.

# RoPE

**Normalization before/after each layer?**
