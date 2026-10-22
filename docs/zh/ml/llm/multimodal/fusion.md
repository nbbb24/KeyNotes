!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Representation Fusion

Learn a joint representation that models cross-modal interactions between individual elements of different modalities

![image.png](../../../../assets/notion/067-00.png)

### Early fusion and Late fusion

![image.png](../../../../assets/notion/067-01.png)

### Linear Regression

![image.png](../../../../assets/notion/067-02.png)

![image.png](../../../../assets/notion/067-03.png)

![image.png](../../../../assets/notion/067-04.png)

![image.png](../../../../assets/notion/067-05.png)

### Additive Fusion

1-layer neural network can be seen as additive (late fusion)

![image.png](../../../../assets/notion/067-06.png)

### Multiplicative Fusion

simple and bilinear

![image.png](../../../../assets/notion/067-07.png)

### Tensor Fusion

have unimodal, bimodal, trimodal

![image.png](../../../../assets/notion/067-08.png)

### Low-Rank Fusion

![image.png](../../../../assets/notion/067-09.png)

![image.png](../../../../assets/notion/067-10.png)

### Tensor vs Low-Rank Fusion

![image.png](../../../../assets/notion/067-11.png)

### Gated Fusion

![image.png](../../../../assets/notion/067-12.png)

### Modality Shifting Fusion

![image.png](../../../../assets/notion/067-13.png)

### Non-Additive Interaction

![image.png](../../../../assets/notion/067-14.png)

![image.png](../../../../assets/notion/067-15.png)

### Heterogeneity-aware Fusion

![image.png](../../../../assets/notion/067-16.png)

![image.png](../../../../assets/notion/067-17.png)

### Improve Optimization

sometimes multimodal doesn’t help!

1. Multimodal networks are more prone to overfitting due to increased complexity
2. Different modalities overfit and generalize at different rates

![image.png](../../../../assets/notion/067-18.png)

![image.png](../../../../assets/notion/067-19.png)
