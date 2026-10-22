!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# DeepseekV3

[https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4](https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4)

paper[https://arxiv.org/pdf/2412.19437](https://arxiv.org/pdf/2412.19437)

### Multi-Head Latent Attention

map down→map up

to reduce KV cache

![image.png](../../../../assets/notion/030-00.png)

### DeepseekMoE

![image.png](../../../../assets/notion/030-01.png)

Ns: shared experts, Nr: routed experts

**Auxiliary-Loss-Free Load Balancing**

**routing collapse** some experts being overused

bias term is only used for routing

decrease the bias if expert is overloaded, increase if underloaded

![image.png](../../../../assets/notion/030-02.png)

**Complementary Sequence-Wise Auxiliary Loss**

prevent extreme imbalance within single sequence

![image.png](../../../../assets/notion/030-03.png)

Pi: the average routing weight of expert_i

fi: frequency expert_i selected

### Muti-Token Prediction

![image.png](../../../../assets/notion/030-04.png)
