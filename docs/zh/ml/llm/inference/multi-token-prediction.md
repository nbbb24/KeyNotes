!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Multi Token prediction

https://medium.com/yugen-ai-technology-blog/deepseek-v3-advances-in-moe-load-balancing-and-multi-token-prediction-training-f6d68c59749c

![image.png](../../../../assets/notion/057-00.png)

![image.png](../../../../assets/notion/057-01.png)

1. Input x1, x2, …., xt are fed to a shared transformer trunk to produce a hidden representation z
2. n  independent output heads, each head responsible for predicting a position-specific token
3. A unembedding matrix that is shared by all prediction heads

![image.png](../../../../assets/notion/057-02.png)

### Deepseek V3

sequential prediction

**D sequential MTP modules** are used to **predict D additional tokens**, each have:

1. A shared embedding layer
2. A shared output head
3. A transformer block and,
4. A projection matrix

![image.png](../../../../assets/notion/057-03.png)

combine representations of the **current token at the previous depth k-1 and the embedding of the future token.**

1. RMS Norm to both before concatenated
2. combination is multiplied by projection matrix - specific to the prediction depth k
3. output is then fed to a transformer block
4. output is passed on to an output head, calculates prob dis for kth additional token

During inference, MTP modules can be discarded
