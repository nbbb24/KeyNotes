!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Layer Normalization

# Layer Norm

compute by col

add **scale and shift** parameter

$$
μ=\frac{1}{d}\sum_{i=1}^{d}x_i,  \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

# RMS Normalization

remove shift parameter

mean is not zero

but faster and stable

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
