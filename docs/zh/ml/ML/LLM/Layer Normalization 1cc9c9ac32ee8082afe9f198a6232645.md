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