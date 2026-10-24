# ⚡ Activation functions: why depth needs nonlinearity

Stacking linear layers alone does not create a genuinely deeper function family: $W_2(W_1x+b_1)+b_2$ is still one affine transformation. Activation functions break that collapse. They determine how signals and gradients pass through a network, while output activations also encode assumptions about the target.

## 🧭 The main functions and their roles

| Function | Definition | Useful interpretation |
| --- | --- | --- |
| Sigmoid | $1/(1+e^{-x})$ | Maps a score to a binary probability |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | Bounded, zero-centered hidden signal |
| ReLU | $\max(0,x)$ | Passes positive inputs; blocks negative ones |
| SiLU / Swish with $\beta=1$ | $x\sigma(x)$ | Smooth, input-dependent gating |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | A distribution over mutually exclusive alternatives |

The original notes' “SWISS” refers to **Swish/SiLU**. SiLU is the $\beta=1$ case of the more general $x\sigma(\beta x)$ family. “Smooth” does not mean “always better”: activation, initialization, normalization, and architecture interact.

## 🎯 Follow the gradient

Sigmoid has derivative $\sigma(x)(1-\sigma(x))$, at most $1/4$, and approaches zero for large-magnitude inputs. Repeated multiplication by small derivatives can weaken gradient signals through depth. Tanh also saturates at its extremes.

ReLU has derivative one for positive inputs and zero for negative inputs, apart from the nondifferentiable origin where implementations choose a convention. A unit that stays negative for every relevant input can stop receiving useful gradient through that path. Leaky ReLU retains a small negative-side slope; SiLU uses a smooth gate. Neither eliminates every optimization problem.

## 🧩 Worked example: stable softmax

For logits `[1000, 1001, 1002]`, directly evaluating exponentials can overflow. Subtracting the maximum does not change softmax because the common exponential factor cancels.

```python
# runnable-example
from math import exp
z = [1000, 1001, 1002]
a = [exp(v - max(z)) for v in z]
p = [v / sum(a) for v in a]
assert abs(sum(p) - 1) < 1e-12
assert abs(p[2] - 0.6652409558) < 1e-9
```

Adding the same constant to every logit leaves the probabilities unchanged. Multiplying logits by a positive constant changes sharpness; it is equivalent to changing temperature.

![Activation functions: why depth needs nonlinearity, original figure 1](../../../assets/notion/080-00.png)

The sigmoid curve becomes nearly flat at both extremes; its middle is the most sensitive region.

![Activation functions: why depth needs nonlinearity, original figure 2](../../../assets/notion/080-01.png)

Tanh is centered at zero but still saturates. A zero-centered output does not prevent vanishing gradients.

![Activation functions: why depth needs nonlinearity, original figure 3](../../../assets/notion/080-02.png)

ReLU preserves the positive half-line and sets the negative half to zero; this explains both its simple gradient and inactive-unit failure mode.

![Activation functions: why depth needs nonlinearity, original figure 4](../../../assets/notion/080-03.png)

This original softmax plot illustrates probabilities for one chosen vector of scores. Softmax is a vector-valued normalization, not a standalone scalar curve: changing any competing logit changes the other probabilities too.

![Activation functions: why depth needs nonlinearity, original figure 5](../../../assets/notion/080-04.png)

The original Swish plot includes its derivative. For SiLU, $f'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))$. For general $x\sigma(\beta x)$, the second term also contains a factor of $\beta$. Read the parameter convention before copying a derivative.

## 📦 Match the output to the task

Use independent sigmoids for labels that can coexist, such as “contains a person” and “contains a bicycle.” Use softmax when exactly one class is selected from the modeled alternatives. A continuous regression target often needs no bounded output activation; bounding it without justification can prevent valid predictions.

When a library loss expects logits, pass logits. Applying softmax before a combined cross-entropy implementation can distort optimization and waste numerical stability. An activation's mathematical output range does not by itself guarantee calibrated probabilities.

## 🚀 Going further

GELU weights an input by a Gaussian cumulative probability and is common in Transformer feed-forward blocks. Gated variants such as SwiGLU apply a nonlinear branch that multiplies another learned projection. They change both expressiveness and parameter count, so comparisons should control compute and width.

**Exercise:** why can a two-layer linear model not learn XOR? **Answer:** the composition remains affine, while XOR requires a nonlinear separation. Adding a hidden nonlinearity allows different regions of the input space to be treated differently.

Continue with [loss functions](loss-functions.md), [normalization](../llm/components/normalization.md), and [training diagnosis](training-challenges.md).

## 📚 Sources and further reading

- [PyTorch: SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) specifies the activation.
- [GELU paper](https://arxiv.org/abs/1606.08415) and [GLU variants](https://arxiv.org/abs/2002.05202) explain alternatives used in deep networks.
