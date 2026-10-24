# 🧮 Optimizers: turn gradients into parameter updates

Backpropagation computes derivatives; an optimizer decides how to use them. A useful optimizer discussion specifies the gradient, learning rate, state, regularization, and update schedule. “Adam is fast” or “SGD generalizes better” is not a universal selection rule.

## 🧭 SGD and momentum

Mini-batch stochastic gradient descent updates

$$
\theta_{t+1}=\theta_t-\eta_tg_t,
$$

where $g_t$ estimates the gradient of the training objective. A batch averages observations to reduce noise, but the estimate need not be unbiased when sampling, weighting, or preprocessing changes the distribution.

Momentum maintains a running direction, for example $v_t=\beta v_{t-1}+g_t$, then steps along $v_t$. Conventions differ by a factor of $1-\beta$, so compare implementations before copying hyperparameters. Momentum can reduce zigzagging in narrow valleys; excessive step sizes can still diverge.

## 🧩 RMSProp and Adam

RMSProp tracks an exponential average of squared gradients and divides the current gradient by its square root. Adam additionally tracks the first moment:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

With zero initial states, bias-corrected estimates are $\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$. The update is $-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$, applied elementwise. The second moment is an average of **squared gradients**, not the centered variance by itself.

Adam rescales coordinates according to gradient history. It does not eliminate vanishing gradients, guarantee convergence for arbitrary settings, or make learning-rate tuning unnecessary.

## 🧩 Worked example: the first Adam step

For a scalar first gradient $g_1=2$, $\beta_1=0.9$, and $\beta_2=0.999$, the raw moments are 0.2 and 0.004. Bias correction gives 2 and 4. Ignoring tiny $\epsilon$, the normalized step magnitude is one, so learning rate 0.001 changes the parameter by approximately 0.001.

This also explains why changing the scale of a gradient does not translate directly into the same change in an adaptive update. Later steps depend on the full history, and $\epsilon$ matters for small values.

## 🧠 AdamW and memory costs

AdamW decouples weight decay from gradient adaptation:

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-
\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Adding an L2 derivative to an Adam gradient instead changes the moment estimates and adaptive scaling. That is generally not the same operation. Decide which parameters receive decay; biases and normalization parameters are often treated separately, according to the training recipe.

Adam keeps two optimizer-state arrays. With two FP32 moments, that is eight bytes per parameter before gradients, model weights, master copies, or temporary buffers. This connects optimizer choice to [DeepSpeed and ZeRO](../llm/tools/deepspeed.md).

## 🔄 A reliable tuning workflow

First verify that a tiny batch can be fit. Then inspect the gradient norm and update-to-weight ratio while sweeping learning rate. Add warmup or decay based on the architecture and training budget. If accumulating gradients, normalize the loss correctly and step the optimizer at the intended effective-batch boundary.

Zero-initializing every weight in a symmetric hidden layer prevents units from differentiating. However, zero initialization of biases or selected residual/adaptation branches can be intentional; “no parameter may start at zero” is incorrect.

**Question:** why save optimizer state in a training checkpoint? **Answer:** restoring only weights resets momentum and adaptive histories, changing the future trajectory even with the same data order.

## 📚 Sources and further reading

- [Adam paper](https://arxiv.org/abs/1412.6980) defines moments and bias correction.
- [AdamW paper](https://arxiv.org/abs/1711.05101) distinguishes decoupled decay from L2 regularization.
