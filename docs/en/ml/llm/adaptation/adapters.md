# Adapters: learn a small task-specific addition

Full fine-tuning creates a complete updated model for each task. Adapters instead insert small trainable modules into a largely frozen network. The base model can be shared while each task stores its own compact addition.

## Bottleneck computation

A common adapter maps hidden width $d$ down to a smaller width $r$, applies a nonlinearity, then maps back:

$$
h'=h+W_{up}\,\phi(W_{down}h+b_{down})+b_{up}.
$$

The main weight count is $2dr$, compared with $d^2$ for a square dense matrix. With $d=768$ and $r=64$, that is 98,304 weights per adapter, excluding biases. Placement and the number of adapters per layer determine the total.

The residual path and near-identity initialization help preserve pretrained behavior at the start. The inserted nonlinear computation gives the adaptation its own capacity. Depending on the recipe, layer-normalization parameters or task heads may also be trained; record the actual trainable parameter list.

## A practical use case

A shared encoder serves sentiment classification and support-ticket routing. Each task loads its own adapter and classifier while retaining the same base weights. This reduces checkpoint storage and optimizer-state memory, but the base model still must be loaded and executed.

## Tradeoffs versus LoRA

A bottleneck adapter typically adds operations on the inference path. LoRA learns a low-rank weight update that can often be merged into a compatible dense weight. Neither method guarantees full-fine-tuning quality on every task. Compare under the same data, learning-rate search, and evaluation budget.

Freezing a layer does not mean all its intermediate computation can be skipped during training: gradients may need to flow through it to reach an earlier adapter. Activation memory can therefore remain substantial.

**Exercise.** If 24 adapters each contain 98,304 weights, how many adapter weights are stored? 2,359,296, before biases and heads. This count is separate from the frozen backbone.

Reference: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751). Continue with [LoRA](lora.md) and [SFT](sft.md).
