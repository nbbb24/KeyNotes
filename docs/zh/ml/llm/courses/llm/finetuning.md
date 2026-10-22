!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Finetuning

![image.png](../../../../../assets/notion/042-00.png)

# BERT

![image.png](../../../../../assets/notion/042-01.png)

# Prompt Engineering

Verbalizer: The template we wrap an example in in order to perform the task.

# Parameter Efficient Tuning

![image.png](../../../../../assets/notion/042-02.png)

1. Addition: What if we introduce additional trainable parameters to the neural network and just train those?
2. Specification: What if we pick a small subset of the parameters of the neural network and just tune those?
3. Reparameterization: What if we re-parameterize the model into something that is more efficient to train?

## prompt tuning

prompt+query, ONLY in input

Create a new learnable embedding matrix 𝐏 ∈ ℝ𝑘×𝑑
○ Set the first k input embeddings to be learnable.
○ k is a hyperparameter up to the choice of the implementer.

![image.png](../../../../../assets/notion/042-03.png)

![image.png](../../../../../assets/notion/042-04.png)

## prefix tuning

at each layer of the Transformer

![image.png](../../../../../assets/notion/042-05.png)

![image.png](../../../../../assets/notion/042-06.png)

## Adapter

new NN modules that are added between layers of a pre-trained network

![image.png](../../../../../assets/notion/042-07.png)

linear project → low dimension → non-linear → high dimension

$$
𝑦 = 𝑾_𝟐 GELU (𝑾_𝟏𝒙 + 𝒃_1)+ 𝒃_2
$$

Pros:

● Faster to converge

● Have been shown to be quite effective in multi-task settings

Cons:

● Adding in new layers means

● It also makes the model bigger

## BitFit: Bias-terms Fine-tuning

Only tune the bias terms and final classification layer

![image.png](../../../../../assets/notion/042-08.png)

## DiffPruning

○ For each parameter, finetune a learnable “delta” which gets added to the original parameter value.
○ Use an L0-norm penalty to encourage sparsity in the deltas.

## Re-parameterizing the model

reparameterize a subset of the original model parameters with low-dimensional proxy parameters, and just optimize the proxy.

## LoRA

injects trainable rank decomposition matrices into each layer.

![image.png](../../../../../assets/notion/042-09.png)

## (IA)3 : Infused Adapter by Inhibiting and Amplifying Inner Activations

Rescale inner activations with lower-dimensional learned vectors, which are injected into the attention and feedforward modules

**VS LoRA:**

○ LoRA learns low-rank updates to the attention weights
○ (IA)3 learns injectable vectors.
