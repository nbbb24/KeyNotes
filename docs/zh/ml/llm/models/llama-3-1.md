!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Llama 3.1

[https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

Multilingual

# Multi-model encoder pre-training

separate encoders for images and speech

1. image encoder on image-text pairs
2. speech encoder by self-supervised (mask and reconstruct)

# Vision adapter training

integrate image encoder into LM

cross-attention image+language

during training update image encoder but not LM

video adapter top of Image adapter on video-text

# Speech adapter training

integrate speech encoder into model

# Model Architecture

![image.png](../../../../assets/notion/051-00.png)

**Grouped query attention:** 8 k-v heads

**Attention mask**

# Post-Training

![image.png](../../../../assets/notion/051-01.png)

## Reward Modeling

ranking (edited > chosen > rejected).

## DPO

**Masking out formatting tokens including header/termination tokens**

→undesired model behaviors such as tail repetition or abruptly generating termination tokens

**Regularization with NLL loss**

additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2

→further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses

## Rejection sampling

sample K outputs from latest/best model policy

use reward model to select best candidate

## Tool usage

![image.png](../../../../assets/notion/051-02.png)

# Inference

## FP8 Quantization

parameters and activations in FFN, 50% inference time

no quantization in first and last Transformer layers

# Vision Experiments

![image.png](../../../../assets/notion/051-03.png)

## Model

Image encoder: ViT
