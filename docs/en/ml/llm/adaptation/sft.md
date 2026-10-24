# 🎓 Supervised fine-tuning: teach the desired response behavior

Supervised fine-tuning, or SFT, trains a pretrained model on examples of inputs and desired outputs. For a chat model, this usually means conversations whose assistant responses demonstrate the target style, task, and behavior.

## 🎯 The objective and the data format

For prompt $x$ and response tokens $y_1,\ldots,y_T$:

$$
L_{SFT}=-\sum_{t=1}^T\log\pi_\theta(y_t\mid x,y_{<t}).
$$

A response-only setup masks prompt tokens out of the loss. Other recipes train on more of the conversation, so state the convention. Apply the model's chat template consistently: role delimiters, end-of-turn tokens, and special tokens are part of the sequence the model learns.

For a support assistant, a useful example includes the user's problem, relevant context, a correct response, and appropriate uncertainty when information is missing. Repeating polished but factually wrong answers teaches the wrong behavior efficiently.

## 🔄 A reliable training workflow

1. Define the task and collect representative, reviewed examples.
2. Split by source/user/time where needed to avoid near-duplicate leakage.
3. Inspect tokenized examples, truncation, masks, and target boundaries.
4. Train either all weights or an adaptation such as [LoRA](lora.md).
5. Compare with the base model on held-out task quality and retained general capabilities.

Packing examples saves padding but must preserve boundaries and the intended attention visibility. A lower training loss may reflect memorization or duplicated data rather than better behavior.

## ⚠️ What SFT can and cannot do

SFT is especially useful for output formats, domain procedures, and following instructions demonstrated in the data. It is not a dependable mechanism for continuously updating a large factual database; [retrieval](../rag.md) can provide changing evidence at inference time. Fine-tuning may also cause forgetting, so include regression tests for capabilities the application still needs.

**Exercise.** If an answer is truncated halfway through during preprocessing, what might the model learn? An incomplete response pattern or an incorrect end boundary. Inspect actual tokenized samples before interpreting training curves.

Reference: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). Continue with [RLHF](../alignment/rlhf.md).
