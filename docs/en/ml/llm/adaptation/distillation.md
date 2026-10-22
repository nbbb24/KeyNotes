# Knowledge distillation: train a student from a teacher

Distillation transfers behavior from a teacher model to a student. It is useful when the teacher is accurate but expensive, or when a smaller model should inherit task-specific behavior. The teacher can be one model or an ensemble.

## Soft targets contain more than the top label

For teacher logits $z^T$, student logits $z^S$, and temperature $T$:

$$
p_i^T=\operatorname{softmax}(z^T/T)_i,\quad
p_i^S=\operatorname{softmax}(z^S/T)_i.
$$

A common objective combines hard-label cross-entropy and soft-target matching:

$$
L=(1-\lambda)L_{hard}+\lambda T^2D_{KL}(p^T\Vert p^S).
$$

Higher temperature exposes relative probabilities among alternatives. The $T^2$ factor compensates for the approximate gradient-scale change in the standard setup. Teacher targets are detached from gradient updates when the teacher is fixed.

If a teacher assigns `[0.7, 0.25, 0.05]` to dog, wolf, and airplane, the student learns that wolf is a more plausible confusion than airplane. A one-hot dog label lacks that relationship. This information is useful only to the extent that the teacher's distinctions are useful.

## Other forms

Feature distillation matches hidden states, sometimes through a projection when widths differ. Sequence distillation trains on teacher-generated outputs. Reasoning-data distillation uses curated teacher solutions, but copying long explanations does not guarantee the student acquired reliable reasoning. Validate answers and process quality separately.

For different tokenizers, direct token-logit matching is not straightforward because output coordinates refer to different units. Sequence-level supervision or an explicit alignment method may be more suitable.

## Evaluate the student independently

Compare quality, latency, memory, and robustness at the same task settings. A smaller student can inherit teacher mistakes or amplify dataset shortcuts. Distillation data must cover the deployment distribution; a strong teacher queried only on easy examples may leave important behaviors untaught.

**Exercise.** Is every small connector a distillation module? No. Size reduction alone is insufficient; distillation uses a teacher-derived learning signal.

Reference: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). Related: [DeepSeek-R1](../models/deepseek-r1.md), [cross-entropy and KL](../components/cross-entropy-kl.md).
