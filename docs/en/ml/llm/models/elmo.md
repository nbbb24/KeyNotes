# 🔤 ELMo: a word representation that changes with context

Static word embeddings give a word type one vector. ELMo addresses polysemy: `bank` should have different representations in a financial sentence and a river sentence. It learns contextual features from a bidirectional language model, then exposes those features to downstream tasks.

## ⚙️ How the representation is constructed

The original system uses character-based token representations followed by stacked forward and backward LSTMs. The forward model predicts from the left; the backward model predicts from the right. Their states are concatenated at corresponding positions. This is more precise than saying one recurrent model can see both directions while performing a causal prediction.

A downstream task learns a weighted combination of layers:

$$
\operatorname{ELMo}^{task}_k=\gamma^{task}\sum_{j=0}^{L}s_j^{task}h^{LM}_{k,j},
$$

where $s_j$ are softmax-normalized layer weights, $h_{k,j}$ are token representations from different levels, and $\gamma$ is a learned scale. The token layer and different recurrent layers may supply different information; using only the top layer discards this flexibility.

## 🧩 Work through an example

In “The bank raised rates,” the left/right recurrent states encode clues such as `raised` and `rates`. In “The bank was muddy,” different context changes the vector even though the spelling is identical. A sentiment or entity model can use these vectors as additional input features without relearning all language structure from its small labeled dataset.

## 🎯 What ELMo teaches about transfer learning

Pretraining and the downstream task are distinct. A language-model objective supplies broadly useful representations; a task-specific readout learns how to use them. Contextualization does not guarantee factual correctness or eliminate domain shift. Character input helps with morphology and unusual spellings but does not make unknown terminology automatically understood.

The LSTM computation remains sequential within each direction, which limits training parallelism compared with a Transformer. This historical tradeoff helps explain the transition to [BERT](bert.md), while ELMo remains a clear example of feature-based transfer and learned layer mixing.

**Exercise.** If a downstream task changes only its layer-mixing weights, have the language-model weights necessarily changed? No. Feature extraction and fine-tuning are separate choices.

Reference: [Deep contextualized word representations](https://arxiv.org/abs/1802.05365).
