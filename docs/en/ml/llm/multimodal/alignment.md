# Cross-modal alignment: connect corresponding information

Alignment makes information from different modalities correspond in a way useful for a task. Image–caption retrieval, matching video frames to spoken words, and learning a sensor representation from textual descriptions are different alignment problems. They need not use the same loss or the same granularity.

## Global representation alignment

Two encoders and projection heads can map inputs to comparable vectors:

$$
z_x=\operatorname{normalize}(W_xf_x(x)),\quad
z_y=\operatorname{normalize}(W_yf_y(y)).
$$

A contrastive objective rewards paired examples relative to alternatives:

$$
L_i=-\log\frac{\exp(s(z_i^x,z_i^y)/\tau)}{\sum_j\exp(s(z_i^x,z_j^y)/\tau)}.
$$

Matching dimensions only makes a dot product possible; the training signal gives it meaning. L2 or cosine losses can directly match features, but using only positive-pair similarity can permit collapse unless other constraints preserve information. KL divergence compares probability distributions, so arbitrary signed feature vectors cannot simply be passed into it as probabilities.

## Interaction and temporal alignment

Cross-attention lets one modality query another:

$$
\operatorname{Attention}(Q_x,K_y,V_y)=
\operatorname{softmax}(Q_xK_y^\top/\sqrt{d_k}+M)V_y.
$$

A question token can retrieve relevant image patches; audio frames can be matched to a transcript at a finer timescale. Co-attention exchanges information in both directions. Learned query tokens compress one modality into an interface for another, as in [BLIP-2](../models/blip.md). Attention weights are learned compatibility patterns, not automatically verified alignments.

## Semantic supervision and transfer

Concept labels, text embeddings, or a teacher model can guide a less richly labeled modality. For example, a machine vibration segment and its maintenance report can provide paired supervision. Reports may contain information absent from the measured interval, so pairing at the document level does not guarantee every sentence is observable in the signal.

Not every multimodal task requires one shared metric space. A generator can use modality-specific representations connected by cross-attention. Choose the interface according to retrieval, classification, grounding, or generation needs.

## Evaluate correspondence, not just similarity

Use retrieval Recall@K for global pairs, localization/timing metrics for local alignment, and downstream tests for semantic usefulness. Include hard negatives that differ in one attribute. Check duplicates, many-to-many matches, and leakage across subjects or recording sessions.

**Exercise.** Two captions correctly describe the same image. Should one be a negative simply because it is off the batch diagonal? No. Multi-positive labels or appropriate sampling can avoid contradictory supervision.

References: [CLIP](https://arxiv.org/abs/2103.00020), [ALBEF](https://arxiv.org/abs/2107.07651). Continue with [fusion](fusion.md).
