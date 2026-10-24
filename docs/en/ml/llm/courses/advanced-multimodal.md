# 🖼️ Advanced multimodal learning: turn a method into a research question

The advanced track focuses on heterogeneity, connections, interactions, and how to quantify them. Begin after the [multimodal foundations route](multimodal.md), with a working understanding of representation learning and controlled experiments.

## 🧭 Three questions to investigate

**Heterogeneity:** How do modalities differ in sampling rate, noise, structure, dimensionality, and semantic abstraction? A waveform and a sentence should not automatically share preprocessing or a complete encoder.

**Connections:** Which observations refer to the same event or concept? Correspondence can be temporal, spatial, instance-level, or semantic. Weak pairing may be useful but must not be mistaken for precise local alignment.

**Interactions:** Is the task solved by adding independent signals, or does one modality change the meaning of another? Compare additive baselines with [non-additive fusion](../multimodal/fusion.md), and inspect the residual benefit.

## 🔎 A research-style experiment

State a falsifiable hypothesis, such as “temporal alignment improves event localization when audio and video are both informative.” Define a baseline, an alignment intervention, a matched-cost control, and a held-out metric. Perturb alignment while keeping content constant. Report uncertainty and examples that contradict the hypothesis as well as those that support it.

Transfer experiments can suggest parameter sharing across tasks or modalities. Positive transfer in one direction does not guarantee symmetric transfer. A transfer matrix is evidence about a particular training setup, not an immutable taxonomy of modalities.

## 🔍 Reading papers critically

Identify what changed: data, objective, architecture, optimization, or evaluation. Check whether a claimed interaction survives strong unimodal baselines. Distinguish predictive association from causal conclusions. Examine missing-modality assumptions and whether the evaluation split excludes subject/session leakage.

**Exercise.** A fusion model beats a small unimodal baseline but not a capacity-matched one. Does that establish useful cross-modal interaction? No. Capacity and optimization remain alternative explanations.

Original resource: [CMU Advanced Topics in Multimodal Machine Learning, Spring 2023](https://cmu-multicomp-lab.github.io/adv-mmml-course/spring2023/schedule/). Related: [alignment](../multimodal/alignment.md) and [fusion](../multimodal/fusion.md).
