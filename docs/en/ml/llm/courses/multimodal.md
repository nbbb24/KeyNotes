# Multimodal machine learning: a practical study route

This route follows the broad themes of the original CMU multimodal course: representations, alignment, interaction, generation, and evaluation. The aim is to explain when multiple modalities add useful information and how to demonstrate that contribution.

## Prerequisites and sequence

Start with [learning paradigms](../../mle/learning-paradigms.md), [loss functions](../../mle/loss-functions.md), and [attention](../components/attention.md). Then study:

1. [Multimodal foundations](../multimodal/foundations.md): distinguish the core questions.
2. [CNNs](../models/cnn.md), [ViT](../models/vit.md), and [word representations](advanced-nlp/word-representations.md): understand modality-specific encoders.
3. [Alignment](../multimodal/alignment.md) and [CLIP](../models/clip.md): connect corresponding examples.
4. [Fusion](../multimodal/fusion.md): model additive and interactive information.
5. [BLIP-2](../models/blip.md), [Flamingo](../models/flamingo.md), and [LLaVA](../models/llava.md): compare interfaces to generation.
6. [Speech evaluation](../speech/asr.md): match metrics to the actual task.

## Project: test whether a second modality helps

Choose a paired dataset with a clear task. Train or evaluate each modality alone, simple late fusion, and one interactive model. Keep data splits and evaluation conditions fixed. Include missing-modality and shuffled-pair tests. If the interactive model improves, identify examples where the extra modality supplies information absent from the first.

For sentiment in video, words may convey the proposition while voice and expression modify it. A dataset with strong textual shortcuts may show little need for other modalities. That is a property to measure, not an excuse to assume the model learned interaction.

## Advanced extension

Test temporal misalignment, noisy observations, and domain shift. Report both average quality and failure subsets. Adding modalities increases collection, synchronization, and serving costs; an improvement must be interpreted with those costs and availability constraints.

**Exercise.** If removing video at test time does not change predictions, what are two possible explanations? The model ignores video, or the task/data do not require it. Targeted examples help separate them.

Original resource: [CMU Multimodal Machine Learning, Fall 2023](https://cmu-multicomp-lab.github.io/mmml-course/fall2023/schedule/).
