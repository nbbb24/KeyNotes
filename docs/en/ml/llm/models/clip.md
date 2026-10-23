# CLIP: match images and text in an embedding space

CLIP trains an image encoder and a text encoder so matching image–caption pairs receive high similarity. At inference, either side can be encoded independently, which makes the design useful for retrieval and classification with textual labels.

## The contrastive matrix

Normalize image vectors $v_i$ and text vectors $t_j$. For a batch of $N$ paired examples, form $s_{ij}=v_i^\top t_j/\tau$, with a learned temperature-related scale. The image-to-text loss is

$$
L_{I\to T}=-\frac1N\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}.
$$

The text-to-image loss applies the same idea down columns. Their average trains both retrieval directions. The diagonal is the designated match; off-diagonal pairs supply negatives. For a three-pair batch, the score matrix has nine entries, not three independent classification outputs.

![CLIP: match images and text in an embedding space, original figure 1](../../../../assets/notion/024-00.png)

![CLIP: match images and text in an embedding space, original figure 2](../../../../assets/notion/024-01.png)

The first diagram moves from paired training to text-described classes at inference. The second shows one direction of the contrastive objective; a full CLIP loss uses both directions.

## Zero-shot classification as retrieval

Encode prompts such as `a photo of a dog` and `a photo of a cat`, then compare them with the image embedding. The highest similarity determines the selected class. Prompt wording and candidate labels matter. Softmax over these scores gives a relative distribution over the supplied candidates, not a universally calibrated probability that the image contains the concept.

The original work studies both modified ResNet and ViT image encoders. Its ResNet variant uses attention pooling; this is different from claiming every CLIP model uses a flattened patch sequence as its final retrieval embedding. The retrieval representation is a pooled, projected vector.

## Limits and useful experiments

A caption may describe only part of an image. Two off-diagonal examples may actually share valid semantics, producing false negatives. Object presence can be easier than counting or compositional relationships such as “red cube left of blue sphere.” Evaluate these separately.

**Exercise.** Can image embeddings be precomputed for a retrieval database? Yes, because the image encoder does not depend on the query text. A cross-encoder generally cannot offer the same independence.

Reference: [CLIP paper](https://arxiv.org/abs/2103.00020). Compare [SigLIP](siglip.md) and [ALBEF](albef.md).
