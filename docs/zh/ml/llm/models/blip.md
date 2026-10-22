!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# BLIP (Bootstrapping Language-Image Pre-training)

[https://arxiv.org/pdf/2201.12086](https://arxiv.org/pdf/2201.12086)

![image.png](../../../../assets/notion/021-00.png)

## Unimodal Encoder (ITC)

ITC loss aligns global image and text embeddings in a shared semantic space by maximizing similarity for matched pairs and minimizing it for mismatched ones, often using momentum encoders for soft labels to improve alignment precision.

Contrast learning similar to CLIP

## Image-grounded text Encoder (ITM)

As a binary classification loss, ITM loss trains the model to make fine-grained judgments of image-text semantic matching via cross-attention, with hard negative sampling strengthening its ability to distinguish subtle semantic discrepancies between pairs.

use cross attention to identify whether the image-text match

cross attention+linear head

## Image-grounded text Decoder (LM)

LM loss (usually autoregressive cross-entropy) optimizes the decoder to generate coherent, image-relevant text by maximizing the likelihood of correct next tokens, which endows the model with cross-modal generative capabilities for tasks like image captioning.

cross-entropy to maximize the likelihood of text in autoregressive manner

Cross attention with image-features

---

# BLIP2

[https://arxiv.org/pdf/2301.12597](https://arxiv.org/pdf/2301.12597)

![image.png](../../../../assets/notion/021-01.png)

![image.png](../../../../assets/notion/021-02.png)

1. **Frozen Vision Encoder**

    The image is first encoded by a *frozen* ViT, producing visual patch embeddings.

    These embeddings are **not updated during training**.


---

1. **Learnable Query Tokens**

    A fixed number of learnable query tokens (e.g., 32) are initialized.

    These tokens act as “semantic probes” that will extract high-level information from the image.


---

1. **Q-Former Transformer**

    The Q-Former consists of Transformer layers with:

    - **Self-attention** among the query tokens (to let them coordinate and specialize).
    - **Cross-attention** where query tokens attend to the frozen visual features

        (queries = Q, visual patches = K/V), enabling semantic extraction.


    The output is a compact set of visual embeddings aligned to language semantics.


---

1. **Linear Projection to LLM Space**

    The Q-Former outputs are passed through a **trainable linear projection**

    to map them into the embedding space of the frozen LLM.


---

1. **Training with Three Losses**

    Only the Q-Former (including query tokens + Transformer layers + linear projector) is trained.

    Both the Vision Encoder and the LLM remain frozen.

    The Q-Former is optimized using three objectives:

    - **ITC (Image–Text Contrastive Loss):**

        Aligns visual query embeddings with text embeddings.

    - **ITM (Image–Text Matching Loss):**

        Determines whether an image–text pair is matched.

    - **LM (Language Modeling Loss):**

        The projected query embeddings are used as prefix tokens for the frozen LLM,

        and standard LM loss trains the Q-Former to produce better visual prefix representations.


## Q-former

1)image transformer decoder interacts with frozen image encoder for visual feature extraction

2)text transformer both a encoder and decoder

### Query-Image

left transformer interacts with image encoder

Use query embeddings and cross attention to extract features

### Query-Text

right transformer is both encoder and decoder

self-attention between left and right transformer

## Image-text contrastive learning (ITC)

contrastive learning

## Image-grounded Text Generation (ITG)

## Image-Text Matching (ITM)

![image.png](../../../../assets/notion/021-03.png)

**1. What is the purpose of the Q-former in BLIP-2? Why not use image features directly?**

Q-former keeps the image encoder frozen, use learnable queries to extract image features instead of passing all image tokens, just like distillation.

**2. How do learned query tokens in the Q-former differ from standard input tokens in transformers?**

Q-former queries are learned parameters that do not correspond to any specific image or text input. These queries adapt through training and become highly specialized for downstream tasks such as captioning or question answering

**3. What’s the difference between self-attention and cross-attention in the Q-former architecture?**

Self-attention in the Q-former operates over the set of learned query tokens (and optionally over text tokens, depending on the attention mask), allowing them to communicate and refine one another.

Cross-attention, on the other hand, allows these queries to attend to the output of the image encoder. The cross-attention mechanism injects visual information into the queries, while self-attention ensures that these queries are contextually aware and collaborative.

**4. What are the roles of different attention masks used in Q-former during ITC, ITM, and LM tasks?**

In the Image-Text Contrastive (ITC) task, the attention mask prevents interaction between query and text tokens, ensuring that they are processed separately before computing similarity via InfoNCE loss.

In Image-Text Matching (ITM), a bidirectional attention mask is used to allow full multimodal fusion between query tokens and text tokens.

In language modeling (LM), a causal attention mask is applied to text tokens (so they only see previous tokens), while queries can still be attended to by the text decoder, enabling image-conditioned generation.

**5. Compare how Q-former operates in BLIP-2 vs. how CLIP processes image-text pairs. What are the key differences?**

In CLIP, image and text encoders are separate and independent, and their embeddings are aligned through contrastive learning. There is no interaction between modalities during encoding.

In contrast, the Q-former in BLIP-2 allows for direct interaction between image and text tokens, particularly during ITM and generation stages, via cross-attention and multimodal self-attention.

Moreover, CLIP uses full image patch sequences for embedding, while BLIP-2 compresses these into a small number of query embeddings, making it more efficient and generative-capable.

**6. How could you extend the Q-former to handle video frames instead of a single image?**

add positional embedding for each frame.
