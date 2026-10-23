# BLIP and BLIP-2: understanding, generation, and a learned bridge

BLIP and BLIP-2 are related but distinct designs. BLIP combines visual understanding and text generation while improving noisy captions. BLIP-2 connects pretrained frozen vision and language models through a trainable Querying Transformer, or Q-Former.

## BLIP: three operating modes

![BLIP and BLIP-2: understanding, generation, and a learned bridge, original figure 1](../../../../assets/notion/021-00.png)

*BLIP's contrastive encoder, image-grounded matching encoder, and causal text decoder share a common multimodal design but use different information flow.*

ITC aligns independently encoded image/text representations. ITM uses cross-attention to judge whether a pair matches, including hard negatives. The language-modeling mode generates text conditioned on image features. The CapFilt data procedure generates synthetic captions and filters noisy captions, making data quality part of the method rather than relying only on architecture.

## BLIP-2 stage 1: learn a visual representation interface

![BLIP and BLIP-2: understanding, generation, and a learned bridge, original figure 2](../../../../assets/notion/021-01.png)

*The two-stage bridge keeps the large image encoder and language model frozen while learning the interface.*

A fixed number of learned query vectors—32 in the paper's design—cross-attend to image features. Their outputs form a compact visual representation. The queries are parameters, not words supplied by the user, and each image produces different output query states.

Stage 1 trains Q-Former using image–text contrastive learning, image–text matching, and image-grounded text generation. These objectives use different attention masks:

- ITC separates query and text streams before similarity computation.
- ITM permits bidirectional query/text interaction.
- Image-grounded generation allows text to read queries and previous text, while preventing future-text leakage.

![BLIP and BLIP-2: understanding, generation, and a learned bridge, original figure 3](../../../../assets/notion/021-02.png)

*The masking diagrams show why changing the objective requires changing permitted query/text interactions.*

## BLIP-2 stage 2: connect to a frozen language model

A learned linear projection maps query outputs to the LLM's input width. These visual prefix embeddings condition a frozen decoder-only or encoder–decoder language model, and a generative objective trains the bridge. This is separate from the three representation-learning objectives of stage 1; combining them into one undifferentiated training stage obscures the method.

![BLIP and BLIP-2: understanding, generation, and a learned bridge, original figure 4](../../../../assets/notion/021-03.png)

*The top path conditions a decoder-only LM; the bottom path connects to an encoder–decoder LM. Both use projected visual query outputs.*

## Why compress visual features?

If a vision encoder emits hundreds of patch states, a fixed query set reduces the interface sequence length. That saves downstream work but introduces a bottleneck: tiny text or several small objects may need more visual detail. Compression alone is not knowledge distillation; distillation requires a teacher-derived learning target.

A video extension would need temporal information, frame selection, a visual-token budget, and training for temporal tasks. Merely adding frame positions does not guarantee temporal understanding.

**Exercise.** Can gradients train a bridge through a frozen LLM? Yes. Freezing weights prevents their update; differentiation through the LLM can still deliver gradients to its input embeddings and the bridge.

References: [BLIP](https://arxiv.org/abs/2201.12086), [BLIP-2](https://arxiv.org/abs/2301.12597). Compare [Flamingo](flamingo.md) and [LLaVA](llava.md).
