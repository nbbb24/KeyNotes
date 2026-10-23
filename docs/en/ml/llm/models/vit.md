# Vision Transformer: turning an image into a sequence

ViT applies a Transformer encoder to image patches. Its starting point is simple: if a language model can mix information among token vectors, an image model can mix information among vectors representing local image regions.

## Patch embedding, carefully counted

An image of height $H$, width $W$, and $C$ channels is split into $P\times P$ patches. Assuming divisible dimensions, the number of patches is $N=HW/P^2$. Each flattened patch has $P^2C$ values and is projected to hidden width $d$.

For a $224\times224$ RGB image and $16\times16$ patches:

- There are $14\times14=196$ patches.
- Each flattened patch has $16\times16\times3=768$ values.
- ViT-Base uses hidden width 768, but this matching number is a configuration choice.
- Adding a class token gives 197 tokens before the encoder.

A convolution with kernel size and stride both equal to the patch size can implement this projection. Patches are concatenated along the sequence dimension, not flattened into one giant hidden vector. Learned positional embeddings distinguish patch locations.

![Vision Transformer: turning an image into a sequence, original figure 1](../../../../assets/notion/073-00.png)

**Figure guide.** Read the image from bottom to top: image patches are flattened and projected into tokens, position information and a class token are added, and a transformer encoder exchanges information between patches. The classification head reads the class representation. Patch count controls attention cost, so image resolution is part of the model's systems design.

## What the encoder learns

Self-attention allows a patch to exchange information with distant patches from the first layer. A class-token state or a pooling operation supplies a representation for classification. Compared with CNNs, a plain ViT has weaker built-in locality assumptions and relies more on data, pretraining, and augmentation to learn useful visual structure.

## Resolution is a systems decision

Reducing patch width from 16 to 8 at fixed image size quadruples the number of patch tokens. Dense attention score size then grows by roughly sixteen times, ignoring the class token. Increasing resolution can help small-object recognition or OCR but increases cost. Positional embeddings may need interpolation when the patch grid changes; verify the model's prescribed preprocessing.

**Exercise.** A $384\times384$ image with 16-pixel patches produces how many patch tokens? $24^2=576$, or 577 with a class token. Hidden width need not change.

ViT features can feed [CLIP](clip.md), [LLaVA](llava.md), and other multimodal systems. A vision encoder alone does not generate text; a connector and text-side model define that interface.

Reference: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
