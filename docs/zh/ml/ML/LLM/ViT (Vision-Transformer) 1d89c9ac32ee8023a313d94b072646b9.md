# ViT (Vision-Transformer)

![image.png](ViT%20(Vision-Transformer)/image.png)

divide image into patches → flattened → concatenate + positional encoding

**Input**: (224/16)**2=196, input [1,196,768]

ViT-base **hidden dimension**: 768 (bert)

[cls] + flattened patches → Encoder → MLP head

the linear projection is actually 2d Conv for patch embedding