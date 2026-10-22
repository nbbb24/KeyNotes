!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# ViT (Vision-Transformer)

![image.png](../../../../assets/notion/073-00.png)

divide image into patches → flattened → concatenate + positional encoding

**Input**: (224/16)**2=196, input [1,196,768]

ViT-base **hidden dimension**: 768 (bert)

[cls] + flattened patches → Encoder → MLP head

the linear projection is actually 2d Conv for patch embedding
