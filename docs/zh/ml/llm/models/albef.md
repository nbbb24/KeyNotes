!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# ALBEF

https://arxiv.org/abs/2107.07651

![image.png](../../../../assets/notion/010-00.png)

# **🔶**

# **1. Encoders**

### **Online Encoders (trainable)**

- **Vision encoder**: ViT
- **Text encoder**: BERT
- **Multimodal fusion encoder**: Cross-attention Transformer

These are updated by backpropagation from all losses.

### **Momentum Encoders (not trainable)**

- A momentum copy of the vision/text encoders
- Updated by EMA:

    \theta_m \leftarrow \alpha \theta_m + (1-\alpha)\theta

    They provide **stable image/text representations** for contrastive learning.


---

# **🔶**

# **2. Training Losses**

ALBEF is trained using **three objectives**:

### **(1) ITC — Image–Text Contrastive Loss**

- Uses online encoders as queries
- Uses momentum encoders as keys/targets
- Aligns image and text embeddings in a joint space
- Prevents collapse and improves global alignment

### **(2) ITM — Image–Text Matching Loss**

- Uses the multimodal fusion encoder with cross-attention
- A binary classifier predicts whether an image–text pair matches
- Provides fine-grained cross-modal supervision

### **(3) MLM — Masked Language Modeling (with images)**

- Some text tokens are masked
- The multimodal encoder predicts the missing words conditioned on the image
- Strengthens grounding of language in visual content

The final loss is a weighted sum:

\mathcal{L} = \lambda_{ITC}\mathcal{L}_{ITC} + \lambda_{ITM}\mathcal{L}_{ITM} + \lambda_{MLM}\mathcal{L}_{MLM}

---

# **🔶**

# **3. Training Workflow (Step-by-Step)**

### **Step 1 — Encode image and text**

- Image → online vision encoder → patch embeddings
- Text → online BERT encoder → token embeddings

### **Step 2 — Compute momentum representations**

- Same image/text passed through momentum encoders
- These become stable targets for alignment (ITC)

### **Step 3 — Alignment Stage (Align Before Fuse)**

Use **ITC** to force:

- online image embeddings → close to momentum text embeddings
- online text embeddings → close to momentum image embeddings

This builds a shared semantic embedding space **before fusion**.

### **Step 4 — Fusion Stage**

Text tokens attend to image patches in the multimodal fusion encoder (cross-attention):

h = \text{MMEncoder}(t, v)

### **Step 5 — ITM & MLM losses**

- The multimodal [CLS] token is used for ITM (match / mismatch)
- The hidden states predict masked tokens for MLM

### **Step 6 — Update Encoders**

- Online encoders updated via backprop
- Momentum encoders updated via EMA (no gradient)
