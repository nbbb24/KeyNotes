!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Llava

LLaVA’s training is divided into two phases, with the **vision encoder frozen throughout**

① In the *image-text alignment pre-training phase*, only the **projection layer** is trained to align visual and language feature spaces;

② In the *visual instruction tuning phase* , the **projection layer and language model** are jointly trained to enable the model to respond to multimodal instructions.

### **2. Academic Presentation Version**

LLaVA’s training framework is centered on "layered training and cross-modal alignment", with the vision encoder’s weights fixed during the entire process:

- During pre-training, only the **projection layer** (a single linear layer in early versions, upgraded to a 2-layer MLP in LLaVA-1.5) is trained on image-text pairs to achieve initial cross-modal feature space alignment.
- In the instruction tuning phase, end-to-end joint training is conducted on the **projection layer and language model (e.g., Vicuna)**. Leveraging multimodal instruction datasets, the model gains the ability to perform multimodal interaction and reasoning for practical tasks.

### **3. Technical Documentation Version**

LLaVA’s training targets are clearly phase-specific, and the vision encoder (CLIP ViT-L/14 or RICE-ViT) remains frozen at all stages:

- **Image-text alignment pre-training phase**: Only the **projection layer** is trained. Its core goal is to map high-dimensional visual features (from the vision encoder) to the embedding space of the language model, completing basic cross-modal alignment.
- **Visual instruction tuning phase**: The **projection layer and language model** are trained synchronously. This not only optimizes the accuracy of feature alignment but also enables the language model to generate instruction-compliant text responses by integrating visual information, thus adapting to multimodal tasks.
