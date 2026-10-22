!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# cross-modal alignment

## **1. When Cross-Modal Alignment Is Needed**

Cross-modal alignment is required whenever different modalities must share or exchange semantic information. Typical scenarios include:

### **(1) When modalities must represent the same semantic concept**

Examples:

- Image ↔ Text (CLIP, SigLIP)
- ECG ↔ Clinical Report
- PPG/ECG ↔ Diagnostic Labels

The goal is to ensure that semantically corresponding signals—e.g., “atrial fibrillation” and its irregular RR-interval pattern—occupy nearby positions in the embedding space.

---

### **(2) When one modality supervises or constrains another**

For example:

- Text labels guiding vision encoders
- Clinical diagnosis guiding ECG feature learning

Semantic supervision must propagate across modalities, which requires alignment.

---

### **(3) When modalities must be interchangeable or jointly used**

Examples:

- Text-to-Image retrieval
- Image captioning
- Multimodal fusion (vision + signal + text)

Such tasks require the modalities to lie in a **shared representation space**.

---

### **(4) When modalities have different abstraction levels**

Signals provide low-level patterns (e.g., waveform morphology), whereas text encodes high-level semantics.

Alignment ensures that **low-level patterns map coherently to high-level semantic concepts.**

---

## **2. How Cross-Modal Alignment Is Achieved**

Cross-modal alignment strategies fall into three broad categories:

**(A) Representation Alignment**, **(B) Interaction Alignment**, and **(C) Semantic Alignment**.

---

# **A. Representation Alignment (Aligning Feature Spaces)**

The goal is to map modalities into a shared embedding space.

### **1. Contrastive Learning (e.g., InfoNCE, CLIP-style)**

$\mathcal{L}_{\text{InfoNCE}}
= -\log \frac{e^{s(z_i^x, z_i^y)/\tau}}
{\sum_j e^{s(z_i^x, z_j^y)/\tau}}$

Positive pairs (e.g., ECG + its report) are pulled together, while negatives are pushed apart.

This is the most common and scalable alignment method for multimodal systems.

---

### **2. Projection-Layer Alignment**

Each encoder is followed by a projection head:

$h_x = W_x f_x,\qquad h_y = W_y f_y.$

Different modalities are projected into a common dimensional and semantic space.

---

### **3. Distance-Based Regularization**

Use constraints such as:

- L2 distance
- Cosine similarity loss
- KL divergence

These directly force representations to become similar.

---

# **B. Interaction Alignment (Aligning Through Model Interactions)**

Here, modalities interact explicitly through attention mechanisms.

### **1. Cross-Attention**

$\text{Attn}(Q_x, K_y, V_y) = \text{softmax}(Q_x K_y^\top) V_y.$

Effect:

- ECG tokens “attend to” textual tokens
- Image patches “attend to” captions

This creates token-level alignment between modalities.

---

### **2. Co-Attention (Dual Attention)**

Two-way attention:

- X attends to Y
- Y attends to X

Useful for medical images + clinical notes, or ECG + structured EHR.

---

### **3. Shared/Query Tokens (e.g., Q-Former, Flamingo)**

Learnable query tokens act as intermediaries that extract and align features across modalities.

---

# **C. Semantic Alignment (Aligning to Clinical or Linguistic Semantics)**

Here, the model aligns representations to an external semantic space.

### **1. LLM-Guided Alignment**

Embed text through large language models (e.g., ClinicalBERT, LLaMA) and align signal encoders to this semantic space.

---

### **2. Concept-Level Supervision**

Attach supervision from:

- Diagnosis labels (AF, PVC, STEMI, etc.)
- Clinical concepts (SNOMED, ICD-10)

This anchors biomedical signals to interpretable semantic units.

---

### **3. Knowledge Distillation**

Transfer semantic structure from the text encoder to the signal encoder:

h_{\text{signal}} \approx h_{\text{text}}.
