!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# ASR

# **1. Automatic Speech Recognition (ASR) Metrics**

## **1.1 Word Error Rate (WER)**

WER is the most widely used metric for ASR quality. It measures the difference between the predicted transcript and the reference transcript.

$WER = \frac{S + D + I}{N}$

- **S** = substitutions
- **D** = deletions
- **I** = insertions
- **N** = number of words in the reference

A lower WER means better recognition accuracy.

---

## **1.2 Character Error Rate (CER)**

Similar to WER but computed at the **character level** rather than word level.

$CER = \frac{S + D + I}{N}$

Used mainly in languages without explicit word boundaries (Chinese, Japanese, Korean).

---

## **1.3 Phone Error Rate (PER) / Token Error Rate (TER)**

- **PER**: error rate at the phoneme level
- **TER**: error rate at BPE/subword/token level

Useful for phoneme-based ASR or multilingual systems.

---

# **2. Spoken Language Understanding (SLU) Metrics**

SLU includes intent classification, slot filling, and semantic parsing.

### **2.1 Intent Accuracy**

Percentage of correctly predicted intents.

### **2.2 Slot F1 Score**

F1 score for token-level slot predictions (similar to NER):

$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

### **2.3 Semantic Error Rate (SemER)**

Measures semantic parsing errors:

$SemER = \frac{\text{Intent errors + Slot errors}}{\text{Total slots + 1}}$

---

# **3. Text-to-Speech (TTS) Metrics**

## **3.1 Mean Opinion Score (MOS)**

Human raters score naturalness on a **1–5 scale**:

5 = completely natural

1 = very unnatural

Still the most trusted evaluation metric for TTS.

---

## **3.2 Comparative MOS (CMOS)**

Measures preference between two systems (A vs B).

Useful for detecting small improvements.

---

## **3.3 Objective TTS Metrics**

Although they do not fully replace subjective evaluations, they help compare systems.

- **MCD (Mel-Cepstral Distortion):** spectral distortion
- **F0 RMSE:** pitch error
- **Duration RMSE:** rhythm/length error
- **F0 correlation:** how well pitch contours match

---

# **4. Multimodal (Audio–Text / Audio–Image) Alignment Metrics**

Used in models like **CLAP, Whisper-LM, Speech-LLaMA, Qwen-Audio, GPT-4o**.

## **4.1 Recall@K (R@1, R@5, R@10)**

Evaluates retrieval performance:

- Audio → Text retrieval
- Text → Audio retrieval
- Audio → Image retrieval

$R@K = \frac{\text{Correct matches found in top K}}{\text{Total samples}}$

---

## **4.2 Cosine Similarity**

Used for audio-text or audio-image embedding alignment:

$\text{sim}(a,b) = \frac{a \cdot b}{\|a\| \|b\|}$

Forms the basis of contrastive learning (e.g., InfoNCE).

---

## **4.3 mAP (mean Average Precision)**

Common in keyword spotting or audio event retrieval.

---

# **5. Audio Event Detection (AED) Metrics**

Used for tasks such as detecting sirens, dog barks, coughing, etc.

### **5.1 Event-based F1**

Counts an event as correct only if **class and time boundaries match**.

### **5.2 Segment-based F1**

Evaluates predictions in fixed time windows (e.g., 1 second).

### **5.3 Error Rate (ER)**

Used in DCASE challenges:

ER = $\frac{FN + FP + Sub}{N}$

---

# **Summary Table**

| **Task** | **Common Metrics** |
| --- | --- |
| ASR | **WER, CER, TER, PER** |
| SLU | **Intent Accuracy, Slot F1, SemER** |
| TTS | **MOS, CMOS, MCD, F0 RMSE** |
| Multimodal Alignment | **Recall@K, Cosine Similarity, mAP** |
| Audio Event Detection | **Event-F1, Segment-F1, ER** |

---
