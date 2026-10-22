!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# whisper

### **How it works**

- Audio is first converted into an **80-channel log-Mel spectrogram**.
- A **Transformer encoder** processes the audio features.
- A **Transformer decoder** (similar to GPT) autoregressively generates text tokens.
- The decoder uses **self-attention** (for previous text tokens) and **cross-attention** (to the audio representation).

### **Special design**

- Whisper supports **multiple languages**.
- It can **transcribe** or **translate** depending on control tokens (e.g., <|transcribe|>, <|translate|>).
- It predicts **timestamps** using special time tokens.
- It uses a **GPT-2 BPE tokenizer** plus task/format tokens.

### **Why Whisper is strong**

- Trained on **680,000+ hours** of noisy, real-world multilingual audio.
- Extremely robust to accents, background noise, and imperfect speech.
- No fine-tuning required—performs well in zero-shot settings.

### **Main capabilities**

- Speech-to-text
- Speech-to-English translation
- Automatic language detection
- Timestamped transcription
- Highly robust real-world ASR performance
