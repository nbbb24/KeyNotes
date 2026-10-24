# 📚 Learning objectives, tokenization, and pretraining data

A model learns the prediction problem created by its data pipeline. Architecture, objective, tokenization, masking, and batching must agree; a correct Transformer implementation can still learn the wrong task if these pieces are mismatched.

## 🎯 Objective and information visibility

Causal language modeling predicts the next token from earlier tokens. Masked modeling reconstructs selected hidden content from visible context. Prefix modeling can use bidirectional input with causal output. Denoising objectives define a corruption and reconstruction target.

![Learning objectives, tokenization, and pretraining data, original figure 1](../../../../../assets/notion/045-00.png)

*The diagram varies architecture, objective, adaptation, and evaluation separately. A result cannot be attributed to architecture if several other factors changed at once.*

## 🧩 From clean text to subwords

![Learning objectives, tokenization, and pretraining data, original figure 2](../../../../../assets/notion/045-01.png)
![Learning objectives, tokenization, and pretraining data, original figure 3](../../../../../assets/notion/045-02.png)

*Tokenization turns text into IDs; BPE learns frequent pair merges. The training vocabulary and merge rules are then fixed for inference.*

Classical BPE repeatedly merges frequent adjacent units. Byte-level variants can represent arbitrary byte sequences, avoiding the same unknown-character issue as a limited character alphabet. Pre-tokenization and normalization are implementation choices; not every subword tokenizer requires the same word-level split. Runtime complexity depends on data structures and implementation, so a blanket $O(n^2)$ claim is insufficient.

Vocabulary size trades sequence length against embedding/output-table size. A tokenizer that produces many more tokens for one language increases cost and reduces the amount of text fitting in a context window. Evaluate coverage and compression by language/domain.

## ⚖️ Padding versus packing

![Learning objectives, tokenization, and pretraining data, original figure 4](../../../../../assets/notion/045-03.png)
![Learning objectives, tokenization, and pretraining data, original figure 5](../../../../../assets/notion/045-04.png)
![Learning objectives, tokenization, and pretraining data, original figure 6](../../../../../assets/notion/045-05.png)

*The same documents can be padded separately or packed into fixed-size blocks. The packed view saves empty positions but introduces document-boundary decisions.*

Padding needs an attention mask and usually a loss mask. Packing may permit cross-document attention or use segment boundaries to prevent it; state which objective is intended. End-of-document tokens identify boundaries but do not by themselves enforce attention isolation. Truncation can remove crucial targets or split structured examples.

## 📦 Data quality before scale

Deduplicate near-identical content, separate evaluation sources, record data provenance, and inspect language/domain balance. Avoid fitting preprocessing on held-out evaluation data. Contamination can raise benchmark scores without improving generalization. More tokens of repeated low-quality material are not equivalent to more useful evidence.

**Exercise.** If a packed batch has no padding, are all loss positions automatically valid? No. Boundary, prompt-only, or otherwise excluded positions may still need masking.

References: [subword BPE for neural translation](https://arxiv.org/abs/1508.07909), [T5 objective study](https://arxiv.org/abs/1910.10683). Continue with [fine-tuning](finetuning.md).
