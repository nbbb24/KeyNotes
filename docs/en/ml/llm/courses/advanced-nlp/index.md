# 🎓 Advanced NLP learning route

This course track connects practical text modeling to the representations used by modern language models. Begin with a measurable baseline, then add context and architectural complexity one step at a time.

## 🔄 Suggested sequence

1. [NLP introduction](introduction.md): define the task, construct bag-of-words features, and recognize what orderless representations miss.
2. [Word representations](word-representations.md): understand lookup tables, CBOW, skip-gram, and distributional similarity.
3. [RNNs](../../models/rnn.md) and [ELMo](../../models/elmo.md): learn how sequence context changes a token representation.
4. [Attention](../../components/attention.md) and [Transformers](../../word2vec-to-transformer.md): track information flow and tensor shapes.
5. [BERT](../../models/bert.md), [GPT](../../models/gpt.md), and [T5/BART](../../models/t5-bart.md): distinguish architectures from pretraining objectives.

## 🧩 A project that connects the chapters

Choose one text-classification dataset. Keep its train/validation/test split fixed. Build a sparse linear baseline, then a model using learned embeddings, then a pretrained contextual model. Record parameter count, training cost, inference latency, class-level errors, and behavior on negation or rare terminology. Change one major factor at a time so improvements have an interpretable cause.

For advanced readers, test whether the apparent gain survives a domain or time shift. Examine label ambiguity before attributing every error to model capacity. A baseline that performs similarly at a fraction of the cost can be the better system for a constrained application.

## ✅ Completion check

You should be able to explain why one-hot lookup and embedding indexing are equivalent, why CBOW loses context order, why causal and bidirectional masks serve different objectives, and why a benchmark score depends on the split and metric. If any answer is unclear, revisit the corresponding chapter before scaling the model.

Original course resources: [CMU Advanced NLP 2024 lectures](https://phontron.com/class/anlp2024/lectures/) and [course code](https://github.com/neubig/anlp-code). These notes are a learning companion, not a claim to reproduce the entire course.
