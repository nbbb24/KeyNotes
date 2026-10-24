# 📝 T5 and BART: learning to generate from corrupted input

Encoder–decoder models separate input understanding from output generation. The encoder reads the source bidirectionally; the decoder generates a target causally while attending to encoder states. T5 and BART use this structure with different pretraining formulations.

## 🧭 T5: a common text-to-text interface

T5 represents tasks as text input and text output. A task prefix such as `translate English to German:` describes the operation. Classification can output a label word; semantic similarity can output a textual score. CoLA is a grammatical-acceptability classification task, while STS-B measures sentence-pair similarity—sharing an interface does not make their evaluation metrics identical.

In span-corruption pretraining, selected spans are replaced by distinct sentinel tokens. The target contains the missing spans separated by sentinels. For example, `The <extra_id_0> sat on <extra_id_1>.` can require recovering `cat` and `the mat`. The decoder learns conditional reconstruction rather than ordinary continuation of the visible source.

## 🧩 BART: reconstruct the original document

BART corrupts text and learns to generate the original sequence. The paper studies token masking, token deletion, text infilling, sentence permutation, and document rotation. Infilling replaces a span with a single mask, so the model must infer both missing content and its length. The studied span lengths use a Poisson distribution with mean 3. Different corruption schemes create different reconstruction problems; the list is not a requirement to apply all of them to every sample.

![T5 and BART: learning to generate from corrupted input, original figure 1](../../../../assets/notion/070-00.png)

![T5 and BART: learning to generate from corrupted input, original figure 2](../../../../assets/notion/070-01.png)

![T5 and BART: learning to generate from corrupted input, original figure 3](../../../../assets/notion/070-02.png)

**Figure guide.** The first image contrasts encoder–decoder, causal, and prefix visibility. The second shows BART's corrupted-input encoder and autoregressive decoder beside BERT and GPT. The third shows two fine-tuning routes: classification from a learned representation and sequence generation for translation. These illustrations distinguish architecture from the corruption objective used in pretraining.

## ⚖️ Compare the learning signals

| Aspect | T5 | BART |
| --- | --- | --- |
| Task interface | Unified text-to-text formulation | Denoising sequence-to-sequence backbone |
| Typical corruption | Spans replaced by distinct sentinels | Infilling and sentence permutation in the selected recipe |
| Reconstruction target | Missing spans with sentinels | Original text |
| Shared property | Bidirectional source encoding and autoregressive target decoding | Same |

For summarization, the encoder reads the document and the decoder produces a shorter text. Neither architecture guarantees factual summaries: a fluent decoder can introduce unsupported details. Evaluate content coverage, factual consistency, length, and the intended audience separately.

**Exercise.** Why can a corrupted-input model use bidirectional attention without leaking the target? The encoder reads the corrupted source; the decoder's own target stream remains causally masked.

References: [T5](https://arxiv.org/abs/1910.10683), [BART](https://arxiv.org/abs/1910.13461). Review [attention masks](../components/attention-masks.md).
