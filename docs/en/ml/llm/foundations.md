# 📚 Language-model foundations

A language model assigns probabilities to token sequences. A tokenizer converts text to discrete IDs; the model turns those IDs into contextual states and predicts distributions. A token may be a word, part of a word, punctuation, or a byte-related unit. Token counts are not word counts.

## 🔄 Factor a sequence into predictions

The chain rule gives

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^T p(x_t\mid x_{<t}).
$$

Training a causal language model usually minimizes negative log likelihood, summed over eligible target positions. Padding, packed examples, and instruction-tuning prompt tokens may be excluded, so “loss on every token” is not universally correct. Teacher forcing supplies the true previous tokens during training; generation feeds back the model's own outputs.

## 🏗️ Architecture and visibility

| Architecture | Input visibility | Typical objective/application |
| --- | --- | --- |
| Encoder | Bidirectional within input | Mask recovery, classification, embeddings |
| Causal decoder | Earlier/current input positions | Next-token generation |
| Prefix decoder | Bidirectional prompt, causal response | Conditional generation |
| Encoder–decoder | Bidirectional source; causal target with cross-attention | Translation, denoising, summarization |

A causal decoder does not see the whole future sentence during prediction. A prefix decoder's output can attend to the prompt **and** previous output tokens. Encoder–decoder models do not inherently require multitask fine-tuning to function; their usefulness depends on objective, data, scale, and deployment constraints.

![Language-model foundations, original figure 1](../../../assets/notion/022-00.png)

The three columns contrast an encoder–decoder, causal language model, and prefix language model. Trace which tokens each predicted position can inspect, not merely how many boxes the drawing contains.

![Language-model foundations, original figure 2](../../../assets/notion/022-01.png)

Blue, green, and yellow regions in the attention-mask diagram represent different visibility rules. A prefix may attend bidirectionally within itself while generated positions still cannot inspect future output tokens.

![Language-model foundations, original figure 3](../../../assets/notion/022-02.png)

The language-model formula in the source image sums log probabilities. That quantity is a log-likelihood to maximize; a training loss to minimize uses its negative.

![Language-model foundations, original figure 4](../../../assets/notion/022-03.png)

The denoising-image formula expresses reconstruction conditioned on a corrupted input. Its notation is abbreviated: the model receives the corrupted sequence and predicts the original sequence, usually by minimizing negative log-likelihood.

Read each diagram by separating the architecture's attention pattern from its training loss. The same architecture can support several training recipes.

## 💡 Why decoder-only models became common

A single autoregressive interface accommodates continuation, dialogue, code, and in-context examples. Large-scale next-token training and efficient serving infrastructure made it a practical general-purpose choice. This is an empirical engineering trajectory, not a theorem that bidirectional attention is inferior or that decoder-only models are always more parameter-efficient. Pure-attention rank-collapse results do not directly establish such a comparison for full Transformers with residual and feed-forward layers.

## 🧩 A concrete example and limits

For `Paris is in France`, the model learns predictions at successive token boundaries. At inference, `Paris is in` can yield a high probability for `France`. That probability is learned from data; it is not a database query or a proof of factual correctness. Retrieval, tools, and evaluation are separate components.

**Exercise.** Does fluent generation prove factual knowledge is current? No. Training data can be outdated, and plausible continuations can be incorrect.

Continue with [Word2Vec to Transformer](word2vec-to-transformer.md), [BERT](models/bert.md), and [decoding](inference/decoding.md). References: [Transformer](https://arxiv.org/abs/1706.03762), [T5 architecture/objective study](https://arxiv.org/abs/1910.10683).
