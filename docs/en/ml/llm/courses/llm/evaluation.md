# 📊 Automatic evaluation: measure the behavior you need

A metric compresses behavior into a number. Choose it according to the task, state its conventions, and inspect examples where it disagrees with the intended outcome. No single metric measures fluency, factuality, reasoning, and user usefulness simultaneously.

## 🧭 Predictive fit: perplexity

$$
\operatorname{PPL}=\exp\left(-\frac1N\sum_t\log p(y_t\mid y_{<t})\right).
$$

![Automatic evaluation: measure the behavior you need, original figure 1](../../../../../assets/notion/041-00.png)

*Perplexity is the exponentiated mean token negative log likelihood. Compare it under compatible tokenization, masking, and data conditions.*

A uniform model over four choices has perplexity 4. Lower perplexity does not by itself establish better instruction following or factual reliability.

## 🧩 Lexical overlap

![Automatic evaluation: measure the behavior you need, original figure 2](../../../../../assets/notion/041-01.png)
![Automatic evaluation: measure the behavior you need, original figure 3](../../../../../assets/notion/041-02.png)

*BLEU clips n-gram counts to avoid rewarding repeated matches and includes a brevity penalty. ROUGE variants measure overlap or longest common subsequences; many implementations report precision, recall, and F1, so “ROUGE is recall” is incomplete.*

BLEU traditionally aggregates modified n-gram precision geometrically. Corpus and sentence BLEU differ, especially with smoothing. ROUGE-N uses n-gram overlap; ROUGE-L uses longest-common-subsequence structure. Both can penalize valid paraphrases and reward text that shares words but changes meaning.

![Automatic evaluation: measure the behavior you need, original figure 4](../../../../../assets/notion/041-03.png)
![Automatic evaluation: measure the behavior you need, original figure 5](../../../../../assets/notion/041-04.png)

*Character overlap can capture morphology. The original chrF screenshot has a sign typo: the factor is $1+\beta^2$, not $1-\beta^2$.*

$$
F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}.
$$

For P=R=0.8, F is 0.8 for any positive beta. This simple check exposes the screenshot's typo. Character order, whitespace, and n-gram settings should be recorded.

## 📊 Contextual and learned metrics

![Automatic evaluation: measure the behavior you need, original figure 6](../../../../../assets/notion/041-05.png)
![Automatic evaluation: measure the behavior you need, original figure 7](../../../../../assets/notion/041-06.png)

*BERTScore matches contextual token embeddings using cosine similarity, then aggregates precision/recall-like scores. It can recognize some paraphrases but can miss subtle negation or numerical errors.*

![Automatic evaluation: measure the behavior you need, original figure 8](../../../../../assets/notion/041-07.png)
![Automatic evaluation: measure the behavior you need, original figure 9](../../../../../assets/notion/041-08.png)

*COMET-style diagrams use learned representations and human-quality supervision; regression and ranking variants have different objectives. Specify the exact metric checkpoint and input requirements.*

Learned metrics inherit training biases and domain limits. LLM judges also need calibrated rubrics, order controls, and checks for verbosity or model-family bias. Human review remains useful on targeted error sets.

## 📊 A complete evaluation record

Record dataset/version, split, preprocessing, model/checkpoint, prompt/template, decoding, number of samples, metric implementation, and uncertainty. For code, execute tests; for extraction, compare fields; for RAG, check both answer correctness and citation support. Use bootstrap or repeated-run intervals when variation affects the decision.

**Exercise.** “The dose is 5” and “The dose is 50” share most text. Can high overlap establish correctness? No. Task-critical fields need explicit checks.

References: [BLEU](https://aclanthology.org/P02-1040/), [chrF](https://aclanthology.org/W15-3049/), [BERTScore](https://arxiv.org/abs/1904.09675), [COMET](https://arxiv.org/abs/2009.09025).
