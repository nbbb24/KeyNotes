# 🚪 NLP introduction: from text to measurable predictions

Natural language processing begins by defining a task, a representation, and an evaluation protocol. A large model is one possible component; it does not replace a clear target or a trustworthy dataset.

## 🧭 Start with a bag-of-words baseline

A bag-of-words vector counts vocabulary items in a document. With vocabulary `[good, bad, movie]`, `good good movie` becomes `[2,0,1]`. A linear classifier learns weights on these features:

$$
z=w^\top x+b,\qquad p(y=1\mid x)=\sigma(z).
$$

The word counts themselves are not learned classifier weights. TF–IDF reweights counts using corpus statistics; those statistics should be fitted on training data. A sparse baseline is fast, interpretable, and often competitive on keyword-driven tasks.

![NLP introduction: from text to measurable predictions, original figure 1](../../../../../assets/notion/013-00.png)

The first diagram separates representation $h=f(x)$, task scores, and the final decision. Changing the representation can improve a classifier without changing its decision rule.

![NLP introduction: from text to measurable predictions, original figure 2](../../../../../assets/notion/013-01.png)

The bag-of-words diagram counts word identities before applying learned weights. Its summed features do not retain the order of “not” and “love.”

![NLP introduction: from text to measurable predictions, original figure 3](../../../../../assets/notion/013-02.png)

The limitations slide lists morphology, synonyms, combinations, and sentence structure. Adding n-grams can recover some local combinations, but increases sparsity and does not solve arbitrary long-range structure.

## 🧩 What the representation misses

`dog bites person` and `person bites dog` have the same unigram counts. Negation, order, long-range dependencies, and polysemy need richer features or contextual models. N-grams recover local order but expand the feature space and still provide limited generalization to unseen combinations.

Embeddings replace sparse lexical identity with learned dense vectors. Contextual models then make a token's representation depend on its sentence. This progression adds capability and cost; use a baseline to determine which added complexity actually helps.

## 🛠️ A practical experiment

For review classification, split by the unit that could leak information—such as product, user, or time—before fitting the vectorizer. Train a bag-of-words logistic classifier and a contextual model. Compare class-specific precision/recall and inspect errors involving negation, sarcasm, and domain vocabulary. A higher aggregate score can hide regressions on a small but important subgroup.

**Exercise.** Can a unigram model distinguish `not good` from `good`? It can learn a contribution from `not`, but cannot reliably model all context-dependent negation scopes. The issue is expressiveness, not absolute inability to use that word.

Continue with [word representations](word-representations.md). Course reference: [CMU Advanced NLP 2024](https://phontron.com/class/anlp2024/lectures/).
