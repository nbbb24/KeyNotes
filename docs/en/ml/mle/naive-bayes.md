# 🎲 Naive Bayes: a useful classifier built from a strong assumption

Naive Bayes applies Bayes' rule while assuming that features are conditionally independent given the class. The assumption is often false, yet the resulting classifier can be a useful, inexpensive baseline, especially for text. Understand both the computational benefit and the probability limitations.

## 🧮 Derive the classification rule

For class $c$ and features $x_1,\ldots,x_d$,

$$
p(c\mid x)\propto p(c)p(x\mid c)\approx p(c)\prod_jp(x_j\mid c).
$$

The class prior $p(c)$ captures how common a class is. Feature likelihoods describe how observations behave within a class. For classification, choose the largest posterior score; the shared evidence term $p(x)$ cancels in the comparison.

Compute in log space:

$$
\operatorname{score}(c)=\log p(c)+\sum_j\log p(x_j\mid c).
$$

This turns many small products into sums and avoids numerical underflow. Normalizing scores into probabilities still requires a stable log-sum-exp operation.

## 🧩 Worked example: a spam signal

Suppose spam has prior 0.2 and legitimate mail prior 0.8. For two binary features, “contains offer” and “contains link,” assume their presence probabilities are `(0.8, 0.7)` in spam and `(0.1, 0.2)` in legitimate mail. A message containing both receives unnormalized scores $0.2\times0.8\times0.7=0.112$ and $0.8\times0.1\times0.2=0.016$. The normalized spam probability is 0.875 under this model.

This example deliberately assumes conditional independence. If “offer” and “link” are nearly redundant within a class, multiplying both pieces of evidence can exaggerate confidence.

## 📌 Choose the likelihood family

| Variant | Feature interpretation | Typical use |
| --- | --- | --- |
| Gaussian | Continuous values with class-conditional Gaussian marginals | Small numeric baselines |
| Multinomial | Nonnegative event counts | Word-count text classification |
| Bernoulli | Presence or absence | Binary indicator features |
| Complement | Statistics derived from other classes | Some imbalanced text tasks |

Do not pass arbitrary signed standardized values into a count-based likelihood. The model's mathematical assumptions constrain valid inputs.

For word counts, additive smoothing estimates $p(w\mid c)=(N_{wc}+\alpha)/(N_c+\alpha V)$. It prevents an unseen word from making the entire class likelihood zero. The vocabulary size $V$, tokenization, and smoothing strength all matter.

## 🚀 Going further

Naive Bayes can classify well even when its posterior probabilities are poorly calibrated. A correct ordering of class scores is a weaker requirement than an accurate probability. If probabilities drive decisions, validate calibration separately.

Fit tokenization and vocabulary only on training data. Preserve the real class prevalence in evaluation, and consider prior shift if deployment class frequencies differ. Compare against a regularized linear classifier: similar computational simplicity can come with different robustness to correlated features.

**Exercise:** why is the independence assumption conditional rather than unconditional? **Answer:** features may be correlated overall because classes differ. The model assumes they become independent after the class is known, which is still a strong restriction.

Continue with [linear and logistic regression](linear-logistic.md) and [metrics](metrics.md).

## 📚 Sources and further reading

- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) defines variants and limitations.
- [Stanford: Introduction to Information Retrieval, Naive Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html) derives the text classifier.
