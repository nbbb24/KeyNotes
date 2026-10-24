# 🎓 Learning paradigms: where the training signal comes from

A learning algorithm needs a rule for deciding whether a change is useful. The main paradigms differ in **where that signal comes from**, not simply in which neural-network architecture they use. Before choosing a model, identify the information available during training and the decision that must be made after deployment.

## 🧩 Start with one running example

Suppose a service receives customer messages. You could build several different systems from the same text:

| Task | Training signal | Paradigm |
| --- | --- | --- |
| Predict a human-assigned support category | Message–category pairs | Supervised learning |
| Find recurring themes without categories | Similarity or a representation objective | Unsupervised learning |
| Predict missing words in raw messages | Targets constructed from the text itself | Self-supervised learning |
| Combine a small labeled set with many unlabeled messages | Labels plus assumptions about unlabeled data | Semi-supervised learning |
| Choose a sequence of help actions and optimize eventual resolution | Rewards produced after actions | Reinforcement learning |

These categories overlap. A language model can receive self-supervised pretraining, supervised instruction tuning, and reward-based post-training. That does not make its architecture change paradigms; its training objective and data do.

## ⚙️ How the objectives differ

Supervised learning minimizes an empirical loss, such as

$$
\hat R(\theta)=\frac1n\sum_{i=1}^{n}\ell(f_\theta(x_i),y_i).
$$

Here $x_i$ is an input, $y_i$ its target, and $\theta$ the learned parameters. A separate evaluation set estimates how well that fitting procedure generalizes.

Unsupervised methods also optimize explicit objectives. K-means minimizes distances to cluster centers; it does not discover a uniquely correct meaning of “customer type.” The features, distance measure, and number of clusters encode assumptions.

Self-supervision creates a prediction problem from otherwise unlabeled observations. In next-token prediction, the prefix is the input and the following token is the label. “No manual labels” therefore does not mean “no targets.”

Reinforcement learning optimizes expected return, often $E[\sum_t\gamma^t r_t]$. Actions affect later observations, rewards may be delayed, and collecting better data is itself part of the problem. Replacing a classification label with a numeric score does not automatically create a meaningful sequential decision task.

## 📌 A small design exercise

You have 200 labeled messages and 200,000 unlabeled messages. First establish a supervised baseline using a pretrained text representation. Then evaluate whether self-supervised adaptation or pseudo-labeling improves a held-out set. Do not assume the unlabeled data help: a different language, time period, or customer population can introduce distribution shift.

If the task is routing rather than content generation, a small classifier may be easier to evaluate and maintain than an instruction-following model. Conversely, a fixed label set is a poor fit for open-ended summarization.

## 🚀 Going further

Separate **learning paradigm**, **model family**, and **evaluation protocol**. A decision tree and a neural network can both perform supervised learning. A classifier can be trained offline yet operate in a changing online environment. Active learning describes how examples are selected for labeling; it is not a substitute for specifying a loss.

For temporal or repeated-entity data, split by time or entity before training. Random splits can leak the same underlying situation into both sets, making every paradigm look better than it is.

## ✅ Check your understanding

**Question:** Is next-token pretraining supervised or unsupervised?

**Answer:** It is commonly called self-supervised: the data provide their own labels. Its optimization is still a supervised prediction loss. State the signal construction rather than arguing only about the label.

Continue with [data cleaning](data-cleaning.md), [loss functions](loss-functions.md), and [RLHF](../llm/alignment/rlhf.md).

## 📚 Sources and further reading

- [scikit-learn: semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html) explains assumptions behind using unlabeled examples.
- [Sutton and Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html) develops states, actions, rewards, and return.
