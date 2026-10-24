# 🌲 Random forests: average trees that make different mistakes

A random forest combines decision trees trained with randomized data and feature choices. It is a strong baseline for many tabular tasks because it captures nonlinear interactions without requiring a manually specified functional form.

## 🧭 From one tree to a forest

A deep tree is sensitive to its sample: a slightly different dataset can select another early split. A forest reduces that instability by combining many trees. In a common configuration, each tree receives a bootstrap sample; at each node, only a random subset of features is considered for splitting.

Feature subsampling matters because a very strong feature could otherwise dominate every tree. Lower correlation among tree errors makes averaging more useful. The number of candidate features trades tree strength against diversity.

For regression, average the tree predictions. Classification aggregation depends on the implementation: scikit-learn averages class-probability estimates across trees before selecting a class. This is related to, but not always identical to, majority voting over each tree's hard label.

## 🧩 Worked example: averaging probabilities

Three trees estimate positive-class probabilities `[0.51, 0.51, 0.01]`. Hard voting predicts positive because two trees exceed 0.5. Averaging gives about 0.343, which predicts negative at threshold 0.5. Therefore, inspect the aggregation rule before reproducing a library result by hand.

```python
# runnable-example
p = [0.51, 0.51, 0.01]
assert sum(x > 0.5 for x in p) == 2
assert sum(p) / len(p) < 0.5
```

## 📊 Out-of-bag evaluation

An observation has probability $(1-1/n)^n\approx e^{-1}$ of being absent from a bootstrap sample. About 36.8% are therefore out-of-bag for a given tree in the large-$n$ limit. Predictions from trees that did not train on a row provide an internal evaluation estimate.

This does not automatically respect groups or time. If related rows from the same user appear in a tree's training sample, the out-of-bag row may still have an overly optimistic estimate. Use a deployment-appropriate external split when needed.

## ⚠️ Tuning and failure modes

Control minimum leaf size, depth, candidate-feature count, and the number of trees. Additional trees often stabilize the ensemble estimate, but increase memory and inference cost. Leaf size can regularize the probability estimates and reduce sensitivity to individual labels.

Trees do not require the same scaling treatment as k-NN, but they are not universally immune to outliers. Extreme target values can influence regression leaf means. Standard regression forests also do not extrapolate trends beyond the range implied by their leaves as a linear model can.

## 🚀 Going further

Impurity importance favors some high-cardinality features. Correlated features can substitute for each other, making a low importance score ambiguous. Inspect held-out permutation importance and task-specific error slices rather than treating a feature ranking as a causal explanation.

For class imbalance, compare class weighting and decision thresholds using precision–recall or application costs. Raw forest probabilities may need calibration when probability quality matters.

**Exercise:** would doubling the trees fix a consistently wrong feature definition? **Answer:** no. More averaging cannot remove an error shared by every tree's inputs.

## 📚 Sources and further reading

- [scikit-learn: forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest) describes aggregation and hyperparameters.
- [Breiman: Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) gives the original formulation.
