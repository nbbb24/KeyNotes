# Metrics: connect a score to a decision

A metric compresses many predictions into a number. The useful question is not “Which metric is best?” but “Which errors matter, at what operating point, for which population?” Define those choices before comparing models.

## Classification starts with the confusion matrix

For a designated positive class, true positives (TP) are correctly detected positives, false positives (FP) are false alarms, and false negatives (FN) are missed positives.

$$
\text{precision}=\frac{TP}{TP+FP},\qquad
\text{recall}=\frac{TP}{TP+FN},\qquad
F_1=\frac{2TP}{2TP+FP+FN}.
$$

Precision asks how trustworthy an alert is. Recall asks how much of the target population is found. F1 combines them but ignores true negatives and does not directly encode the financial or operational cost of an error. Handle undefined denominators explicitly rather than silently treating every empty case as perfect.

## Worked example: why accuracy misleads

Among 1,000 cases, 20 are positive. A model that always predicts negative has 98% accuracy and zero recall. Another model catches 16 positives but raises 24 false alarms. It has TP=16, FP=24, FN=4, TN=956: precision 40%, recall 80%, and accuracy 97.2%.

The second model has lower accuracy but may be far more useful if missing a positive is expensive. It may also overwhelm a team with false alarms. The operational decision determines the preferred trade-off.

```python
# runnable-example
TP, FP, FN, TN = 16, 24, 4, 956
assert TP / (TP + FP) == 0.4
assert TP / (TP + FN) == 0.8
assert (TP + TN) / 1000 == 0.972
```

## Scores, thresholds, and probabilities

Changing a classification threshold changes the confusion matrix without changing the ranking model. ROC curves plot true-positive rate against false-positive rate; precision–recall curves focus on positive detections and are often more revealing under strong imbalance. Their values still depend on the evaluation setup, and precision depends on prevalence.

A high ranking score does not guarantee calibrated probabilities. If 100 cases receive probability 0.8, calibration asks whether roughly 80 are positive over repeated comparable cases. Log loss and Brier score evaluate probabilistic predictions; reliability diagrams make calibration errors visible.

### What ROC AUC measures

AUC is the probability that a randomly selected positive receives a higher score than a randomly selected negative, counting a tie as half:

$$
\mathrm{AUC}=\frac{\sum_{i\in +}\sum_{j\in -}[\mathbf{1}(s_i>s_j)+\tfrac12\mathbf{1}(s_i=s_j)]}{n_+n_-}.
$$

Equivalently, sort scores in ascending order and assign average ranks to ties. If $R_+$ is the positive-rank sum, then $\mathrm{AUC}=[R_+-n_+(n_++1)/2]/(n_+n_-)$. Integrating the empirical ROC curve with trapezoids gives the same tie-aware result. For positive scores 0.9 and 0.5 and negative scores 0.5 and 0.1, three pairs win and one ties, so AUC is $3.5/4=0.875$. AUC is undefined when an evaluation set contains only one class.

![Original Metrics figure 1](../../../assets/notion/099-00.png)

![Original Metrics figure 2](../../../assets/notion/099-01.png)

## Regression and multiclass evaluation

MAE reports average absolute error in target units. RMSE emphasizes large errors while retaining those units. $R^2$ compares squared error with a mean-prediction reference and can be negative on held-out data. MAPE is unstable near zero and unsuitable when percentage error has no meaningful denominator.

In multiclass problems, macro averaging gives each class equal weight; micro averaging pools decisions; weighted averaging weights classes by support. A single weighted score can hide poor minority-class performance. Report per-class or meaningful subgroup results as well.

## Going further

Compare models on the same observations and quantify uncertainty with an appropriate paired resampling method. For repeated users, resample users rather than pretending rows are independent. Choose thresholds on validation data, freeze them, then evaluate the final test set.

**Exercise:** if deployment prevalence drops while sensitivity and specificity stay fixed, what happens to precision? **Answer:** it generally falls because false alarms form a larger share of positive predictions.

Continue with [loss functions](loss-functions.md), [data cleaning](data-cleaning.md), and [LLM evaluation](../llm/courses/llm/evaluation.md).

## Sources and further reading

- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) defines classification and regression metrics.
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html) separates ranking from calibration.
