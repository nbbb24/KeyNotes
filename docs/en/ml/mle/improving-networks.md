# 🛠️ Improving neural networks: run experiments that explain the gain

Improving a model means improving a specified outcome under a specified budget. That might be accuracy, calibration, latency, memory, or robustness. A larger model that gains one metric but misses the deployment budget is not automatically an improvement.

## ⚠️ Establish a baseline and an error taxonomy

Fix the split, metric, preprocessing, seed policy, and training budget. Record a simple baseline, such as a linear model or a small network. Inspect errors by category rather than treating every failure as a capacity problem.

For a support-message classifier, errors might come from ambiguous labels, rare categories, long messages, mixed languages, or shifted terminology. Each suggests a different experiment. Collecting more examples from an already easy category will not necessarily improve the weakest category.

## 🔄 Use learning curves to choose the next step

Plot training and validation performance against both optimization steps and training-set size. High errors on both can point toward underfitting or an optimization defect. A growing train–validation gap suggests overfitting or mismatch. Improvement as data grow suggests useful data collection, but check that the added data are genuinely independent and representative.

A controlled experiment changes one main hypothesis at a time. “More layers, different optimizer, new augmentation, longer training” may improve the score while teaching you nothing about which change mattered.

## 🧩 Worked example: a targeted augmentation

Suppose a classifier scores 94% overall but only 70% on low-light images. Collect or synthesize plausible low-light variation in the **training** data, keep a fixed real low-light validation slice, and compare against the same baseline budget. If overall performance rises to 95% but low-light accuracy stays 70%, the intervention did not address the intended failure.

An augmentation must preserve the target. Horizontal flips can be valid for some object categories but invalid for left-versus-right labels. Text paraphrases can accidentally remove negation. Validate transformations with examples before scaling them up.

## 🏗️ Hyperparameters and architecture

Search a meaningful range of learning rates first, often on a logarithmic scale. Then examine regularization, effective batch size, schedule, and model capacity. Use a fixed tuning budget and avoid selecting on the final test set.

| Evidence | Candidate intervention | Measurement |
| --- | --- | --- |
| Noisy or contradictory targets | Relabel or refine target definition | Agreement and held-out error |
| Rare failure mode | Targeted data collection | Slice-level recall/precision |
| Excess capacity | Regularization or smaller model | Gap and validation loss |
| Underused capacity | Better schedule or architecture | Tiny-set fit and learning curve |
| Serving bottleneck | Quantization, batching, or distillation | Quality–latency–memory trade-off |

Architecture changes should match the structure of the problem: convolutions exploit local spatial structure; attention handles content-dependent interactions. Adding complexity without a reason increases the search space and debugging burden.

## ⚖️ Going further: compare fairly

Repeat promising configurations across seeds or resampled splits when practical. Report uncertainty and the cost of the search, not only the best run. Compare at equal data, compute, or latency budgets depending on the question. Ablations isolate the contribution of each component.

After selection, evaluate once on a held-out test set and then monitor deployment drift. A change in preprocessing or retrieval can invalidate a previously good model comparison.

**Exercise:** three seeds improve by +0.3, −0.2, and +0.1 percentage points. Is the gain established? **Answer:** not from those numbers alone. Inspect uncertainty, paired errors, and whether the difference is operationally meaningful.

## 📚 Sources and further reading

- [scikit-learn: learning curves](https://scikit-learn.org/stable/modules/learning_curve.html) explains training-size and validation diagnostics.
- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html) motivates budget-aware search.
