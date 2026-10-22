# K-means: compress a dataset into representative centers

K-means partitions observations into $K$ groups by minimizing squared Euclidean distance to their assigned centers. It is an unsupervised geometric objective. A cluster is not automatically a meaningful real-world category.

## Why alternate assignment and updating?

For clusters $C_j$ and centers $\mu_j$, the objective is

$$
J=\sum_{j=1}^{K}\sum_{x_i\in C_j}\|x_i-\mu_j\|^2.
$$

With centers fixed, assign each observation to its nearest center. With assignments fixed, the mean minimizes squared distance within each cluster. Repeating these two operations does not increase the objective, but can converge to a local solution. Initialization therefore matters; k-means++ spreads initial centers and multiple restarts can improve reliability.

## Worked example

One-dimensional observations are `[1, 2, 8, 9]`, with initial centers 1 and 8. Assignments are `{1,2}` and `{8,9}`. Updated centers are 1.5 and 8.5. The within-cluster sum of squares is $4\times0.5^2=1$.

```python
# runnable-example
clusters = [[1, 2], [8, 9]]
centers = [sum(c) / len(c) for c in clusters]
sse = sum((x - m) ** 2 for c, m in zip(clusters, centers) for x in c)
assert centers == [1.5, 8.5]
assert sse == 1
```

Compare this with [k-NN](nearest-neighbors.md): K-means learns unlabeled centers; k-NN predicts from labeled neighbors. The letter k does not imply the same learning problem.

## Choosing K

Increasing $K$ can only improve or preserve the best attainable training objective, so lowest training error is not a meaningful selection rule. The elbow heuristic looks for diminishing improvement, but an obvious elbow may not exist.

The silhouette of a sample compares average within-cluster distance $a(i)$ with the smallest average distance to another cluster $b(i)$:

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}.
$$

It is near one for strong separation, near zero at an ambiguous boundary, and can be negative for an unfavorable assignment. It requires a suitable distance and meaningful nontrivial clusters. A high silhouette does not prove that the grouping answers a useful question.

## Where the assumptions fail

Squared Euclidean distance favors compact groups and can struggle with elongated shapes, unequal densities, or heavy outliers. A large outlier can drag a mean. Scaling changes the objective: standardizing a low-variance but noisy feature may amplify noise, so choose scaling based on semantics.

For categorical or mixed data, an arithmetic mean may not represent a valid observation. Do not apply K-means to arbitrary integer-encoded categories as if their codes defined distances.

## Going further

Mini-batch K-means trades exact full-dataset updates for scalable approximate updates. Compare stability across seeds and samples, and inspect centers and representative members. If cluster labels are later used as features, fit the clustering only on the training split.

**Exercise:** can two runs use different numeric cluster labels and still represent the same partition? **Answer:** yes. Cluster identifiers are arbitrary; compare assignments up to permutation or use permutation-invariant metrics.

## Sources and further reading

- [scikit-learn: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) describes initialization and limitations.
- [k-means++ paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) analyzes careful seeding.
