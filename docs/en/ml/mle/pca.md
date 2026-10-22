# PCA: retain directions of large variation

Principal component analysis (PCA) replaces correlated numeric coordinates with orthogonal directions ordered by variance. It is useful for compression, visualization, noise reduction, and conditioning some downstream models. Its objective is geometric; large variance is not automatically useful predictive information.

## Derive the first component

Let $X\in\mathbb R^{n\times d}$ contain centered observations. Along a unit vector $w$, projected values are $Xw$, with sample variance $w^T\Sigma w$, where $\Sigma=X^TX/(n-1)$. Maximizing that variance subject to $\|w\|=1$ leads to the top eigenvector of $\Sigma$.

Subsequent components maximize remaining variance while being orthogonal to earlier ones. With $W_k$ containing the first $k$ directions, scores are $Z=XW_k$ and the rank-$k$ reconstruction is $\hat X=ZW_k^T$. Add the training mean back when reconstructing original coordinates.

## Worked example

A covariance matrix

$$
\Sigma=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

has eigenvalues 3 and 1. The first component points along $[1,1]^T/\sqrt2$ and explains $3/(3+1)=75\%$ of total variance. Keeping only this component compresses two coordinates into one while discarding variation along the opposing direction.

This is a statement about variance, not “75% of the meaning” or “75% of classification accuracy.” A low-variance direction could contain the signal that distinguishes classes.

## Centering and scaling are different choices

Centering subtracts feature means and is fundamental to ordinary covariance-based PCA. Standardizing also divides by feature standard deviations. That can be appropriate for different units, but is not universally required: it deliberately changes which variation matters.

If one sensor is measured in volts and another in millivolts, unit differences can dominate unscaled PCA. If both are meaningful physical quantities with intentionally different variance, standardizing may erase that information. Document the choice and fit all statistics on training data only.

## Going further: SVD and whitening

Compute PCA through singular value decomposition, $X=U\Sigma_XV^T$, without necessarily forming a covariance matrix. Right singular vectors give component directions; squared singular values divided by $n-1$ give explained variances. Forming $X^TX$ can worsen conditioning, while truncated or randomized SVD can reduce work when only a few components are needed.

Whitening additionally rescales component scores to unit variance. It removes relative variance information and can amplify small noisy components. Whether it helps depends on the downstream objective.

Component signs are arbitrary. Repeated or nearly repeated eigenvalues can make individual axes unstable even when the retained subspace is stable. Evaluate reconstruction or downstream performance rather than requiring identical component signs across runs.

## Check your understanding

**Question:** can PCA improve a classifier while discarding some variation? **Answer:** yes, if discarded variation is mostly nuisance or noise. It can also hurt by discarding low-variance predictive signal. Select dimensionality within the validation pipeline.

**Exercise:** if eigenvalues are `[8, 1, 1]`, how much variance do two components explain? **Answer:** 90%, assuming they are retained in descending order.

Continue with [eigenvectors](eigenvectors.md), [LDA](lda.md), and [t-SNE](tsne.md).

## Sources and further reading

- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca) covers solvers and whitening.
- [NumPy: SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) specifies the matrix factorization.
