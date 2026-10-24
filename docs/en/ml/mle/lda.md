# 📐 Linear discriminant analysis: use labels to find separation

Linear discriminant analysis (LDA) has two closely related roles: a probabilistic classifier with shared class covariance, and a supervised dimensionality-reduction method. Unlike PCA, it uses labels to distinguish useful separation from variation within classes.

## 🧭 Fisher's projection criterion

Let $S_W$ summarize within-class scatter and $S_B$ between-class scatter. A projection direction $w$ can be chosen to maximize

$$
J(w)=\frac{w^TS_Bw}{w^TS_Ww}.
$$

The numerator rewards separated class means; the denominator penalizes spread within classes. The corresponding generalized eigenproblem is

$$
S_Bw=\lambda S_Ww.
$$

Writing an explicit inverse $S_W^{-1}S_B$ is a useful theoretical shorthand only when the inverse exists. Numerical implementations should use appropriate solvers rather than blindly inverting a possibly singular matrix.

## 📊 Worked geometric example

Imagine two classes centered at `(−2, 0)` and `(2, 0)`. Each class is stretched vertically, with much larger variance along the vertical axis. PCA may select that high-variance vertical direction even though it does not distinguish the means. LDA favors the horizontal direction because it separates the classes relative to their within-class spread.

This does not mean LDA always beats PCA. The label information may be scarce or noisy, and the assumptions behind a linear separation may be poor.

## 🔍 The classifier interpretation

Assume $x\mid y=c$ is Gaussian with mean $\mu_c$ and a covariance matrix $\Sigma$ shared across classes. Comparing class log posteriors cancels common quadratic terms, leaving linear discriminant scores:

$$
\delta_c(x)=x^T\Sigma^{-1}\mu_c-
\frac12\mu_c^T\Sigma^{-1}\mu_c+\log\pi_c.
$$

The class prior $\pi_c$ influences the decision boundary. Allowing each class its own covariance leads to quadratic discriminant analysis (QDA), which has more flexibility but more parameters to estimate.

## 📦 Dimensionality and data requirements

With $C$ classes, the between-class mean structure has rank at most $C-1$. LDA therefore provides at most $\min(d,C-1)$ discriminative components in the usual formulation. Two classes provide at most one such component, regardless of the original feature count.

When features outnumber observations or are strongly correlated, covariance estimates can be singular or noisy. Shrinkage toward a simpler covariance structure can improve estimation. Scaling and preprocessing still need to be fitted within training folds.

## 🚀 Going further

LDA's projection emphasizes class-mean differences under its scatter criterion. Classes that differ mainly through nonlinear geometry or covariance may not separate well in that subspace. Inspect class-conditional distributions rather than treating a two-dimensional plot as proof of a complete model.

The abbreviation LDA also means Latent Dirichlet Allocation in topic modeling. These are different algorithms; state the full name when context is ambiguous.

**Exercise:** can a three-class dataset yield five standard LDA components? **Answer:** no. At most two discriminant components are available, even if there are hundreds of features.

Continue with [PCA](pca.md), [covariance](covariance-correlation.md), and [logistic regression](linear-logistic.md).

## 📚 Sources and further reading

- [scikit-learn: linear and quadratic discriminant analysis](https://scikit-learn.org/stable/modules/lda_qda.html) explains the probabilistic model, projection, and shrinkage.
- [scikit-learn: LDA API](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) specifies solver and component constraints.
