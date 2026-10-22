# Covariance, correlation, bias, and variance

These terms describe different kinds of variation. **Covariance and correlation** summarize how variables move together. **Bias and variance** describe properties of a prediction procedure across possible training samples. Confusing the two makes it harder to diagnose a model.

## From joint movement to a scale-free number

For paired observations $(x_i,y_i)$, sample covariance is

$$
s_{xy}=\frac{1}{n-1}\sum_i(x_i-\bar x)(y_i-\bar y).
$$

If large values of $x$ tend to occur with large values of $y$, the products are often positive. Covariance has the product of the variables' units. Converting meters to centimeters multiplies covariance by 100, without changing the underlying relationship.

Pearson correlation divides by the two sample standard deviations:

$$
r=\frac{s_{xy}}{s_xs_y}.
$$

It lies between −1 and 1 when both variances are positive. If one variable is constant, the denominator is zero: the correlation is undefined, not zero.

## Worked example

Let $x=[1,2,3]$ and $y=[2,4,6]$. Their means are 2 and 4. The covariance is $(2+0+2)/2=2$. The standard deviations are 1 and 2, so correlation is 1. Changing $y$ to centimeters multiplies covariance and $s_y$ by 100; correlation remains 1.

Now let $x=[-1,0,1]$ and $y=x^2=[1,0,1]$. Covariance is zero because the negative and positive contributions cancel. Nevertheless, knowing $x$ completely determines $y$. Zero linear correlation does not mean statistical independence.

## How this helps model building

A correlation matrix can reveal redundant numeric features, incorrect units, or obvious collection artifacts. It cannot establish causal direction. A common cause can drive both variables, and selecting a subgroup can create misleading correlations. Always inspect scatter plots and domain context alongside a coefficient.

For nonlinear monotonic relationships, rank-based measures such as Spearman correlation answer a different question. Neither measure replaces evaluation of the actual prediction task. Computing correlations with the target over the whole dataset and selecting features before splitting is leakage.

## Bias and variance: a different experiment

Imagine repeatedly drawing a fresh training dataset and fitting the same procedure. At a fixed input $x$, squared-error prediction risk can be decomposed, under the usual regression assumptions, into

$$
E[(Y-\hat f(x))^2]=\operatorname{Bias}[\hat f(x)]^2+
\operatorname{Var}[\hat f(x)]+\sigma^2(x).
$$

Bias is the difference between the procedure's average prediction and the true regression function. Variance measures how predictions change with the sampled training data. The final term is irreducible conditional noise. Bias is therefore **not defined as training error**, and variance is not simply “risk of bad predictions.”

A very shallow tree may miss a curved relationship consistently. A very deep tree may chase different noise patterns in each sample. More data, regularization, and ensembling affect this balance differently; dataset size alone does not determine the correct model class.

## Check your understanding

**Question:** Why can a high training score coexist with high bias? **Answer:** finite-sample fit is not the definition of statistical bias; the estimator's behavior across datasets and the target function matter. In practice, learning curves are diagnostic evidence, not a direct measurement of the theoretical decomposition.

Continue with [PCA](pca.md), [random forests](random-forests.md), and [overfitting](overfitting.md).

## Sources and further reading

- [NumPy: covariance](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) gives normalization conventions.
- [scikit-learn: bias–variance example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html) compares single trees and bagging.
