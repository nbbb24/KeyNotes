# Support vector machines: classify with a margin

A support vector machine (SVM) chooses a decision boundary while controlling the margin around it. For a linear binary classifier, the score is $f(x)=w^Tx+b$ and the prediction is its sign. The geometric margin connects classification to regularization.

## From a separating line to a soft margin

A hard-margin SVM seeks a wide separating gap when the data are separable. Real datasets often contain overlap or mislabeled observations, so a soft-margin objective allows violations:

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i,
\quad y_i(w^Tx_i+b)\geq1-\xi_i,\quad\xi_i\geq0.
$$

Labels are $y_i\in\{-1,1\}$. The equivalent unconstrained form uses hinge loss. $C$ trades the penalty on margin violations against the weight norm. A larger $C$ emphasizes fitting violations, but its actual effect on generalization must be validated.

Support vectors are observations with nonzero coefficients in the dual solution. In a soft-margin model they can lie on the margin, inside it, or on the wrong side of the boundary; “the nearest correctly classified points” is incomplete.

## Worked example: margin violations

For a positive label, scores 2, 0.5, and −1 have hinge losses 0, 0.5, and 2. The middle point is correctly classified but still inside the desired unit functional margin. This explains why a correctly classified sample can influence training.

```python
# runnable-example
hinge = lambda y, score: max(0, 1 - y * score)
assert [hinge(1, s) for s in [2, 0.5, -1]] == [0, 0.5, 2]
```

![Original Support Vectors Machine figure 1](../../../assets/notion/106-00.png)

## Kernels: change the similarity without listing every feature

A kernel computes an inner product in a feature space, $K(x,z)=\langle\phi(x),\phi(z)\rangle$. Linear, polynomial, and radial-basis-function kernels represent different assumptions. For an RBF kernel,

$$
K(x,z)=\exp(-\gamma\|x-z\|^2).
$$

Large $\gamma$ makes influence highly local; small $\gamma$ makes similarity vary more slowly. Tune it jointly with $C$ after scaling features. A polynomial kernel models certain feature interactions; its degree also controls complexity. Sigmoid-shaped kernels are not positive semidefinite for every parameter choice and are less straightforward.

A kernel does not guarantee that the real problem becomes usefully separable. Too much flexibility can fit noise, and the representation may still miss the signal.

## Going further

Kernel methods can require substantial memory and computation as sample count grows; prediction also depends on the number of support vectors. For large sparse text datasets, a linear solver may be more practical than an RBF kernel. A kernel choice should follow the data and budget rather than a universal “default best” rule.

SVM scores are not automatically probabilities. Probability calibration introduces another estimation step and must respect the train/validation split. Feature scaling belongs inside cross-validation because distance-based kernels are sensitive to units.

**Exercise:** why does scaling only one input feature change an RBF model? **Answer:** it changes squared distances and therefore the similarity geometry, even when the underlying observations are the same.

## Sources and further reading

- [scikit-learn: support vector machines](https://scikit-learn.org/stable/modules/svm.html) covers kernels, scaling, and solver behavior.
- [LIBSVM guide](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) gives a practical validation workflow.
