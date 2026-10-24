# 🧭 Eigenvectors: directions that a matrix preserves

A matrix transforms vectors. Most vectors change both length and direction, but some special directions stay on the same line. A nonzero vector $v$ is an eigenvector of a square matrix $A$ if

$$
Av=\lambda v.
$$

The scalar $\lambda$ is its eigenvalue. Positive values scale along the same direction; negative values also reverse orientation. When $\lambda=0$, the vector is sent to zero. Saying that an eigenvector “never changes direction” therefore needs these qualifications.

## 🧭 A concrete two-dimensional example

Consider

$$
A=\begin{bmatrix}3&1\\1&3\end{bmatrix}.
$$

Multiplying $v_1=[1,1]^T$ gives $[4,4]^T=4v_1$. Multiplying $v_2=[1,-1]^T$ gives $[2,-2]^T=2v_2$. These two directions are stretched by factors of four and two. A vector such as $[1,0]^T$ combines both directions, so its output $[3,1]^T$ changes direction.

```python
# runnable-example
A = ((3, 1), (1, 3))
def multiply(v):
    return tuple(sum(a * b for a, b in zip(row, v)) for row in A)
assert multiply((1, 1)) == (4, 4)
assert multiply((1, -1)) == (2, -2)
```

Eigenvectors are not unique in scale: multiplying one by any nonzero scalar gives another eigenvector. Numerical software commonly returns unit-length vectors, and the sign can differ across runs without changing the represented direction.

## 💡 Why machine learning uses them

A covariance matrix is real, symmetric, and positive semidefinite. It admits orthonormal eigenvectors and nonnegative eigenvalues. Projecting data along an eigenvector gives variance equal to the associated eigenvalue. This connects matrix geometry to [principal component analysis](pca.md).

Graph methods use related ideas. Eigenvectors of graph Laplacians can expose slowly varying patterns over connected samples. Stability analysis examines whether repeated applications of a transformation amplify or damp perturbations. The interpretation depends on the matrix: a covariance matrix, adjacency matrix, and Hessian do not describe the same quantity.

## ⚖️ Going further: eigenvalues versus singular values

Not every real matrix has a complete real eigenbasis. A planar rotation by 90 degrees has no nonzero real eigenvector. A defective matrix may lack enough independent eigenvectors for diagonalization.

Singular value decomposition applies to rectangular matrices too: $X=U\Sigma V^T$. Singular values are nonnegative and describe directional amplification. For centered data, the right singular vectors of $X$ provide PCA directions without explicitly forming $X^TX$. Forming that product can worsen numerical conditioning.

When eigenvalues are repeated, an individual eigenvector can be unstable while the whole eigenspace remains meaningful. Comparing principal directions by elementwise equality is then misleading; compare the subspaces or reconstruction behavior.

## ✅ Check your understanding

**Question:** Is $[-1,-1]^T$ a different principal direction from $[1,1]^T$? **Answer:** no. They span the same one-dimensional subspace, with opposite orientation.

**Exercise:** replace the off-diagonal entries of $A$ with zero. The coordinate axes become eigenvectors with eigenvalue three; every direction is preserved because the matrix is $3I$.

## 📚 Sources and further reading

- [MIT OpenCourseWare: eigenvalues and eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-21-eigenvalues-and-eigenvectors/) develops the geometric interpretation.
- [NumPy: `eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) describes the symmetric-matrix solver.
