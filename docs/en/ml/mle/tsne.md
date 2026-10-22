# t-SNE: visualize local neighborhoods without overreading the map

t-distributed stochastic neighbor embedding (t-SNE) creates a low-dimensional visualization that emphasizes local neighborhoods. Its output can be useful for exploring representations, but it is not a faithful geographic map of all high-dimensional distances.

## Convert distances into neighborhood probabilities

In the original space, t-SNE assigns high similarity to nearby observations using locally scaled Gaussian neighborhoods. The perplexity parameter controls an effective neighborhood scale; it is related to the entropy of each conditional distribution, not a fixed number of clusters.

In the low-dimensional space, similarities use a heavy-tailed Student-t form. That gives separated points more room and helps address crowding. The optimization minimizes

$$
D_{KL}(P\|Q)=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

Because this direction of KL heavily penalizes separating pairs with large $p_{ij}$, preserving local neighbors is central. It does not impose equal accuracy on all far-away relationships.

## A concrete interpretation example

Suppose a digit representation produces separate islands for several writing styles. Two islands being far apart does not imply that their source-space distance is twice that of two closer islands. A large island does not necessarily contain more variance than a small one, because local scaling and the embedding objective distort density and area.

If class labels are colored on the plot, ask whether the representation was already trained using those labels. A visually clean separation is then not independent evidence that an unsupervised method discovered the categories.

## A responsible workflow

1. Clean and scale features where appropriate; consider PCA to reduce noise and computation.
2. Run several plausible perplexities and random seeds.
3. Compare local-neighbor preservation and stability, not just the prettiest picture.
4. Inspect ambiguous or surprising points in the original data.
5. Validate any claimed groups with an independent analysis or downstream task.

Perplexity must be compatible with sample count. Learning rate, initialization, early exaggeration, iteration budget, and approximation method also affect the result. Keep these settings with the figure.

## Going further

The objective is nonconvex, so different layouts can represent similar local structure. Rotation, reflection, and translation of the whole map do not change its neighborhood meaning. Runs on different datasets cannot automatically be compared by coordinate position.

Standard t-SNE is primarily transductive: it optimizes coordinates for the provided observations. Adding new points or changing the sample can change the layout. Some implementations offer extensions, but a fitted t-SNE map should not be assumed to expose the same straightforward transform as PCA.

Approximate algorithms reduce the cost of repulsive interactions, yet large datasets still require deliberate resource planning. Subsampling can help exploration but may hide rare populations. Always record which observations were plotted.

**Exercise:** a small, tight island appears after changing the random seed. Does it establish a new class? **Answer:** no. Check stability, original-space neighbors, and actual examples before interpreting it.

Continue with [PCA](pca.md), [Isomap](isomap.md), and [clustering](kmeans.md).

## Sources and further reading

- [van der Maaten and Hinton: t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html) introduces the objective.
- [scikit-learn: manifold learning](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) documents practical parameters and caveats.
