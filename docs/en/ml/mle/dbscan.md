# 🔍 DBSCAN: find connected dense regions and leave noise unassigned

DBSCAN groups observations through density connectivity rather than distance to a fixed number of centers. It is useful when clusters have irregular shapes and some points should remain unassigned. Its two key parameters describe a local neighborhood: radius $\varepsilon$ and a minimum neighborhood count.

## 🧭 Core, border, and noise

A core point has enough observations in its $\varepsilon$ neighborhood. A border point is near a core point but does not itself meet the density threshold. Noise is not reached by any cluster under the chosen parameters. In scikit-learn, `min_samples` includes the point itself; other descriptions may use different counting conventions.

Start from a core point and expand through neighboring core points. Border points can be included, but do not extend a cluster through another chain unless they are also core. This distinction prevents a sparse bridge of border points from connecting regions in the same way as a dense chain.

## 🧩 Worked example on a line

Take points `[0.0, 0.1, 0.2, 3.0]`, radius 0.15, and minimum count 3 including the point itself. The point 0.1 sees all three nearby observations and is core. Points 0.0 and 0.2 each see only two points, so they are border points. The point 3.0 is noise. The result is one three-point cluster and one unassigned point.

If the radius grows to 3.1, the previously separate observation may join the dense region. There is no universally correct radius independent of the feature scale.

## 📌 Selecting parameters

Inspect neighbor-distance distributions and test stability over a plausible range. Scale features deliberately and choose a metric that reflects what “near” means. Geographic coordinates, sparse text vectors, and standardized sensor values need different treatment.

A single density threshold may struggle when one genuine cluster is much denser than another. A radius large enough to connect a sparse group may merge dense groups or absorb noise. Methods such as OPTICS or HDBSCAN address related variable-density questions, with their own assumptions.

## 🚀 Going further

Noise is a model result, not a statement that a record is erroneous. A rare valid observation can be labeled noise; a dense collection of corrupted records can form a cluster. Inspect what the labels mean for the application.

A border point adjacent to more than one core component can be assigned according to traversal order in common implementations. Core connectivity is more stable than every border assignment. Do not demand exact label equality across permutations without considering this detail.

Neighbor search dominates cost. Indexes can help in suitable low-dimensional spaces, but high dimension or a large radius can produce very large neighborhoods and memory consumption. Some implementations have quadratic worst-case memory behavior. Benchmark on realistic density, not only row count.

DBSCAN does not inherently provide the same universal out-of-sample prediction rule as a supervised classifier. Adding new observations and rerunning clustering can change density relationships.

**Question:** can a point initially marked noise later become a border point? **Answer:** yes, if expansion from a subsequently discovered core point reaches it. Its local count need not change.

Continue with [K-means](kmeans.md) and [hierarchical clustering](hierarchical-clustering.md).

## 📚 Sources and further reading

- [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) explains neighborhood and memory behavior.
- [Original DBSCAN paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) defines density reachability and connectivity.
