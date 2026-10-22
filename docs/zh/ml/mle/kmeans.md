!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# KMeans

It’s unsupervised

1. randomly initialize k points
2. each point to its closest mean and update the center
3. repeat

It minimizes the **within-cluster sum of squared distances (WCSS)**:

$$
J = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2
$$

## Choosing K

### 1.Elbow Method

**Within-Cluster Sum of Squares (WCSS):** the sum of squared distances between each data point and centroid of its cluster

the point WCSS starts to **decrease more slowly**

### 2.Silhouette Score

how similar a data point to cluster compared to other

The score for each sample iii is:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

where:

- a(i) = average distance from point iii to all other points in the **same cluster** (measures cohesion).
- b(i) = average distance from point iii to points in the **nearest different cluster** (measures separation).

- s(i) ranges from **–1 to +1**:
    - **Close to +1** → well-clustered (good fit)
    - **Around 0** → on the boundary between clusters
    - **Negative** → possibly assigned to the wrong cluster

The **overall silhouette score** is the average of all points’ scores.
