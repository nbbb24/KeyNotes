!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Hierarchical Clustering

**Hierarchical clustering** is an **unsupervised learning** method that builds a hierarchy of clusters — often visualized as a **dendrogram** (a tree diagram showing how clusters are merged or split).

### ⚙️**Two Main Types**

1. **Agglomerative (bottom-up)** — most common
    - Start with each point as its own cluster
    - Iteratively **merge** the two closest clusters
    - Continue until all points form one big cluster
2. **Divisive (top-down)**
    - Start with one cluster containing all points
    - Recursively **split** it into smaller clusters

### **Distance (Linkage) Methods**

When merging, the “closeness” of clusters is measured by a **linkage method**:

- **Single linkage:** minimum distance between two clusters
- **Complete linkage:** maximum distance
- **Average linkage:** average distance
- **Ward’s method:** minimizes variance within clusters
