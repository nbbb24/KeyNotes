!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is an **unsupervised** clustering algorithm that groups together points that are **closely packed (high density)** and marks points in low-density areas as **outliers (noise)**

### **How DBSCAN Works — Step by Step**

1. **Pick an unvisited point.**
    - Calculate how many points fall within its **ε (epsilon)** radius.
2. **Classify the point.**
    - If it has **≥ minPts** neighbors → it’s a **core point** → start a new cluster.
    - If it has **< minPts** → mark as **noise (temporarily)**.
3. **Expand the cluster.**
    - For each neighbor of a core point:
        - If it’s also a core point, include its neighbors too (density reachable).
        - Keep adding reachable points until no more can be added.
4. **Repeat**
    - Continue with remaining unvisited points.
    - If a noise point later becomes reachable from a core point, it’s reclassified into that cluster.
