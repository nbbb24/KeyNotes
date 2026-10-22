# Decision Tree

At each node, it selects the **best feature and threshold** using impurity measures such as:

- **Gini impurity** (CART)
    - $Gini=1−∑p_i^2$
        
        where pi is the probability of class *i* in the node.
        
    - **Meaning:** Measures how *mixed* the classes are — lower Gini means purer nodes.
    - **Intuition:**
        - Gini = 0 → pure node (only one class)
        - Gini = 0.5 → 50/50 mix of two classes
- **Entropy / Information Gain** (ID3, C4.5)
    
    $$
    Entropy=−∑p_ilog_2(p_i)
    $$
    
    - **Information Gain (split criterion):**
    - $IG=Entropy(parent)−children∑\frac{n_{parent}}{n_{child}}Entropy(child)$
- **Variance reduction** (for regression trees)