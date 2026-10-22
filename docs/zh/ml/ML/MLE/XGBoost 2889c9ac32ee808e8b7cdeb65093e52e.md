# XGBoost

- **Regularization (L1 & L2):**
    
    Adds penalties to leaf weights to prevent overfitting — a major improvement over classic GBDT.
    
    $$
    \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2
    $$
    
- **Second-order optimization:**
    
    Uses both **first and second derivatives** of the loss function (gradients & Hessians) for more accurate updates.
    
- **Parallelized tree construction:**
    
    Splits are computed in parallel, making training much faster.