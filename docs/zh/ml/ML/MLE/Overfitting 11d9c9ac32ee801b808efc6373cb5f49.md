# Overfitting

- model learn the training set too well
- impact model’s ability to generalize

ways avoid overfitting:

- **regularization**:
    - L1 (Lasso) Regression: penalizes high-value coefficients **(large number of features)**
        
        $$
        J(θ)=L(θ)+λi∑∣θi∣
        $$
        
        $$
        
        \frac{\partial \Omega}{\partial w_i}=
        \begin{cases}
        \lambda, & w_i > 0 \\
        -\lambda, & w_i < 0 \\
        \in[-\lambda, \lambda], & w_i = 0
        \end{cases}
        
        $$
        
        - The gradient is constant and does not depend on the magnitude of $w_i$.
        - Even if $w_i$ is very small, there is still a fixed “force” pushing it toward 0.
        - As a result, many weights are compressed to **exactly 0**, leading to a **sparse solution** and achieving **feature selection**.
        
        **penalty item**: absolute value of sum of coefficients
        
        - Feature selection
        - sparse models
        - simple model
    - L2 (Ridge) Regression: reduce coefficient close to zero **(correlated features)**
        
        $$
        J(θ)=L(θ)+λi∑θi^2
        $$
        
        $$
        \frac{\partial \Omega}{\partial w_i} = 2 \lambda w_i
        $$
        
        - The gradient is **proportional** to $w_i$.
        - As $w_i$  approaches 0, the gradient also approaches 0, so updates become slower.
        - Therefore, the weights are shrunk to very small values but usually not exactly 0 → it **does not perform feature selection**, instead **retains all features**.
        
        **penalty item**: squared sum of coefficients
        
        - no feature selection (no 0 coefficient)
        - handle multicollinearity
- **simple model**: less variable and parameter
- **cross-validation**:
    1. split the model into k parts
    2. train on k-1 folds and test on the remaining fold
    3. repeat k times