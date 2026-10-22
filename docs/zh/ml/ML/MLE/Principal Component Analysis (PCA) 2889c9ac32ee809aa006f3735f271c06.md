# Principal Component Analysis (PCA)

transforms a set of **correlated variables** into a smaller set of **uncorrelated variables** called **principal components**, while retaining as much **variance** (information) as possible.

PCA finds new axes (directions) that maximize the variance of the data.

## **How does PCA work? (Step-by-step)**

You should always be able to summarize the **PCA pipeline** like this:

1. **Standardize the data**
    - Subtract mean and divide by standard deviation
    - Important because PCA is variance-based and sensitive to scale
2. **Compute the covariance matrix**
    
    $$
    \Sigma = \frac{1}{n-1} X^T X
    $$
    
    where X is the standardized data matrix.
    
3. **Find eigenvalues and eigenvectors**
    - Eigenvectors = principal component directions
    - Eigenvalues = variance explained by each component
4. **Sort eigenvalues in descending order**
    - The first few components capture most of the variance
5. **Project the data onto top** k **components**
    
    $$
    X_{reduced} = X W_k
    $$
    
    where W_k are the top k eigenvectors.
    

## **Why do we need to standardize before PCA?**

**Answer:**

Because PCA is based on the covariance matrix, and covariance depends on the **scale** of variables.

If features have different units or ranges, the feature with the largest variance will dominate the first principal component.