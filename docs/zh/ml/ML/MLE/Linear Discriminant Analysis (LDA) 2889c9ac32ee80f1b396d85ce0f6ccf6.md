# Linear Discriminant Analysis (LDA)

LDA finds a **projection vector**  such that:

- The **between-class variance** is maximized.
- The **within-class variance** is minimized.

$$
J(w)=wTSWwwTSBw
$$

where

- $S_B$ = *between-class scatter matrix* → how far apart the class means are.
- $S_W$ = *within-class scatter matrix* → how spread out points are inside each class.

### Steps of LDA

1. **Compute the mean of each class** and the **overall mean**.
2. **Compute SWS_WSW** (sum of covariance within each class).
3. **Compute SBS_BSB** (scatter due to class means’ difference).
4. Solve the **generalized eigenvalue problem**:$S_W^{−1}S_Bw=λw$
5. Select the top eigenvectors (those with largest eigenvalues). → These are the directions that maximize separability.
6. Project your data onto these directions.