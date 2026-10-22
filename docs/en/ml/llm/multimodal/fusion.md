# Representation fusion: learn when modalities interact

Fusion combines information from different modalities for a prediction. A useful design must express the interactions the task needs without adding more complexity than the data can support. Always compare against strong unimodal models.

## Early and late fusion

![Original Representation Fusion figure 1](../../../../assets/notion/067-00.png)

*Fusion can combine already encoded vectors or heterogeneous raw inputs. Encoding choices determine which information remains available.*

![Original Representation Fusion figure 2](../../../../assets/notion/067-01.png)

*Early fusion combines features before prediction; late fusion combines modality-specific predictions. These describe where interaction occurs, not a universal ranking of methods.*

Concatenation followed by a linear layer is additive: $W[x_A;x_B]=W_Ax_A+W_Bx_B$. A nonlinear network after concatenation can express interactions. Thus, “early fusion” is not synonymous with “multiplicative,” and a single linear layer does not automatically provide rich cross-modal reasoning.

## Work through an interaction

Consider a review score predicted from smiling frequency $a$ and a binary indicator $b$ for professional-critic status:

$$
\hat y=w_0+w_1a+w_2b+w_3ab.
$$

The effect of smiling is $w_1$ when $b=0$ and $w_1+w_3$ when $b=1$. The product term lets one input change how another input matters. It is an association unless the experimental design supports a causal interpretation.

![Original Representation Fusion figure 3](../../../../assets/notion/067-02.png)
![Original Representation Fusion figure 4](../../../../assets/notion/067-03.png)
![Original Representation Fusion figure 5](../../../../assets/notion/067-04.png)
![Original Representation Fusion figure 6](../../../../assets/notion/067-05.png)

*These four slides add an intercept, individual effects, and then an interaction. A frequentist 95% confidence procedure covers the fixed parameter in 95% of repeated samples under its assumptions; a computed interval is not literally a 95% posterior probability for that parameter.*

## Additive, bilinear, and tensor fusion

![Original Representation Fusion figure 7](../../../../assets/notion/067-06.png)
![Original Representation Fusion figure 8](../../../../assets/notion/067-07.png)

*Additive fusion sums independent contributions. Bilinear fusion includes products between coordinates from different modalities.*

For vector features, $x_A^\top Wx_B$ captures pairwise products. Tensor fusion forms an outer product. Appending a constant 1 to each modality retains lower-order terms as well:

$$
[x_A;1]\otimes[x_B;1]\otimes[x_C;1].
$$

This contains unimodal, bimodal, and trimodal terms. With widths $d_A,d_B,d_C$, its size is $(d_A+1)(d_B+1)(d_C+1)$, which grows rapidly.

![Original Representation Fusion figure 9](../../../../assets/notion/067-08.png)

*The appended constants preserve lower-order interactions instead of keeping only the three-way product.*

Low-rank factorization approximates the fusion weight tensor with a sum of simpler factors. Computation can then project each modality and multiply smaller intermediate vectors without materializing the full outer product. Rank controls an expressiveness/cost tradeoff.

![Original Representation Fusion figure 10](../../../../assets/notion/067-09.png)
![Original Representation Fusion figure 11](../../../../assets/notion/067-10.png)
![Original Representation Fusion figure 12](../../../../assets/notion/067-11.png)

*Read these as factorization of the weight tensor followed by rearrangement of the computation, not as an assertion that every full tensor is exactly low rank.*

## Gates and modality-conditioned shifts

A gate can compute $g=\sigma(W[x_A;x_B]+b)$ and blend projected features as $g\odot u_A+(1-g)\odot u_B$. A modality-shifting design adds a gated correction from secondary modalities to a primary representation. For example, tone of voice can modify how a textual phrase is interpreted.

![Original Representation Fusion figure 13](../../../../assets/notion/067-12.png)
![Original Representation Fusion figure 14](../../../../assets/notion/067-13.png)

*Soft gates are differentiable. Hard selection needs an appropriate gradient estimator or separate optimization strategy.*

## Are non-additive interactions actually useful?

An additive approximation can reveal whether a complex model mostly uses independent modality effects. With expectations under a chosen product distribution, an additive projection has the form

$$
\tilde f(a,b)=\mathbb E_B[f(a,B)]+\mathbb E_A[f(A,b)]-\mathbb E_{A,B}[f(A,B)].
$$

The centering term avoids double-counting the global mean; the original slide omits that constant. Residual schemes can learn unimodal effects first, then explain remaining errors with pairwise and higher-order interactions.

![Original Representation Fusion figure 15](../../../../assets/notion/067-14.png)
![Original Representation Fusion figure 16](../../../../assets/notion/067-15.png)

*Compare the full model against its additive approximation using held-out predictions. A larger model is not evidence that interactions improved generalization.*

## Heterogeneity and optimization

Different modalities or tasks need not benefit from identical parameter sharing. Transfer experiments can inform which encoders or interaction modules to share. Such decisions require validation because negative transfer is possible.

![Original Representation Fusion figure 17](../../../../assets/notion/067-16.png)
![Original Representation Fusion figure 18](../../../../assets/notion/067-17.png)

*The diagrams use transfer relationships to group unimodal and cross-modal components.*

Modalities can overfit at different rates. Auxiliary unimodal predictors and training/validation trajectories help identify a dominant or poorly generalizing branch. Gradient blending is one proposed way to adjust contributions using estimated generalization behavior; it requires a defined estimation protocol rather than arbitrary loss weights.

![Original Representation Fusion figure 19](../../../../assets/notion/067-18.png)
![Original Representation Fusion figure 20](../../../../assets/notion/067-19.png)

*The last slides motivate balancing optimization using unimodal generalization signals. More modalities can hurt when complexity or shortcuts dominate.*

**Exercise.** For scalar binary inputs, can a linear additive model express XOR exactly? No. An interaction term or nonlinear fusion is needed. Then test whether the real task needs that interaction as well.

References: [Tensor Fusion Network](https://arxiv.org/abs/1707.07250), [Low-rank Multimodal Fusion](https://arxiv.org/abs/1806.00064), [What Makes Training Multi-Modal Classification Networks Hard?](https://arxiv.org/abs/1905.12681).
