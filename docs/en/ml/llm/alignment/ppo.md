# PPO: improve a policy while moderating update size

Proximal Policy Optimization is a family of policy-gradient methods. The clipped surrogate is widely used because it reuses a batch for several updates while discouraging some overly large probability changes. Clipping is not a hard guarantee that the entire policy stays inside a trust region.

## Ratio and advantage

For a sampled state/action pair:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)},\qquad
L^{clip}=\mathbb E_t\left[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right].
$$

This is maximized; a minimization implementation negates it. The ratio compares the same action under current and old policies. Advantage $A_t$ estimates whether that action's outcome is better than the baseline at that state.

For $A=2$, $r=1.4$, and $\epsilon=0.2$, the objective uses $\min(2.8,2.4)=2.4$, removing the incentive from that term to increase the ratio further. For a negative advantage, the clipping behavior reverses. The ratio itself is not forcibly clamped in the resulting policy.

## Returns, critic, and GAE

An undiscounted return is the sum of future rewards. With discount $\gamma$:

$$
G_t=\sum_{k\ge0}\gamma^k r_{t+k}.
$$

A simple advantage is $G_t-V(s_t)$. Generalized advantage estimation uses temporal-difference residuals

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\quad
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},
$$

with correct termination/bootstrap handling. The critic is trained toward a return target, commonly with squared error. Its coefficient, entropy bonus, and clipping details are hyperparameters; 0.1 is not a universal critic weight.

## PPO in language models

Sample responses from the rollout policy, score them, compute values/advantages, then update policy and critic. A fixed-reference KL penalty can regularize behavior toward an SFT model. This is separate from the current/old-policy ratio and from a measured KL used for early stopping or update control.

![PPO: improve a policy while moderating update size, original figure 1](../../../../assets/notion/059-00.png)

![PPO: improve a policy while moderating update size, original figure 2](../../../../assets/notion/059-01.png)

![PPO: improve a policy while moderating update size, original figure 3](../../../../assets/notion/059-02.png)

![PPO: improve a policy while moderating update size, original figure 4](../../../../assets/notion/059-03.png)

![PPO: improve a policy while moderating update size, original figure 5](../../../../assets/notion/059-04.png)

![PPO: improve a policy while moderating update size, original figure 6](../../../../assets/notion/059-05.png)

![PPO: improve a policy while moderating update size, original figure 7](../../../../assets/notion/059-06.png)

![PPO: improve a policy while moderating update size, original figure 8](../../../../assets/notion/059-07.png)

![PPO: improve a policy while moderating update size, original figure 9](../../../../assets/notion/059-08.png)

![PPO: improve a policy while moderating update size, original figure 10](../../../../assets/notion/059-09.png)

![PPO: improve a policy while moderating update size, original figure 11](../../../../assets/notion/059-10.png)

**Figure guide.** Figures 1–2 introduce the new-to-old policy ratio and an advantage estimate. Figures 3–6 expand the clipped surrogate; the clipping interval changes the objective for sampled actions, not every probability in the policy. Figures 7–9 show KL-penalty variants and adaptive penalty strength; distinguish the rollout policy from the fixed reference policy in language-model alignment. Figure 10 compares unclipped, clipped, and KL-penalized objectives. Figure 11 summarizes the sampled policy-update loop. Several slides repeat notation with different subscripts, so track each denominator before transferring a formula.

*Read the original equations by tracking which policy appears in each denominator. The reference model controls drift from a baseline; the old policy identifies how the sampled actions were collected.*

## Practical failure modes

Large update epochs, poor advantage estimates, reward-scale changes, and stale rollout data can destabilize training. Monitor reward, independent task quality, KL, clipping fraction, entropy, value error, and response length. A low actor loss alone does not establish progress.

**Exercise.** Does clipping at 1.2 ensure every action probability changes by at most 20%? No. Shared parameters affect unsampled actions too, and clipping shapes an objective rather than enforcing a global constraint.

References: [PPO](https://arxiv.org/abs/1707.06347), [GAE](https://arxiv.org/abs/1506.02438). Review [importance sampling](importance-sampling.md).
