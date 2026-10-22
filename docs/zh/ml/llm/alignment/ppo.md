!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# PPO (Proximal Policy Optimization)

[https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)

https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a

PPO updates model to maximize reward, but constrains so that the new policy doesn’t diverge too far

![image.png](../../../../assets/notion/059-00.png)

ratio of how much the new model deviate from the old

KL: how much the new differ from the old in that state

![image.png](../../../../assets/notion/059-01.png)

At is the advantage

---

|
Function
 |
Models Used
 |
Explanation
 |
| --- | --- | --- |
| 1️⃣ PPO Ratio
 | Actor & Old Policy | Measures how much the new policy differs from the previous one. Used for stable updates through clipping. |
| 2️⃣ KL Penalty
 | Actor & Reference Model (SFT)
 | Adds a penalty when the new policy drifts too far from the original SFT policy, keeping behavior aligned. |
| 3️⃣ Reward Calculation
 | Reward Model + Reference Model
 | Combines the human preference score from the reward model with a KL penalty to form the final reward. |
| 4️⃣ Advantage Calculation
 | Actor & Critic
 | Compares the actual reward to the critic’s predicted value to guide policy improvement. |
| 5️⃣ Critic Training
 | Critic
 | Trains the critic to accurately predict expected rewards, providing a stable baseline for the actor. |

---

# PPO

### surrogate loss (actor loss)

![image.png](../../../../assets/notion/059-02.png)

### Advantage Function

![image.png](../../../../assets/notion/059-03.png)

each token is regarded as 1 action, add from position i to the last

$$
\text{return}[i] = \text{reward}[i] + \text{reward}[i+1] + \dots + \text{reward}[N]
$$

so for each token, we have advantage:

$$
a[i] = \text{return}[i] - \text{values}[i]
$$

---

### **Probability Ratio in PPO Clip**

how likely the new policy is compared to the older one

If the ratio exceeds 1, the new policy is more likely.

![image.png](../../../../assets/notion/059-04.png)

---

### Clipping

![image.png](../../../../assets/notion/059-05.png)

ensure that drastic policy updates do not take place

---

### Penalty

![image.png](../../../../assets/notion/059-06.png)

![image.png](../../../../assets/notion/059-07.png)

![image.png](../../../../assets/notion/059-08.png)

### Variants

![image.png](../../../../assets/notion/059-09.png)

constrains the ratio between 1-e,1+e

---

### Critic loss

$$
\mathcal{L}_{\text{critic}}
= \frac{1}{2N}
\sum_{i=1}^{N}
\big( \text{values}[i] - \text{returns}[i] \big)^2
$$

---

### Final loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + 0.1 \times \mathcal{L}_{\text{critic}}
$$

---

# Policy update

![image.png](../../../../assets/notion/059-10.png)

### PPO in LLMs

ot: token generated at time step t

o<t: tokens generated before time step t

# Process

**SFT**

Sample multiple responses for each prompt, then rank

train reward model

### PPO

sample response from current policy

get reward from reward model

estimate advantage=current reward-baseline

compute PPO loss
