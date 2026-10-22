!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# GRPO (Group Relative Policy Optimization)

![image.png](../../../../assets/notion/036-00.png)

- **Eliminates the Value Model** to reduce memory consumption & computational burden
in PPO, advantage At is computed using a GAE, which is based on expected future rewards (`r>=t`) and a learned value function `Vψ`
- **Uses a Group-based Reward comparison** instead of a learned value function.
Multiple outputs, sampled from the old policy for a given question, are fed to a reward model and the reward across all sampled outputs are averaged to calculate the baseline reward.

---

| **Mechanism** | **Effect on Diversity** |
| --- | --- |
| Group-based relative rewards | Allows multiple reasoning paths to succeed |
| No critic model | Removes global bias and template overfitting |
| Per-group normalization | Encourages local exploration |

| **Method** | **Analogy** | **Outcome** |
| --- | --- | --- |
| PPO | One teacher grading by a fixed answer key | All students learn the same solution pattern |
| **GRPO** | Group competitions — best performers in each group are rewarded | Different groups find different valid ways to solve the problem |

---

<aside>

Question: q

Output” o1 (score 8), o2 (score 6)

Average reward (R): 7

Advantage: o1 (8-7)=1 o2 (6-7)=-1

</aside>

### Objective Function

![image.png](../../../../assets/notion/036-01.png)

![image.png](../../../../assets/notion/036-02.png)

First average the surrogate loss across all tokens in a sampled output. Then average across all sampled outputs, so every output contributes equally regardless of length.

### KL Penalty

![image.png](../../../../assets/notion/036-03.png)

### Outcome Supervision

- **When reward is given:** Only **once at the end** of the full output.
- **Reward signal:** A single score per output (e.g., factual accuracy + style).
- **Advantage calculation:** Every token in that output gets the same advantage (normalized reward).
- **Focus:** Encourages the model to produce correct **final answers**, regardless of how it reasons.

### Process Supervision

- **When reward is given:** At the end of **each reasoning step**.
- **Reward signal:** A sequence of scores per output (one per step).
- **Advantage calculation:** For each token, the advantage is the sum (or accumulation) of normalized step-level rewards.
- **Focus:** Encourages the model to generate **correct and logical reasoning chains**, not just the final result.

### GRPO for Iterative RL

During reinforcement learning, the original reward model can become outdated as the policy improves.

To address this, new outputs are sampled and scored to create fresh training data.

The reward model is then retrained on this new data, while mixing in about 10% of past data to prevent catastrophic forgetting.

### Rewarding System

rule based reward system is employed to train deepseek r1 zero

- **Accuracy rewards** — This model evaluates whether or not the response is correct. These rewards in some cases are rule-based.
- **Format rewards** — A format reward enforces the model to put its thinking process between `<think>` and `</think>` tags
