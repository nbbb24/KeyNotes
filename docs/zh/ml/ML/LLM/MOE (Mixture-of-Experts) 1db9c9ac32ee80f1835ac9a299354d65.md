# MOE (Mixture-of-Experts)

[https://cameronrwolfe.substack.com/p/moe-llms](https://cameronrwolfe.substack.com/p/moe-llms)

# Layers

![image.png](MOE%20(Mixture-of-Experts)/image.png)

### **Sparse MoE Layer**

A Sparse Mixture of Experts (MoE) layer replaces the traditional feed-forward network (FFN) in a Transformer block.

Instead of one shared FFN, there are multiple **experts** (independent neural networks), and each token is processed by only a few of them.

This design greatly increases model capacity while keeping computation efficient, since only a subset of experts is activated per token.

### **Gating Network / Router**

The **router** (or gating network) decides **which experts** each token should be sent to.

It computes routing scores for all experts and selects the top-k ones (e.g., top-1 or top-2).

The selected experts process the token, and their outputs are combined using learned gating weights.

The router is trained jointly with the rest of the model and includes a **load-balancing loss** to ensure all experts are used evenly.

## Deepseek MOE

### **Fine grained expert segmentation**

each expert FFN is segmented into m experts

hidden dimension size is reduced to 1/m of their original size

### **Shared expert isolation**

`Ks` experts serve as shared experts

each token is deterministically routed to each shared expert

### Loss-free Load balancing (only for expert selection)

![image.png](MOE%20(Mixture-of-Experts)/image%201.png)

# Routing algorithms

simplest: linear transformation of size **N** to the token. Then apply a softmax to form distribution over experts

**routing collapse:** may repeatedly selecting the same experts, and they are trained more frequently and more favored

![image.png](MOE%20(Mixture-of-Experts)/image%202.png)

# Auxiliary Losses and Expert load Balancing

![image.png](MOE%20(Mixture-of-Experts)/image%203.png)

## **Importance score**

taking a sum of the probabilities assigned to each expert across all tokens in the batch

take the squared [coefficient of variation (CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the expert importance scores (similar expert→low CV)

![image.png](MOE%20(Mixture-of-Experts)/image%204.png)

<aside>
❓

Because experts of equal importance might: a few tokens with high prob/ lots of tokens with low prob

</aside>

## **Load Balancing**

capture both importance and load balancing

![image.png](MOE%20(Mixture-of-Experts)/image%205.png)

dot product of 2 fraction

# Router z-loss

![image.png](MOE%20(Mixture-of-Experts)/image%206.png)

constrain the size of logits, but not probabilities, so it is before softmax

# Expert Capacity

![image.png](MOE%20(Mixture-of-Experts)/image%207.png)

#