# prefix, prompt, p tuning

## **🔹 Prompt Tuning**

**Prompt Tuning** learns a small set of *soft prompt vectors* that are prepended to the input text.

Only these prompt embeddings are trainable; the entire pre-trained model is frozen.

- You add **k learnable prompt tokens** before the input.
- These prompt embeddings guide the model toward the target task.
- Very parameter-efficient.

**Idea:**

> Instead of fine-tuning the whole model, only optimize a small prompt that steers the model.
> 

---

## **🔹 Prefix Tuning**

**Prefix Tuning** is similar, but instead of adding prompt tokens only to the input embeddings, it adds **learnable key/value prefixes to every Transformer layer**.

- Injects trainable vectors into the **attention mechanism**.
- No modification of word embeddings, but modifies internal attention states.
- More expressive than simple prompt tuning.

**Idea:**

> Learn a set of prefix key/value states that control the model at every layer.
> 

---

## **🔹 P-Tuning (v1)**

**P-Tuning** replaces manual hard prompts with **continuous embeddings** inserted into the input through a *prompt encoder* (e.g., an LSTM or MLP).

- Learns soft prompts, but uses an extra encoder for stability.
- Often used in small models where simple prompt tuning is unstable.
- Treats prompt tokens as trainable continuous parameters.

**Idea:**

> Map learnable virtual tokens through an encoder to generate continuous prompts.
> 

---

## **🔹 Summary Table**

| **Method** | **Where prompts are inserted?** | **Trainable params** | **Key benefit** |
| --- | --- | --- | --- |
| **Prompt Tuning** | Only at input embedding | Very small | Simple and efficient |
| **Prefix Tuning** | Key/value states at *every layer* | Small–medium | Much stronger control |
| **P-Tuning** | Soft prompts + encoder at input | Small | More stable, flexible prompts |