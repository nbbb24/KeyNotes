# Attention masks: deciding which information is visible

An attention mask specifies permitted information flow. It is not dropout and does not generally remove a token from the tensor. Masking is essential because a training sequence contains future answers that must remain unavailable to an autoregressive prediction.

## Read the matrix by rows

Rows represent queries and columns represent keys. An additive causal mask for four positions is:

$$
M=\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}.
$$

Adding negative infinity before softmax makes forbidden weights zero. Position 2 can use positions 0, 1, and 2. The training label at position 2 is normally token 3 after shifting inputs and targets. The diagonal is therefore permitted without revealing the next-token label.

| Mask | What it prevents | Typical use |
| --- | --- | --- |
| Causal | Access to future keys | Autoregressive language modeling |
| Padding | Access to padded key positions | Mixed-length batches |
| Prefix | Output leakage while allowing bidirectional prompt access | Prefix language models |
| Segment | Access across independent packed documents | Efficient packed training |
| Local window | Access beyond a neighborhood | Bounded-context attention |

Masks can be combined. A causal mask alone does not prevent one packed training example from reading another example placed before it.

## A small example

With input `The cat sleeps`, a decoder is trained to predict the next token at each position. The state for `The` must not read `cat`, even though both tensors are computed in the same parallel training call. At inference, a single new query can attend to every valid cached past key; query and key lengths are then different, so blindly applying a square triangular mask is wrong.

```python
# runnable-example
length = 4
allowed = [[key <= query for key in range(length)] for query in range(length)]
assert allowed[0] == [True, False, False, False]
assert allowed[3] == [True, True, True, True]
```

## Implementation traps

Boolean-mask conventions vary between APIs. In PyTorch scaled dot-product attention, `True` means allowed; in several padding-mask interfaces it means excluded. Read the particular API contract. A fully masked row can create undefined softmax values. Handle empty/padded queries explicitly. Masking padding keys also does not automatically exclude padded prediction positions from the loss; loss masking is a separate operation.

**Exercise.** A three-token prompt is bidirectional and two response tokens are causal. Which keys may response position 3 read? Positions 0 through 3. Prompt rows may read positions 0 through 2, but never response positions.

Continue with [attention](attention.md) and [KV cache](../inference/kv-cache.md). Reference: [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
