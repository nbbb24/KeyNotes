!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Word2Vec/Seq2Seq/transformer

![image.png](../../../assets/notion/074-00.png)

# Word2Vec

![image.png](../../../assets/notion/074-01.png)

is an **embedding model**!

no activation function after hidden layer because the vector space is linear

sentence→token→token id→one hot→embedding

## skip-gram

predict nearby words of a given word

<aside>
📖

"The cat sat on the mat"

</aside>

![image.png](../../../assets/notion/074-02.png)

![image.png](../../../assets/notion/074-03.png)

![image.png](../../../assets/notion/074-04.png)

w_in is the embedding

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

# Seq2Seq

![image.png](../../../assets/notion/074-05.png)

Encoder → feature vector → Decoder

### Limitation

1. the dimension of feature vector is fixed.
2. For long input, it may loss important information
3. do not have attention

### Add attention

inner product of hidden state of each step in decoder with all the hidden state in encoder

# Transformer

## Encoder

![image.png](../../../assets/notion/074-06.png)

### Data preprocessing

the input each batch is [batch size * sentence length]

### Input Embedding

![image.png](../../../assets/notion/074-07.png)

![image.png](../../../assets/notion/074-08.png)

**Fully Connected Layer**: [#vocab * embedding length]

### Positional Encoding

$$
input_i=word'embedding_i+positional'encoding_i
$$

### Attention

![image.png](../../../assets/notion/074-09.png)

**relative position information** of each word in the sentence

$$
Q=XW^Q,K=XW^K,V=XW^V
$$

d_k is the dimension of Q/K query. This contains all the attention scores. L*dk

$$
z=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

### Multi-head Attention

![image.png](../../../assets/notion/074-10.png)

split the embedding into multiple “heads”

d_model :512→8*64, to make sure the dimension consistent

another **Linear Projection** after concatenate: to learn the features

### LayerNorm + Residual Connection

$$
LayerNorm(x+Sublayer(x))
$$

### Feed-Forward Network

$$
FFN(x) = max(0, xW1 + b1)W2 + b2
$$

512→2048→512

## Decoder

![image.png](../../../assets/notion/074-11.png)

### Masked Multi-Head Self-Attention

prevent see the later words

### Encoder-Decoder Cross Attention

Use Q in decoder with KV in last encoder block

### Linear+Softmax→output

![image.png](../../../assets/notion/074-12.png)

q:[batch, seq_len, d_model]

Q:[batch, seq_len, d_model]

dv=dk=d_model/num_head

Qi=[batch, seq_len, num_head, dk] → [batch, num_head, seq_len, dk]

(QK)= [batch, num_head, seq_len, seq_len]

softmax()V = [batch, num_head, seq_len, dk]

concatenate = [batch, seq_len, d_model]
