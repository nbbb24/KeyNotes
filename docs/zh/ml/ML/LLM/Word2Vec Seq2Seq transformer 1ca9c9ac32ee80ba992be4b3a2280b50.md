# Word2Vec/Seq2Seq/transformer

![image.png](Word2Vec%20Seq2Seq%20transformer/image.png)

# Word2Vec

![image.png](Word2Vec%20Seq2Seq%20transformer/image%201.png)

is an **embedding model**!

no activation function after hidden layer because the vector space is linear

sentence→token→token id→one hot→embedding

## skip-gram

predict nearby words of a given word

<aside>
📖

"The cat sat on the mat"

</aside>

![image.png](Word2Vec%20Seq2Seq%20transformer/image%202.png)

![image.png](Word2Vec%20Seq2Seq%20transformer/image%203.png)

![image.png](Word2Vec%20Seq2Seq%20transformer/image%204.png)

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

![image.png](Word2Vec%20Seq2Seq%20transformer/image%205.png)

Encoder → feature vector → Decoder

### Limitation

1. the dimension of feature vector is fixed.
2. For long input, it may loss important information
3. do not have attention

### Add attention

inner product of hidden state of each step in decoder with all the hidden state in encoder

# Transformer

## Encoder

![image.png](Word2Vec%20Seq2Seq%20transformer/image%206.png)

### Data preprocessing

the input each batch is [batch size * sentence length]

### Input Embedding

![image.png](Word2Vec%20Seq2Seq%20transformer/image%207.png)

![image.png](Word2Vec%20Seq2Seq%20transformer/image%208.png)

**Fully Connected Layer**: [#vocab * embedding length]

### Positional Encoding

$$
input_i=word'embedding_i+positional'encoding_i
$$

### Attention

![image.png](Word2Vec%20Seq2Seq%20transformer/image%209.png)

**relative position information** of each word in the sentence

$$
Q=XW^Q,K=XW^K,V=XW^V
$$

d_k is the dimension of Q/K query. This contains all the attention scores. L*dk

$$
z=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

### Multi-head Attention

![image.png](Word2Vec%20Seq2Seq%20transformer/image%2010.png)

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

![image.png](Word2Vec%20Seq2Seq%20transformer/image%2011.png)

### Masked Multi-Head Self-Attention

prevent see the later words

### Encoder-Decoder Cross Attention

Use Q in decoder with KV in last encoder block

### Linear+Softmax→output

![image.png](Word2Vec%20Seq2Seq%20transformer/image%2012.png)

q:[batch, seq_len, d_model]

Q:[batch, seq_len, d_model]

dv=dk=d_model/num_head

Qi=[batch, seq_len, num_head, dk] → [batch, num_head, seq_len, dk]

(QK)= [batch, num_head, seq_len, seq_len]

softmax()V = [batch, num_head, seq_len, dk]

concatenate = [batch, seq_len, d_model]