# Positional Encoding

# Absolute positional encoding

1 2 3 4 ….

problems: Long sequence→ larger index → larger weights

# Transformer (PE)

![image.png](Positional%20Encoding/image.png)

![image.png](Positional%20Encoding/image%201.png)

![image.png](Positional%20Encoding/image%202.png)

the i is the dimension of embedding

larger i, smaller angular frequency, larger frequency

![image.png](Positional%20Encoding/image%203.png)

# RoPE

turn each 2 dim into a pair

define a position rotation

![image.png](Positional%20Encoding/image%204.png)

![image.png](Positional%20Encoding/image%205.png)

![image.png](Positional%20Encoding/image%206.png)

![image.png](Positional%20Encoding/image%207.png)

![image.png](Positional%20Encoding/image%208.png)

![image.png](Positional%20Encoding/image%209.png)

![image.png](Positional%20Encoding/image%2010.png)

The inner product becomes sensitive to **relative position** p-q.