!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Positional Encoding

# Absolute positional encoding

1 2 3 4 ….

problems: Long sequence→ larger index → larger weights

# Transformer (PE)

![image.png](../../../../assets/notion/060-00.png)

![image.png](../../../../assets/notion/060-01.png)

![image.png](../../../../assets/notion/060-02.png)

the i is the dimension of embedding

larger i, smaller angular frequency, larger frequency

![image.png](../../../../assets/notion/060-03.png)

# RoPE

turn each 2 dim into a pair

define a position rotation

![image.png](../../../../assets/notion/060-04.png)

![image.png](../../../../assets/notion/060-05.png)

![image.png](../../../../assets/notion/060-06.png)

![image.png](../../../../assets/notion/060-07.png)

![image.png](../../../../assets/notion/060-08.png)

![image.png](../../../../assets/notion/060-09.png)

![image.png](../../../../assets/notion/060-10.png)

The inner product becomes sensitive to **relative position** p-q.
