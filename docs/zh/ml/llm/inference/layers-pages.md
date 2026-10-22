!!! info "中文版正在逐篇更新"

    英文优化版已完成。本页暂时保留原笔记，中文版优化正文正在编写。

# Layer&Page

# 计算Kernel

```
layer_first layout:

layer 0:
  page 0, page 1, page 2, page 3

layer 1:
  page 0, page 1, page 2, page 3

layer 2:
  page 0, page 1, page 2, page 3
```

因为对于某一层来说，它的数据更集中，attention kernel 访问更自然。

# 数据存储

```
page_first layout:

page 0:
  layer 0, layer 1, layer 2, layer 3

page 1:
  layer 0, layer 1, layer 2, layer 3

page 2:
  layer 0, layer 1, layer 2, layer 3
```

这样 `page 0` 的完整数据是连续的，直接写入 L3 / 从 L3 读取会更方便。

# 不同Layout

## layer_first

先按 layer 排，再按 token/page 排

```jsx
[ K/V, Layer, Token, Head, Dim ]
```

可以理解成：

```
K:
  Layer 0:
    P0: t0 t1 t2 t3
    P1: t4 t5 t6 t7

  Layer 1:
    P0: t0 t1 t2 t3
    P1: t4 t5 t6 t7

V:
  Layer 0:
    P0: t0 t1 t2 t3
    P1: t4 t5 t6 t7

  Layer 1:
    P0: t0 t1 t2 t3
    P1: t4 t5 t6 t7
```

## page_first

先按 page/token 排，再按 layer 排

```
[ K/V, Token, Layer, Head, Dim ]
```

```jsx
K P0:
	token0-Layer0, token0-Layer1,
	token1-Layer0, token1-Layer1,
	token2-Layer0, token2-Layer1,
	token3-Layer0, token3-Layer1

V P0:
	token0-Layer0, token0-Layer1,
	token1-Layer0, token1-Layer1,
	token2-Layer0, token2-Layer1,
	token3-Layer0, token3-Layer1
```

## page_first_direct

`page_first_direct` 也是 page-first，但它把 **page 内部的 layer-token 排列方式**改了一下。

```
[ K/V, Page, Layer, Token-in-page, Head, Dim ]
```

展开看：

```
K P0:
  L0: t0 t1 t2 t3
  L1: t0 t1 t2 t3

V P0:
  L0: t0 t1 t2 t3
  L1: t0 t1 t2 t3
```

## page_first_kv_split

这个和前面几个不太一样。它不是单纯换 `page/layer/token` 的顺序，而是进一步把 **K-like 部分和 V-like/rope 部分拆开存**。
比如 DeepSeek MLA 类模型里，KV cache 不一定是传统完整 K/V head 形式，而可能是：

```
kv_lora_rank 部分
qk_rope_head_dim 部分
```

在 SGLang 当前代码里，`page_first_kv_split` 对 MLA host pool 的处理是：

```
base_dims = [page_num, layer_num, page_size, 1]

k_buffer shape = [page_num, layer_num, page_size, 1, kv_lora_rank]
v_buffer shape = [page_num, layer_num, page_size, 1, qk_rope_head_dim]
```

也就是把 `kv_lora_rank` 和 `qk_rope_head_dim` 分开放到两个 buffer。

```jsx
page_size = 4
num_layers = 2
kv_lora_rank = 512
qk_rope_head_dim = 64
```

那么 `page_first_kv_split` 可以理解成：

```
page_first_direct:

P0
 ├── L0
 │    └── t0 t1 t2 t3: full KV vector
 └── L1
      └── t0 t1 t2 t3: full KV vector

page_first_kv_split:

P0
 ├── k_buffer
 │    ├── L0: t0 t1 t2 t3
 │    └── L1: t0 t1 t2 t3
 │
 └── v_buffer
      ├── L0: t0 t1 t2 t3
      └── L1: t0 t1 t2 t3
```
