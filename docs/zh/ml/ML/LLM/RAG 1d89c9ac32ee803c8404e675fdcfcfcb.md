# RAG

## 1. 整体流程总览（E2E）

1. **数据摄取与预处理**：抓取/同步 → 清洗/去重 → 分块（chunking）→ 元数据与权限。
2. **嵌入（Embedding）**：文本/表格/图片转向量；可做 L2 归一化。
3. **索引（Indexing）**：构建稠密向量索引（FAISS/HNSW/ScaNN）、稀疏倒排（BM25）、或混合索引。
4. **查询理解/改写**：Query Rewrite / Decomposition / HyDE（假设性文档扩展）。
5. **初始检索（Recall / Retrieve）**：从索引中召回 top-k 候选片段。
6. **重排序（Rerank）**：Cross-Encoder 或 Late Interaction（如 ColBERT）精排，得到最终候选集。
7. **上下文组装（Context Assembly）**：在 token 预算内做去重、聚类、多样性采样与拼接；附带来源/引用。
8. **生成（Generation）**：LLM 在检索证据上回答；结构化输出（JSON/表格）与引用标注。
9. **后处理与安全**：引用去重、敏感信息过滤、提示注入防御。
10. **评估与监控**：召回率/精度、MRR/nDCG、事实一致性（faithfulness）、延迟与成本、A/B 实验；反馈闭环。

## 2. 数据摄取与预处理

- **数据源**：文档库（PDF/HTML/Markdown）、Wiki/Confluence、数据库、API、日志等。
- **清洗**：去脚注/导航、模板与噪声；正则/HTML 解析；语言检测与分段。
- **分块（Chunking）**：
    - 固定窗口（如 400–800 tokens）+ 重叠（如 10–20%）避免跨段语义漂移。
    - 语义分块（按标题/段落/句子）与层级分块（Section→Paragraph→Sentence）。
    - **元数据**：title、url、段内位置、时间、作者、ACL 标签，用于检索过滤与重排特征。
- **去重与版本**：指纹/SimHash；保留时间维度与版本链，便于时序检索。

## 3.嵌入（Embedding）

- **模型选择**：通用 vs. 领域（法律/医疗/代码/多语）；维度（例如 384/768/1024）。
- **归一化与相似度**：常用余弦相似度或点积；向量可作 L2 归一化。

$$
cos⁡(θ)=q⋅d∥q∥ ∥d∥\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{d}\rVert}cos(θ)=∥q∥∥d∥q⋅d
$$

$$
Dot(q,d)=∑i=1nqidi\text{Dot}(\mathbf{q},\mathbf{d}) = \sum_{i=1}^{n} q_i d_iDot(q,d)=∑i=1nqidi
$$

- **稀疏表示**（可混合）：TF–IDF/BM25；数值/实体的精确匹配友好。

$$
TF-IDF(w,d)=tf(w,d)⋅log⁡Ndf(w)\text{TF-IDF}(w,d) = \mathrm{tf}(w,d)\cdot\log\frac{N}{\mathrm{df}(w)}TF-IDF(w,d)=tf(w,d)⋅logdf(w)N
$$

---

## 4. 索引（Indexing）

- **稠密索引**：
    - FAISS：Flat / IVF / (H)NSW / PQ / OPQ；参数如 `nlist`、`nprobe`。
    - HNSW：`M`（图度）、`efConstruction`、`efSearch` 影响召回率/延迟。
- **稀疏索引（倒排）**：BM25，适合关键词与长尾符号（ID/代码/数值）。

---

## 5. 查询理解与改写（Query Understanding）

- **改写**：拼写纠错、别名/同义词扩展、实体规范化（Normalization）。
- **拆分/分解**：复杂问题分解为子查询（Sub-Queries）。
- **HyDE**：先生成“假设性答案/文档”，再向量化检索以提升召回。
- **检索过滤**：时间范围、作者、权限、标签、语言。

---

## 6. 初始检索（Recall / Retrieve）

### （1）基于关键词的检索器

- **倒排索引 / BM25**
    - 最传统的方法，依赖关键词匹配。
    - 优点：速度快、可解释性强。
    - 缺点：无法处理语义相似但用词不同的情况。

### （2）基于向量的检索器（Dense Retriever）

- **原理**：把 query 和文档都编码成向量，用向量相似度（余弦相似度、点积）来检索。
- 常用模型：DPR（Dense Passage Retrieval）、sentence-BERT、E5、OpenAI embeddings 等。
- 优点：能捕捉语义相似度，不依赖关键词。
- 缺点：需要向量化+ANN索引，存储和检索成本更高。

### （3）混合检索器（Hybrid Retriever）

- 结合 **关键词检索** 和 **语义检索**，取两者优点。
- 常见策略：先用 BM25 召回，再用向量检索 rerank；或者直接加权融合分数。

### （4）重排序（Re-ranker）

- 检索器有时分两阶段：
    1. **初步召回**（可能取 top-100 文档）
    2. **精排（re-rank）**：用更复杂的模型（如 cross-encoder）对候选结果重新打分，选最相关的 top-k。
- 好处：减少初筛时的漏召，同时保证最终质量。

---

## 7. 重排序（Reranking）

- **Cross-Encoder 精排**：将 $(q, d)$ 拼接送入 BERT/LLM，输出相关性分数。
- **Pairwise/Listwise 训练目标**：

$$
Lpairwise=max⁡(0,  m−s(q,d+)+s(q,d−))
$$

$$
{\text{Lsoftmax}} = -\log\frac{\exp(s(q,d^{+}))}{\sum_{d\in C} \exp(s(q,d))}
$$

- **特征融合**：BM25 分数、位置/层级、时间、新鲜度、点击率、来源可信度等作为附加特征。

---

## 8. 上下文组装（Context Assembly）

- **目标**：在 token 预算 $T$ 内，最大化有用信息与多样性，避免冗余与冲突。
- **0-shot 选择**：按精排分数排序，去重后逐条放入直到超出 $T$ 停止。
- **多样性与背压**：按来源/主题聚类（Maximal Marginal Relevance, 时间去偏）。

---

## 9. 生成（Generation）

- **提示模板**：System 约束 + 指令 + 检索证据 + 输出格式（JSON/表格/Markdown）。
- **引用**：在段落或要点级别附上 `[出处: 标题/链接/时间]`。
- **防幻觉**：在 Prompt 中显式要求“仅依据证据回答；若证据不足则说明未知”。
- **结构化输出**：使用模式示例（Schema + Few-shot）让 LLM 严格对齐。

---

## 10.评判指标

### 1. Recall@k（召回率）

**定义**：在前 *k* 个检索结果里，是否包含至少一个正确答案。

**计算方法**：

- 每个 query 有一组“相关文档”标签（人工标注或任务答案对应的文档）。
- 看前 *k* 个检索结果里，命中多少相关文档。
- Recall@k = 命中的相关文档数 ÷ 总相关文档数。

## 2. Precision@k（准确率）

**定义**：前 *k* 个检索结果中，有多少比例是相关的。

**计算方法**：

Precision@k = 相关文档数 ÷ k

## 3. MRR（Mean Reciprocal Rank）

**定义**：看第一个正确文档在结果列表里的排名，越靠前越好。

**计算方法**：

- Reciprocal Rank = 1 ÷ (第一个正确答案的排名位置)
- MRR = 所有 query 的 Reciprocal Rank 的平均值。