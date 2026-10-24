# 🐼 Pandas：保留每一行的意义

Pandas 提供了带有标签的一维 `Series` 和二维 `DataFrame` 对象。它的价值不仅在于方便的语法：标签、类型、连接和分组让你能够表达数据转换。危险在于一个转换可能成功运行，但改变了某一行的意义。

## 📦 创建并检查一个小数据集

```python
# runnable-example
import pandas as pd

df = pd.DataFrame({
    "name": ["Tom", "Jerry", "Ada"],
    "department": ["R&D", "R&D", "HR"],
    "age": [20, 22, None],
    "salary": [5000, 7000, 6000],
    "date": ["2025-09-01", "2025-09-02", "2025-10-01"],
})
assert df.shape == (3, 5)
assert df["salary"].mean() == 6000
print(df.head())
```

使用 `tail(3)`、`columns`、`info()` 和 `describe()` 来检查形状、标签、数据类型和分布。一个看起来像数字的标识符可能需要字符串存储以保留前导零。`read_csv`/`to_csv(index=False)` 处理 CSV；`read_excel`/`to_excel(index=False)` 处理带有适当引擎的电子表格。指定重要的数据类型和日期解释，而不是完全依赖推断。

## 📍 标签不是位置

`df['age']` 返回一个 Series；`df[['name', 'age']]` 返回一个 DataFrame。`iloc[0:2]` 选择位置 0 和 1。在一个有序的整数索引上，`loc[0:2]` 包括端点标签，并且因此可以选择三行。标签不需要是连续的或唯一的。

使用 `loc` 一次完成筛选和赋值：`df.loc[df['age'] > 20, 'salary'] = 7500`。避免链式赋值，如 `df[df['age'] > 20]['salary'] = 7500`；它不能可靠地表达对原始对象的修改。通过标签对 Series 进行对齐也会与位置赋值重新排序值。请参阅 [索引指南](https://pandas.pydata.org/docs/user_guide/indexing.html)。

## 📦 按目的清理数据

| 任务 | 示例 | 应采取的决策 |
| --- | --- | --- |
| 缺失值 | `df.isna().sum()` | 缺失、零和未知是不同的 |
| 删除/填充 | `df.dropna(subset=['salary'])`; `df['age'].fillna(21)` | 哪个变量和哪种策略？ |
| 重命名 | `df.rename(columns={'name': 'full_name'})` | 保持列模式清晰 |
| 转换 | `df['age'].astype('Float64')` | 可空与不可空的数据类型 |
| 重复项 | `df.duplicated(subset=['name'])` | 名称真的是实体键吗？ |
| 排序／排名 | `df.sort_values('salary')`; `df['salary'].rank(method='dense')` | 明确并列值与缺失值的处理方式 |

不要用 `age` 列的平均值填充每一列。只在训练集上计算填补缺失值所需的统计量。高于第 95 百分位数的工资只是高值，不一定是错误；在删除之前检查域约束。`drop_duplicates` 可能会丢弃真正的重复事件，如果键是错误的。

## ✅ 聚合、组合和验证基数

`df.groupby('department')['salary'].mean()` 将行减少到每个组一个值。命名聚合可以使输出意义明确：

```python
summary = df.groupby("department", dropna=False).agg(
    mean_salary=("salary", "mean"),
    max_age=("age", "max"),
    rows=("name", "size"),
)
```

`transform` 返回与原始行对齐的结果；`agg` 进行汇总。决定是否应该使用 `dropna=False` 来形成缺失分组键的组。多个键，如部门和角色，创建更细的分组粒度。请参阅 [分组语义](https://pandas.pydata.org/docs/user_guide/groupby.html)。

`merge` 基于键进行组合，`join` 通常使用索引进行组合，而 `concat` 在轴上堆叠。连接不会像 SQL `UNION` 那样移除重复项；行堆叠更接近 `UNION ALL`，带有标签对齐的细节。在附加唯一查找表时，请使用显式键和 `validate='many_to_one'`。两边都有重复键会增加行数。Pandas 可以匹配空连接键彼此，而普通 SQL 等值连接不这样做；请务必检查这一点。请参阅 [`merge`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html)。

## 🏗️ 日期、表达式和层次索引

使用 `pd.to_datetime` 解析；在你同时检查新出现的缺失值时，才使用 `errors='coerce'`。对于时间戳，定义时区处理。在 `set_index('date')` 之后，`resample('MS')['salary'].sum()` 按月开始分组；在当前 pandas 中使用 `ME` 表示月末。`df.index.month` 提取月份数字，但会丢失年份，因此按其分组可能会混合年份。请参阅 [时间序列操作](https://pandas.pydata.org/docs/user_guide/timeseries.html)。

`query('age > 20 and salary < 8000')` 过滤和 `eval('annual = salary * 12')` 计算表达式。将这些视为表达语言，而不是存放不受信任用户输入的地方。从 `set_index(['department', 'name']).sort_index()` 创建的 MultiIndex 支持元组查找，如 `.loc[('HR', 'Ada')]`；它不保证键的唯一性。

练习：附加一个部门查找表，断言行数未改变，计算每月总和，并证明它们的总和在你的缺失值策略下等于原始总和。与 [SQL](sql.md) 进行比较，然后回顾 [数据泄漏](mle/data-cleaning.md)。
