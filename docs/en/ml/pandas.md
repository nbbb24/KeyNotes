# Pandas: preserve the meaning of every row

Pandas supplies labeled one-dimensional `Series` and two-dimensional `DataFrame` objects. Its value is not just convenient syntax: labels, types, joins, and grouping let you express a data transformation. The danger is that a transformation can run successfully while changing what one row means.

## Create and inspect a small dataset

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

Use `tail(3)`, `columns`, `info()`, and `describe()` to inspect shape, labels, dtypes, and distributions. A numeric-looking identifier may need string storage to preserve leading zeros. `read_csv`/`to_csv(index=False)` handle CSV; `read_excel`/`to_excel(index=False)` handle spreadsheets with an appropriate engine installed. Specify important dtypes and date interpretation instead of relying entirely on inference.

## Labels are not positions

`df['age']` returns a Series; `df[['name', 'age']]` returns a DataFrame. `iloc[0:2]` selects positions 0 and 1. On an ordered integer index, `loc[0:2]` includes both endpoint labels and can therefore select three rows. Labels need not be consecutive or unique.

Filter and assign in one operation: `df.loc[df['age'] > 20, 'salary'] = 7500`. Avoid chained assignment such as `df[df['age'] > 20]['salary'] = 7500`; it does not reliably express a mutation of the original object. Aligning a Series by label can also reorder values compared with positional assignment. See the [indexing guide](https://pandas.pydata.org/docs/user_guide/indexing.html).

## Clean with a reason

| Task | Example | Decision to make |
| --- | --- | --- |
| Missingness | `df.isna().sum()` | Missing, zero, and unknown are different |
| Remove/fill | `df.dropna(subset=['salary'])`; `df['age'].fillna(21)` | Which variable and which policy? |
| Rename | `df.rename(columns={'name': 'full_name'})` | Preserve a clear schema |
| Convert | `df['age'].astype('Float64')` | Nullable versus non-nullable dtype |
| Duplicates | `df.duplicated(subset=['name'])` | Is name really the entity key? |
| Sort/rank | `df.sort_values('salary')`; `df['salary'].rank(method='dense')` | Specify ties and missing-value behavior |

Do not fill every column with the mean of `age`. Fit imputation statistics on the training split only. A salary above the 95th percentile is a high observation, not automatically an error; inspect domain constraints before removal. `drop_duplicates` can discard genuine repeated events if the key is wrong.

## Aggregate, combine, and validate cardinality

`df.groupby('department')['salary'].mean()` reduces rows to one value per group. Named aggregation can make output meaning explicit:

```python
summary = df.groupby("department", dropna=False).agg(
    mean_salary=("salary", "mean"),
    max_age=("age", "max"),
    rows=("name", "size"),
)
```

`transform` returns results aligned with original rows; `agg` summarizes. Decide whether missing grouping keys should form a group using `dropna=False`. Multiple keys, such as department and role, create more granular groups. See [grouping semantics](https://pandas.pydata.org/docs/user_guide/groupby.html).

`merge` combines on keys, `join` commonly combines using indexes, and `concat` stacks along an axis. Concatenation does not remove duplicates like SQL `UNION`; row stacking is closer to `UNION ALL`, with label-alignment details. Use explicit keys and `validate='many_to_one'` when attaching a unique lookup table. Duplicate keys on both sides multiply rows. Pandas can match null join keys to one another, unlike ordinary SQL equality joins; check this deliberately. See [`merge`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html).

## Dates, expressions, and hierarchical indexes

Parse with `pd.to_datetime`; use `errors='coerce'` only when you also inspect newly missing values. For timestamps, define timezone handling. After `set_index('date')`, `resample('MS')['salary'].sum()` groups by month start; use `ME` for month end in current pandas. `df.index.month` extracts month numbers but loses the year, so grouping on it can mix years. See [time-series operations](https://pandas.pydata.org/docs/user_guide/timeseries.html).

`query('age > 20 and salary < 8000')` filters and `eval('annual = salary * 12')` evaluates an expression. Treat these as expression languages, not a place for untrusted user input. A MultiIndex from `set_index(['department', 'name']).sort_index()` supports tuple lookup such as `.loc[('HR', 'Ada')]`; it does not guarantee key uniqueness.

Exercise: attach a department lookup, assert row count is unchanged, compute monthly totals, and prove their sum equals the original total under your missing-value policy. Compare with [SQL](sql.md), then review [data leakage](mle/data-cleaning.md).
